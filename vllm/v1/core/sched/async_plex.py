# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM's PLEX binding: an `EnginePort` and the scheduler-facing wrapper.

The v0.7 wire format, request identity, plan freshness, feedback delivery and
the fallback decision all live in `plex.engine`, which ships with the contract.
What is left here is the part only vLLM knows: where the scheduler keeps its
queues and what its per-request counters are called.
"""

from __future__ import annotations

import itertools
import time
from typing import TYPE_CHECKING, Any

from plex.engine import (
    NO_SIGNALS,
    AdmissionCapacity,
    CacheCapacity,
    CacheDecision,
    PolicyController,
    RequestSignals,
    ScheduleCapacity,
    SchedulePlan,
)

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request

logger = init_logger(__name__)

DEFAULT_PRINCIPAL = "vllm-default"

# Requests parked on something other than capacity cannot be selected, so
# offering them to the policy would ask it to rank what it cannot start.
UNSELECTABLE = (
    RequestStatus.WAITING_FOR_REMOTE_KVS,
    RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
    RequestStatus.WAITING_FOR_STREAMING_REQ,
)

# `plex.engine` owns the plan and decision shapes; these aliases keep the
# scheduler's imports and type hints unchanged.
PlexSchedulePlan = SchedulePlan
PlexCacheDecision = CacheDecision


class VllmRequest:
    """One `Request`, answering the questions `plex.engine` asks."""

    __slots__ = ("_request", "_scheduler", "_signals")

    def __init__(
        self,
        request: Request,
        scheduler: Scheduler,
        signals: RequestSignals | None = None,
    ) -> None:
        self._request = request
        self._scheduler = scheduler
        self._signals = signals if signals is not None else NO_SIGNALS

    @property
    def engine_id(self) -> str:
        return self._request.request_id

    def plex_config(self) -> dict[str, Any] | None:
        params = self._request.sampling_params
        if params is None or params.extra_args is None:
            return None
        return params.extra_args.get("plex")

    def body(self) -> dict[str, Any]:
        request = self._request
        return {
            "prompt_token_ids": list(request.prompt_token_ids),
            "max_tokens": request.max_tokens,
            "priority": request.priority,
        }

    def facts(self) -> dict[str, Any]:
        request = self._request
        signals = self._signals
        waiting_ms = max(int((time.time() - request.arrival_time) * 1000), 0)
        running = request.status == RequestStatus.RUNNING
        prompt_tokens = request.num_prompt_tokens
        hit = min(signals.lpm_hit_tokens, prompt_tokens)
        uncached = prompt_tokens - hit
        return {
            "queue_member": not running,
            "scheduler_state": "running" if running else "waiting",
            "attained_service": request.num_computed_tokens,
            "service_tokens": request.num_computed_tokens,
            "dispatch_input_tokens": max(
                prompt_tokens - request.num_computed_tokens, 0
            ),
            "generated_tokens": len(request.output_token_ids),
            "preempted": request.num_preemptions > 0,
            "preemptions": request.num_preemptions,
            "waiting_ms": waiting_ms,
            "call_wait_us": waiting_ms * 1000,
            "arrival_ms": int(request.arrival_time * 1000),
            "arrival_seq": signals.arrival_seq,
            "cache_ready": request.num_computed_tokens > 0,
            "cached_tokens": request.num_computed_tokens,
            "prompt_tokens": prompt_tokens,
            "computation_length": prompt_tokens + len(request.output_token_ids),
            "lpm_hit_tokens": hit,
            "uncached_tokens": uncached,
            # What is left to prefill after the cache, not after progress: a
            # policy weighing admission wants the work the request will cost.
            "new_prefill_tokens": max(
                uncached - max(request.num_computed_tokens - hit, 0), 0
            ),
            "prefix_hit_ratio_ppm": (
                hit * 1_000_000 // prompt_tokens if prompt_tokens else 0
            ),
            "current_queue_ms": 0 if running else waiting_ms,
            "kv_overloaded": under_pressure(self._scheduler),
            "now_ms": int(time.time() * 1000),
        }

    def cache_facts(self) -> dict[str, Any]:
        request = self._request
        return {
            "actual_size_bytes": self.actual_size_bytes(),
            "leaf": True,
            "cached_length": request.num_computed_tokens,
            "computation_length": (
                request.num_prompt_tokens + len(request.output_token_ids)
            ),
            "last_access_ms": self._signals.last_access_ms,
            "state_kind": request.status.name.lower(),
            "tier": "gpu",
        }

    def token_budget(self) -> int:
        request = self._request
        scheduler = self._scheduler
        pending = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
            if request.status == RequestStatus.RUNNING
            else request.num_tokens - request.num_computed_tokens
        )
        return max(
            min(
                pending,
                scheduler.max_model_len
                - request.num_computed_tokens
                - scheduler.num_sampled_tokens_per_step,
                scheduler.max_num_scheduled_tokens,
            ),
            0,
        )

    def size_bytes(self) -> int:
        """One reclaim unit.

        vLLM frees whole pages, so what the policy can act on is "how many
        requests to drop", not "how many bytes". Reporting a uniform unit
        makes the byte budget a request count; the real size is a fact.
        """
        return reclaim_unit(self._scheduler)

    def actual_size_bytes(self) -> int:
        scheduler = self._scheduler
        blocks = scheduler.kv_cache_manager.get_blocks(
            self._request.request_id
        ).blocks
        return sum(
            len(group_blocks) * group.kv_cache_spec.page_size_bytes
            for group_blocks, group in zip(
                blocks, scheduler.kv_cache_config.kv_cache_groups
            )
        )

    def reload_cost(self) -> int:
        return self._request.num_computed_tokens


def reclaim_unit(scheduler: Scheduler) -> int:
    """Bytes freed by evicting one page from every KV cache group."""
    return max(
        sum(
            group.kv_cache_spec.page_size_bytes
            for group in scheduler.kv_cache_config.kv_cache_groups
        ),
        1,
    )


def block_tokens(scheduler: Scheduler) -> int:
    """Tokens held by one block of the largest KV cache group."""
    return max(
        (
            group.kv_cache_spec.block_size
            for group in scheduler.kv_cache_config.kv_cache_groups
        ),
        default=1,
    )


def under_pressure(scheduler: Scheduler) -> bool:
    pool = scheduler.kv_cache_manager.block_pool
    return pool.num_gpu_blocks > 0 and (
        pool.get_num_free_blocks() * 2 <= pool.num_gpu_blocks
    )


#: How deep into the queue a lookahead policy may read. The same bound the
#: contract puts on any other working set, so a deep backlog costs a bounded
#: scan rather than the engine's step time.
MAX_LOOKAHEAD_REQUESTS = 256


def shared_beneficiaries(
    scheduler: Scheduler, request_ids: list[str]
) -> dict[str, list[str]]:
    """For each resident, every request whose work its blocks serve.

    A prefix-caching engine hands the same block to every request whose
    prompt shares that prefix, so a resident's real beneficiary set is the
    requests sitting on its blocks -- not, as the contract's default
    assumes, the one request that happens to own the entry. Without this
    every resident reports a beneficiary count of exactly 1, and a policy
    that ranks residents by demand ranks them all equal: it evicts in list
    order, which is arbitrary, where vLLM would have evicted by recency.

    The set has two halves, and a policy needs both:

      running   requests already holding the same block. Evicting one of
                these frees the least, because the block stays referenced.
      waiting   requests whose prompt would hit the block if it survives.
                This is the half the lookahead papers are named for -- Peek
                keeps what pending demand will reuse -- and it is invisible
                from the running set alone. Measured with only the running
                half, Peek scored 0.69 on prefix hit rate; the demand it is
                supposed to read was all in the queue.
    """
    holders: dict[int, list[str]] = {}
    for request_id in request_ids:
        try:
            groups = scheduler.kv_cache_manager.get_blocks(request_id).blocks
        except (KeyError, AttributeError):
            continue
        for group_blocks in groups:
            for block in group_blocks:
                holders.setdefault(block.block_id, []).append(request_id)

    sharing: dict[str, set[str]] = {request_id: set() for request_id in request_ids}
    for block_holders in holders.values():
        if len(block_holders) < 2:
            continue
        for request_id in block_holders:
            sharing[request_id].update(block_holders)
            sharing[request_id].discard(request_id)

    # The queue carries the same bound as any other working set, so a deep
    # backlog costs a bounded scan rather than the engine's step time.
    for waiting in itertools.islice(scheduler.waiting, MAX_LOOKAHEAD_REQUESTS):
        for block in _cache_hit_blocks(scheduler, waiting):
            for request_id in holders.get(block.block_id, ()):
                if request_id != waiting.request_id:
                    sharing[request_id].add(waiting.request_id)

    return {
        request_id: sorted(peers)
        for request_id, peers in sharing.items()
        if peers
    }


def _cache_hit_blocks(scheduler: Scheduler, request: Request) -> list[KVCacheBlock]:
    """Blocks a not-yet-running request would be handed by the prefix cache.

    Read-only: the lookup walks the block-hash table and takes no
    references, so asking on behalf of a queued request cannot change what
    the engine would then do.
    """
    manager = scheduler.kv_cache_manager
    if not manager.prefix_cache_lookup_enabled(request):
        return []
    groups, _, _ = manager.coordinator.find_longest_cache_hit(
        request.block_hashes, max(request.num_tokens - 1, 0)
    )
    return [block for group_blocks in groups for block in group_blocks]


def prefix_hit_tokens(scheduler: Scheduler, request: Request) -> int:
    """Prompt tokens already in the prefix cache, read without disturbing it.

    vLLM answers this itself, but only inside the scheduling loop and only
    for the request it has decided to run -- and `take_prefill_stats` then
    consumes the answer. A policy choosing *which* request to run needs it
    one step earlier, so ask the coordinator directly. The lookup walks the
    block-hash table and takes no references, so it is safe to repeat.
    """
    manager = scheduler.kv_cache_manager
    if not manager.prefix_cache_lookup_enabled(request):
        return 0
    _, hit_tokens, _ = manager.coordinator.find_longest_cache_hit(
        request.block_hashes, max(request.num_tokens - 1, 0)
    )
    return hit_tokens


class VllmEnginePort:
    """The scheduler, seen through the contract's vocabulary."""

    name = "vllm"
    default_principal = DEFAULT_PRINCIPAL
    # vLLM cannot guarantee atomic enactment of a multi-request unit, so a
    # wider selection has to sink the plan rather than be applied piecewise.
    max_requests_per_selection = 1

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        self._signals: dict[str, RequestSignals] = {}
        self._arrivals = 0
        self._probed_tokens = 0
        self._hit_tokens = 0
        self._pending_finish: set[str] = set()
        self._aborted: list[tuple[str, int]] = []

    def observe(self, request: Request) -> RequestSignals:
        """Record what only arrival order and the cache-at-arrival can say."""
        hit = prefix_hit_tokens(self.scheduler, request)
        self._arrivals += 1
        self._probed_tokens += request.num_prompt_tokens
        self._hit_tokens += min(hit, request.num_prompt_tokens)
        signals = RequestSignals(
            arrival_seq=self._arrivals - 1, lpm_hit_tokens=hit
        )
        self._signals[request.request_id] = signals
        return signals

    def touch(self, request: Request) -> None:
        signals = self._signals.get(request.request_id)
        if signals is not None:
            signals.touch()

    def forget(self, request_id: str) -> None:
        self._signals.pop(request_id, None)

    def view(self, request: Request) -> VllmRequest:
        return VllmRequest(
            request, self.scheduler, self._signals.get(request.request_id)
        )

    def candidates(self) -> list[VllmRequest]:
        scheduler = self.scheduler
        seen: set[str] = set()
        candidates: list[VllmRequest] = []
        for request in (
            *scheduler.running,
            *scheduler.waiting,
            *scheduler.skipped_waiting,
        ):
            if (
                request.request_id in seen
                or request.is_finished()
                or request.status in UNSELECTABLE
            ):
                continue
            seen.add(request.request_id)
            candidates.append(self.view(request))
        return candidates

    def residents(self) -> list[VllmRequest]:
        return [self.view(request) for request in self.scheduler.running]

    def capacity(self) -> ScheduleCapacity:
        scheduler = self.scheduler
        selections = min(
            len(self.candidates()), scheduler.max_num_running_reqs
        )
        return ScheduleCapacity(
            max_selections=selections,
            max_requests=selections,
            max_total_tokens=scheduler.max_num_scheduled_tokens,
        )

    def cache_capacity(self, residents: list[VllmRequest]) -> CacheCapacity:
        """Budget one unit less than the residents occupy.

        vLLM asks only when it is about to preempt something, so the answer it
        needs is which request to drop. Budgeting the resident total would let
        a plan that frees nothing be valid, and vLLM would then preempt on its
        own having asked for nothing.
        """
        unit = reclaim_unit(self.scheduler)
        return CacheCapacity(
            max_bytes=max(unit * len(residents) - unit, 0),
            fixed_bytes=0,
            facts={"virtual_request_pressure": True},
        )

    def under_pressure(self) -> bool:
        return under_pressure(self.scheduler)

    #: vLLM starts a request only when `schedule` picks it out of `waiting`,
    #: so a request can be held between arrival and the first schedule with
    #: no engine state to unwind. That window is what makes real admission
    #: control expressible here.
    #:
    #: Opt-in, because holding is only correct when the policy answers: a
    #: policy with no `admit` of its own returns `fallback-required`, and
    #: every arrival would then pay the deadline before starting. The
    #: deployment says which it is.
    @property
    def admission_controlled(self) -> bool:
        return envs.PLEX_ADMISSION_CONTROL

    def admission_capacity(self) -> AdmissionCapacity:
        scheduler = self.scheduler
        return AdmissionCapacity(
            max_accepted=max(
                scheduler.max_num_running_reqs - len(scheduler.running), 0
            ),
            max_tokens=scheduler.max_num_scheduled_tokens,
        )

    def beneficiaries(
        self, residents: list[VllmRequest]
    ) -> dict[str, list[str]]:
        return shared_beneficiaries(
            self.scheduler, [resident.engine_id for resident in residents]
        )

    # --- mechanics -----------------------------------------------------------
    #
    # Only what vLLM really performs. Notably absent:
    #
    #   schedule.atomic-enqueue@1  a guarantee, and vLLM cannot give it: the
    #                              running loop stops on the token budget, so a
    #                              selection can be admitted in part. This is
    #                              the same reason `max_requests_per_selection`
    #                              is 1.
    #   request.pause@1            vLLM preempts on its own terms and has no
    #                              entry point to park a named request with its
    #                              state preserved.
    #   cache.prefetch@1           no API to warm a prefix that is not attached
    #   cache.move@1               to a request already in the engine.
    #   request.rebalance@1        one engine, nowhere to rebalance to.
    mechanics = ("request.finish@1", "group.cancel@1")

    def enact(self, method: str, args: dict[str, Any]) -> bool:
        """Accept a finish or cancel, to be applied before the next decision.

        The work is queued rather than done here. `enact` is reached from
        inside `Scheduler.schedule`, and `finish_requests` rebinds
        `self.running`; doing that under the running loop would corrupt the
        index it walks. Draining at the top of `poll_schedule` applies the
        action before any request is looked at, which is the earliest point it
        can take effect and still be safe.
        """
        if method == "plex.request.finish@1":
            request_id = args.get("request_id")
            if not isinstance(request_id, str) or args.get("disposition") not in (
                "cancelled",
                "completed",
            ):
                return False
            return self._stage_finish([request_id])
        if method == "plex.group.cancel@1":
            group_id = args.get("group_id")
            if not isinstance(group_id, str) or args.get("propagation") not in (
                "group-only",
                "live-requests",
            ):
                return False
            if args.get("propagation") == "group-only":
                # Nothing to do in the engine: the group has no engine-side
                # existence beyond its live requests, so cancelling the group
                # without them is a policy-side bookkeeping change.
                return True
            members = args.get("request_ids")
            if not isinstance(members, list):
                return False
            return self._stage_finish(
                [member for member in members if isinstance(member, str)]
            )
        return False

    def _stage_finish(self, request_ids: list[str]) -> bool:
        staged = [
            request_id
            for request_id in request_ids
            if (request := self.scheduler.requests.get(request_id)) is not None
            and not request.is_finished()
        ]
        self._pending_finish.update(staged)
        # An action naming only requests the engine no longer has is not a
        # failure: the policy acted on a view one step old, and the outcome it
        # asked for already holds.
        return True

    def drain_actions(self) -> int:
        """Apply staged finishes. Safe only outside the scheduler's loops.

        The aborted requests are retained rather than dropped: the client is
        still awaiting output for a request the *engine* chose to end, so it
        has to be told. `finish_requests` only detaches the request inside the
        scheduler.
        """
        if not self._pending_finish:
            return 0
        request_ids = sorted(self._pending_finish)
        self._pending_finish.clear()
        finished = self.scheduler.finish_requests(
            request_ids, RequestStatus.FINISHED_ABORTED
        )
        self._aborted.extend(finished)
        return len(finished)

    def take_aborted(self) -> list[tuple[str, int]]:
        """Hand back the (request id, client index) pairs still unreported."""
        aborted = self._aborted
        self._aborted = []
        return aborted

    def note_aborted(self, finished: list[tuple[str, int]]) -> None:
        """Queue an abort the scheduler performed on the policy's behalf."""
        self._aborted.extend(finished)

    def engine_facts(self) -> dict[str, Any]:
        scheduler = self.scheduler
        pool = scheduler.kv_cache_manager.block_pool
        free_blocks = pool.get_num_free_blocks()
        total_blocks = pool.num_gpu_blocks
        page_bytes = reclaim_unit(scheduler)
        per_block = block_tokens(scheduler)
        return {
            "queue_depth": len(scheduler.waiting) + len(scheduler.skipped_waiting),
            "running_requests": len(scheduler.running),
            "batch_size": len(scheduler.running),
            "max_batch_size": scheduler.max_num_running_reqs,
            "max_total_tokens": scheduler.max_num_scheduled_tokens,
            "free_kv_blocks": free_blocks,
            "total_kv_blocks": total_blocks,
            "free_kv_tokens": free_blocks * per_block,
            "total_kv_tokens": total_blocks * per_block,
            "used_kv_ppm": (
                (total_blocks - free_blocks) * 1_000_000 // total_blocks
                if total_blocks
                else 0
            ),
            "memory_capacity": total_blocks * page_bytes,
            "active_kv_bytes": (total_blocks - free_blocks) * page_bytes,
            "hit_ratio_ppm": (
                self._hit_tokens * 1_000_000 // self._probed_tokens
                if self._probed_tokens
                else 0
            ),
            "kv_overloaded": under_pressure(scheduler),
            "now_ms": int(time.time() * 1000),
        }


class AsyncPlexPolicyController:
    """Scheduler-facing wrapper over `plex.engine.PolicyController`.

    Keeps the call sites in `scheduler.py` unchanged: they pass `Request`
    objects and this turns them into the views the controller expects.
    """

    def __init__(
        self, controller: PolicyController, port: VllmEnginePort
    ) -> None:
        self.controller = controller
        self.port = port

    @classmethod
    def from_policy(
        cls,
        policy: str,
        scheduler: Scheduler,
        *,
        model: str,
        target_id: str,
    ) -> AsyncPlexPolicyController:
        """Build a controller for `policy`, or raise if PLEX is unusable.

        The controller ships with the runtime now, so there is no version
        range to police and no check that `AsyncRuntime` still has the methods
        this file needs: a mismatch is not expressible.
        """
        try:
            import plex.engine  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "PLEX policy configured but plex is not installed. "
                "Install vLLM with the 'plex' extra or install plex directly."
            ) from error
        port = VllmEnginePort(scheduler)
        logger.info("Enabling PLEX with policy %s", policy)
        return cls(
            PolicyController.from_policy(
                policy, port, model=model, target_id=target_id
            ),
            port,
        )

    def tracks(self, engine_id: str) -> bool:
        return self.controller.tracks(engine_id)

    def register_request(self, request: Request) -> None:
        self.port.observe(request)
        self.controller.register_request(self.port.view(request))

    def mark_scheduled(
        self,
        plan: SchedulePlan | None,
        num_scheduled_tokens: dict[str, int],
    ) -> None:
        self.controller.mark_scheduled(plan, num_scheduled_tokens)

    def mark_progress(
        self,
        request: Request,
        *,
        committed_tokens: int,
        output_tokens: int,
    ) -> None:
        self.port.touch(request)
        self.controller.mark_progress(
            self.port.view(request),
            committed_tokens=committed_tokens,
            output_tokens=output_tokens,
        )

    def mark_preempted(self, request: Request) -> None:
        self.controller.mark_preempted(self.port.view(request))

    def mark_finished(self, request: Request, reason: str) -> None:
        self.controller.mark_finished(self.port.view(request), reason)
        self.port.forget(request.request_id)

    def mark_cache_enacted(
        self, decision: CacheDecision, preempted: Request | None
    ) -> None:
        self.controller.mark_cache_enacted(decision, preempted is not None)

    def publish(self) -> None:
        self.controller.publish()

    def poll_schedule(self) -> SchedulePlan | None:
        plan = self.controller.poll_schedule()
        # Actions staged while polling are applied here, before `schedule`
        # looks at a single request. Anything that arrived on the cache
        # channel mid-loop waits for the next step, which is the earliest
        # moment `finish_requests` can rebind `self.running` safely.
        self.port.drain_actions()
        return plan

    def take_aborted(self) -> list[tuple[str, int]]:
        """Requests the policy ended, whose clients have not been told yet."""
        return self.port.take_aborted()

    def awaiting_admission(self) -> bool:
        return self.controller.awaiting_admission()

    def holds(self, engine_id: str) -> bool:
        return self.controller.holds(engine_id)

    def take_admissions(self) -> dict[str, bool]:
        return self.controller.take_admissions()

    def note_finished(self, finished: list[tuple[str, int]]) -> None:
        """Route a rejection's abort through the same delivery as any other."""
        self.port.note_aborted(finished)

    def cached_preemption(self) -> CacheDecision | None:
        return self.controller.cached_reclaim()

    def close(self) -> None:
        self.controller.close()
