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
    events,
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

# The host refuses a working set wider than this, and the controller submits a
# leading prefix of `candidates()` to stay inside it. Imported rather than
# mirrored: a copy of a contract bound is exactly the kind of duplicate that
# drifts a revision behind without anyone noticing.
MAX_RANKABLE_CANDIDATES = events.MAX_WORKING_SET_REQUESTS

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

    __slots__ = (
        "_request",
        "_scheduler",
        "_signals",
        "_cached_blocks",
        "_attained_service",
        "_child_count",
    )

    def __init__(
        self,
        request: Request,
        scheduler: Scheduler,
        signals: RequestSignals | None = None,
        cached_blocks: int | None = None,
        attained_service: int | None = None,
        child_count: int = 0,
    ) -> None:
        self._request = request
        self._scheduler = scheduler
        self._signals = signals if signals is not None else NO_SIGNALS
        self._cached_blocks = cached_blocks
        # Preemption zeroes `num_computed_tokens` before the port is told, so a
        # record built afterwards reports that the request has been served
        # nothing. The engine is right -- it threw the work away -- but a policy
        # reading this as cumulative service then re-ranks a request preempted
        # at 4,000 tokens as the least-served candidate in the queue. The caller
        # passes what the request had accrued at the moment of the transition.
        self._attained_service = attained_service
        # How many other requests are sitting on, or would hit, this resident's
        # blocks. Preble's eviction key is
        # `(replicated, interior, last_access_ms, index)` with
        # `interior = child_count > 0`, so leaving it unpublished collapsed the
        # key to recency alone -- the LRU baseline the policy is measured
        # against. vLLM keeps no prefix tree, but the sharing set the port
        # already computes for `beneficiaries` is the same relationship: an
        # entry with dependents is exactly an interior node.
        self._child_count = child_count

    @property
    def engine_id(self) -> str:
        return self._request.request_id

    def set_child_count(self, count: int) -> None:
        """Set after construction: sharing is a property of the whole set."""
        self._child_count = count

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
        served = (
            request.num_computed_tokens
            if self._attained_service is None
            else self._attained_service
        )
        prompt_tokens = request.num_prompt_tokens
        hit = min(signals.lpm_hit_tokens, prompt_tokens)
        uncached = prompt_tokens - hit
        return {
            "queue_member": not running,
            "scheduler_state": "running" if running else "waiting",
            "attained_service": served,
            "service_tokens": served,
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
            # "Some of this request's KV is already resident" -- which is true
            # of a running request's own computed prefix *and* of a waiting
            # request whose prompt matched the prefix cache. Testing only the
            # first answered `false` for every request still in the queue, no
            # matter how much of it was already cached, which is precisely the
            # request a cache-aware scheduler wants to prefer. Found by
            # replaying one situation through both bindings: SGLang tested only
            # the second and was wrong the other way round, on the running case.
            "cache_ready": (
                request.num_computed_tokens > 0 or hit > 0
            ),
            # A waiting request has computed nothing, so `num_computed_tokens`
            # is 0 for every arrival however much of it the cache already holds
            # -- and a policy ordering the queue by cache affinity then sorts a
            # column of zeros and collapses to arrival order. The prefix probe
            # two lines down is the quantity meant here; `num_computed_tokens`
            # is the right answer only once the request is running.
            "cached_tokens": request.num_computed_tokens if running else hit,
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
            "leaf": self._child_count == 0,
            "child_count": self._child_count,
            "cached_length": request.num_computed_tokens,
            "computation_length": (
                request.num_prompt_tokens + len(request.output_token_ids)
            ),
            # The later of "this request last made progress" and "something hit
            # this request's cached prefix". Only the first was reported, so a
            # parked resident aged while new arrivals were reusing it -- and a
            # recency-ranking policy evicted the hottest prefix in the pool.
            "last_access_ms": max(
                self._signals.last_access_ms,
                self._scheduler.kv_cache_manager.block_pool.plex_owner_last_hit_ms(
                    request.request_id
                )
                or 0,
            ),
            # The state of the cached *object*, not of the request that owns
            # it. Publishing `request.status.name.lower()` here put vLLM's
            # request lifecycle ("waiting", "running") under a name SGLang
            # answers with the object's residency ("resident", "retracted"),
            # so a policy branching on it read two different vocabularies
            # through one fact. `preempted` on the request layer already
            # carries the lifecycle; this layer answers about the KV.
            "state_kind": (
                "retracted"
                if self._cached_blocks is not None
                or request.status == RequestStatus.PREEMPTED
                else "resident"
            ),
            "tier": "gpu",
        }

    def token_budget(self) -> int:
        request = self._request
        scheduler = self._scheduler
        if request.status == RequestStatus.RUNNING:
            pending = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
        else:
            # A waiting request costs what the cache does not already hold. This
            # counted the whole prompt, so a queued cache-hitter was priced at
            # full cost while `dispatch_input_tokens` and `new_prefill_tokens`
            # in the same fact set already subtracted the hit -- the policy was
            # given two different answers to one question and the one it packs a
            # step against was the wrong one.
            #
            # Floored at 1, which is vLLM's own rule rather than a guard invented
            # here: on a full prompt hit the scheduler re-computes the last token
            # (`num_computed_tokens = num_tokens - 1`). Without the floor a fully
            # cached arrival would price at zero, and the controller submits only
            # candidates with a positive budget -- so the request the policy most
            # wants to admit would vanish from the set it is allowed to rank.
            already = max(
                request.num_computed_tokens,
                min(self._signals.lpm_hit_tokens, request.num_prompt_tokens),
            )
            pending = max(request.num_tokens - already, 1)
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
        if self._cached_blocks is not None:
            # A parked resident holds no allocation the KV manager still knows
            # about -- its blocks were freed and stayed in the prefix cache.
            # Its size is what those blocks occupy.
            return self._cached_blocks * page_size_bytes(scheduler)
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


def page_size_bytes(scheduler: Scheduler) -> int:
    """Bytes held by one block of the largest KV cache group.

    `_plex_cached_owner` counts individual blocks without recording which
    group each came from, so a multi-group config over-reports a parked
    resident's size. Over-reporting is the safe direction: a policy asked to
    free a byte budget frees at least it.
    """
    return max(
        (
            group.kv_cache_spec.page_size_bytes
            for group in scheduler.kv_cache_config.kv_cache_groups
        ),
        default=1,
    )


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
        # Requests that have left the scheduler but whose KV is still in the
        # prefix cache. They are the only reason a prefix eviction has anything
        # to choose between, and nothing else in vLLM holds a handle to them
        # once `finish_requests` is done.
        self._parked: dict[str, Request] = {}
        # Evictions charged at the previous `cache_capacity`; the difference is
        # what the engine consumed while the last plan was in force.
        # How many residents each ask told the policy to give up, against how
        # many it was offered. The ranking the engine later walks is exactly
        # the set the policy named, and it names only as many as `max_bytes`
        # forces, so this pair is what separates a starved budget from a policy
        # that ranks few of its own accord.
        self._plex_ask_calls = 0
        self._plex_wanted_sum = 0
        self._plex_offered_sum = 0

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
        self._parked.pop(request_id, None)

    def park(self, request: Request) -> None:
        """Keep a finished request reachable while its KV is still cached.

        Called instead of `forget`. `residents()` prunes it as soon as the
        block pool reports no cached blocks under its id, so the map is bounded
        by the cache, not by the run.
        """
        self._parked[request.request_id] = request

    def view(
        self, request: Request, attained_service: int | None = None
    ) -> VllmRequest:
        return VllmRequest(
            request,
            self.scheduler,
            self._signals.get(request.request_id),
            attained_service=attained_service,
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

    def cached_resident_ids(self) -> frozenset[str]:
        """Requests the block pool still holds cached KV for.

        The controller uses this to keep a finished request resolvable while
        its bytes are still resident. Without it the host refuses to resolve
        the id and every parked resident is dropped from the cache submission
        -- which is to say the policy never sees the objects it exists to rank.
        """
        pool = self.scheduler.kv_cache_manager.block_pool
        return frozenset(pool.plex_cached_owners())

    def engine_stats(self) -> dict[str, int]:
        """Who decided each cached block's eviction, policy or LRU.

        The pair is the whole point: `prefix_evicted_by_policy` alone cannot
        distinguish a ranking that decided every eviction from one that decided
        two, and it was exactly that blindness -- a cache channel answered
        1,800 times per run while vLLM's LRU made every real eviction -- that
        let eight prefix-caching replications be measured against a decision
        they never made.
        """
        stats = dict(
            self.scheduler.kv_cache_manager.block_pool.plex_eviction_stats()
        )
        stats["ask_calls"] = self._plex_ask_calls
        stats["ask_wanted_sum"] = self._plex_wanted_sum
        stats["ask_offered_sum"] = self._plex_offered_sum
        # Whether the scheduler ever had a choice for the schedule channel to
        # make. A ranking over an empty queue is arithmetic, not a decision.
        for key, value in getattr(self.scheduler, "_plex_choice", {}).items():
            stats[f"sched_{key}"] = value
        return stats

    def prune_parked(self) -> int:
        """Drop parked requests the block pool no longer holds any KV for.

        `park` retains a whole `Request` -- prompt and output token ids -- for
        every finished request, and the only place `forget` was ever called from
        is `residents()`, which the controller reaches only while
        `under_pressure()`. So an engine that never crosses the pressure line
        retained one `Request` plus one `RequestSignals` per completed request
        for the lifetime of the process, and once `_parked` grew past the
        working-set bound it began pushing real residents out of the cache
        submission.

        Called from `poll_schedule`, which runs every step, so the map is bounded
        by the cache as its docstring always claimed.
        """
        if not self._parked:
            return 0
        census = self.scheduler.kv_cache_manager.block_pool.plex_cached_owners()
        stale = [
            request_id for request_id in self._parked if request_id not in census
        ]
        for request_id in stale:
            self.forget(request_id)
        return len(stale)

    def take_prefix_evictions(self) -> dict[str, int]:
        """Owners the ranking evicted since the last call, for feedback.

        The preemption seam reports through `mark_cache_enacted`; prefix
        eviction had no equivalent, so the channel that does essentially all of
        the cache work told the policy nothing about whether the object it
        ranked was actually reclaimed.
        """
        pool = self.scheduler.kv_cache_manager.block_pool
        return pool.take_plex_evicted_owners()

    def residents(self) -> list[VllmRequest]:
        """What the policy may reclaim, cached prefixes first.

        For a long time this was `scheduler.running`, which is the set vLLM
        offers when it is about to *preempt* -- a live request whose KV is
        referenced. That is a real decision, but it is not the one the
        prefix-caching papers describe and it is not the one `prefix_hit_rate`
        measures: every actual prefix eviction happens in `BlockPool`, off the
        LRU tail of the free queue, and never reached PLEX at all. A policy
        replicating Preble or Peek was ranking preemption victims while
        believing it was ranking cache entries.

        Both are published now, and they are distinguishable: a parked resident
        reports `state_kind: retracted` and a size drawn from its cached blocks,
        a running one reports `resident`. Cached prefixes come first because a
        policy that reclaims the least it can should reach for the entry nobody
        is using before it reaches for a running request.
        """
        scheduler = self.scheduler
        census = scheduler.kv_cache_manager.block_pool.plex_cached_owners()
        residents: list[VllmRequest] = []
        for request_id in list(self._parked):
            blocks = census.get(request_id, 0)
            if blocks == 0:
                # Its last cached block was evicted or hit; there is nothing
                # left to rank and nothing left to keep it alive for.
                self.forget(request_id)
                continue
            residents.append(
                VllmRequest(
                    self._parked[request_id],
                    scheduler,
                    self._signals.get(request_id),
                    cached_blocks=blocks,
                )
            )
        residents.extend(self.view(request) for request in scheduler.running)
        # One pass over the whole set: sharing is only visible when every
        # resident is considered together, so this cannot be computed per view.
        sharing = shared_beneficiaries(
            scheduler, [resident.engine_id for resident in residents]
        )
        for resident in residents:
            resident.set_child_count(len(sharing.get(resident.engine_id, ())))
        return residents

    def capacity(self) -> ScheduleCapacity:
        """How many candidates the policy may rank this opportunity.

        Capping this at `max_num_running_reqs` conflates two quantities: how
        many requests the engine can *run*, and how many the policy may *rank*.
        They are the same number only when nothing is queued. A plan's ranks are
        populated from the policy's selections alone, so once the running set is
        full it consumes the entire allowance and no waiting request can be
        ranked -- while `candidates()` has been publishing the waiting queue to
        the policy the whole time, and
        `Scheduler._select_waiting_request_for_scheduling` has been ready to act
        on such a rank. That is why `waiting_picks` read 0 with 144 requests
        queued: not because the engine refused the decision, but because the
        port never let the policy express it.

        Lifting the cap changes no vLLM invariant. The seat-limit break stays,
        the engine still admits only as slots free, and it still never demotes a
        running request. The one thing that changes is which waiting request
        takes a freed seat: native FIFO before, the policy's ranking after.

        Bounded by the protocol's working-set limit because that is what the
        controller will actually submit -- it takes a leading prefix of
        `candidates()`, so the policy sees the running set plus the head of the
        queue, and an allowance larger than the submission would be an
        over-report.

        Off by default: both states are measured on one build, per
        `experiments/regime-preregistration.md` section 6.
        """
        scheduler = self.scheduler
        candidates = len(self.candidates())
        if envs.PLEX_SCHEDULE_RANK_WAITING:
            selections = min(candidates, MAX_RANKABLE_CANDIDATES)
        else:
            selections = min(candidates, scheduler.max_num_running_reqs)
        return ScheduleCapacity(
            max_selections=selections,
            max_requests=selections,
            max_total_tokens=scheduler.max_num_scheduled_tokens,
        )

    def cache_capacity(self, residents: list[VllmRequest]) -> CacheCapacity:
        """Budget the shortfall the engine is about to take from the cache.

        This used to budget exactly one unit less than the residents occupy,
        which asks "which single request do I drop" -- the right question for
        preemption, and the only question the cache channel used to reach.
        Now that the channel drives prefix eviction, a one-victim plan is a
        near-empty answer: measured live, the policy decided 147 of 38,785
        prefix evictions (0.4%) because it named one resident and vLLM's LRU
        took every block after that resident's few ran out.

        So the demand is the real deficit: how far below the pressure line the
        free pool has fallen, converted to residents at their mean size. The
        policy then ranks enough residents to cover what the engine will
        actually take before it asks again.

        Capped at the number of residents whose KV is genuinely reclaimable.
        Demanding more than can be freed makes every plan over budget, and the
        contract refuses an over-budget plan *whole* -- which is the S6.24
        defect class, and it would turn a starved channel into a dead one.

        The deficit alone is a *level*, and the engine consumes at a *rate*: it
        evicts roughly 90 blocks/s while the channel is invoked ~2.5 times/s, so
        a plan sized to the instantaneous shortfall is spent within milliseconds
        and LRU decides the rest of the interval.

        A rate-based demand was computed here for exactly that reason and then
        overridden two lines later by `max(wanted, len(census))`, which asks for
        the whole census unconditionally -- so the docstring described a
        self-correcting mechanism that the code did not run. The census-wide ask
        is the behaviour that was actually measured (Preble 12.8% -> 34.5% of
        prefix evictions), so the dead arithmetic is gone rather than the ask:
        one fewer thing claiming to be measured when it is not.
        """
        unit = reclaim_unit(self.scheduler)
        pool = self.scheduler.kv_cache_manager.block_pool
        census = pool.plex_cached_owners()
        # Sizing the ask in bytes is still not enough, because the *answer* is
        # a list of requests and the engine spends it at allocation granularity.
        # Measured on ShareGPT: 119 residents offered per ask, 7.8 demanded, and
        # the ranking `_plex_take` actually held averaged 3.5 of an 85-request
        # census -- against 158 allocations per ask. Those 3.5 owners are drained
        # in the first few allocations and the remaining 96% fall through to LRU.
        # A budget that is correct in bytes for the interval is therefore still
        # wrong in *entries* for the interval, so the demand asks the policy to
        # order the whole cached census: enough order to survive the gap between
        # asks. Measured, Preble's share of prefix evictions went 12.8% -> 34.5%
        # at half the census and the policy named exactly what it was asked with
        # no plan refused, which is what says the ranking was budget-limited
        # rather than policy-limited.
        #
        # One resident of headroom is always left. `max_bytes` is the only lever
        # the contract gives, so demanding the entire residency sets the budget
        # to zero, and a plan that retains anything at all is then over budget
        # and refused *whole* -- the S6.24 class, which turns a starved channel
        # into a dead one. Over-naming otherwise costs nothing: `_plex_take`
        # walks the order and takes only the blocks it needs, so a longer
        # ranking evicts no more, it just keeps deciding for longer.
        #
        # The headroom has to survive the single-resident case, and it did not:
        # flooring `wanted` at 1 while capping it at `len(residents) - 1` gave
        # `wanted == 1` against one resident, so `max_bytes` was 0 and any plan
        # retaining that resident was over budget and refused whole -- the S6.24
        # class this paragraph exists to prevent, reached by the arithmetic meant
        # to prevent it. Asking for nothing when there is nothing to spare is the
        # honest answer, and it leaves a budget the policy can satisfy.
        #
        # Capping at the offered count also bounds a census that is wider than
        # what `residents()` can offer -- preempted and locally-freed requests
        # appear in it and are not rankable -- so an inflated census can no
        # longer drive the budget to zero.
        wanted = min(len(census), max(len(residents) - 1, 0))
        self._plex_ask_calls += 1
        self._plex_wanted_sum += wanted
        self._plex_offered_sum += len(residents)
        # Amendment 5: `fixed_bytes` MUST include bytes held by objects that are
        # not offered in `resident[]`. It was 0, so the retained-bytes equation
        # was evaluated against a fiction -- and the objects it omits are real
        # and numerous: preempted requests keep cached blocks in the census and
        # `residents()` never offers them, and anything past the working-set
        # truncation is in the same position. Priced in the same reclaim unit as
        # `max_bytes` so the two are commensurable.
        offered = {resident.engine_id for resident in residents}
        unoffered = sum(1 for owner in census if owner not in offered)
        return CacheCapacity(
            max_bytes=max(unit * len(residents) - unit * wanted, 0),
            fixed_bytes=unit * unoffered,
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
        self._eviction_order: list[str] | None = None

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
        controller = cls(
            PolicyController.from_policy(
                policy, port, model=model, target_id=target_id
            ),
            port,
        )
        # Hand vLLM's prefix eviction to the policy. Without this the cache
        # channel only ever reaches `scheduler.py`'s preemption path, and every
        # real eviction stays with the LRU free queue.
        scheduler.kv_cache_manager.block_pool.set_plex_eviction_order(
            controller.eviction_order
        )
        return controller

    def eviction_order(self) -> list[str]:
        """Request ids whose cached prefix the policy wants gone first.

        Memoised for the step. `BlockPool.get_new_blocks` runs many times per
        scheduler step and this is on that path, so polling the runtime per
        allocation would put a PyO3 call between vLLM and every block it takes.
        Within one step the answer cannot usefully change: the policy is not
        being asked again until the next `publish`.
        """
        if self._eviction_order is None:
            self._eviction_order = self.controller.cached_reclaim_order()
        return self._eviction_order

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

    def mark_preempted(
        self, request: Request, attained_service: int | None = None
    ) -> None:
        self.controller.mark_preempted(
            self.port.view(request, attained_service=attained_service)
        )

    def mark_finished(self, request: Request, reason: str) -> None:
        self.controller.mark_finished(self.port.view(request), reason)
        # Not `forget`: its KV may still be in the prefix cache, and while it is
        # the request is a cache resident the policy is entitled to rank.
        self.port.park(request)

    def mark_cache_enacted(
        self, decision: CacheDecision, preempted: Request | None
    ) -> None:
        self.controller.mark_cache_enacted(decision, preempted is not None)

    def publish(self) -> None:
        self.controller.publish()

    def poll_schedule(self) -> SchedulePlan | None:
        # One step, one reading of the eviction order. Dropping the memo here
        # rather than in `publish` keeps it tied to the step boundary the
        # scheduler actually has.
        self._eviction_order = None
        # Bound `_parked` on the step boundary rather than only under pressure.
        self.port.prune_parked()
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
