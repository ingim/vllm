# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM's PLEX binding: an `EnginePort` and the scheduler-facing wrapper.

The v0.7 wire format, request identity, plan freshness, feedback delivery and
the fallback decision all live in `plex.engine`, which ships with the contract.
What is left here is the part only vLLM knows: where the scheduler keeps its
queues and what its per-request counters are called.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from plex.engine import (
    CacheCapacity,
    CacheDecision,
    PolicyController,
    ScheduleCapacity,
    SchedulePlan,
)

from vllm.logger import init_logger
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
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

    __slots__ = ("_request", "_scheduler")

    def __init__(self, request: Request, scheduler: Scheduler) -> None:
        self._request = request
        self._scheduler = scheduler

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
        waiting_ms = max(int((time.time() - request.arrival_time) * 1000), 0)
        running = request.status == RequestStatus.RUNNING
        return {
            "queue_member": not running,
            "scheduler_state": "running" if running else "waiting",
            "attained_service": request.num_computed_tokens,
            "service_tokens": request.num_computed_tokens,
            "dispatch_input_tokens": max(
                request.num_prompt_tokens - request.num_computed_tokens, 0
            ),
            "generated_tokens": len(request.output_token_ids),
            "preempted": request.num_preemptions > 0,
            "preemptions": request.num_preemptions,
            "waiting_ms": waiting_ms,
            "call_wait_us": waiting_ms * 1000,
            "arrival_ms": int(request.arrival_time * 1000),
            "cache_ready": request.num_computed_tokens > 0,
            "cached_tokens": request.num_computed_tokens,
        }

    def cache_facts(self) -> dict[str, Any]:
        return {
            "actual_size_bytes": self.actual_size_bytes(),
            "leaf": True,
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


class VllmEnginePort:
    """The scheduler, seen through the contract's vocabulary."""

    name = "vllm"
    default_principal = DEFAULT_PRINCIPAL
    # vLLM cannot guarantee atomic enactment of a multi-request unit, so a
    # wider selection has to sink the plan rather than be applied piecewise.
    max_requests_per_selection = 1

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler

    def view(self, request: Request) -> VllmRequest:
        return VllmRequest(request, self.scheduler)

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
            candidates.append(VllmRequest(request, scheduler))
        return candidates

    def residents(self) -> list[VllmRequest]:
        return [
            VllmRequest(request, self.scheduler)
            for request in self.scheduler.running
        ]

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
        pool = self.scheduler.kv_cache_manager.block_pool
        return (
            pool.num_gpu_blocks > 0
            and pool.get_num_free_blocks() * 2 <= pool.num_gpu_blocks
        )

    def engine_facts(self) -> dict[str, Any]:
        scheduler = self.scheduler
        pool = scheduler.kv_cache_manager.block_pool
        return {
            "queue_depth": len(scheduler.waiting) + len(scheduler.skipped_waiting),
            "running_requests": len(scheduler.running),
            "free_kv_blocks": pool.get_num_free_blocks(),
            "total_kv_blocks": pool.num_gpu_blocks,
            "total_kv_tokens": pool.num_gpu_blocks,
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
        self.controller.mark_progress(
            self.port.view(request),
            committed_tokens=committed_tokens,
            output_tokens=output_tokens,
        )

    def mark_preempted(self, request: Request) -> None:
        self.controller.mark_preempted(self.port.view(request))

    def mark_finished(self, request: Request, reason: str) -> None:
        self.controller.mark_finished(self.port.view(request), reason)

    def mark_cache_enacted(
        self, decision: CacheDecision, preempted: Request | None
    ) -> None:
        self.controller.mark_cache_enacted(decision, preempted is not None)

    def publish(self) -> None:
        self.controller.publish()

    def poll_schedule(self) -> SchedulePlan | None:
        return self.controller.poll_schedule()

    def cached_preemption(self) -> CacheDecision | None:
        return self.controller.cached_reclaim()

    def close(self) -> None:
        self.controller.close()
