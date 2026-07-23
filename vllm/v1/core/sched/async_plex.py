# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgspec

from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from pie_plex import AsyncRuntime

    from vllm.v1.core.sched.scheduler import Scheduler

PLEX_API_VERSION = "pie.plex.engine@1"


@dataclass(frozen=True)
class PlexSchedulePlan:
    token_budgets: dict[str, int]
    ranks: dict[str, int]

    def selects(self, request_id: str) -> bool:
        return request_id in self.token_budgets

    def token_budget(self, request_id: str) -> int:
        return self.token_budgets[request_id]

    def rank(self, request_id: str) -> int | None:
        return self.ranks.get(request_id)


class AsyncPlexPolicyController:
    def __init__(
        self,
        runtime: AsyncRuntime,
        *,
        model: str,
        target_id: str,
    ) -> None:
        self.runtime = runtime
        self.model = model
        self.target_id = target_id
        self.epoch = 0
        self.dirty = False
        self.evict_dirty = False
        self.feedback_sequence = 0
        self._engine_to_logical: dict[str, str] = {}
        self._logical_to_engine: dict[str, str] = {}
        self._request_metadata: dict[str, dict[str, Any]] = {}
        self._terminal_on_complete: dict[str, bool] = {}
        self._completion_event: dict[str, str] = {}
        self._pending_request_events: deque[dict[str, Any]] = deque()
        self._pending_feedback: deque[dict[str, Any]] = deque()
        self._pending_finishes: deque[str] = deque()
        self._submitted_candidates: dict[int, tuple[str, ...]] = {}
        self._submitted_residents: dict[int, tuple[str, ...]] = {}
        self._seen_schedule_epoch = 0
        self._seen_evict_epoch = 0
        self._resolved_schedule_epoch = 0
        self._resolved_evict_epoch = 0
        self._schedule_plan: tuple[int, PlexSchedulePlan] | None = None
        self._evict_victim: tuple[int, str] | None = None

    @classmethod
    def from_policy(
        cls,
        policy: str,
        *,
        model: str,
        target_id: str,
    ) -> AsyncPlexPolicyController:
        try:
            from pie_plex import AsyncRuntime
        except ImportError as error:
            raise ImportError(
                "PLEX policy configured but pie-plex is not installed. "
                "Install vLLM with the 'plex' extra or install pie-plex directly."
            ) from error
        return cls(
            AsyncRuntime(policy, queue_capacity=256),
            model=model,
            target_id=target_id,
        )

    def register_request(self, request: Request) -> None:
        (
            logical_id,
            generation_id,
            metadata,
            terminal,
            completion_event,
        ) = self._request_identity(request)
        previous = self._logical_to_engine.get(logical_id)
        if previous is not None and previous != request.request_id:
            raise ValueError(
                f"PLEX logical request {logical_id!r} is already active as "
                f"engine request {previous!r}"
            )
        self._engine_to_logical[request.request_id] = logical_id
        self._logical_to_engine[logical_id] = request.request_id
        self._request_metadata[logical_id] = metadata
        self._terminal_on_complete[request.request_id] = terminal
        self._completion_event[request.request_id] = completion_event
        request_event = {
            "op": "create" if generation_id == 0 else "continue",
            "request_id": logical_id,
            "facts": {
                "generation_id": generation_id,
                "engine_request_id": request.request_id,
                "arrival_ms": int(request.arrival_time * 1000),
                "attained_service": request.num_computed_tokens,
            },
            "fields": self._request_fields(request, metadata),
        }
        self._pending_request_events.append(request_event)
        self._invalidate()

    def mark_preempted(self, request: Request) -> None:
        logical_id = self._engine_to_logical.get(request.request_id)
        if logical_id is not None:
            self._pending_feedback.append(
                {
                    "event": "preempted",
                    "request_id": logical_id,
                    "facts": {"attained_service": request.num_computed_tokens},
                }
            )
        self._invalidate()

    def mark_finished(self, request: Request, reason: str) -> None:
        logical_id = self._engine_to_logical.get(request.request_id)
        if logical_id is None:
            return
        terminal = self._terminal_on_complete.get(request.request_id, True)
        self._pending_feedback.append(
            {
                "event": self._completion_event.get(
                    request.request_id,
                    "finished" if terminal else "generation-finished",
                ),
                "request_id": logical_id,
                "facts": {
                    "reason": reason,
                    "attained_service": request.num_computed_tokens,
                    "generated_tokens": len(request.output_token_ids),
                },
            }
        )
        if terminal:
            self._pending_finishes.append(logical_id)
        self._forget_request(
            request.request_id,
            preserve_logical_state=not terminal,
        )
        self._invalidate()

    def publish(self, scheduler: Scheduler) -> None:
        if (
            not self.dirty
            and not self._pending_request_events
            and not self._pending_feedback
        ):
            return

        if self.dirty:
            candidates = self._candidates(scheduler)
            schedule_event = self._schedule_event(
                scheduler,
                candidates,
                list(self._pending_request_events),
            )
            if not self.runtime.try_submit_bytes(
                "schedule",
                self.epoch,
                msgspec.json.encode(schedule_event),
            ):
                return
            self._pending_request_events.clear()
            self._submitted_candidates[self.epoch] = tuple(
                request.request_id for request in candidates
            )
            self._trim_submissions(self._submitted_candidates)

            residents = [
                request
                for request in scheduler.running
                if request.request_id in self._engine_to_logical
            ]
            if residents:
                evict_event = self._evict_event(scheduler, residents)
                if self.runtime.try_submit_bytes(
                    "evict",
                    self.epoch,
                    msgspec.json.encode(evict_event),
                ):
                    self._submitted_residents[self.epoch] = tuple(
                        request.request_id for request in residents
                    )
                    self._trim_submissions(self._submitted_residents)
            self.dirty = False
            self.evict_dirty = False

        if self._pending_feedback:
            self.feedback_sequence += 1
            feedback = list(self._pending_feedback)
            finishes = list(self._pending_finishes)
            event = {
                "api_version": PLEX_API_VERSION,
                "hook": "feedback",
                "context": {
                    "delivery_id": (f"vllm:{self.target_id}:{self.feedback_sequence}"),
                    "records": feedback,
                    "context": self._hook_context(),
                },
                "request_events": [
                    {"op": "finish", "request_id": logical_id}
                    for logical_id in finishes
                ],
            }
            if self.runtime.try_submit_bytes(
                "feedback",
                self.epoch,
                msgspec.json.encode(event),
            ):
                self._pending_feedback.clear()
                self._pending_finishes.clear()

    def poll_schedule(self) -> PlexSchedulePlan | None:
        if self._resolved_schedule_epoch == self.epoch:
            return (
                self._schedule_plan[1]
                if self._schedule_plan is not None
                and self._schedule_plan[0] == self.epoch
                else None
            )
        result = self.runtime.latest("schedule", self._seen_schedule_epoch)
        if result is not None:
            epoch, outcome = result
            self._seen_schedule_epoch = epoch
            request_ids = self._submitted_candidates.pop(epoch, ())
            if outcome.get("status") == "success":
                ranks: dict[str, int] = {}
                token_budgets: dict[str, int] = {}
                for rank, item in enumerate(
                    outcome.get("decision", {}).get("selected", [])
                ):
                    candidate_index = item.get("candidate_index")
                    if (
                        isinstance(candidate_index, int)
                        and not isinstance(candidate_index, bool)
                        and 0 <= candidate_index < len(request_ids)
                    ):
                        request_id = request_ids[candidate_index]
                        ranks[request_id] = rank
                        # Async plans carry standing rank/selection only. Native
                        # scheduling owns the current-step token budget.
                        token_budgets[request_id] = (1 << 63) - 1
                self._schedule_plan = (
                    epoch,
                    PlexSchedulePlan(token_budgets, ranks),
                )
            else:
                self._schedule_plan = None
            if epoch == self.epoch:
                self._resolved_schedule_epoch = epoch

        if self._schedule_plan is None or self._schedule_plan[0] != self.epoch:
            return None
        return self._schedule_plan[1]

    def cached_preemption(self) -> str | None:
        if self._resolved_evict_epoch == self.epoch:
            return (
                self._evict_victim[1]
                if self._evict_victim is not None
                and self._evict_victim[0] == self.epoch
                else None
            )
        result = self.runtime.latest("evict", self._seen_evict_epoch)
        if result is not None:
            epoch, outcome = result
            self._seen_evict_epoch = epoch
            request_ids = self._submitted_residents.pop(epoch, ())
            selected = outcome.get("decision", {}).get("selected", [])
            if outcome.get("status") == "success" and selected:
                candidate_index = selected[0].get("candidate_index")
                if (
                    isinstance(candidate_index, int)
                    and not isinstance(candidate_index, bool)
                    and 0 <= candidate_index < len(request_ids)
                ):
                    self._evict_victim = (epoch, request_ids[candidate_index])
                else:
                    self._evict_victim = None
                if epoch == self.epoch:
                    self._resolved_evict_epoch = epoch
            else:
                self._evict_victim = None

        if self._evict_victim is None or self._evict_victim[0] != self.epoch:
            return None
        return self._evict_victim[1]

    def close(self) -> None:
        self.runtime.shutdown()

    def _invalidate(self) -> None:
        self.epoch += 1
        self.dirty = True
        self.evict_dirty = True
        self._schedule_plan = None
        self._evict_victim = None
        self._resolved_schedule_epoch = 0
        self._resolved_evict_epoch = 0

    def _forget_request(
        self,
        request_id: str,
        *,
        preserve_logical_state: bool,
    ) -> None:
        logical_id = self._engine_to_logical.pop(request_id, None)
        self._terminal_on_complete.pop(request_id, None)
        self._completion_event.pop(request_id, None)
        if logical_id is not None:
            self._logical_to_engine.pop(logical_id, None)
            if not preserve_logical_state:
                self._request_metadata.pop(logical_id, None)

    def _candidates(self, scheduler: Scheduler) -> list[Request]:
        candidates: list[Request] = []
        seen: set[str] = set()
        for request in (
            *scheduler.running,
            *scheduler.waiting,
            *scheduler.skipped_waiting,
        ):
            if (
                request.request_id in seen
                or request.request_id not in self._engine_to_logical
                or request.is_finished()
                or request.status
                in (
                    RequestStatus.WAITING_FOR_REMOTE_KVS,
                    RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
                    RequestStatus.WAITING_FOR_STREAMING_REQ,
                )
            ):
                continue
            candidates.append(request)
            seen.add(request.request_id)
        return candidates

    def _schedule_event(
        self,
        scheduler: Scheduler,
        candidates: list[Request],
        lifecycle_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        max_budget = max(
            (self._max_token_budget(scheduler, request) for request in candidates),
            default=0,
        )
        return {
            "api_version": PLEX_API_VERSION,
            "hook": "schedule",
            "context": {
                "runnable": [
                    {
                        "request_id": self._engine_to_logical[request.request_id],
                        "facts": self._facts(request),
                        "max_token_budget": self._max_token_budget(scheduler, request),
                    }
                    for request in candidates
                ],
                "capacity": {
                    "max_selected": min(
                        len(candidates), scheduler.max_num_running_reqs
                    ),
                    "max_total_tokens": scheduler.max_num_scheduled_tokens,
                    "max_token_budget": max_budget,
                },
                "context": self._hook_context(
                    {"mode": "async-indexed", "epoch": self.epoch}
                ),
            },
            "request_events": [
                *lifecycle_events,
                *(
                    {
                        "op": "merge-facts",
                        "request_id": self._engine_to_logical[request.request_id],
                        "facts": self._facts(request),
                    }
                    for request in candidates
                ),
            ],
        }

    def _evict_event(
        self,
        scheduler: Scheduler,
        residents: list[Request],
    ) -> dict[str, Any]:
        return {
            "api_version": PLEX_API_VERSION,
            "hook": "evict",
            "context": {
                "resident": [
                    {
                        "id": request.request_id,
                        "request_id": self._engine_to_logical[request.request_id],
                        "size_bytes": self._request_size_bytes(scheduler, request),
                        "facts": {
                            **self._facts(request),
                            "reload_cost": request.num_computed_tokens,
                        },
                    }
                    for request in residents
                ],
                "bytes_needed": max(self._minimum_reclaim_bytes(scheduler), 1),
                "context": self._hook_context(
                    {"mode": "async-indexed", "epoch": self.epoch}
                ),
            },
            "request_events": [],
        }

    def _hook_context(self, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        context = {
            "engine": "vllm",
            "model": self.model,
            "target_id": self.target_id,
            "capabilities": {"queries": []},
        }
        context.update(extra or {})
        return context

    def _target_facts(self, scheduler: Scheduler) -> dict[str, Any]:
        pool = scheduler.kv_cache_manager.block_pool
        return {
            "queue_depth": len(scheduler.waiting) + len(scheduler.skipped_waiting),
            "running_requests": len(scheduler.running),
            "free_kv_blocks": pool.get_num_free_blocks(),
            "total_kv_blocks": pool.num_gpu_blocks,
        }

    def _facts(self, request: Request) -> dict[str, Any]:
        return {
            "engine_request_id": request.request_id,
            "attained_service": request.num_computed_tokens,
            "generated_tokens": len(request.output_token_ids),
            "preempted": request.num_preemptions > 0,
            "preemptions": request.num_preemptions,
            "waiting_ms": max(int((time.time() - request.arrival_time) * 1000), 0),
        }

    @staticmethod
    def _max_token_budget(scheduler: Scheduler, request: Request) -> int:
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

    @staticmethod
    def _minimum_reclaim_bytes(scheduler: Scheduler) -> int:
        return sum(
            group.kv_cache_spec.page_size_bytes
            for group in scheduler.kv_cache_config.kv_cache_groups
        )

    @staticmethod
    def _request_size_bytes(scheduler: Scheduler, request: Request) -> int:
        blocks = scheduler.kv_cache_manager.get_blocks(request.request_id).blocks
        return sum(
            len(group_blocks) * group.kv_cache_spec.page_size_bytes
            for group_blocks, group in zip(
                blocks, scheduler.kv_cache_config.kv_cache_groups
            )
        )

    @staticmethod
    def _trim_submissions(submissions: dict[int, tuple[str, ...]]) -> None:
        while len(submissions) > 256:
            submissions.pop(next(iter(submissions)))

    @staticmethod
    def _request_fields(
        request: Request, metadata: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "body": {
                "prompt_token_ids": (
                    list(request.prompt_token_ids)
                    if request.prompt_token_ids is not None
                    else None
                ),
                "max_tokens": request.max_tokens,
                "priority": request.priority,
            },
            "metadata": dict(metadata),
        }

    @staticmethod
    def _request_identity(
        request: Request,
    ) -> tuple[str, int, dict[str, Any], bool, str]:
        config: Mapping[str, Any] = {}
        if (
            request.sampling_params is not None
            and request.sampling_params.extra_args is not None
        ):
            raw = request.sampling_params.extra_args.get("plex")
            if raw is not None:
                if not isinstance(raw, Mapping):
                    raise ValueError(
                        "sampling_params.extra_args['plex'] must be an object"
                    )
                config = raw
        logical_id = config.get("logical_request_id", request.request_id)
        generation_id = config.get("generation_id", 0)
        terminal = config.get("terminal", True)
        completion_event = config.get(
            "completion_event",
            "finished" if terminal else "generation-finished",
        )
        if not isinstance(logical_id, str) or not logical_id:
            raise ValueError("PLEX logical_request_id must be a non-empty string")
        if (
            not isinstance(generation_id, int)
            or isinstance(generation_id, bool)
            or generation_id < 0
        ):
            raise ValueError("PLEX generation_id must be a non-negative integer")
        if not isinstance(terminal, bool):
            raise ValueError("PLEX terminal must be a boolean")
        if not isinstance(completion_event, str) or not completion_event:
            raise ValueError("PLEX completion_event must be a non-empty string")
        metadata = config.get("metadata")
        if metadata is None:
            metadata = {
                key: value
                for key, value in config.items()
                if key
                not in {
                    "logical_request_id",
                    "generation_id",
                    "terminal",
                    "completion_event",
                }
            }
        if not isinstance(metadata, Mapping):
            raise ValueError("PLEX metadata must be an object")
        json.dumps(metadata)
        return (
            logical_id,
            generation_id,
            dict(metadata),
            terminal,
            completion_event,
        )
