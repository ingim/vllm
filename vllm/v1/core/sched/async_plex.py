# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from typing import TYPE_CHECKING, Any

import msgspec
from packaging.version import InvalidVersion, Version

from vllm.logger import init_logger
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from pie_plex import AsyncRuntime

    from vllm.v1.core.sched.scheduler import Scheduler

PLEX_API_VERSION = "pie.plex.engine@2"
MIN_PIE_PLEX_VERSION = Version("0.6")
MAX_PIE_PLEX_VERSION = Version("0.7")
logger = init_logger(__name__)


@dataclass(frozen=True)
class PlexScheduleSelection:
    index: int
    request_ids: tuple[str, ...]
    token_budgets: tuple[int, ...]


@dataclass(frozen=True)
class PlexSchedulePlan:
    opportunity_id: str
    submitted_at: float
    token_budgets: dict[str, int]
    ranks: dict[str, int]
    selections: tuple[PlexScheduleSelection, ...]

    def selects(self, request_id: str) -> bool:
        return request_id in self.token_budgets

    def token_budget(self, request_id: str) -> int:
        return self.token_budgets[request_id]

    def rank(self, request_id: str) -> int | None:
        return self.ranks.get(request_id)


@dataclass(frozen=True)
class PlexCacheDecision:
    opportunity_id: str
    submitted_at: float
    object_id: str
    request_id: str
    object_index: int


@dataclass(frozen=True)
class _ScheduleSubmission:
    membership_epoch: int
    opportunity_id: str
    request_ids: tuple[str, ...]
    submitted_at: float
    usable: bool


@dataclass(frozen=True)
class _CacheSubmission:
    membership_epoch: int
    opportunity_id: str
    request_ids: tuple[str, ...]
    submitted_at: float


class AsyncPlexPolicyController:
    def __init__(
        self,
        runtime: AsyncRuntime,
        *,
        model: str,
        target_id: str,
        plan_ttl_s: float = 0.25,
        publish_interval_s: float = 0.0,
    ) -> None:
        if plan_ttl_s <= 0:
            raise ValueError("PLEX plan_ttl_s must be positive")
        if publish_interval_s < 0:
            raise ValueError("PLEX publish_interval_s cannot be negative")
        self.runtime = runtime
        self.model = model
        self.target_id = target_id
        self.plan_ttl_s = plan_ttl_s
        self.publish_interval_s = publish_interval_s
        self.epoch = 0
        self._submission_epoch = 0
        self._last_publish_at = 0.0
        self._urgent_feedback = False
        self.schedule_dirty = False
        self.cache_dirty = False
        self.feedback_sequence = 0
        self._engine_to_request: dict[str, str] = {}
        self._request_to_engine: dict[str, str] = {}
        self._request_generation: dict[str, int] = {}
        self._request_principal: dict[str, str] = {}
        self._request_group: dict[str, str | None] = {}
        self._request_metadata: dict[str, dict[str, Any]] = {}
        self._canonical_fields: dict[str, dict[str, Any]] = {}
        self._known_groups: dict[str, str] = {}
        self._terminal_on_complete: dict[str, bool] = {}
        self._close_group_on_complete: dict[str, bool] = {}
        self._completion_outcome: dict[str, str] = {}
        self._pending_lifecycle: deque[dict[str, Any]] = deque()
        self._pending_feedback: deque[dict[str, Any]] = deque()
        self._pending_progress: dict[str, dict[str, Any]] = {}
        self._pending_request_cleanup: deque[dict[str, Any]] = deque()
        self._pending_group_cleanup: deque[dict[str, Any]] = deque()
        self._pending_steps: dict[str, deque[tuple[int, float]]] = {}
        self._submitted_candidates: dict[int, _ScheduleSubmission] = {}
        self._submitted_residents: dict[int, _CacheSubmission] = {}
        self._seen_schedule_epoch = 0
        self._seen_cache_epoch = 0
        self._seen_feedback_epoch = 0
        self._feedback_inflight: tuple[int, dict[str, Any], int] | None = None
        self._schedule_plan: PlexSchedulePlan | None = None
        self._cache_victims: deque[PlexCacheDecision] = deque()
        self._rejected_outcomes = 0
        self._successful_outcomes = 0
        self._fallback_outcomes = 0
        self._unavailable_outcomes = 0
        self._successful_by_operation = {
            "schedule": 0,
            "cache": 0,
            "feedback": 0,
        }
        self._schedule_enactments = 0
        self._schedule_partial_enactments = 0
        self._cache_enactments = 0
        self._last_schedule_outcome_at: float | None = None
        self._last_cache_outcome_at: float | None = None

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
        try:
            version = distribution_version("pie-plex")
        except PackageNotFoundError as error:
            raise ImportError(
                "PLEX policy configured but the pie-plex distribution metadata "
                "is unavailable. Reinstall pie-plex 0.6.x."
            ) from error
        cls._validate_runtime_version(version)
        required_methods = {
            "latest",
            "shutdown",
            "stats",
            "try_submit_bytes",
        }
        missing = sorted(
            method for method in required_methods if not hasattr(AsyncRuntime, method)
        )
        if missing:
            raise RuntimeError(
                "pie-plex 0.6.x AsyncRuntime is missing required methods: "
                + ", ".join(missing)
            )
        logger.info("Enabling PLEX with pie-plex %s", version)
        return cls(
            AsyncRuntime(policy, queue_capacity=256),
            model=model,
            target_id=target_id,
            publish_interval_s=0.025,
        )

    @staticmethod
    def _validate_runtime_version(version: str) -> None:
        try:
            parsed = Version(version)
        except InvalidVersion as error:
            raise RuntimeError(f"invalid pie-plex version {version!r}") from error
        if not MIN_PIE_PLEX_VERSION <= parsed < MAX_PIE_PLEX_VERSION:
            raise RuntimeError(
                f"vLLM PLEX requires pie-plex >=0.6,<0.7; found {version}"
            )

    def register_request(self, request: Request) -> None:
        (
            request_id,
            generation_id,
            principal_id,
            group_id,
            group_limits,
            metadata,
            terminal,
            close_group,
            completion_outcome,
        ) = self._request_identity(request)
        previous = self._request_to_engine.get(request_id)
        if previous is not None and previous != request.request_id:
            raise ValueError(
                f"PLEX request {request_id!r} is already active as "
                f"engine request {previous!r}"
            )
        if group_id is not None:
            previous_principal = self._known_groups.get(group_id)
            if previous_principal is None:
                self._known_groups[group_id] = principal_id
                self._pending_lifecycle.append(
                    {
                        "event": "create-group",
                        "group_id": group_id,
                        "principal_id": principal_id,
                        "limits": group_limits,
                        "facts": {"engine": "vllm"},
                    }
                )
            elif previous_principal != principal_id:
                raise ValueError(
                    f"PLEX group {group_id!r} belongs to principal "
                    f"{previous_principal!r}, not {principal_id!r}"
                )
        self._engine_to_request[request.request_id] = request_id
        self._request_to_engine[request_id] = request.request_id
        self._request_generation[request.request_id] = generation_id
        self._request_principal[request.request_id] = principal_id
        self._request_group[request.request_id] = group_id
        self._request_metadata[request_id] = metadata
        self._terminal_on_complete[request.request_id] = terminal
        self._close_group_on_complete[request.request_id] = close_group
        self._completion_outcome[request.request_id] = completion_outcome
        fields = self._request_fields(request, metadata)
        self._canonical_fields[request_id] = fields
        facts = {
            **self._facts(request),
            "generation_id": generation_id,
        }
        if generation_id == 0:
            self._pending_lifecycle.append(
                {
                    "event": "create-request",
                    "request_id": request_id,
                    "principal_id": principal_id,
                    "group_id": group_id,
                    "fields": fields,
                    "facts": facts,
                }
            )
            self._pending_lifecycle.append(
                {"event": "admit-request", "request_id": request_id}
            )
        else:
            self._pending_lifecycle.append(
                {
                    "event": "continue-request",
                    "request_id": request_id,
                    "fields": fields,
                    "facts": facts,
                }
            )
        self._pending_lifecycle.append(
            {"event": "activate-request", "request_id": request_id}
        )
        self._invalidate()

    def mark_scheduled(
        self,
        plan: PlexSchedulePlan | None,
        num_scheduled_tokens: Mapping[str, int],
    ) -> None:
        scheduled_at = time.monotonic()
        for engine_request_id, token_count in num_scheduled_tokens.items():
            if engine_request_id not in self._engine_to_request:
                continue
            self._pending_steps.setdefault(engine_request_id, deque()).append(
                (token_count, scheduled_at)
            )

        if plan is not None:
            for selection in plan.selections:
                scheduled = [
                    num_scheduled_tokens.get(request_id, 0)
                    for request_id in selection.request_ids
                ]
                requested_tokens = sum(selection.token_budgets)
                scheduled_tokens = sum(scheduled)
                status = (
                    "enacted"
                    if scheduled == list(selection.token_budgets)
                    else "partially-enacted"
                    if scheduled_tokens > 0
                    else "not-enacted"
                )
                if status == "enacted":
                    self._schedule_enactments += 1
                elif status == "partially-enacted":
                    self._schedule_partial_enactments += 1
                self._pending_feedback.append(
                    {
                        "subject": {
                            "kind": "schedule-selection",
                            "value": {
                                "opportunity_id": plan.opportunity_id,
                                "selection_index": selection.index,
                            },
                        },
                        "outcome": "progress",
                        "facts": {
                            "status": status,
                            "request_ids": list(selection.request_ids),
                            "requested_tokens": requested_tokens,
                            "scheduled_tokens": scheduled_tokens,
                        },
                    }
                )
            self._schedule_plan = None

        if plan is not None or num_scheduled_tokens:
            self.schedule_dirty = True

    def mark_progress(
        self,
        request: Request,
        *,
        committed_tokens: int,
        output_tokens: int,
    ) -> None:
        request_id = self._engine_to_request.get(request.request_id)
        if request_id is None or committed_tokens <= 0:
            return
        scheduled_at = None
        pending_steps = self._pending_steps.get(request.request_id)
        if pending_steps:
            _, scheduled_at = pending_steps.popleft()
            if not pending_steps:
                self._pending_steps.pop(request.request_id, None)
        service_us = (
            max(int((time.monotonic() - scheduled_at) * 1_000_000), 0)
            if scheduled_at is not None
            else 0
        )
        input_tokens = max(committed_tokens - output_tokens, 0)
        request_key = f"request:{request_id}"
        record = self._pending_progress.get(request_key)
        if record is None:
            record = {
                "subject": {"kind": "request", "value": request_id},
                "outcome": "progress",
                "facts": self._feedback_facts(request),
            }
            self._pending_progress[request_key] = record
        facts = record["facts"]
        facts.update(self._feedback_facts(request))
        for field, value in (
            ("committed_tokens", committed_tokens),
            ("service_tokens", committed_tokens),
            ("input_tokens", input_tokens),
            ("output_tokens", output_tokens),
            ("service_us", service_us),
        ):
            facts[field] = facts.get(field, 0) + value
        group_id = self._request_group.get(request.request_id)
        if group_id is not None:
            group_key = f"group:{group_id}"
            group_record = self._pending_progress.get(group_key)
            if group_record is None:
                group_record = {
                    "subject": {"kind": "work-group", "value": group_id},
                    "outcome": "progress",
                    "facts": {"service_us": 0},
                }
                self._pending_progress[group_key] = group_record
            group_record["facts"]["service_us"] += service_us

    def mark_cache_enacted(
        self,
        decision: PlexCacheDecision,
        request: Request | None,
    ) -> None:
        enacted = request is not None and request.request_id == decision.request_id
        if enacted:
            self._cache_enactments += 1
        self._pending_feedback.append(
            {
                "subject": {"kind": "cache-object", "value": decision.object_id},
                "outcome": "progress",
                "facts": {
                    "opportunity_id": decision.opportunity_id,
                    "status": "reclaimed" if enacted else "not-enacted",
                    "engine_request_id": decision.request_id,
                    "object_index": decision.object_index,
                },
            }
        )
        self.cache_dirty = True
        self._urgent_feedback = True

    def mark_preempted(self, request: Request) -> None:
        request_id = self._engine_to_request.get(request.request_id)
        if request_id is not None:
            self._pending_feedback.append(
                {
                    "subject": {"kind": "request", "value": request_id},
                    "outcome": "progress",
                    "facts": {
                        **self._feedback_facts(request),
                        "preempted": True,
                    },
                }
            )
            self._urgent_feedback = True
        self._invalidate()

    def mark_finished(self, request: Request, reason: str) -> None:
        request_id = self._engine_to_request.get(request.request_id)
        if request_id is None:
            return
        terminal = self._terminal_on_complete.get(request.request_id, True)
        if terminal:
            outcome, status = self._terminal_outcome(
                self._completion_outcome.get(request.request_id, "auto"),
                reason,
            )
            self._pending_feedback.append(
                {
                    "subject": {"kind": "request", "value": request_id},
                    "outcome": outcome,
                    "facts": {
                        "initiator": "host",
                        "reason": reason,
                        **self._feedback_facts(request),
                    },
                }
            )
            self._pending_request_cleanup.append(
                {"request_id": request_id, "status": status}
            )
            group_id = self._request_group.get(request.request_id)
            if group_id is not None and self._close_group_on_complete.get(
                request.request_id, False
            ):
                group_status = (
                    "cancelled"
                    if status == "cancelled"
                    else "expired"
                    if status == "expired"
                    else "closed"
                )
                group_outcome = (
                    "cancelled"
                    if group_status == "cancelled"
                    else "expired"
                    if group_status == "expired"
                    else "completed"
                )
                self._pending_feedback.append(
                    {
                        "subject": {"kind": "work-group", "value": group_id},
                        "outcome": group_outcome,
                        "facts": {"initiator": "host", "reason": reason},
                    }
                )
                self._pending_group_cleanup.append(
                    {"group_id": group_id, "status": group_status}
                )
        else:
            self._pending_feedback.append(
                {
                    "subject": {"kind": "request", "value": request_id},
                    "outcome": "progress",
                    "facts": {
                        "boundary": True,
                        "reason": reason,
                        **self._feedback_facts(request),
                    },
                }
            )
        self._forget_request(
            request.request_id,
            preserve_request_state=not terminal,
        )
        self._urgent_feedback = True
        self._invalidate()

    def publish(self, scheduler: Scheduler) -> None:
        if not self._poll_feedback_outcome():
            return
        self._poll_cache_outcome()
        if (
            self._cache_under_pressure(scheduler)
            and not self._cache_victims
            and not self._submitted_residents
        ):
            self.cache_dirty = True
        if (
            not self.schedule_dirty
            and not self.cache_dirty
            and not self._pending_lifecycle
            and not self._pending_feedback
            and not self._pending_progress
            and self._feedback_inflight is None
        ):
            return
        if (
            not self._pending_lifecycle
            and not self._pending_request_cleanup
            and not self._pending_group_cleanup
            and not self._urgent_feedback
            and time.monotonic() - self._last_publish_at < self.publish_interval_s
        ):
            return

        has_feedback = bool(
            self._pending_feedback
            or self._pending_progress
            or self._feedback_inflight is not None
        )
        if self._pending_lifecycle:
            if not self._submit_schedule(
                scheduler,
                list(self._pending_lifecycle),
                usable=not has_feedback,
            ):
                return
            self._pending_lifecycle.clear()
            self.schedule_dirty = False

        if has_feedback:
            if not self._submit_feedback():
                return
            self.schedule_dirty = True
            self.cache_dirty = True

        if self.schedule_dirty:
            if not self._submit_schedule(scheduler, [], usable=True):
                return
            self.schedule_dirty = False

        if self.cache_dirty:
            if self._cache_under_pressure(scheduler):
                self.cache_dirty = not self._submit_cache(scheduler)
            else:
                self.cache_dirty = False

    def poll_schedule(self) -> PlexSchedulePlan | None:
        if self._schedule_plan is not None:
            if self._is_fresh(self._schedule_plan.submitted_at):
                return self._schedule_plan
            self._schedule_plan = None
            self.schedule_dirty = True
            self._rejected_outcomes += 1

        result = self.runtime.latest("schedule", self._seen_schedule_epoch)
        if result is not None:
            epoch, outcome = result
            self._seen_schedule_epoch = epoch
            self._record_outcome("schedule", outcome)
            submission = self._submitted_candidates.pop(epoch, None)
            self._last_schedule_outcome_at = time.monotonic()
            if (
                submission is not None
                and submission.usable
                and submission.membership_epoch == self.epoch
                and self._is_fresh(submission.submitted_at)
                and self._adapter_safe(outcome)
            ):
                self._schedule_plan = self._parse_schedule_plan(outcome, submission)
                if self._schedule_plan is None and outcome.get("status") == "success":
                    self._rejected_outcomes += 1
                    self.schedule_dirty = True
            elif outcome.get("status") == "success" and (
                submission is None or submission.usable
            ):
                self._rejected_outcomes += 1
                self.schedule_dirty = True

        return self._schedule_plan

    def cached_preemption(self) -> PlexCacheDecision | None:
        while self._cache_victims:
            decision = self._cache_victims.popleft()
            if self._is_fresh(decision.submitted_at):
                if not self._cache_victims:
                    self.cache_dirty = True
                return decision
            self._rejected_outcomes += 1
        self.cache_dirty = True
        self._poll_cache_outcome()
        if not self._cache_victims:
            return None
        decision = self._cache_victims.popleft()
        if not self._cache_victims:
            self.cache_dirty = True
        return decision

    def _poll_cache_outcome(self) -> None:
        result = self.runtime.latest("cache", self._seen_cache_epoch)
        if result is not None:
            epoch, outcome = result
            self._seen_cache_epoch = epoch
            self._record_outcome("cache", outcome)
            submission = self._submitted_residents.pop(epoch, None)
            for stale_epoch in [
                submitted_epoch
                for submitted_epoch in self._submitted_residents
                if submitted_epoch < epoch
            ]:
                self._submitted_residents.pop(stale_epoch, None)
            self._last_cache_outcome_at = time.monotonic()
            if (
                submission is not None
                and submission.membership_epoch == self.epoch
                and self._is_fresh(submission.submitted_at)
                and self._adapter_safe(outcome)
            ):
                self._cache_victims.extend(
                    self._parse_cache_decisions(outcome, submission)
                )
                if not self._cache_victims and outcome.get("status") == "success":
                    self._rejected_outcomes += 1
            elif outcome.get("status") == "success":
                self._rejected_outcomes += 1

    def _submit_schedule(
        self,
        scheduler: Scheduler,
        lifecycle: list[dict[str, Any]],
        *,
        usable: bool,
    ) -> bool:
        candidates = self._candidates(scheduler)
        submission_epoch = self._next_submission_epoch()
        opportunity_id = self._opportunity_id("schedule", submission_epoch)
        event = self._schedule_event(
            scheduler,
            candidates,
            lifecycle,
            submission_epoch,
        )
        if not self.runtime.try_submit_bytes(
            "schedule",
            submission_epoch,
            msgspec.json.encode(event),
        ):
            return False
        self._submitted_candidates[submission_epoch] = _ScheduleSubmission(
            membership_epoch=self.epoch,
            opportunity_id=opportunity_id,
            request_ids=tuple(request.request_id for request in candidates),
            submitted_at=time.monotonic(),
            usable=usable,
        )
        self._trim_submissions(self._submitted_candidates)
        self._last_publish_at = time.monotonic()
        return True

    def _submit_cache(self, scheduler: Scheduler) -> bool:
        residents = [
            request
            for request in scheduler.running
            if request.request_id in self._engine_to_request
        ]
        if not residents:
            return True
        submission_epoch = self._next_submission_epoch()
        opportunity_id = self._opportunity_id("cache", submission_epoch)
        event = self._cache_event(scheduler, residents, submission_epoch)
        if not self.runtime.try_submit_bytes(
            "cache",
            submission_epoch,
            msgspec.json.encode(event),
        ):
            return False
        self._submitted_residents[submission_epoch] = _CacheSubmission(
            membership_epoch=self.epoch,
            opportunity_id=opportunity_id,
            request_ids=tuple(request.request_id for request in residents),
            submitted_at=time.monotonic(),
        )
        self._trim_submissions(self._submitted_residents)
        self._last_publish_at = time.monotonic()
        return True

    def _submit_feedback(self) -> bool:
        retrying = self._feedback_inflight is not None
        if retrying:
            inflight_epoch, event, attempts = self._feedback_inflight
            if inflight_epoch != 0:
                return False
        else:
            attempts = 1
            feedback_sequence = self.feedback_sequence + 1
            terminal_request_ids = {
                terminal["request_id"] for terminal in self._pending_request_cleanup
            }
            terminal_group_ids = {
                terminal["group_id"] for terminal in self._pending_group_cleanup
            }
            terminal_records: list[dict[str, Any]] = []
            nonterminal_records: list[dict[str, Any]] = []
            for record in self._pending_feedback:
                subject = record["subject"]
                terminal = (
                    subject["kind"] == "request"
                    and subject["value"] in terminal_request_ids
                    or subject["kind"] == "work-group"
                    and subject["value"] in terminal_group_ids
                )
                (terminal_records if terminal else nonterminal_records).append(record)
            split_before_cleanup = bool(
                (terminal_request_ids or terminal_group_ids)
                and (self._pending_progress or nonterminal_records)
            )
            if split_before_cleanup:
                records = [
                    *self._pending_progress.values(),
                    *nonterminal_records,
                ]
                cleanup = {"requests": [], "groups": []}
            else:
                records = [
                    *self._pending_progress.values(),
                    *self._pending_feedback,
                ]
                cleanup = {
                    "requests": list(self._pending_request_cleanup),
                    "groups": list(self._pending_group_cleanup),
                }
            event = {
                "api_version": PLEX_API_VERSION,
                "operation": "feedback",
                "context": {
                    "delivery_id": f"vllm:{self.target_id}:{feedback_sequence}",
                    "records": records,
                },
                "cleanup": cleanup,
            }
        submission_epoch = self._next_submission_epoch()
        if not self.runtime.try_submit_bytes(
            "feedback",
            submission_epoch,
            msgspec.json.encode(event),
        ):
            return False
        self._feedback_inflight = (submission_epoch, event, attempts)
        if not retrying:
            self.feedback_sequence = feedback_sequence
            self._pending_progress.clear()
            if split_before_cleanup:
                self._pending_feedback = deque(terminal_records)
            else:
                self._pending_feedback.clear()
                self._pending_request_cleanup.clear()
                self._pending_group_cleanup.clear()
        self._urgent_feedback = bool(
            self._pending_feedback
            or self._pending_progress
            or self._pending_request_cleanup
            or self._pending_group_cleanup
        )
        self._last_publish_at = time.monotonic()
        return True

    def _poll_feedback_outcome(self) -> bool:
        if self._feedback_inflight is None:
            return True
        inflight_epoch, event, attempts = self._feedback_inflight
        if inflight_epoch == 0:
            return True
        result = self.runtime.latest("feedback", self._seen_feedback_epoch)
        if result is None:
            return False
        epoch, outcome = result
        self._seen_feedback_epoch = epoch
        if epoch != inflight_epoch:
            raise RuntimeError(
                "PLEX feedback outcome does not match the in-flight delivery"
            )
        self._record_outcome("feedback", outcome)
        status = outcome.get("status")
        if status in {"success", "unavailable"}:
            self._feedback_inflight = None
            if (
                not self._pending_feedback
                and not self._pending_progress
                and not self._pending_request_cleanup
                and not self._pending_group_cleanup
            ):
                self._urgent_feedback = False
            return True
        failure = outcome.get("failure")
        failure_kind = failure.get("kind") if isinstance(failure, Mapping) else None
        if failure_kind == "state-conflict" and attempts < 3:
            self._feedback_inflight = (0, event, attempts + 1)
            self._urgent_feedback = True
            return True
        logger.error(
            "PLEX feedback delivery %s failed: %s",
            event["context"]["delivery_id"],
            outcome,
        )
        self._feedback_inflight = None
        return True

    def close(self) -> None:
        self.runtime.shutdown()
        self._poll_feedback_outcome()
        self._poll_cache_outcome()
        self.poll_schedule()
        submitted, dropped, completed = self.runtime.stats()
        logger.info(
            "PLEX worker stopped: submitted=%d dropped=%d completed=%d "
            "success=%d fallback=%d unavailable=%d rejected=%d "
            "schedule_success=%d cache_success=%d feedback_success=%d "
            "schedule_enacted=%d schedule_partial=%d cache_enacted=%d",
            submitted,
            dropped,
            completed,
            self._successful_outcomes,
            self._fallback_outcomes,
            self._unavailable_outcomes,
            self._rejected_outcomes,
            self._successful_by_operation["schedule"],
            self._successful_by_operation["cache"],
            self._successful_by_operation["feedback"],
            self._schedule_enactments,
            self._schedule_partial_enactments,
            self._cache_enactments,
        )

    def _invalidate(self) -> None:
        self.epoch += 1
        self.schedule_dirty = True
        self.cache_dirty = True
        self._schedule_plan = None
        self._cache_victims.clear()

    def _forget_request(
        self,
        engine_request_id: str,
        *,
        preserve_request_state: bool,
    ) -> None:
        request_id = self._engine_to_request.pop(engine_request_id, None)
        self._request_generation.pop(engine_request_id, None)
        self._request_principal.pop(engine_request_id, None)
        self._request_group.pop(engine_request_id, None)
        self._terminal_on_complete.pop(engine_request_id, None)
        self._close_group_on_complete.pop(engine_request_id, None)
        self._completion_outcome.pop(engine_request_id, None)
        self._pending_steps.pop(engine_request_id, None)
        if request_id is not None:
            self._request_to_engine.pop(request_id, None)
            if not preserve_request_state:
                self._request_metadata.pop(request_id, None)
                self._canonical_fields.pop(request_id, None)

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
                or request.request_id not in self._engine_to_request
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
        submission_epoch: int,
    ) -> dict[str, Any]:
        return {
            "api_version": PLEX_API_VERSION,
            "operation": "schedule",
            "context": {
                "meta": self._decision_meta("schedule", submission_epoch),
                "cause": "capacity-changed",
                "runnable": [
                    {
                        "request": self._request_ref(request),
                        "facts": self._facts(request),
                        "max_token_budget": self._max_token_budget(scheduler, request),
                    }
                    for request in candidates
                ],
                "capacity": {
                    "max_selections": min(
                        len(candidates), scheduler.max_num_running_reqs
                    ),
                    "max_requests": min(
                        len(candidates), scheduler.max_num_running_reqs
                    ),
                    "max_total_tokens": scheduler.max_num_scheduled_tokens,
                    "facts": self._engine_facts(scheduler),
                },
            },
            "lifecycle": [
                *lifecycle_events,
                *(
                    {
                        "event": "merge-request-facts",
                        "request_id": self._engine_to_request[request.request_id],
                        "facts": self._facts(request),
                    }
                    for request in candidates
                ),
            ],
        }

    def _cache_event(
        self,
        scheduler: Scheduler,
        residents: list[Request],
        submission_epoch: int,
    ) -> dict[str, Any]:
        reclaim_unit = max(self._minimum_reclaim_bytes(scheduler), 1)
        represented_bytes = reclaim_unit * len(residents)
        return {
            "api_version": PLEX_API_VERSION,
            "operation": "cache",
            "context": {
                "meta": self._decision_meta("cache", submission_epoch),
                "cause": "pressure",
                "resident": [
                    {
                        "object": {
                            "object_id": f"vllm-kv:{request.request_id}",
                            "size_bytes": reclaim_unit,
                            "beneficiaries": [
                                {
                                    "kind": "request",
                                    "id": self._engine_to_request[request.request_id],
                                }
                            ],
                            "beneficiary_count": 1,
                            "facts": {
                                **self._facts(request),
                                "reload_cost": request.num_computed_tokens,
                                "actual_size_bytes": self._request_size_bytes(
                                    scheduler, request
                                ),
                                "leaf": True,
                            },
                        },
                        "reclaimable": True,
                    }
                    for request in residents
                ],
                "prospective": [],
                "capacity": {
                    "max_bytes": max(represented_bytes - reclaim_unit, 0),
                    "fixed_bytes": 0,
                    "facts": {
                        **self._engine_facts(scheduler),
                        "virtual_request_pressure": True,
                    },
                },
                "episode": None,
            },
            "lifecycle": [],
        }

    def _parse_schedule_plan(
        self,
        outcome: Mapping[str, Any],
        submission: _ScheduleSubmission,
    ) -> PlexSchedulePlan | None:
        plan = outcome.get("plan")
        if not isinstance(plan, Mapping) or plan.get("operation") != "schedule":
            return None
        body = plan.get("plan")
        if not isinstance(body, Mapping):
            return None
        raw_selections = body.get("selections")
        if not isinstance(raw_selections, list):
            return None

        ranks: dict[str, int] = {}
        token_budgets: dict[str, int] = {}
        selections: list[PlexScheduleSelection] = []
        for selection_index, raw_selection in enumerate(raw_selections):
            if not isinstance(raw_selection, Mapping):
                return None
            indices = raw_selection.get("requests")
            budgets = raw_selection.get("token_budgets")
            if (
                not isinstance(indices, list)
                or not isinstance(budgets, list)
                or len(indices) != 1
                or len(budgets) != 1
            ):
                # vLLM cannot guarantee atomic enactment of multi-request units.
                return None
            candidate_index = indices[0]
            token_budget = budgets[0]
            if (
                not self._is_int(candidate_index)
                or not 0 <= candidate_index < len(submission.request_ids)
                or not self._is_int(token_budget)
                or token_budget <= 0
            ):
                return None
            engine_request_id = submission.request_ids[candidate_index]
            if engine_request_id in token_budgets:
                return None
            token_budgets[engine_request_id] = token_budget
            ranks[engine_request_id] = selection_index
            selections.append(
                PlexScheduleSelection(
                    index=selection_index,
                    request_ids=(engine_request_id,),
                    token_budgets=(token_budget,),
                )
            )

        return PlexSchedulePlan(
            opportunity_id=submission.opportunity_id,
            submitted_at=submission.submitted_at,
            token_budgets=token_budgets,
            ranks=ranks,
            selections=tuple(selections),
        )

    def _parse_cache_decisions(
        self,
        outcome: Mapping[str, Any],
        submission: _CacheSubmission,
    ) -> list[PlexCacheDecision]:
        plan = outcome.get("plan")
        if not isinstance(plan, Mapping) or plan.get("operation") != "cache":
            return []
        body = plan.get("plan")
        if not isinstance(body, Mapping):
            return []
        reclaim = body.get("reclaim")
        if not isinstance(reclaim, list):
            return []

        decisions: list[PlexCacheDecision] = []
        for object_index in reclaim:
            if not self._is_int(object_index) or not 0 <= object_index < len(
                submission.request_ids
            ):
                return []
            request_id = submission.request_ids[object_index]
            decisions.append(
                PlexCacheDecision(
                    opportunity_id=submission.opportunity_id,
                    submitted_at=submission.submitted_at,
                    object_id=f"vllm-kv:{request_id}",
                    request_id=request_id,
                    object_index=object_index,
                )
            )
        return decisions

    def _adapter_safe(self, outcome: Mapping[str, Any]) -> bool:
        if outcome.get("status") != "success" or outcome.get("actions"):
            return False
        state_update = outcome.get("state_update")
        if not isinstance(state_update, Mapping):
            return False
        request_updates = state_update.get("requests")
        if not isinstance(request_updates, list):
            return False
        for update in request_updates:
            if not isinstance(update, Mapping):
                return False
            fields = update.get("fields")
            if fields is None:
                continue
            request_id = update.get("request_id")
            canonical = self._canonical_fields.get(request_id)
            if (
                not isinstance(fields, Mapping)
                or canonical is None
                or fields.get("body") != canonical["body"]
            ):
                if canonical is not None and isinstance(request_id, str):
                    self._pending_lifecycle.append(
                        {
                            "event": "replace-request-fields",
                            "request_id": request_id,
                            "fields": canonical,
                        }
                    )
                    self._invalidate()
                return False
        return True

    def _record_outcome(
        self,
        operation: str,
        outcome: Mapping[str, Any],
    ) -> None:
        status = outcome.get("status")
        if status == "success":
            self._successful_outcomes += 1
            self._successful_by_operation[operation] += 1
        elif status == "fallback":
            self._fallback_outcomes += 1
        elif status == "unavailable":
            self._unavailable_outcomes += 1

    def _decision_meta(
        self,
        operation: str,
        submission_epoch: int,
    ) -> dict[str, Any]:
        return {
            "opportunity_id": self._opportunity_id(operation, submission_epoch),
            "snapshot": {"id": "host-filled", "revision": 0},
            "attempt": 0,
            "mechanics": [],
        }

    def _opportunity_id(self, operation: str, submission_epoch: int) -> str:
        return f"vllm:{self.target_id}:{operation}:{submission_epoch}"

    @staticmethod
    def _cache_under_pressure(scheduler: Scheduler) -> bool:
        pool = scheduler.kv_cache_manager.block_pool
        return (
            pool.num_gpu_blocks > 0
            and pool.get_num_free_blocks() * 2 <= pool.num_gpu_blocks
        )

    def _engine_facts(self, scheduler: Scheduler) -> dict[str, Any]:
        pool = scheduler.kv_cache_manager.block_pool
        submitted, dropped, completed = self.runtime.stats()
        return {
            "engine": "vllm",
            "model": self.model,
            "target_id": self.target_id,
            "membership_epoch": self.epoch,
            "queue_depth": len(scheduler.waiting) + len(scheduler.skipped_waiting),
            "running_requests": len(scheduler.running),
            "free_kv_blocks": pool.get_num_free_blocks(),
            "total_kv_blocks": pool.num_gpu_blocks,
            "schedule_plan_age_ms": self._age_ms(self._last_schedule_outcome_at),
            "cache_plan_age_ms": self._age_ms(self._last_cache_outcome_at),
            "async_submitted": submitted,
            "async_dropped": dropped,
            "async_completed": completed,
            "successful_outcomes": self._successful_outcomes,
            "schedule_successes": self._successful_by_operation["schedule"],
            "cache_successes": self._successful_by_operation["cache"],
            "feedback_successes": self._successful_by_operation["feedback"],
            "schedule_enactments": self._schedule_enactments,
            "schedule_partial_enactments": self._schedule_partial_enactments,
            "cache_enactments": self._cache_enactments,
            "fallback_outcomes": self._fallback_outcomes,
            "unavailable_outcomes": self._unavailable_outcomes,
            "rejected_outcomes": self._rejected_outcomes,
        }

    def _facts(self, request: Request) -> dict[str, Any]:
        principal_id = self._request_principal[request.request_id]
        request_id = self._engine_to_request[request.request_id]
        metadata = self._request_metadata.get(request_id, {})
        policy_facts = metadata.get("facts", {})
        return {
            **policy_facts,
            "engine_request_id": request.request_id,
            "client_id": principal_id,
            "attained_service": request.num_computed_tokens,
            "service_tokens": request.num_computed_tokens,
            "dispatch_input_tokens": max(
                request.num_prompt_tokens - request.num_computed_tokens,
                0,
            ),
            "generated_tokens": len(request.output_token_ids),
            "preempted": request.num_preemptions > 0,
            "preemptions": request.num_preemptions,
            "waiting_ms": max(int((time.time() - request.arrival_time) * 1000), 0),
            "cache_ready": request.num_computed_tokens > 0,
            "cached_tokens": request.num_computed_tokens,
        }

    def _feedback_facts(self, request: Request) -> dict[str, Any]:
        return {
            **self._facts(request),
            "committed_tokens": 0,
            "service_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "service_us": 0,
        }

    def _request_ref(self, request: Request) -> dict[str, Any]:
        return {
            "request_id": self._engine_to_request[request.request_id],
            "generation_id": self._request_generation[request.request_id],
            "group_id": self._request_group[request.request_id],
            "principal_id": self._request_principal[request.request_id],
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

    def _next_submission_epoch(self) -> int:
        self._submission_epoch += 1
        return self._submission_epoch

    def _is_fresh(self, submitted_at: float) -> bool:
        return time.monotonic() - submitted_at <= self.plan_ttl_s

    @staticmethod
    def _is_int(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    @staticmethod
    def _age_ms(timestamp: float | None) -> int | None:
        if timestamp is None:
            return None
        return max(int((time.monotonic() - timestamp) * 1000), 0)

    @staticmethod
    def _trim_submissions(submissions: dict[int, Any]) -> None:
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
    def _terminal_outcome(configured: str, reason: str) -> tuple[str, str]:
        if configured != "auto":
            mapping = {
                "completed": ("completed", "completed"),
                "failed": ("failed", "failed"),
                "cancelled": ("cancelled", "cancelled"),
                "expired": ("expired", "expired"),
            }
            return mapping[configured]
        lower = reason.lower()
        if "abort" in lower or "cancel" in lower:
            return "cancelled", "cancelled"
        if "error" in lower or "fail" in lower:
            return "failed", "failed"
        if "expire" in lower or "timeout" in lower:
            return "expired", "expired"
        return "completed", "completed"

    @staticmethod
    def _request_identity(
        request: Request,
    ) -> tuple[
        str,
        int,
        str,
        str | None,
        dict[str, int],
        dict[str, Any],
        bool,
        bool,
        str,
    ]:
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
        request_id = config.get(
            "request_id",
            config.get("logical_request_id", request.request_id),
        )
        legacy_request_id = config.get("logical_request_id")
        if (
            "request_id" in config
            and legacy_request_id is not None
            and legacy_request_id != request_id
        ):
            raise ValueError(
                "PLEX request_id and logical_request_id must match when both set"
            )
        generation_id = config.get("generation_id", 0)
        principal_id = config.get(
            "principal_id",
            config.get("tenant", "vllm-default"),
        )
        group_id = config.get("group_id")
        terminal = config.get("terminal", True)
        close_group = config.get("close_group", False)
        completion_outcome = config.get("completion_outcome", "auto")
        group_limits = config.get(
            "group_limits",
            {"max_members": 256, "max_scratch_bytes": 65536},
        )
        AsyncPlexPolicyController._validate_id("request_id", request_id)
        if (
            not isinstance(generation_id, int)
            or isinstance(generation_id, bool)
            or generation_id < 0
            or generation_id > 2**64 - 1
        ):
            raise ValueError("PLEX generation_id must be an unsigned 64-bit integer")
        AsyncPlexPolicyController._validate_id("principal_id", principal_id)
        if group_id is not None:
            AsyncPlexPolicyController._validate_id("group_id", group_id)
        if not isinstance(terminal, bool) or not isinstance(close_group, bool):
            raise ValueError("PLEX terminal and close_group must be booleans")
        if completion_outcome not in {
            "auto",
            "completed",
            "failed",
            "cancelled",
            "expired",
        }:
            raise ValueError("PLEX completion_outcome is invalid")
        if not isinstance(group_limits, Mapping):
            raise ValueError("PLEX group_limits must be an object")
        max_members = group_limits.get("max_members", 256)
        max_scratch_bytes = group_limits.get("max_scratch_bytes", 65536)
        if (
            not isinstance(max_members, int)
            or isinstance(max_members, bool)
            or max_members <= 0
            or max_members > 2**32 - 1
            or not isinstance(max_scratch_bytes, int)
            or isinstance(max_scratch_bytes, bool)
            or max_scratch_bytes <= 0
            or max_scratch_bytes > 2**64 - 1
        ):
            raise ValueError("PLEX group limits exceed the v0.6 integer bounds")
        metadata = config.get("metadata")
        if metadata is None:
            metadata = {
                key: value
                for key, value in config.items()
                if key
                not in {
                    "request_id",
                    "logical_request_id",
                    "generation_id",
                    "principal_id",
                    "tenant",
                    "group_id",
                    "group_limits",
                    "terminal",
                    "close_group",
                    "completion_outcome",
                }
            }
        if not isinstance(metadata, Mapping):
            raise ValueError("PLEX metadata must be an object")
        json.dumps(metadata, allow_nan=False)
        policy_facts = metadata.get("facts", {})
        if not isinstance(policy_facts, Mapping) or any(
            not isinstance(key, str) for key in policy_facts
        ):
            raise ValueError("PLEX metadata.facts must be an object")
        return (
            request_id,
            generation_id,
            principal_id,
            group_id,
            {
                "max_members": max_members,
                "max_scratch_bytes": max_scratch_bytes,
            },
            dict(metadata),
            terminal,
            close_group,
            completion_outcome,
        )

    @staticmethod
    def _validate_id(name: str, value: object) -> None:
        if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 128:
            raise ValueError(
                f"PLEX {name} must be a non-empty string of at most 128 bytes"
            )
