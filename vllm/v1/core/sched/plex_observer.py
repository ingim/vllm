"""PLEX v2 observer: the scheduler, described in the contract's vocabulary.

Stage ① of the phased attach in `.wiki/v2/plan.md`: events and facts flowing,
zero influence. The engine runs natively and a policy watches.

## The constraint this file exists to satisfy

Phase 3's standing rule is "no engine-core semantic changes, nothing upstream
would refuse to merge". v0.7 is the cautionary measurement, not a template:
1,146 lines added to vLLM's own files, 611 of them in `scheduler.py` with 50
deletions. A patch that rewrites a tenth of the scheduler is not one upstream
merges, whatever its merits.

Breaking that number down by attach stage
(`scripts/measure-vllm-invasiveness.py`) gives the useful finding: only 31 of
those 611 lines are stage ①. The rest are enactment — reordering and
preemption — which genuinely has to touch the scheduler. Observation does not.

## Why this is smaller, and it is not cleverness

v0.7 derived facts inside the engine: `async_plex.py` is 1,485 lines and most
of it computes quantities — prefix hit ratios, virtual usage, cache capacity —
from vLLM state. v2 moves that out. `plex-port-vllm` turns a JSON step
document into the contract's world, so this file only *reads* what vLLM
already holds.

The test of that split is arithmetic: nothing here computes a number the port
could compute itself. Where a division appears it is because the engine holds
both operands and the port holds neither — blocks to tokens, which needs the
block size.

## What attaching this can change

Nothing. There is no branch on policy output, because stage ① has none. That
is what makes observer parity checkable: run the engine with and without it
and the outputs must be identical. If they are not, this file is doing
something it should not.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request


class PlexObserver:
    """One scheduler, in the contract's vocabulary. Holds no policy.

    Emits one JSON document per scheduler step. What consumes it is the
    port's business — an observer that owns its transport is an observer
    that can block the engine.
    """

    def __init__(self, scheduler: Scheduler, sink: Any, target: str = "vllm-0") -> None:
        self._scheduler = scheduler
        self._sink = sink
        self._target = target
        self._step = 0
        # Arrival order, which nothing in vLLM records. The only quantity
        # here the engine does not already hold, and it is unavoidable: a
        # fairness policy that cannot tell which request came first has no
        # fairness to enforce, and `arrival_time` is a float whose ties are
        # broken by nothing.
        self._arrival_seq: dict[str, int] = {}
        self._arrivals = 0
        # Edges since the last document. Cleared on emit, because an event
        # delivered twice is one thing a policy counts as two.
        self._admitted: list[str] = []
        # Stage 3, if a source was named. `None` costs nothing, which is
        # what lets stage 1 be reviewed on its own.
        from vllm.v1.core.sched.plex_verbs import PlexVerbs

        self._verbs = PlexVerbs.maybe(scheduler)
        self._finished: list[list[str]] = []

    # ── the hooks, and there are two ─────────────────────────────────────

    def on_request_added(self, request: Request) -> None:
        """A request entered the scheduler."""
        if request.request_id not in self._arrival_seq:
            self._arrival_seq[request.request_id] = self._arrivals
            self._arrivals += 1
        self._admitted.append(request.request_id)

    def on_request_freed(self, request: Request) -> None:
        """A request left, by any path.

        Hooked at `_free_request`, which both the natural-stop path
        (`update_from_output`) and the external-abort path
        (`finish_requests`) pass through. One hook rather than two, and no
        way for a terminal edge to be missed by hooking only one of them.

        `host` as initiator because the engine ended it. A policy-initiated
        finish is stage ③ and does not exist yet; recording it as `policy`
        now would attribute the engine's own decisions to a component that
        made none.
        """
        self._arrival_seq.pop(request.request_id, None)
        self._finished.append([request.request_id, "completed", "host"])

    def emit_step(self) -> None:
        """Write one step document to the sink, and never fail the engine.

        A port that can raise inside the scheduler is a port that can take
        the engine down, and stage 1 is supposed to be unable to change
        anything — including whether the step completes. So the write is
        guarded and a failure disables the observer rather than propagating:
        losing observation is a measurement problem, losing the step is an
        outage.
        """
        try:
            self._sink.write(self.on_step() + "\n")
            self._sink.flush()
        except Exception:  # noqa: BLE001 - see docstring
            self._sink = None
            self._scheduler.plex_observer = None

    def drain_verbs(self) -> int:
        """Enact staged verbs. Called before a scheduling pass begins.

        Separate from `on_step` because a verb is a *write* and the
        observer's hook sits inside `schedule()`. `pause` removes from
        `scheduler.running`, and doing that after the scheduling loops
        have run trips the engine's own assert on resumed requests — the
        second time a PLEX write applied at the wrong moment has crashed
        a real engine. Observation is safe mid-pass; writes are not.
        """
        if self._verbs is None:
            return 0
        return self._verbs.drain()

    def on_step(self) -> str:
        """One scheduler step, as a document the port reads."""
        # A standing table lands here, between scheduling passes, because
        # this is the one place per step where the scheduler is provably
        # not iterating its own queue. Reloading inside `pop_request`
        # crashed a real engine with `KeyError`: vLLM peeks a request,
        # allocates blocks for it, then pops, and a re-sort in between
        # hands it a request it never allocated for.
        reload_table = getattr(self._scheduler.waiting, "reload", None)
        if reload_table is not None:
            reload_table()
        self._step += 1
        document_now_ms = int(time.time() * 1000)
        document = {
            "step": self._step,
            "now-ms": document_now_ms,
            "target": self._target,
            "subjects": self._subjects(),
            "facts": self._facts(document_now_ms),
            "events": self._events(),
        }
        self._admitted.clear()
        self._finished.clear()
        return json.dumps(document)

    # ── scraping ─────────────────────────────────────────────────────────

    def _tracked(self) -> list[Request]:
        scheduler = self._scheduler
        return [*scheduler.waiting, *scheduler.running]

    def _subjects(self) -> dict[str, list[str]]:
        return {
            "request": [request.request_id for request in self._tracked()],
            "target": [self._target],
        }

    def _facts(self, now_ms: int) -> dict[str, dict[str, Any]]:
        running_ids = {request.request_id for request in self._scheduler.running}
        facts: dict[str, dict[str, Any]] = {}
        for request in self._tracked():
            running = request.request_id in running_ids
            prompt = request.num_prompt_tokens
            computed = request.num_computed_tokens
            generated = len(request.output_token_ids)
            facts[request.request_id] = {
                "state": {"text": "active" if running else "admitted"},
                "arrival_seq": {"num": self._arrival_seq.get(request.request_id, 0)},
                "arrival_ms": {"num": int(request.arrival_time * 1000)},
                "prompt_tokens": {"num": prompt},
                "generated_tokens": {"num": generated},
                "computation_length": {"num": prompt + generated},
                "dispatch_input_tokens": {"num": max(prompt - computed, 0)},
                "cached_tokens": {"num": computed},
                "queue_member": {"flag": not running},
                "preempted": {"flag": request.num_preemptions > 0},
                # How long it has been here. The engine holds both
                # operands and the port holds neither, which is the
                # standing test for whether a derivation belongs on this
                # side.
                "waiting_ms": {"num": max(now_ms - int(request.arrival_time * 1000), 0)},
            }
        facts[self._target] = self._target_facts()
        return facts

    def _target_facts(self) -> dict[str, Any]:
        scheduler = self._scheduler
        pool = scheduler.kv_cache_manager.block_pool
        block_size = scheduler.cache_config.block_size
        total_blocks = pool.num_gpu_blocks
        free_blocks = pool.get_num_free_blocks()
        running = list(scheduler.running)
        max_running = scheduler.max_num_running_reqs
        # Output tokens the engine still owes, and prompt tokens it has
        # yet to prefill. Both are sums over state the scheduler already
        # holds; neither is a model of anything.
        pending_decode = sum(
            max(request.max_tokens - len(request.output_token_ids), 0)
            for request in running
        )
        queued_tokens = sum(
            max(request.num_prompt_tokens - request.num_computed_tokens, 0)
            for request in scheduler.waiting
        )
        decoding = sum(1 for request in running if request.output_token_ids)
        return {
            "queue_depth": {"num": len(scheduler.waiting)},
            "running_requests": {"num": len(running)},
            "batch_size": {"num": len(running)},
            "decode_batch_size": {"num": len(running)},
            "max_batch_size": {"num": max_running},
            # An alias the corpus reads under a second name (helium,
            # llumnix). Published rather than left absent: a policy
            # reading it gets `unknown-key` and silently takes a default,
            # which is how `targets` made thirteen cache policies inert.
            "max_requests": {"num": max_running},
            "free_decode_slots": {"num": max(max_running - len(running), 0)},
            "pending_decode_tokens": {"num": pending_decode},
            "queued_tokens": {"num": queued_tokens},
            "decoder_ratio_ppm": {
                "num": min(decoding * 1_000_000 // len(running), 1_000_000)
                if running
                else 0
            },
            "kv_overloaded": {"flag": free_blocks < total_blocks / 10},
            "used_kv_ppm": {
                "num": min(
                    (total_blocks - free_blocks) * 1_000_000 // total_blocks,
                    1_000_000,
                )
                if total_blocks
                else 0
            },
            # Blocks to tokens: the one derivation here, and only because
            # the block size lives on the engine and the port has no way
            # to know it.
            "total_kv_tokens": {"num": total_blocks * block_size},
            "free_kv_tokens": {"num": free_blocks * block_size},
            "max_total_tokens": {"num": total_blocks * block_size},
        }

    def _events(self) -> dict[str, Any]:
        events: dict[str, Any] = {}
        if self._admitted:
            events["admitted"] = list(self._admitted)
        if self._finished:
            events["finished"] = [list(entry) for entry in self._finished]
        return events


def maybe_plex_observer(scheduler: Scheduler) -> PlexObserver | None:
    """Attach an observer if one was asked for, and otherwise cost nothing.

    Environment rather than config, deliberately. Stage 1 changes no
    behaviour, so it needs no place in a config schema a user has to
    understand — and a feature that is off by default and invisible when
    off is the easiest kind to review.

        VLLM_PLEX_OBSERVE=/path/to/steps.jsonl

    A path that cannot be opened disables the observer rather than failing
    startup, for the same reason `emit_step` swallows: observation must not
    be able to stop the engine.
    """
    path = os.environ.get("VLLM_PLEX_OBSERVE")
    if not path:
        return None
    try:
        sink = open(path, "a", encoding="utf-8")  # noqa: SIM115 - lives with the scheduler
    except OSError:
        return None
    target = os.environ.get("VLLM_PLEX_TARGET", "vllm-0")
    return PlexObserver(scheduler, sink, target)
