"""PLEX v2 stage-3 attach: the verbs, mapped onto vLLM's own primitives.

Stage ③ of the phased attach. A policy stages a verb and the engine either
enacts it through a primitive it already has, or **declines with a reason**.

## The two rules this file exists to keep

**Declines are declared, not discovered.** `.wiki/v2/plan.md` calls this the
hooks §8 lesson. A verb that quietly does nothing is indistinguishable from one
that lost a race, and the two call for opposite responses: a policy that lost a
race should retry, and a policy whose verb this engine cannot perform should
stop asking. `plex-port-vllm::capabilities()` declares which is which, and
[`enact`] refuses to invent a mechanism for the ones declared unavailable.

**A refusal is an answer.** Every path returns a `Refusal` rather than a bool,
because "it did not happen" is not a reason, and `on-refused` carries the
reason to the policy so its beliefs can be corrected. v0.7's post-mortem is
largely about beliefs that could not be corrected.

## What vLLM can and cannot do

  pause      preemption. `_preempt_request` returns a running request to the
             waiting queue and zeroes its computed tokens: vLLM recomputes
             rather than swapping, so `preserve` cannot be honoured and a
             pause here is always `release`.
  finish     `finish_requests`, the abort path the API server already uses.
  prefetch   declined. vLLM computes KV, it does not load it — a block exists
             because a request produced it, and there is no primitive that
             makes a cold prefix resident without running the prefill.
  move       declined. No CPU-offload path in the v1 scheduler a port may
             drive; the swap machinery is the scheduler's own.
  rebalance  declined. One engine, one target: rebalance is the router
             world's verb.

The interesting entry is `pause`. The contract lets a policy ask for the KV to
be preserved, and vLLM cannot — so the port declines *that disposition*
specifically rather than the verb, which is a finer answer than "no" and the
one a policy can act on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler


class Refusal(NamedTuple):
    """Why a staged verb did not take effect.

    A closed set, mirroring `wit/io.wit`'s `refusal` variant. Free-text
    would let a port invent reasons a policy cannot match on, which is the
    same failure as a fact name nobody publishes.
    """

    verb: str
    subject: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"verb": self.verb, "subject": self.subject, "reason": self.reason}


#: `reason` values, closed. Each maps to a `refusal` case in the contract.
UNKNOWN_SUBJECT = "unknown-subject"
WRONG_STATE = "wrong-request-state"
UNSUPPORTED = "unsupported-on-this-engine"
UNSUPPORTED_DISPOSITION = "unsupported-kv-disposition"


class PlexVerbs:
    """Staged verbs, enacted through vLLM's own primitives.

    Holds no policy and stages nothing itself: it is handed intents that
    have already been committed by the host transaction, and its whole job
    is to turn each into a primitive call or a refusal.
    """

    @staticmethod
    def maybe(scheduler: Scheduler) -> PlexVerbs | None:
        """Attach verbs if a source was named, and otherwise cost nothing.

        `VLLM_PLEX_VERBS=/path/to/verbs.jsonl` — one staged verb per
        line, drained at the step boundary. A file rather than a callback
        for the same reason the table is one: the policy runs in the PLEX
        host, and the engine must not import a runtime, block on one, or
        be able to fail because one is slow.
        """
        import os

        if not os.environ.get("VLLM_PLEX_VERBS"):
            return None
        return PlexVerbs(scheduler)

    def drain(self) -> int:
        """Enact everything staged since the last call.

        **Called at the step boundary, never mid-pass.** A verb is a
        write, and vLLM's scheduling loop is not reentrant with respect
        to its own running list — `_pause` removes from `running`, which
        is exactly what the loop is iterating. The standing table learned
        this by crashing the engine; the verbs get it for free by
        draining where the table reloads.

        Returns how many took effect. Lines that cannot be read are
        skipped rather than failing the engine: a malformed instruction
        is the policy's error and stopping inference is not the
        proportionate response.
        """
        import json
        import os

        path = os.environ.get("VLLM_PLEX_VERBS")
        if not path:
            return 0
        try:
            with open(path, encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            return 0
        if len(lines) <= self._drained:
            return 0

        applied = 0
        for line in lines[self._drained :]:
            try:
                staged = json.loads(line)
                verb = str(staged["verb"])
                subject = str(staged["subject"])
            except (ValueError, KeyError, TypeError):
                continue
            kwargs = {
                key: value
                for key, value in staged.items()
                if key not in ("verb", "subject")
            }
            if self.enact(verb, subject, **kwargs):
                applied += 1
        self._drained = len(lines)
        return applied

    def __init__(self, scheduler: Scheduler) -> None:
        self._scheduler = scheduler
        # How much of the staged file has already been enacted. A verb is
        # an instruction, not a document: replaying one would pause a
        # request twice.
        self._drained = 0
        self._refusals: list[Refusal] = []

    def take_refusals(self) -> list[Refusal]:
        """Drain what was refused, for delivery as `on-refused`.

        Drained rather than read, because a refusal delivered twice is one
        correction a policy applies twice.
        """
        refusals, self._refusals = self._refusals, []
        return refusals

    def enact(self, verb: str, subject: str, **kwargs: object) -> bool:
        """Enact one verb. Returns whether it took effect.

        The bool is for the caller's own accounting; the *reason* goes to
        the policy through `take_refusals`, because a bool cannot say why
        and a policy that cannot tell a lost race from an unsupported verb
        will either retry forever or stop asking when it should not.
        """
        handler = {
            "pause": self._pause,
            "finish": self._finish,
        }.get(verb)
        if handler is None:
            # Declared unavailable rather than unrecognised. The
            # distinction matters: `prefetch` is a verb the contract has
            # and this engine cannot perform, which is a different fact
            # from a typo.
            self._refuse(verb, subject, UNSUPPORTED)
            return False
        return handler(subject, **kwargs)

    # ── the two vLLM can do ──────────────────────────────────────────────

    def _pause(self, subject: str, kv: str = "release", **_: object) -> bool:
        request = self._scheduler.requests.get(subject)
        if request is None:
            self._refuse("pause", subject, UNKNOWN_SUBJECT)
            return False
        if kv == "preserve":
            # The finer answer. vLLM's preemption zeroes `num_computed_tokens`
            # and frees the blocks — it recomputes rather than swapping — so
            # the disposition cannot be honoured even though the verb can.
            # Refusing the verb outright would tell the policy less than it
            # needs: it may well want the pause anyway.
            self._refuse("pause", subject, UNSUPPORTED_DISPOSITION)
            return False

        from vllm.v1.request import RequestStatus

        if request.status != RequestStatus.RUNNING:
            # `_preempt_request` asserts this, and an assert is not an
            # answer. A policy pausing something that already stopped has
            # lost a race, which is exactly what `wrong-request-state`
            # says.
            self._refuse("pause", subject, WRONG_STATE)
            return False

        # The scheduler's own contract: pop from running first.
        import time

        self._scheduler.running.remove(request)
        self._scheduler._preempt_request(request, time.time())
        return True

    def _finish(self, subject: str, **_: object) -> bool:
        request = self._scheduler.requests.get(subject)
        if request is None:
            self._refuse("finish", subject, UNKNOWN_SUBJECT)
            return False

        from vllm.v1.request import RequestStatus

        if request.is_finished():
            self._refuse("finish", subject, WRONG_STATE)
            return False
        self._scheduler.finish_requests(subject, RequestStatus.FINISHED_ABORTED)
        return True

    def _refuse(self, verb: str, subject: str, reason: str) -> None:
        self._refusals.append(Refusal(verb, subject, reason))
