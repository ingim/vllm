"""PLEX v2 stage-2 attach: the policy's order, as a request queue.

Stage ② of the phased attach in `.wiki/v2/plan.md`: the standing tables.
A policy installs an order and the engine follows it.

## Why this is a queue and not a patch to the scheduler

v0.7 reordered inside `schedule()`, which is why its scheduler diff is 611
lines. vLLM already has the extension point this needs: `RequestQueue` is an
abstract base with two implementations, and `create_request_queue` picks one
from a policy enum. A standing schedule table *is* a queue policy — an order
over waiting requests — so expressing it as one costs a subclass rather than a
rewrite, and every caller in the scheduler keeps working because the interface
is unchanged.

The measured difference: stage ② here is a handful of lines at the factory,
against 179 in v0.7's `scheduler.py`.

## What "follow the table" means, precisely

The table is a *total order over the requests it names*. It is not a promise
that those requests run — capacity decides that, and the scheduler's own
budget logic is untouched. So:

- A request the table names is ordered where the table puts it.
- A request the table does not name sorts after every request it does, in
  arrival order among themselves. Unnamed is not "last by preference", it is
  "the policy expressed no view", and arrival order is what the engine would
  have done unaided.
- A table naming a request that has since left is ignored for that request.
  Tables are standing documents and the world moves under them; `.wiki` calls
  this table age, and the log records it so a stale plan is visible rather
  than silently obeyed.

## What this cannot do, and must not pretend to

It cannot preempt, pause, or evict — those are verbs, and verbs are stage ③.
A queue can only order what is waiting. A policy whose table implies a running
request should yield gets no such effect here, and that is the honest
behaviour: the alternative is a port that half-enacts and reports success.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

from vllm.v1.core.sched.request_queue import RequestQueue

if TYPE_CHECKING:
    from vllm.v1.request import Request


class PlexRequestQueue(RequestQueue):
    """Waiting requests, ordered by the policy's standing table.

    Holds the same requests a `FCFSRequestQueue` would and answers the
    same interface; only the order differs, and only for requests the
    table names.
    """

    def __init__(self, source: str | None = None) -> None:
        self._requests: deque[Request] = deque()
        # Where the policy's table arrives from. A path rather than a
        # callback because the policy runs in the PLEX host, not in the
        # engine: the engine must not import a runtime, block on one, or
        # be able to fail because one is slow.
        self._source = source if source else os.environ.get("VLLM_PLEX_TABLE")
        self._source_stamp: tuple[int, int] | None = None

        # ── admission hold (the route channel) ───────────────────────────
        #
        # A gate rules on requests that have arrived and have not yet been
        # admitted. vLLM has no such state: `add_request` puts a request
        # straight into the waiting queue, so a policy answering `reject`
        # would be answering about something already accepted.
        #
        # Holding arrivals here creates the state the verdict is about.
        # It is a behaviour change and therefore stage 2, which is where
        # it belongs: admission control that cannot withhold admission is
        # not admission control.
        self._gate = os.environ.get("VLLM_PLEX_GATE")
        self._held: dict[str, tuple[Request, float]] = {}
        self._verdict_stamp: tuple[int, int] | None = None
        # How long a request may be held with no verdict.
        #
        # Not optional, and not large. If the bridge dies, every held
        # request would otherwise be stranded forever — a far worse
        # failure than a stale table, because a stale table still serves
        # everyone. `hooks.md` pairs a deadline with a *declared default*
        # so that silence and slowness produce the same behaviour, and
        # the declared default here is the engine's own: admit.
        #
        # Admit rather than reject because stage 1's principle carries
        # forward — a policy that does not answer must not change what
        # the engine would have done.
        self._hold_ms = float(os.environ.get("VLLM_PLEX_GATE_MS", "50"))
        self.released_by_deadline = 0
        # request id -> rank. Replaced wholesale on install: a table is a
        # document, not a stream of edits, so a partially-applied one is
        # not representable.
        self._rank: dict[str, int] = {}
        # Installs seen, and how many arrivals ago the current table was
        # installed. Age travels with the decision so a plan that ruled on
        # a queue that no longer exists is visible — v0.7 had one arm's
        # plan running 222 arrivals stale and no way to see it.
        self._installs = 0
        # Instrumentation; see `_resort`.
        self.resorts = 0
        self.queued = 0
        self.ranked = 0
        self.moved = 0
        self._arrivals = 0
        self._installed_at_arrival = 0
        # Requests a policy refused entry. Kept rather than counted: the
        # engine has to answer the caller, and "rejected by policy" is a
        # different answer from "failed".
        self.rejected_by_policy: list[str] = []

    # ── the standing table ───────────────────────────────────────────────

    def install(self, order: list[str]) -> None:
        """Replace the standing order.

        Whole-document, because that is what makes an install atomic and
        idempotent: a policy that reinstalls says exactly what it wants
        and cannot half-apply. The measured cost is a few hundred bytes
        per install, which does not buy delta encoding.
        """
        self._rank = {request_id: rank for rank, request_id in enumerate(order)}
        self._installs += 1
        self._installed_at_arrival = self._arrivals
        self._resort()

    @property
    def table_age(self) -> int:
        """Arrivals since the current table was installed."""
        return self._arrivals - self._installed_at_arrival

    @property
    def installs(self) -> int:
        return self._installs

    def _sort_key(self, request: Request) -> tuple[int, int, float]:
        # Named requests first, in the table's order. Unnamed after, in
        # arrival order among themselves — "the policy expressed no view"
        # rather than "the policy ranked these last", which are different
        # claims and only one of them is true.
        rank = self._rank.get(request.request_id)
        if rank is None:
            return (1, 0, request.arrival_time)
        return (0, rank, request.arrival_time)

    def _resort(self) -> None:
        before = [request.request_id for request in self._requests]
        ordered = sorted(self._requests, key=self._sort_key)
        self._requests = deque(ordered)

        # Did the table actually move anything?
        #
        # `decision alone = 1.000x` has two explanations that look
        # identical from outside: the engine honoured an order that
        # happened not to matter, or the engine never had one to honour.
        # The counters separate them. `ranked` is how many of the queued
        # requests the policy named -- a policy ranking none of them has
        # decided nothing whatever the metric says -- and `moved` is how
        # many changed position, which is the reordering the engine
        # actually performed.
        after = [request.request_id for request in self._requests]
        self.resorts += 1
        self.ranked += sum(1 for rid in before if rid in self._rank)
        self.queued += len(before)
        self.moved += sum(1 for a, b in zip(before, after) if a != b)
        if self.resorts % 50 == 0:
            print(
                f"[plex-queue] resorts={self.resorts} queued={self.queued} "
                f"ranked={self.ranked} moved={self.moved} "
                f"installs={self._installs} table={len(self._rank)}",
                file=sys.stderr,
                flush=True,
            )

    # ── RequestQueue ─────────────────────────────────────────────────────

    def reload(self) -> None:
        """Pick up a newly written table, if there is one.

        **Call this between scheduling passes, never during one.** The
        first version polled inside `pop_request`, and a real engine
        crashed with `KeyError` on a request id: vLLM's scheduling loop
        peeks a request, allocates blocks for it, then pops — so a
        re-sort between the peek and the pop hands it a request it never
        allocated for. The contract already says this at the policy
        level ("facts do not move under a policy mid-call"); a standing
        table owes the engine the same courtesy.

        Polling rather than notification, and on the engine's own thread.
        The alternative is a watcher that can wake the scheduler, which
        makes a policy able to affect *when* the engine runs rather than
        only what it prefers — a much larger claim than stage 2 makes.

        A table that is missing, unreadable or malformed leaves the
        current one standing. The engine has a working order either way,
        and an order it can explain is better than one it half-applied.
        """
        if not self._source:
            return
        try:
            stat = os.stat(self._source)
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp == self._source_stamp:
                return
            with open(self._source, encoding="utf-8") as handle:
                order = json.load(handle)
            if not isinstance(order, list):
                return
            self._source_stamp = stamp
        except (OSError, ValueError):
            return
        self.install([str(entry) for entry in order])

    # ── the gate ─────────────────────────────────────────────────────────

    def _pick_up_verdicts(self) -> None:
        """Apply any verdicts the policy has written.

        `{"request-id": "assign"|"defer"|"reject"}`. A verdict names a
        request the gate was offered; anything else is ignored, because a
        policy ruling on a request this engine does not hold has stale
        beliefs and acting on it would be acting on someone else's world.
        """
        if not self._gate:
            return
        try:
            stat = os.stat(self._gate)
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp == self._verdict_stamp:
                return
            with open(self._gate, encoding="utf-8") as handle:
                verdicts = json.load(handle)
            if not isinstance(verdicts, dict):
                return
            self._verdict_stamp = stamp
        except (OSError, ValueError):
            return

        for request_id, verdict in verdicts.items():
            entry = self._held.get(str(request_id))
            if entry is None:
                continue
            request, _ = entry
            if verdict == "assign":
                del self._held[str(request_id)]
                self._requests.append(request)
            elif verdict == "reject":
                # Released *and* recorded, not dropped.
                #
                # A first version deleted the request from the hold and
                # queued nothing. The engine hung: the caller was still
                # waiting for an answer that no longer had any code path
                # to produce. A refusal is a kind of *ending*, not a kind
                # of forgetting, and a request that vanishes without an
                # outcome is worse than one that is served.
                #
                # So it goes back into the queue and its id is recorded;
                # the observer reports it and the scheduler ends it
                # through the engine's own terminal path. `reject` stays
                # a verdict rather than a verb because the decision is
                # about *entry* — no standing table can say "never" — but
                # enacting it still has to use the one way this engine
                # finishes things.
                del self._held[str(request_id)]
                self._requests.append(request)
                self.rejected_by_policy.append(str(request_id))
            # `defer` keeps it held, which is the whole point of `defer`.

    def _release_expired(self) -> None:
        """Admit anything held past the deadline.

        The declared default. A policy that never answers and a policy
        that answers slowly produce the same behaviour, and that
        behaviour is what the engine would have done unaided.
        """
        if not self._held:
            return
        now = time.monotonic()
        expired = [
            request_id
            for request_id, (_, since) in self._held.items()
            if (now - since) * 1000.0 >= self._hold_ms
        ]
        for request_id in expired:
            request, _ = self._held.pop(request_id)
            self._requests.append(request)
            self.released_by_deadline += 1

    def held_requests(self) -> list[Request]:
        """Requests awaiting a verdict — the contract's `pending`."""
        return [request for request, _ in self._held.values()]

    def _gate_tick(self) -> None:
        self._pick_up_verdicts()
        self._release_expired()

    # ── RequestQueue ─────────────────────────────────────────────────────

    def add_request(self, request: Request) -> None:
        self._arrivals += 1
        if self._gate:
            # Held rather than queued: this is the moment a gate rules
            # on, and it does not otherwise exist in vLLM.
            self._held[request.request_id] = (request, time.monotonic())
            return
        self._requests.append(request)
        self._resort()

    def pop_request(self) -> Request:
        return self._requests.popleft()

    def peek_request(self) -> Request:
        if not self._requests:
            raise IndexError("peek from empty queue")
        return self._requests[0]

    def prepend_request(self, request: Request) -> None:
        # Deliberately not re-sorted. `prepend` is the scheduler putting a
        # request back after failing to schedule it, and it means "this one
        # goes next" regardless of any table — a policy's order must not be
        # able to send a request the engine just took back to the end of
        # the queue and starve it.
        self._requests.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        self._requests.extendleft(reversed(list(requests)))

    def remove_request(self, request: Request) -> None:
        self._requests.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        removing = set(requests)
        self._requests = deque(
            request for request in self._requests if request not in removing
        )

    def __bool__(self) -> bool:
        # The hold is advanced here, and its result is what is reported.
        #
        # Two wrong versions came first, and both were instructive.
        #
        # Reporting only `bool(self._requests)` **hung a real engine**:
        # with every arrival held the queue looked empty, the scheduler
        # concluded there was nothing to do and stopped stepping, so the
        # observer stopped running, so the deadline never advanced, so
        # nothing was ever released. A deadline only checked while the
        # engine is busy is not a deadline.
        #
        # Reporting `_requests or _held` **crashed it**: the scheduler
        # believed there was work, called `peek_request`, and found an
        # empty deque. Held is not schedulable — that is the entire
        # point of holding it.
        #
        # So: tick first, then answer about what is genuinely
        # schedulable. The tick is what makes the declared default real;
        # the answer is what keeps the engine's own contract.
        if self._held:
            self._gate_tick()
        return bool(self._requests)

    def __len__(self) -> int:
        if self._held:
            self._gate_tick()
        return len(self._requests)

    def __iter__(self) -> Iterator[Request]:
        return iter(self._requests)

    def __reversed__(self) -> Iterator[Request]:
        return reversed(self._requests)
