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

    def __init__(self) -> None:
        self._requests: deque[Request] = deque()
        # request id -> rank. Replaced wholesale on install: a table is a
        # document, not a stream of edits, so a partially-applied one is
        # not representable.
        self._rank: dict[str, int] = {}
        # Installs seen, and how many arrivals ago the current table was
        # installed. Age travels with the decision so a plan that ruled on
        # a queue that no longer exists is visible — v0.7 had one arm's
        # plan running 222 arrivals stale and no way to see it.
        self._installs = 0
        self._arrivals = 0
        self._installed_at_arrival = 0

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
        ordered = sorted(self._requests, key=self._sort_key)
        self._requests = deque(ordered)

    # ── RequestQueue ─────────────────────────────────────────────────────

    def add_request(self, request: Request) -> None:
        self._arrivals += 1
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
        return bool(self._requests)

    def __len__(self) -> int:
        return len(self._requests)

    def __iter__(self) -> Iterator[Request]:
        return iter(self._requests)

    def __reversed__(self) -> Iterator[Request]:
        return reversed(self._requests)
