"""The policy's standing table, as a request queue.

Stage 2 of the phased attach. A standing schedule table is an order over
waiting requests, which is exactly what `RequestQueue` abstracts, so
expressing it as one costs a subclass rather than a rewrite of
`schedule()` -- v0.7 spent 179 lines in the scheduler doing this and the
factory hook here is 7.

These tests hold the queue to the three claims that make "follow the
table" mean something precise, and to the one it must not make.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vllm.v1.core.sched.plex_queue import PlexRequestQueue  # noqa: E402


class FakeRequest:
    """A stand-in for `Request`, hashable by identity as the real one is.

    `SimpleNamespace` is not hashable, and using one here made
    `remove_requests` look broken when it matches upstream's own
    implementation exactly (`request_queue.py:112`). The stand-in has to
    stand in for the properties the code under test relies on, and
    hashability is one of them.
    """

    def __init__(self, rid, arrival):
        self.request_id = rid
        self.arrival_time = arrival


def req(rid, arrival):
    return FakeRequest(rid, arrival)


def ids(queue):
    return [request.request_id for request in queue]


def filled(*pairs):
    queue = PlexRequestQueue()
    for rid, arrival in pairs:
        queue.add_request(req(rid, arrival))
    return queue


def test_without_a_table_the_order_is_arrival():
    # The engine's own behaviour, which is what an uninstalled table must
    # reproduce exactly: stage 2 with no policy attached is stage 1.
    queue = filled(("c", 3.0), ("a", 1.0), ("b", 2.0))
    assert ids(queue) == ["a", "b", "c"]


def test_a_table_reorders_the_requests_it_names():
    queue = filled(("a", 1.0), ("b", 2.0), ("c", 3.0))
    queue.install(["c", "a", "b"])
    assert ids(queue) == ["c", "a", "b"]


def test_an_unnamed_request_sorts_after_named_ones_in_arrival_order():
    # "The policy expressed no view" is not "the policy ranked these
    # last", and only one of those is true. Among themselves they keep
    # the order the engine would have used unaided.
    queue = filled(("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0))
    queue.install(["d"])
    assert ids(queue) == ["d", "a", "b", "c"]


def test_a_request_arriving_after_the_install_is_placed_by_the_table():
    queue = filled(("a", 1.0))
    queue.install(["b", "a"])
    queue.add_request(req("b", 2.0))
    assert ids(queue) == ["b", "a"], "the table is standing, not a one-shot"


def test_a_table_naming_a_departed_request_is_harmless():
    queue = filled(("a", 1.0), ("b", 2.0))
    queue.install(["gone", "b", "a"])
    assert ids(queue) == ["b", "a"]


def test_installing_replaces_rather_than_merges():
    # A table is a document, so a partially-applied one is not
    # representable. The second install must not leave the first's
    # ranking behind for requests it does not mention.
    queue = filled(("a", 1.0), ("b", 2.0), ("c", 3.0))
    queue.install(["c", "b", "a"])
    assert ids(queue) == ["c", "b", "a"]
    queue.install(["b"])
    assert ids(queue) == ["b", "a", "c"], "a and c revert to arrival order"


def test_prepend_beats_the_table():
    # `prepend` is the scheduler putting a request back after failing to
    # schedule it: it means "this one goes next". A policy's order must
    # not be able to send it to the end and starve it.
    queue = filled(("a", 1.0), ("b", 2.0))
    queue.install(["a", "b"])
    queue.remove_request(next(r for r in queue if r.request_id == "b"))
    queue.prepend_request(req("b", 2.0))
    assert ids(queue) == ["b", "a"]


def test_table_age_is_counted_in_arrivals():
    # A plan that ruled on a queue that no longer exists is a real v0.7
    # finding -- one arm's plan was 222 arrivals old with no way to see it.
    queue = filled(("a", 1.0))
    queue.install(["a"])
    assert queue.table_age == 0
    queue.add_request(req("b", 2.0))
    queue.add_request(req("c", 3.0))
    assert queue.table_age == 2
    queue.install(["c"])
    assert queue.table_age == 0


def test_the_queue_still_answers_the_whole_interface():
    # Every scheduler caller keeps working, which is the point of using
    # the abstraction rather than patching around it.
    queue = filled(("a", 1.0), ("b", 2.0), ("c", 3.0))
    assert len(queue) == 3 and bool(queue)
    assert queue.peek_request().request_id == "a"
    assert queue.pop_request().request_id == "a"
    queue.remove_requests([next(r for r in queue if r.request_id == "b")])
    assert ids(queue) == ["c"]
    assert list(reversed(list(queue))) == list(queue)
    queue.pop_request()
    assert not queue and len(queue) == 0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL {name}: {error}")
    raise SystemExit(1 if failures else 0)
