"""Staged verbs, mapped onto vLLM's own primitives -- or declined.

Stage 3 of the phased attach. The two rules under test:

Declines are declared, not discovered. A verb that quietly does nothing
is indistinguishable from one that lost a race, and the two call for
opposite responses -- retry, or stop asking.

A refusal is an answer. Every path produces a reason, because "it did not
happen" is not one, and `on-refused` carries the reason so a policy's
beliefs can be corrected. v0.7's post-mortem is largely about beliefs
that could not be.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


class FakeRequest:
    def __init__(self, rid, status="RUNNING"):
        self.request_id = rid
        self.status = status
        self.num_computed_tokens = 4
        self.num_preemptions = 0
        self.spec_token_ids = []

    def is_finished(self):
        return self.status == "FINISHED"


class FakeScheduler:
    def __init__(self, requests):
        self.requests = {r.request_id: r for r in requests}
        self.running = [r for r in requests if r.status == "RUNNING"]
        self.preempted = []
        self.finished = []

    def _preempt_request(self, request, timestamp):
        assert request.status == "RUNNING"
        request.status = "PREEMPTED"
        request.num_computed_tokens = 0
        request.num_preemptions += 1
        self.preempted.append(request.request_id)

    def finish_requests(self, request_ids, finished_status):
        self.finished.append(request_ids)


def verbs(*requests):
    from vllm.v1.core.sched.plex_verbs import PlexVerbs

    scheduler = FakeScheduler(list(requests))
    return scheduler, PlexVerbs(scheduler)


def test_pause_preempts_a_running_request():
    scheduler, v = verbs(FakeRequest("a"))
    assert v.enact("pause", "a") is True
    assert scheduler.preempted == ["a"]
    assert scheduler.requests["a"].num_computed_tokens == 0
    assert v.take_refusals() == []


def test_pause_with_preserve_refuses_the_disposition_not_the_verb():
    # The finer answer. vLLM recomputes rather than swapping, so the KV
    # cannot be preserved -- but the policy may want the pause anyway, and
    # refusing the verb outright would tell it less than it needs.
    scheduler, v = verbs(FakeRequest("a"))
    assert v.enact("pause", "a", kv="preserve") is False
    assert scheduler.preempted == []
    (refusal,) = v.take_refusals()
    assert refusal.reason == "unsupported-kv-disposition"
    assert refusal.verb == "pause"


def test_pausing_something_that_already_stopped_is_a_lost_race():
    # `_preempt_request` asserts on this, and an assert is not an answer.
    scheduler, v = verbs(FakeRequest("a", status="FINISHED"))
    assert v.enact("pause", "a") is False
    (refusal,) = v.take_refusals()
    assert refusal.reason == "wrong-request-state"


def test_an_unknown_subject_is_named_as_such():
    _, v = verbs()
    assert v.enact("pause", "ghost") is False
    (refusal,) = v.take_refusals()
    assert refusal.reason == "unknown-subject"
    assert refusal.subject == "ghost"


def test_finish_aborts():
    scheduler, v = verbs(FakeRequest("a"))
    assert v.enact("finish", "a") is True
    assert scheduler.finished == ["a"]


def test_finishing_a_finished_request_is_a_lost_race():
    _, v = verbs(FakeRequest("a", status="FINISHED"))
    assert v.enact("finish", "a") is False
    (refusal,) = v.take_refusals()
    assert refusal.reason == "wrong-request-state"


def test_a_verb_this_engine_cannot_perform_is_declined_with_a_reason():
    # Not silence, and not a crash. `prefetch` is a verb the contract has
    # and vLLM cannot perform, which is a different fact from a typo.
    _, v = verbs(FakeRequest("a"))
    for verb in ["prefetch", "move", "rebalance"]:
        assert v.enact(verb, "a") is False
    refusals = v.take_refusals()
    assert [r.verb for r in refusals] == ["prefetch", "move", "rebalance"]
    assert all(r.reason == "unsupported-on-this-engine" for r in refusals)


def test_refusals_are_drained_not_repeated():
    # A refusal delivered twice is one correction a policy applies twice.
    _, v = verbs()
    v.enact("pause", "ghost")
    assert len(v.take_refusals()) == 1
    assert v.take_refusals() == []


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
