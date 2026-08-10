"""The observer emits a document the v2 port can read, without vLLM running.

Stage 1 must be reviewable and testable without a GPU, or it is not the
cheap patch it claims to be. Everything the observer touches is read off
attributes, so a stand-in with those attributes exercises the whole path.
"""

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vllm.v1.core.sched.plex_observer import PlexObserver  # noqa: E402


class FakeRequest:
    def __init__(self, rid, prompt=8, computed=0, generated=0, arrival=1.0):
        self.request_id = rid
        self.num_prompt_tokens = prompt
        self.num_computed_tokens = computed
        self.output_token_ids = [0] * generated
        self.arrival_time = arrival
        self.num_preemptions = 0


class FakeScheduler:
    def __init__(self, waiting, running):
        self.waiting = waiting
        self.running = running
        self.max_num_running_reqs = 8
        pool = types.SimpleNamespace(
            num_gpu_blocks=100, get_num_free_blocks=lambda: 40
        )
        self.kv_cache_manager = types.SimpleNamespace(block_pool=pool)
        self.cache_config = types.SimpleNamespace(block_size=16)
        self.plex_observer = None


def observer(waiting, running):
    scheduler = FakeScheduler(waiting, running)
    sink = types.SimpleNamespace(write=lambda _: None, flush=lambda: None)
    return PlexObserver(scheduler, sink, "vllm-0")


def test_a_step_document_has_the_shape_the_port_parses():
    a = FakeRequest("r0", prompt=8, computed=8, generated=2)
    b = FakeRequest("r1", prompt=16)
    obs = observer([b], [a])
    obs.on_request_added(a)
    obs.on_request_added(b)

    doc = json.loads(obs.on_step())

    assert doc["step"] == 1
    assert doc["target"] == "vllm-0"
    assert set(doc["subjects"]["request"]) == {"r0", "r1"}
    assert doc["subjects"]["target"] == ["vllm-0"]
    assert doc["events"]["admitted"] == ["r0", "r1"]

    r0 = doc["facts"]["r0"]
    assert r0["state"] == {"text": "active"}
    assert r0["queue_member"] == {"flag": False}
    assert r0["computation_length"] == {"num": 10}
    assert r0["dispatch_input_tokens"] == {"num": 0}

    r1 = doc["facts"]["r1"]
    assert r1["state"] == {"text": "admitted"}
    assert r1["dispatch_input_tokens"] == {"num": 16}

    target = doc["facts"]["vllm-0"]
    assert target["total_kv_tokens"] == {"num": 1600}
    assert target["free_kv_tokens"] == {"num": 640}
    assert target["queue_depth"] == {"num": 1}


def test_arrival_order_is_recorded_because_nothing_else_records_it():
    a, b = FakeRequest("a"), FakeRequest("b")
    obs = observer([a, b], [])
    obs.on_request_added(a)
    obs.on_request_added(b)
    facts = json.loads(obs.on_step())["facts"]
    assert facts["a"]["arrival_seq"] == {"num": 0}
    assert facts["b"]["arrival_seq"] == {"num": 1}


def test_events_are_drained_not_repeated():
    a = FakeRequest("a")
    obs = observer([a], [])
    obs.on_request_added(a)
    assert json.loads(obs.on_step())["events"]["admitted"] == ["a"]
    # An event delivered twice is one thing a policy counts as two.
    assert "admitted" not in json.loads(obs.on_step())["events"]


def test_a_terminal_edge_is_reported_once():
    a = FakeRequest("a")
    obs = observer([], [])
    obs.on_request_added(a)
    obs.on_request_freed(a)
    finished = json.loads(obs.on_step())["events"]["finished"]
    assert finished == [["a", "completed", "host"]]


def test_a_failing_sink_disables_the_observer_and_not_the_engine():
    # Stage 1 must be unable to change anything, including whether the
    # step completes.
    a = FakeRequest("a")
    scheduler = FakeScheduler([a], [])

    def explode(_):
        raise OSError("disk full")

    scheduler.plex_observer = PlexObserver(
        scheduler, types.SimpleNamespace(write=explode, flush=lambda: None), "vllm-0"
    )
    scheduler.plex_observer.emit_step()  # must not raise
    assert scheduler.plex_observer is None


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
