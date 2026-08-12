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


def _client_of(request_id: str) -> str:
    """The tenant a request belongs to, from the caller's own id."""
    head, sep, _ = request_id.partition("::")
    return head if sep else request_id


def _max_tokens_of(request: Any) -> int:
    """The output ceiling the caller asked for, if the engine kept it."""
    params = getattr(request, "sampling_params", None)
    for name in ("max_tokens", "max_new_tokens"):
        value = getattr(params, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return 0


def _field_of(request_id: str, index: int) -> str:
    """The nth `::`-separated field of a request id, or the last there is.

    Degrading to a coarser level rather than to nothing matters: a
    workload that names a user and no application should look like one
    application per user, not like an absent fact that a policy silently
    defaults.
    """
    parts = request_id.split("::")
    if not parts:
        return request_id
    return parts[min(index, len(parts) - 1)]


def _group_of(request_id: str) -> str:
    """The group a request belongs to, from the caller's own id.

    `<client>::<group>::<rest>`. Falls back to the client when there is
    no second separator, so a workload that names no groups puts each
    tenant in one group rather than each request in its own -- the
    latter is what "no groups" used to mean and it is the degenerate
    case both group-aware policies are written against.
    """
    parts = request_id.split("::")
    if len(parts) >= 3:
        return f"{parts[0]}::{parts[1]}"
    return parts[0]


def _page_id(block_hash: Any) -> str:
    """A page's name, stable across processes.

    `BlockHash` is a `bytes` newtype. Python randomises the hash of
    `bytes` per interpreter, so `hash(block_hash)` names a different page
    in every process — including between the engine that offers a page
    and any later run that tries to name it back. The bytes themselves do
    not move, so the id is taken from them.

    Truncated to eight hex digits because the id is a *name*, not a
    checksum: it identifies a page among the few dozen the engine offers
    at once, and it is read by humans in traces.
    """
    if isinstance(block_hash, (bytes, bytearray)):
        digest = bytes(block_hash)
    else:
        digest = str(block_hash).encode("utf-8")
    value = int.from_bytes(digest[:8].ljust(8, b"\0"), "big") & 0xFFFFFFFF
    return f"p{value:08x}"


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
        # How often a page has been offered and survived, which is the
        # closest thing to a hit count the free queue can support: a page
        # that keeps coming back to the offer without being taken is one
        # the engine keeps deciding not to evict.
        self._page_hits: dict[str, int] = {}
        self._arrivals = 0
        # Edges since the last document. Cleared on emit, because an event
        # delivered twice is one thing a policy counts as two.
        self._admitted: list[str] = []
        # Stage 3, if a source was named. `None` costs nothing, which is
        # what lets stage 1 be reviewed on its own.
        from vllm.v1.core.sched.plex_verbs import PlexVerbs

        self._verbs = PlexVerbs.maybe(scheduler)
        # How many eviction candidates to offer per step. A bound rather
        # than the whole pool: the cost of describing pages has to stay
        # proportional to how many the engine is about to touch, not to
        # how many exist.
        self._page_budget = int(os.environ.get("VLLM_PLEX_PAGE_BUDGET", "64"))
        # A short window of observed step durations, for the timing
        # facts. Bounded so it tracks the engine's current behaviour
        # rather than averaging over a run whose shape has changed.
        self._step_ms_window: list[int] = []
        self._last_step_ms: int | None = None
        self._offered_last: set[str] = set()
        self._bytes_per_token = int(os.environ.get("VLLM_PLEX_BYTES_PER_TOKEN", "0"))
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

    def _finish_rejected(self) -> None:
        """End requests the gate refused, through the engine's own path.

        `reject` is the one entry decision no standing table can express,
        but enacting it still has to use the single way this engine
        finishes a request. Doing it here — between passes, with the
        verbs — keeps it on the write-safe side of the rule two crashes
        established.
        """
        queue = self._scheduler.waiting
        refused = getattr(queue, "rejected_by_policy", None)
        if not refused:
            return
        ids = list(refused)
        refused.clear()
        from vllm.v1.request import RequestStatus

        self._scheduler.finish_requests(ids, RequestStatus.FINISHED_ABORTED)

    def drain_verbs(self) -> int:
        """Enact staged verbs and refusals. Called before a pass begins.

        Separate from `on_step` because a verb is a *write* and the
        observer's hook sits inside `schedule()`. `pause` removes from
        `scheduler.running`, and doing that after the scheduling loops
        have run trips the engine's own assert on resumed requests — the
        second time a PLEX write applied at the wrong moment has crashed
        a real engine. Observation is safe mid-pass; writes are not.
        """
        # Refusals go here too. `finish_requests` mutates the same
        # bookkeeping `schedule()` builds its lists from, and doing it at
        # the observer's step hook — which is *inside* `schedule()` —
        # tripped the engine's assert on resumed requests for the third
        # time. Third occurrence, same cause, same fix: writes belong
        # before the pass.
        self._finish_rejected()
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
        # Verdicts and the hold deadline advance here too: a released
        # request joins the schedulable set, which is a write, and this
        # is between passes.
        gate_tick = getattr(self._scheduler.waiting, "_gate_tick", None)
        if gate_tick is not None:
            gate_tick()
        self._step += 1
        document_now_ms = int(time.time() * 1000)
        if self._last_step_ms is not None:
            self._step_ms_window.append(max(document_now_ms - self._last_step_ms, 0))
            if len(self._step_ms_window) > 32:
                self._step_ms_window.pop(0)
        self._last_step_ms = document_now_ms
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
        held = getattr(scheduler.waiting, "held_requests", None)
        return [
            *(held() if held is not None else []),
            *scheduler.waiting,
            *scheduler.running,
        ]

    def _held_ids(self) -> set[str]:
        held = getattr(self._scheduler.waiting, "held_requests", None)
        if held is None:
            return set()
        return {request.request_id for request in held()}

    def _subjects(self) -> dict[str, list[str]]:
        subjects = {
            "request": [request.request_id for request in self._tracked()],
            "target": [self._target],
        }
        pages = self._offered_pages()
        if pages:
            subjects["page"] = [page_id for page_id, _ in pages]
        return subjects

    def _offered_pages(self) -> list[tuple[str, Any]]:
        """Cached blocks that are candidates for eviction, coldest first.

        **Offered, not enumerated.** A page-level snapshot of the whole
        pool would be per-block state on every step, which is a different
        order of cost from the per-request scrape the rest of this file
        does — and it would hand a policy thousands of subjects to answer
        about when the engine is only about to reuse a handful.

        So this is the narrower question the contract already has a word
        for: the pages the engine would take next. `free_block_queue` is
        vLLM's own eviction order, so reading it is reading the answer
        rather than modelling it.

        Identity is the block's content hash where it has one, because
        that is what makes a page the *same page* across steps — a
        content-addressed name survives eviction and readmission, and a
        hotness ledger keyed on it does not have to be reaped.

        Derived from the hash **bytes**, not from `hash()` of them.
        `BlockHash` is a `bytes` newtype and Python randomises the hash
        of `bytes` per interpreter, so an id built that way is stable
        within one process and meaningless between two. It read as
        content-addressed and was process-addressed, which is the worst
        version: a policy naming a page it saw last run names nothing,
        silently, and the engine evicts exactly as it always would.
        Measured that way, an eviction order that should have wrecked the
        hit rate changed it by 0.0000.
        """
        try:
            pool = self._scheduler.kv_cache_manager.block_pool
            queue = pool.free_block_queue
        except AttributeError:
            return []

        offered: list[tuple[str, Any]] = []
        block = getattr(queue, "fake_free_list_head", None)
        block = getattr(block, "next_free_block", None) if block else None
        tail = getattr(queue, "fake_free_list_tail", None)
        # The budget bounds the **offer**, and the walk is bounded
        # separately.
        #
        # It used to bound the walk, and the two are not the same on this
        # queue: `free_blocks` prepends blocks with no hash and appends
        # blocks with one, so a pool with plenty of never-used blocks
        # puts them all in front. The budget was spent walking blocks
        # that are not pages, and the offer came back empty or short --
        # measured, zero pages offered on a fresh pool where thousands
        # of cached blocks sat further down.
        #
        # A policy that is offered nothing decides nothing, and it
        # reports no difficulty while doing it.
        walk_limit = max(self._page_budget * 16, 4096)
        walked = 0
        while (
            block is not None
            and block is not tail
            and len(offered) < self._page_budget
            and walked < walk_limit
        ):
            block_hash = getattr(block, "block_hash", None)
            if block_hash is not None:
                offered.append((_page_id(block_hash), block))
            block = getattr(block, "next_free_block", None)
            walked += 1
        return offered

    def _facts(self, now_ms: int) -> dict[str, dict[str, Any]]:
        running_ids = {request.request_id for request in self._scheduler.running}
        held_ids = self._held_ids()
        facts: dict[str, dict[str, Any]] = {}
        for request in self._tracked():
            running = request.request_id in running_ids
            prompt = request.num_prompt_tokens
            computed = request.num_computed_tokens
            generated = len(request.output_token_ids)
            facts[request.request_id] = {
                # `pending` only when the request is genuinely being
                # held out of the schedulable set. Publishing it for a
                # queued request would be a lie: the engine has already
                # accepted that one, and a policy answering `reject`
                # would be ruling on something already admitted.
                "state": {
                    "text": "pending"
                    if request.request_id in held_ids
                    else ("active" if running else "admitted")
                },
                "arrival_seq": {"num": self._arrival_seq.get(request.request_id, 0)},
                # Who the request belongs to.
                #
                # vLLM has no tenant concept, so this is the caller's own
                # label: whatever precedes the first `::` in the request
                # id, and the whole id when there is no separator. A
                # front end that wants fairness across tenants says so by
                # naming them, which is the only place the information
                # exists -- an engine cannot invent an ownership it was
                # never told about.
                #
                # Without it every request reports the same client and a
                # fairness policy sees one tenant. Measured on `vtc`:
                # baseline and treatment arms byte-identical at a gap of
                # 7.000, because the policy had nothing to be fair
                # between.
                "client_id": {"text": _client_of(request.request_id)},
                # The group a request belongs to, from the caller's id.
                #
                # `justitia` prices memory-time "over the whole agent,
                # not one request" and `qlm` orders "request groups";
                # both read `group`, and vLLM has no such concept. The
                # caller names it, the same way it names the tenant:
                # everything between the first `::` and the second.
                #
                # Without it both policies see every request as its own
                # group of one, which is the shape their papers exist to
                # improve on. Measured that way, decision alone 1.000x
                # and 1.007x.
                "group": {"text": _group_of(request.request_id)},
                # The caller's own identifiers, from the same id.
                #
                # `fairserve` reads `user-id`, `application-id` and
                # `stage-id` and limits each against an rpm budget. None
                # of the three is anything vLLM knows -- a user is a
                # deployment's concept, an application is its caller's,
                # and a stage is a step within one. All three arrive with
                # the request or not at all, and without them the policy
                # sees one user running one application at one stage,
                # which is the case it has nothing to say about.
                #
                # `<user>::<application>::<stage>::<unique>`, with each
                # falling back to what precedes it so a workload that
                # names fewer levels degrades to coarser accounting
                # rather than to none.
                "user_id": {"text": _field_of(request.request_id, 0)},
                "application_id": {"text": _field_of(request.request_id, 1)},
                "stage_id": {"text": _field_of(request.request_id, 2)},
                # Spellings the ports use for quantities already
                # published under another name. A port that reads
                # `input-tokens` and is handed `prompt_tokens` sees
                # `unknown-key`, defaults, and decides on a constant --
                # which is not a disagreement about vocabulary, it is a
                # policy that has been switched off one fact at a time.
                "input_tokens": {"num": prompt},
                "output_tokens": {"num": generated},
                # How much output this request asked for.
                #
                # `chameleon` weights request size from predicted output
                # and `qlm` divides slack by it; neither could see it, so
                # every request was the same size and there was nothing
                # to separate or to order. The engine knows the number --
                # it is the caller's own `max_tokens` and the sampler
                # holds it -- so this is a fact vLLM has and did not
                # publish rather than one it would have to invent.
                #
                # A *requested* maximum, not a prediction of what will
                # actually be produced. The distinction matters and the
                # ports treat the number as an upper bound, which is what
                # it is.
                "predicted_output_tokens": {
                    "num": int(_max_tokens_of(request) or 0)
                },
                "arrival_ms": {"num": int(request.arrival_time * 1000)},
                "prompt_tokens": {"num": prompt},
                "generated_tokens": {"num": generated},
                "computation_length": {"num": prompt + generated},
                "dispatch_input_tokens": {"num": max(prompt - computed, 0)},
                "cached_tokens": {"num": computed},
                "queue_member": {"flag": not running and request.request_id not in held_ids},
                "preempted": {"flag": request.num_preemptions > 0},
                # How long it has been here. The engine holds both
                # operands and the port holds neither, which is the
                # standing test for whether a derivation belongs on this
                # side.
                "waiting_ms": {"num": max(now_ms - int(request.arrival_time * 1000), 0)},
                # The same quantity the corpus also reads under a
                # second name. Published rather than left absent:
                # a policy reading it gets `unknown-key` and
                # silently takes a default.
                "current_queue_ms": {
                    "num": 0
                    if running
                    else max(now_ms - int(request.arrival_time * 1000), 0)
                },
            }
        # Pages that stopped being offered have been taken; forgetting
        # them keeps the ledger the size of the offer rather than of
        # every page the run ever saw.
        offered = self._offered_pages()
        live = {page_id for page_id, _ in offered}
        if len(self._page_hits) > 4 * max(len(live), 1):
            self._page_hits = {
                page: count for page, count in self._page_hits.items()
                if page in live
            }
        # Position in the engine's own eviction order, which is what
        # `last-access-ms` means to a policy: the head of the free queue
        # is the least recently used, and the queue *is* vLLM's recency
        # ranking. Converting rank to a timestamp rather than publishing
        # the rank keeps the fact in the corpus's vocabulary -- every
        # cache port reads `last-access-ms` and none reads a rank.
        #
        # Without it, `hotprefix` and `pythia` compute hotness from
        # `unknown-key`, score every page identically, and install an
        # order that is the offer's own order with extra steps. Measured:
        # four cache policies within 4% of LRU, none beating it, on two
        # different corpora including one built so that recency and
        # frequency disagree.
        for rank, (page_id, block) in enumerate(offered):
            facts[page_id] = {
                # Resident, because an eviction candidate is by
                # definition still in the pool — it is on the free list,
                # not gone.
                "resident": {"flag": True},
                "targets": {"ids": [self._target]},
                "tier": {"text": "gpu"},
                "pinned": {"flag": getattr(block, "ref_cnt", 0) > 0},
                "page-tokens": {
                    "num": int(getattr(block, "_block_hash_num_tokens", 0) or 0)
                },
                "size-tokens": {
                    "num": int(getattr(block, "_block_hash_num_tokens", 0) or 0)
                },
                # Older by rank, so the head of the queue is the oldest.
                "last-access-ms": {"num": max(now_ms - rank, 0)},
                # Every offered page is a leaf. vLLM's pool is flat --
                # there is no tree, so no page has a page below it --
                # and a cache port that filters on `leaf` should see all
                # of them rather than none.
                "leaf": {"flag": True},
                "hit-count": {
                    "num": int(self._page_hits.get(page_id, 0))
                },
            }
            # Counted here, once per step the page is offered. A page
            # that keeps reappearing in the offer is one the engine has
            # repeatedly declined to take, which is the only frequency
            # signal a free queue carries. Incremented after the fact is
            # read so the first sighting reports zero rather than one.
            self._page_hits[page_id] = self._page_hits.get(page_id, 0) + 1
        facts[self._target] = self._target_facts()
        return facts


    def _step_timing(self) -> tuple[int, int]:
        """Median step wall clock, and the decode cost it implies.

        **Measured here, not read from the engine.** Neither engine keeps
        a per-step duration an observer can read — `forward_ct` is a
        count — and the honest options were to publish nothing or to
        measure. Publishing nothing loses four names the corpus reads;
        inventing a plausible constant would hand a policy a confident
        number about a machine nobody timed.

        So the observer times its own steps and says so. It is the
        *observer's* view of the step, which includes anything else the
        engine did between two documents — and that is the quantity a
        policy reasoning about "how long until my turn" actually wants.

        Median over a short window rather than a mean: one slow step
        (a cold kernel, a neighbour on the GPU) would drag a mean for
        many steps afterwards, and a policy acting on it would be acting
        on an outlier that has already passed.
        """
        if len(self._step_ms_window) < 3:
            return (0, 0)
        window = sorted(self._step_ms_window)
        median = window[len(window) // 2]
        return (median, median * 1000)

    def _target_facts(self) -> dict[str, Any]:
        scheduler = self._scheduler
        pool = scheduler.kv_cache_manager.block_pool
        block_size = scheduler.cache_config.block_size
        # `getattr` rather than attribute access: an engine build that
        # names these differently must lose the facts, not the engine.
        # A test caught this — a pool without `num_gpu_blocks` raised
        # straight out of `emit_step`, and observation is never allowed
        # to stop inference.
        total_blocks = int(getattr(pool, "num_gpu_blocks", 0) or 0)
        get_free = getattr(pool, "get_num_free_blocks", None)
        free_blocks = int(get_free() if get_free else 0)
        running = list(scheduler.running)
        max_running = int(getattr(scheduler, "max_num_running_reqs", 0) or 0)
        queued = len(scheduler.waiting)
        step_ms, step_us = self._step_timing()
        # Output tokens the engine still owes, and prompt tokens it has
        # yet to prefill. Both are sums over state the scheduler already
        # holds; neither is a model of anything.
        pending_decode = sum(
            max(int(getattr(request, "max_tokens", 0) or 0)
                - len(request.output_token_ids), 0)
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
            "step_ms": {"num": step_ms},
            # Time per output token, from the observer's own timing. The
            # engine keeps no such number, so this is measured rather
            # than read — and it is the observer's view of a step, which
            # is what a policy asking "how long until my turn" wants.
            "decode_ms_per_token": {"num": step_ms},
            "tpot_us": {"num": step_us},
            # How long a newcomer waits: the requests ahead of it,
            # divided by how many run at once, times a step.
            "estimated_wait_ms": {
                "num": ((queued + max_running - 1) // max_running) * step_ms if max_running else 0
            },
            # Guarantee 4: the tier vocabulary is declared, never guessed.
            # vLLM's v1 scheduler has no CPU-offload path a port may
            # drive, so it has exactly one tier and says so — which is a
            # different statement from publishing nothing and letting a
            # policy assume whatever its paper assumed.
            "tiers": {"ids": ["gpu"]},
            # The pool in bytes as well as tokens. Declared rather than
            # derived: bytes-per-token depends on the model's layers,
            # heads and dtype, and vLLM's scheduler holds the block size
            # but not the byte size. A port that assumed one would
            # publish a confident figure for a different model.
            #
            # Zero when undeclared, which reads as "no bytes accounted"
            # rather than as a small pool — and a policy comparing it
            # against a capacity gets a ratio of zero rather than a wrong
            # one.
            "memory_capacity": {
                "num": total_blocks * block_size * self._bytes_per_token
            },
            "active_kv_bytes": {
                "num": (total_blocks - free_blocks) * block_size * self._bytes_per_token
            },
            # The engine's own ceiling on work per step. Read, not
            # modelled: `max_num_scheduled_tokens` is exactly the budget
            # the scheduler spends.
            "throughput_token_cap": {
                "num": int(getattr(scheduler, "max_num_scheduled_tokens", 0) or 0)
            },
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
        offered = [page_id for page_id, _ in self._offered_pages()]
        # Only pages that were not offered last step. `on-offered` means
        # "this is newly on the table", and re-raising it every step for
        # the same page would make a cache policy count one offer as many
        # — every accumulator in the corpus is vulnerable to that.
        fresh = [page_id for page_id in offered if page_id not in self._offered_last]
        if fresh:
            events["offered"] = fresh
        self._offered_last = set(offered)
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
