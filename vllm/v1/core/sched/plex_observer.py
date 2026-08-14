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


def _program_of(request_id: str) -> str:
    """The program a request belongs to, from the caller's own id.

    Same `program::step` convention as `client_id`, with the engine's own
    `cmpl-` prefix stripped so the name is the caller's. A session policy
    ranks by which program a page serves, and vLLM has no session
    concept, so this is the caller's label or nothing. Measured:
    `continuum` read `program-id` 113,034 times without an answer, ranked
    every page identically and sat on the baseline across three corpora,
    which read as a policy that does not reproduce.
    """
    head = _client_of(request_id)
    for prefix in ("cmpl-", "chatcmpl-"):
        if head.startswith(prefix):
            return head[len(prefix):]
    return head


def _program_plan(request_id: str) -> dict[str, Any]:
    """The tool structure the caller encoded in its request id.

    `program::step::tool::next-tool::finished`, written by
    `cache-regime.request_name`. These are properties of the program,
    not of the engine: vLLM has no tool concept and cannot invent one,
    so a policy written about tool structure either reads the caller's
    declaration or reads nothing. Measured: `continuum` sat exactly on
    the baseline at every model size and on three corpora.

    A `-` marks a field the corpus lacked, and produces no fact at all
    rather than a fact naming a tool called "-".
    """
    parts = request_id.split("::")
    if len(parts) < 5:
        return {}
    tool, following, finished = parts[2], parts[3], parts[4]
    plan: dict[str, Any] = {"program-finished": {"flag": finished == "1"}}
    if tool and tool != "-":
        plan["tool-id"] = {"text": tool}
    if following and following != "-":
        plan["next-tool-id"] = {"text": following}
    return plan


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


# The prefix a page belongs to, by page id.
#
# A page id names a page, and a page is gone the moment the engine takes
# it: 97.6% of the names a policy pinned were absent from the free queue
# by the time the engine read them. A *prefix* outlives its pages. When
# the same prompt is computed again it produces the same chain, so a
# name taken from the chain's first block still means something after
# every page under it has been evicted and recreated.
#
# Bounded, because it is a cache of names and not a ledger. The oldest
# entries go first; a prefix nobody has touched in 200,000 pages is not
# one a policy is still protecting.
ROOT_OF_PAGE: dict[str, str] = {}
ROOT_TABLE_LIMIT = 200_000

# How far into a single request's prefix the lookahead reads. A request
# reads its blocks in order, so the pages beyond this bound are the ones
# furthest from being needed -- the cheapest possible thing to leave
# unanswered, and it keeps the scan proportional to the queue rather
# than to the longest prompt in it.
LOOKAHEAD_PAGE_LIMIT = 512

# The requests a page would serve, by page id.
#
# `beneficiaries` is the fact a cache policy uses to get from a page to
# whoever wants it, and nothing published it: `continuum` reads it to
# find which program a page belongs to, found nothing, fell back to
# "default", and pinned zero pages on a workload made of exactly the
# sessions it exists to hold. A policy whose mechanism has nothing to
# attach to is indistinguishable at runtime from one that does not work.
#
# A *set*, and this is the whole point. It held one id -- the request
# that hashed the page -- on the argument that vLLM's pool is flat and a
# block belongs to whoever computed it. That is true of authorship and
# false of demand, and `peek` is judged on demand: its kernel is
# `beneficiaries x prefix depth`, so a beneficiary count that is 1 by
# construction reduces it to "evict the largest page". Measured on 40
# ShareGPT sessions at 30 rps: 12,897 offered pages, every one of them
# with exactly one beneficiary, and a policy that reordered 84% of the
# offer and moved pages a mean of 181 ranks changed the hit rate by
# 0.0000. It was ranking by a constant.
#
# `note_chain` already sees every request that hashes the chain,
# including the later ones that hit an existing prefix -- it was
# overwriting rather than accumulating, so the demand was computed and
# then discarded.
BENEFICIARIES_OF_PAGE: dict[str, set[str]] = {}

# Per page, because the fact is a decision input and not a ledger. A
# page wanted by more requests than this is already maximally wanted as
# far as any ranking is concerned.
BENEFICIARY_LIMIT = 64

# What a preempted token costs to recompute, and what it costs to swap.
#
# `_MS_PER_PREFILL_TOKEN` is deliberately coarse: the comparison policies
# make with it is against a sentinel, so its precision cannot change any
# decision, and a per-request measured rate would imply an accuracy this
# does not have.
#
# `_NO_SWAP_MS` is not "very expensive". vLLM v1 has no CPU-offload path
# a port may drive, so swap is unavailable, and a policy comparing
# against it must reach the same verdict for every request regardless of
# size. Stated as a number because the fact vocabulary carries numbers;
# the capability declaration in `plex_verbs.py` is where it is stated as
# a capability.
_MS_PER_PREFILL_TOKEN = 0.05
_NO_SWAP_MS = 1_000_000_000

# The scheduling step most recently observed, and the step at which each
# page was last actually demanded. Module-level because `note_demand` is
# a hook on the cache manager and has no observer to ask.
CURRENT_STEP = 0
LAST_DEMAND_STEP: dict[str, int] = {}

# Which reading of `steps-to-execution` to publish.
#
#   live      -- the distance to a live reader, or nothing (see §107).
#   recorded  -- when no live reader exists, substitute the number of
#                steps since the page was last demanded.
#
# `live` is the default because it is the honest one. `recorded` exists
# to be *measured*, not to be believed: it is the same substitution made
# for `beneficiaries` (live demand for an eviction candidate is empty by
# construction, so recorded demand stands in), applied to a quantity
# where the substitution is not obviously legitimate. A count of past
# demand is a real count. A distance into the past is not a distance
# into the future, and a policy that ranks the same either way was
# ranking on recency all along.
LOOKAHEAD_READING = os.environ.get("PLEX_LOOKAHEAD", "live")


def note_chain(blocks: list, request_id: str | None = None) -> None:
    """Record which prefix each block of a request belongs to.

    Called where vLLM hashes a request's blocks, which is the only place
    the chain is visible in order. The first block's id is the prefix's
    name: every request sharing that prefix produces the same first
    block hash, which is what prefix caching is.
    """
    if not blocks:
        return
    root = None
    for block in blocks:
        block_hash = getattr(block, "block_hash", None)
        if block_hash is None:
            continue
        page = _page_id(block_hash)
        if root is None:
            root = page
        ROOT_OF_PAGE[page] = root
        if request_id is not None:
            wanted = BENEFICIARIES_OF_PAGE.setdefault(page, set())
            if len(wanted) < BENEFICIARY_LIMIT:
                wanted.add(request_id)
    if len(ROOT_OF_PAGE) > ROOT_TABLE_LIMIT:
        for page in list(ROOT_OF_PAGE)[: len(ROOT_OF_PAGE) - ROOT_TABLE_LIMIT]:
            ROOT_OF_PAGE.pop(page, None)
            BENEFICIARIES_OF_PAGE.pop(page, None)
            LAST_DEMAND_STEP.pop(page, None)


def note_demand(groups: list, request_id: str) -> None:
    """Record that a request reached these already-cached pages.

    `note_chain` records authorship -- who computed a page. This records
    demand -- who wants it now. They are different populations and only
    the second one is what a policy ranking pending demand is asking
    for.

    `groups` is what `find_longest_cache_hit` returns: one list of blocks
    per KV cache group, already narrowed to the blocks the request will
    reuse.
    """
    if not request_id:
        return
    for blocks in groups:
        for block in blocks:
            block_hash = getattr(block, "block_hash", None)
            if block_hash is None:
                continue
            page = _page_id(block_hash)
            LAST_DEMAND_STEP[page] = CURRENT_STEP
            wanted = BENEFICIARIES_OF_PAGE.setdefault(page, set())
            if len(wanted) < BENEFICIARY_LIMIT:
                wanted.add(request_id)


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
        # Read once. These are deployment settings and do not change
        # under a run; re-reading the environment every step would be
        # cost for nothing.
        self._rate_limits: dict[str, dict[str, Any]] = {}
        for name, key in (
            ("VLLM_PLEX_USER_RPM", "user-rpm-limit"),
            ("VLLM_PLEX_APP_RPM", "app-rpm-limit"),
            ("VLLM_PLEX_RPM_WINDOW_MS", "rpm-window-ms"),
        ):
            raw = os.environ.get(name)
            if raw and raw.isdigit():
                self._rate_limits[key.replace("-", "_")] = {"num": int(raw)}
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
        # Cumulative microseconds of engine time per request; see the
        # `service_us` fact.
        self._service_us: dict[str, int] = {}
        # The deployment's deadline, in milliseconds, or 0 for none.
        #
        # Read by chameleon, dualmap, goodserve, pard, qlm, slos-serve
        # and branch-regulation, and published by neither engine until
        # now. `branch-regulation` sizes its whole slack auction from
        # `slo-ms` minus `waiting-ms`, so without it the budget is zero,
        # the auction never opens and only trunks are ever funded --
        # the policy runs, decides, and expresses a fraction of itself.
        #
        # A deadline is a property of the deployment and not something
        # vLLM has a concept of, so the operator states it, exactly as
        # the tenant label is stated in the request id.
        # One deadline, or one per tenant.
        #
        # `VLLM_PLEX_SLO_MS=30000` states a deployment-wide deadline.
        # `VLLM_PLEX_SLO_MS=c0=60000,c1=15000,default=30000` states one
        # per tenant, and the second form is not a convenience.
        #
        # A deadline-ordering policy sorts by `slo-ms - waiting-ms`.
        # Hold `slo-ms` constant across every request and that
        # expression is a strictly decreasing function of waiting time,
        # so **the earliest-deadline order is the arrival order** and
        # EDF is FCFS with extra arithmetic. Measured: `slos-serve`
        # ranked 98% of the queue and moved 6.1% of it, and its
        # treatment arm matched its own control to four digits.
        # Heterogeneous deadlines are what make a deadline schedulable.
        self._slo_ms = 0
        self._slo_by_client: dict[str, int] = {}
        raw = (os.environ.get("VLLM_PLEX_SLO_MS", "") or "").strip()
        if "=" in raw:
            for item in raw.split(","):
                name, _, value = item.partition("=")
                try:
                    number = int(float(value))
                except ValueError:
                    continue
                if name.strip() == "default":
                    self._slo_ms = number
                else:
                    self._slo_by_client[name.strip()] = number
        elif raw:
            try:
                self._slo_ms = int(float(raw))
            except ValueError:
                self._slo_ms = 0
        # The service classes a deployment sells, stated by the operator.
        #
        # `is-best-effort` and `tpot-ms` are the two facts SLOs-Serve is
        # written about, and neither is something an engine can derive:
        # whether a request is allowed to be deferred is a commercial
        # fact about the tenant, and a per-token latency target is a
        # promise someone made. vLLM has concepts for neither, so the
        # operator states them exactly as the tenant label is stated in
        # the request id and the deadline in `VLLM_PLEX_SLO_MS`.
        #
        # Absent, every request defaults to `is-best-effort: false` --
        # so a policy whose mechanism is "defer the deferrable to
        # protect the promised" has an empty set to defer and cannot act
        # however hard it ranks. Measured: slos-serve read
        # `is-best-effort` 4878 times in one trace, got nothing every
        # time, and returned a treatment arm identical to its own
        # control in every digit.
        self._best_effort = {
            name.strip()
            for name in os.environ.get("VLLM_PLEX_BEST_EFFORT", "").split(",")
            if name.strip()
        }
        self._tpot_ms = int(os.environ.get("VLLM_PLEX_TPOT_MS", "0") or 0)
        self._offered_last: set[str] = set()
        self._progress_last: dict[str, int] = {}
        self._bytes_per_token = int(os.environ.get("VLLM_PLEX_BYTES_PER_TOKEN", "0"))
        if not self._bytes_per_token:
            self._bytes_per_token = self._derive_bytes_per_token()
        # First sighting of each program, so a session policy can price
        # how long a program has been resident. Bounded with the page
        # tables below: a program nobody has sent for is not one a policy
        # is still holding.
        self._program_arrival: dict[str, int] = {}
        # When each program's previous request finished, and how long it
        # was away before the next one arrived. From the engine's side
        # that gap *is* the tool duration: the program left, something
        # happened elsewhere, it came back. It is observed rather than
        # declared, which is why it is computed here and not carried in
        # the request id like the rest of the plan.
        self._program_last_finish: dict[str, int] = {}
        self._program_gap: dict[str, int] = {}
        self._finished: list[list[str]] = []
        # The pause edge. vLLM does not announce a preemption; it
        # increments a counter on the request and puts it back on the
        # waiting queue. So the edge is the *rise* in that counter, and
        # holding the last seen value per request is the only way to see
        # it. Without this the contract's `on-paused` hook can never
        # fire against this engine, which is not the same as the engine
        # never preempting.
        self._paused: list[list[str]] = []
        # Requests that left this step, kept as subjects for exactly the
        # one document that announces their finish. The contract offers
        # `on-finished` with fact access, and a port that drops a
        # request from `subjects` in the same document it reports the
        # request finishing makes those facts unreadable at the only
        # moment they are wanted. Measured: 248 of 248 finished events
        # named a request carrying no facts, so `continuum` -- whose
        # whole mechanism runs off `on_finished` -- installed `pinned=0`
        # pins while being handed every fact it asked for.
        self._finishing: list[Request] = []
        self._preemptions: dict[str, int] = {}

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
        self._finishing.append(request)
        finished_program = _program_of(request.request_id)
        self._program_last_finish[finished_program] = int(time.time() * 1000)
        # Cleared so the program's *next* step measures its own gap
        # rather than reporting the first one forever.
        self._program_gap.pop(finished_program, None)
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

    @staticmethod
    def _tenant_of(request_id: str) -> str:
        """The tenant name an operator would write down.

        `_client_of` returns the id's first field, which the API server
        has already prefixed -- `cmpl-c0`, not `c0`. That prefix is the
        engine's, not the caller's, and an operator declaring a service
        class writes the tenant they sell to. Stripping it here rather
        than asking every declaration to know about vLLM's id format.
        """
        head = _client_of(request_id)
        for prefix in ("cmpl-", "chat-", "chatcmpl-", "embd-"):
            if head.startswith(prefix):
                return head[len(prefix):]
        return head

    def _slo_of(self, request_id: str) -> int:
        """This request's deadline: its tenant's, or the deployment's."""
        return self._slo_by_client.get(self._tenant_of(request_id), self._slo_ms)

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

    def _note_preemptions(self) -> None:
        """Raise the pause edge for any request preempted since last step.

        `num_preemptions` only ever rises, so a difference against the
        last seen value is exactly one edge per preemption, and a
        request preempted twice raises it twice. The initiator is
        `host`: vLLM decided this, not a policy -- and reporting it as
        `policy` would credit the policy with an eviction it never
        ordered.

        Requests are forgotten when they leave, so the table cannot
        outgrow the engine's own request set.
        """
        live = set()
        for request in self._tracked():
            request_id = request.request_id
            live.add(request_id)
            count = int(getattr(request, "num_preemptions", 0) or 0)
            if count > self._preemptions.get(request_id, 0):
                self._paused.append([request_id, "host"])
            self._preemptions[request_id] = count
        for gone in [key for key in self._preemptions if key not in live]:
            self._preemptions.pop(gone, None)

    def _charge_service(self, elapsed_ms: int) -> None:
        """Charge the step that just ended to whoever was running in it.

        The interval is measured between step documents, so it is the
        time the engine spent, not the time the port spent looking. It
        is charged to `scheduler.running` because that is the set the
        engine actually computed over.

        Requests that have left are dropped rather than kept: the fact
        is only ever read for a request that is present, and a ledger
        that outlives its subjects is a leak on a long run.
        """
        if elapsed_ms <= 0:
            return
        charge = elapsed_ms * 1000
        live = set()
        for request in getattr(self._scheduler, "running", ()) or ():
            request_id = getattr(request, "request_id", None)
            if request_id is None:
                continue
            live.add(request_id)
            self._service_us[request_id] = (
                self._service_us.get(request_id, 0) + charge
            )
        if len(self._service_us) > len(live) + 4096:
            self._service_us = {
                key: value
                for key, value in self._service_us.items()
                if key in live
            }

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
        global CURRENT_STEP
        CURRENT_STEP = self._step
        self._note_preemptions()
        document_now_ms = int(time.time() * 1000)
        elapsed_ms = (
            max(document_now_ms - self._last_step_ms, 0)
            if self._last_step_ms is not None
            else 0
        )
        if self._last_step_ms is not None:
            self._step_ms_window.append(elapsed_ms)
            if len(self._step_ms_window) > 32:
                self._step_ms_window.pop(0)
        self._last_step_ms = document_now_ms
        self._charge_service(elapsed_ms)
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
        self._paused.clear()
        self._finishing.clear()
        return json.dumps(document)

    # ── scraping ─────────────────────────────────────────────────────────

    def _steps_to_execution(self) -> dict[str, int]:
        """For each page, how many scheduling turns until it is next read.

        The engine cannot see the future, but it can see its own queue,
        and that is a real answer rather than a guess: a request that is
        running reads its prefix now, and a request `k` places back in
        the waiting queue reads its prefix in `k` turns. Pages are named
        by the prefix hashes the requests already carry, so this asks the
        engine only for what it has already computed.

        A page that no live request will read is left out. That absence
        is the honest reading of "infinitely far", which is exactly the
        fallback a lookahead policy already applies to a page it cannot
        place -- inventing a distance for it would be a lie that ranks.
        """
        distance: dict[str, int] = {}
        queues = (
            (0, self._scheduler.running),
            (1, self._scheduler.waiting),
        )
        for base, queue in queues:
            for offset, request in enumerate(queue):
                position = base * (offset + 1)
                hashes = getattr(request, "block_hashes", None) or ()
                for block_hash in hashes[:LOOKAHEAD_PAGE_LIMIT]:
                    page = _page_id(block_hash)
                    if distance.get(page, position + 1) > position:
                        distance[page] = position
        if LOOKAHEAD_READING != "recorded":
            return distance
        # The substitution under test. Steps since the page was last
        # demanded, for pages no live request will read -- which, on a
        # free queue, is nearly all of them.
        for page, step in LAST_DEMAND_STEP.items():
            distance.setdefault(page, max(CURRENT_STEP - step, 0))
        return distance

    def _tracked(self) -> list[Request]:
        scheduler = self._scheduler
        held = getattr(scheduler.waiting, "held_requests", None)
        live = [
            *(held() if held is not None else []),
            *scheduler.waiting,
            *scheduler.running,
        ]
        if not self._finishing:
            return live
        # Requests leaving this step are still subjects of this step, so
        # `on-finished` can read their facts. They are gone from the
        # engine's own queues, so they are appended rather than merged,
        # and they are dropped again when the step's buffers clear.
        seen = {request.request_id for request in live}
        return live + [
            request
            for request in self._finishing
            if request.request_id not in seen
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

    def _wanted_by(self, page_id: str, live_ids: set[str]) -> list[str]:
        """The requests that have reached this page.

        **Not filtered to requests still in the engine, and that is the
        finding.** A page is only offered as an eviction candidate when
        no request holds it -- vLLM's free queue *is* the unreferenced
        set -- so intersecting an offered page's beneficiaries with the
        live set is guaranteed to be empty, by construction. Measured:
        every one of 24,472 offered pages reported exactly one live
        beneficiary, which was the fallback firing, while the recorded
        counts behind it ranged over 1..6.

        So demand for a candidate is necessarily demand in the recent
        past: who has been hitting this prefix. That is the quantity
        `peek` ranks by -- a page many conversations keep returning to is
        worth keeping -- and it is the only one an eviction candidate can
        carry.
        """
        wanted = BENEFICIARIES_OF_PAGE.get(page_id)
        if not wanted:
            return []
        return list(wanted)

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

    def _derive_bytes_per_token(self) -> int:
        """How wide a token's KV is, from the engine's own cache spec.

        The env var stays the override. Without a width, `size-bytes` is
        unanswerable and a size-aware policy ranks nothing -- `marconi`
        prices recency against size and read `size-bytes` 38,207 times
        without an answer -- so it is worth deriving rather than
        requiring every caller to know it. Returns 0 when the build does
        not expose a spec, because a made-up width is worse than none.
        """
        try:
            groups = self._scheduler.kv_cache_config.kv_cache_groups
            block_size = self._scheduler.cache_config.block_size
            page_bytes = groups[0].kv_cache_spec.page_size_bytes
            return int(page_bytes) // max(int(block_size), 1)
        except (AttributeError, IndexError, TypeError, ZeroDivisionError):
            return 0

    def _facts(self, now_ms: int) -> dict[str, dict[str, Any]]:
        running_ids = {request.request_id for request in self._scheduler.running}
        held_ids = self._held_ids()
        finishing_ids = {
            request.request_id for request in self._finishing
        }
        facts: dict[str, dict[str, Any]] = {}
        for request in self._tracked():
            program = _program_of(request.request_id)
            self._program_arrival.setdefault(program, now_ms)
            if program not in self._program_gap:
                previous = self._program_last_finish.get(program)
                if previous is not None:
                    self._program_gap[program] = max(now_ms - previous, 0)
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
                    "text": "completed"
                    if request.request_id in finishing_ids
                    else "pending"
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
                # The program the request belongs to, and when that
                # program first arrived. A session policy prices a
                # program's whole residency, so it needs both the name
                # and the age; neither exists in an engine that has no
                # session concept, and both are in the caller's id.
                "program-id": {"text": _program_of(request.request_id)},
                "program-arrival": {
                    "num": self._program_arrival.get(
                        _program_of(request.request_id), now_ms
                    )
                },
                # The tool structure the caller declared, if it declared
                # any. Absent for every corpus that has none, so this
                # adds no fact where there is nothing to say.
                **_program_plan(request.request_id),
                **(
                    {
                        "tool-duration-ms": {
                            "num": self._program_gap[
                                _program_of(request.request_id)
                            ]
                        }
                    }
                    if _program_of(request.request_id) in self._program_gap
                    else {}
                ),
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
                # The deployment's own rate limits, from the environment.
                #
                # `fairserve`'s throttle only reorders when someone is
                # over their limit, and with no limit published every
                # user sits under an infinite budget and it never fires
                # -- measured, 296,330 requests ranked and **zero**
                # moved. A rate limit with no rate is not a rate limit,
                # and a policy given one correctly declines to reorder.
                #
                # An operator's setting, so it comes from the operator:
                # `VLLM_PLEX_USER_RPM`, `VLLM_PLEX_APP_RPM` and
                # `VLLM_PLEX_RPM_WINDOW_MS`. Unset, nothing is published
                # and the policy behaves as it did.
                **self._rate_limits,
                # Whether the pool is under pressure, on the request as
                # well as on the target.
                #
                # `fairserve` reads `kv-overloaded` from the *request*
                # and it was published only on the target, so the flag
                # guarding its entire throttle was never true. It is a
                # property of the engine either way -- a request served
                # by an overloaded engine is an overloaded request -- and
                # publishing it in one place only made the fact
                # unreachable from where a policy looks.
                "kv_overloaded": {"flag": self._kv_overloaded()},
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
                # What this request was *expected* to cost, against which
                # `output_tokens` is the actual.
                #
                # `fairserve` is the only reader and no engine published
                # them, which does not make them optional: its weighted
                # service counter is `actual / expected`, and a missing
                # `expected` falls back to `actual`, so the ratio is
                # identically 1 and every completed request charges the
                # same 1000000 whatever its size. A counter meant to
                # measure work degenerates into a count of completions,
                # and the policy ranks lowest-completion-count-first
                # while believing it ranks least-served-first. Silent,
                # and it still produces a plausible fairness ordering --
                # just not the paper's.
                #
                # The expectation for input is exact: the prompt is
                # already here and its length is not a guess. For output
                # it is the caller's `max_tokens`, the only statement of
                # intended size anyone made. A request that stops early
                # then charges less than one that runs to its cap, which
                # is what `actual / expected` is for.
                "expected_input_tokens": {"num": prompt},
                "expected_output_tokens": {
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
                # Engine time this request has accrued, cumulative.
                #
                # Read by nine policies -- agentix, branch-regulation,
                # chameleon, dynasor, goodserve, infercept, pythia, qlm
                # and saga -- and published by neither engine until now,
                # so every one of them differenced a constant zero and
                # took its default. `branch-regulation` derives its
                # microseconds-per-token from exactly this difference,
                # which is why it read `not reproduced` with a service
                # vector that had barely moved.
                #
                # **A running request is charged the whole step**, not a
                # share of it. The alternative -- dividing the step by
                # the batch size -- would make a request's own service
                # depend on who else happened to be batched with it,
                # so the same request in a quiet batch and a busy one
                # would report different service for identical work.
                # The papers that read this treat it as "how long has
                # the engine been working on you", and in a batched
                # engine every member of the batch is being worked on
                # for the whole step.
                #
                # Consequence worth stating: the sum over requests
                # exceeds wall clock whenever the batch holds more than
                # one, so this is not a partition of engine time and
                # must not be summed to get utilisation.
                "service_us": {"num": self._service_us.get(request.request_id, 0)},
                # The two costs a policy compares when it decides how to
                # evict, published because this engine is the only party
                # that knows either.
                #
                # `qlm` stages `pause(preserve)` iff recompute costs more
                # than swap. Both facts were unanswered -- 581 times in a
                # single trace -- so it fell back to `computation-length`
                # vs `cached-tokens`, two token counts standing in for two
                # durations, and concluded `preserve` on every one. vLLM
                # declines that disposition by design, so all 581 pauses
                # were refused and its entire overload defence never ran.
                #
                # Recompute is honest arithmetic: the tokens preemption
                # would discard, at this request's own observed prefill
                # rate. Swap is `_NO_SWAP_MS`, a sentinel, and it is the
                # true answer rather than a large guess -- v1 has no
                # CPU-offload path a port may drive, so swapping is not
                # expensive here, it is unavailable, and the comparison a
                # policy makes against an unavailable option should come
                # out the same way every time. `plex_verbs.py` already
                # declares this in `capabilities()`; this states the same
                # fact in the vocabulary policies actually read.
                "recompute_cost_ms": {
                    "num": int(computed * _MS_PER_PREFILL_TOKEN)
                },
                "swap_cost_ms": {"num": _NO_SWAP_MS},
                # Published only when stated. A deadline of zero is not
                # "no deadline" to a policy that subtracts from it, it
                # is a deadline already missed, and the two must not
                # read the same.
                **(
                    {"slo_ms": {"num": self._slo_of(request.request_id)}}
                    if self._slo_of(request.request_id) > 0
                    else {}
                ),
                # The service class, per request, from the tenant.
                #
                # Published unconditionally once any tenant is declared
                # best-effort, because here `false` is a real answer --
                # "this request is promised" -- and must not read the
                # same as "nobody said". When no class is declared at
                # all the fact is absent, which is the honest reading of
                # a deployment that sells one tier.
                **(
                    {
                        "is_best_effort": {
                            "flag": self._tenant_of(request.request_id)
                            in self._best_effort
                        }
                    }
                    if self._best_effort
                    else {}
                ),
                # The per-token latency target, when one was promised.
                **({"tpot_ms": {"num": self._tpot_ms}} if self._tpot_ms > 0 else {}),
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
        step_distance = self._steps_to_execution()
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
        # Which requests are still in the engine. A page wanted only by
        # requests that have already finished is wanted by nobody, and
        # counting them would make demand climb monotonically over a run.
        live_ids = {
            request.request_id
            for request in (
                *self._scheduler.running,
                *self._scheduler.waiting,
            )
        }
        for rank, (page_id, block) in enumerate(offered):
            page_tokens = int(getattr(block, "_block_hash_num_tokens", 0) or 0)
            beneficiary_steps = step_distance.get(page_id)
            facts[page_id] = {
                # Resident, because an eviction candidate is by
                # definition still in the pool — it is on the free list,
                # not gone.
                "resident": {"flag": True},
                "targets": {"ids": [self._target]},
                "tier": {"text": "gpu"},
                "pinned": {"flag": getattr(block, "ref_cnt", 0) > 0},
                "page-tokens": {"num": page_tokens},
                "size-tokens": {"num": page_tokens},
                # Older by rank, so the head of the queue is the oldest.
                "last-access-ms": {"num": max(now_ms - rank, 0)},
                # Every offered page is a leaf. vLLM's pool is flat --
                # there is no tree, so no page has a page below it --
                # and a cache port that filters on `leaf` should see all
                # of them rather than none.
                "leaf": {"flag": True},
                # The prefix this page belongs to. A policy that wants a
                # prefix kept can name this instead of the page, and the
                # name survives the page.
                "prefix": {"text": ROOT_OF_PAGE.get(page_id, page_id)},
                # Who would benefit if this page survived: the requests
                # still in the engine whose chain includes it. Not "who
                # computed it" -- that is authorship, it is 1 by
                # construction, and `peek` ranks by demand.
                "beneficiaries": {"ids": self._wanted_by(page_id, live_ids)},
                # How many requests were ever recorded as wanting this
                # page, live or not. Published next to `beneficiaries`
                # because the two together say which of the two ways a
                # demand count can be 1 actually happened: nothing was
                # recorded, or everything recorded has finished.
                "beneficiaries-recorded": {
                    "num": len(BENEFICIARIES_OF_PAGE.get(page_id, ()))
                },
                "hit-count": {
                    "num": int(self._page_hits.get(page_id, 0))
                },
                # Flat pool, stated rather than left unanswered. `leaf`
                # already says the same thing; a policy that ranks by
                # how many pages sit below this one is entitled to the
                # answer zero instead of a fallback that ties every
                # page with every other.
                "child-count": {"num": 0},
                "object-kind": {"text": "kv"},
                # What it costs to get this page back: the tokens the
                # engine would have to prefill again. A page is worth
                # keeping in proportion to what losing it costs, and an
                # eviction-cost policy that is told no cost ranks
                # nothing. Measured: `infercept` read `reload-cost`
                # 76,414 times without an answer.
                "reload-cost": {"num": page_tokens},
                "recompute-tokens": {"num": page_tokens},
                # Free to take: an offered page with no reference on it
                # is exactly what the engine is about to reclaim.
                "reclaimable": {"flag": getattr(block, "ref_cnt", 0) == 0},
                # Only when the engine was told the width of a token.
                # A size in bytes computed from a zero is not a size.
                **(
                    {"size-bytes": {"num": page_tokens * self._bytes_per_token}}
                    if self._bytes_per_token
                    else {}
                ),
                # The program this page serves, inherited from the
                # request that computed it. This is what makes a page
                # attributable to a session at all.
                #
                # Any one of the recorded beneficiaries answers this:
                # they all reached the same prefix, so they are the same
                # program. The demand *count* is the other fact and is
                # published as `beneficiaries`.
                **(
                    {
                        "program-id": {
                            "text": _program_of(
                                next(iter(BENEFICIARIES_OF_PAGE[page_id]))
                            )
                        }
                    }
                    if BENEFICIARIES_OF_PAGE.get(page_id)
                    else {}
                ),
                # When this page is next needed, as far as a queue can
                # say. Published only when the beneficiary is still live:
                # a lookahead policy told a distance for a request that
                # has finished is being lied to, and the absence already
                # means "infinitely far".
                **(
                    {"steps-to-execution": {"num": beneficiary_steps}}
                    if beneficiary_steps is not None
                    else {}
                ),
                **(
                    {"beneficiary-steps": {"nums": [beneficiary_steps]}}
                    if beneficiary_steps is not None
                    else {}
                ),
            }
            # Counted here, once per step the page is offered. A page
            # that keeps reappearing in the offer is one the engine has
            # repeatedly declined to take, which is the only frequency
            # signal a free queue carries. Incremented after the fact is
            # read so the first sighting reports zero rather than one.
            self._page_hits[page_id] = self._page_hits.get(page_id, 0) + 1
        facts[self._target] = self._target_facts()
        return facts


    def _kv_overloaded(self) -> bool:
        """Is the block pool under pressure?

        One definition, read from two places. It was inline in the
        target facts and `fairserve` reads it on the request, so a
        second copy would have been two thresholds that agreed until
        someone changed one.
        """
        try:
            pool = self._scheduler.kv_cache_manager.block_pool
            free_blocks = pool.get_num_free_blocks()
            total_blocks = pool.num_gpu_blocks
        except AttributeError:
            return False
        return bool(total_blocks) and free_blocks < total_blocks / 10

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
            "kv_overloaded": {"flag": self._kv_overloaded()},
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

    def _progress(self) -> list[list[Any]]:
        """Tokens produced since the last step, per request.

        `on-progress` was the one hook no engine raised. Nine policies
        subscribe to it -- agentix, branch-regulation, dlpm, dynasor,
        justitia, llumnix, pard, slos-serve, vtc -- and several learn
        their service rate from it and can do nothing until it arrives.
        branch-regulation is the clearest case: its price per token is
        `service-us` differenced against these deltas, its auction is
        shut while that price is zero, and a shut auction ranks every
        branch at a cost of zero, which sorts back to arrival order. The
        policy was not disagreeing with FCFS, it was never given the one
        number it prices with.

        A delta, not a total. `host/src/events.rs` has a test named
        `progress_is_a_delta_not_a_total`, and a policy that accumulates
        deltas would integrate the whole history at every step if handed
        a cumulative count.

        Deltas are dropped for requests seen for the first time. The
        first sighting's tokens were produced before anyone was
        watching, and charging them to one step would price that step at
        a rate no step ever ran at.
        """
        deltas: list[list[Any]] = []
        live: set[str] = set()
        for request in self._tracked():
            request_id = request.request_id
            live.add(request_id)
            produced = len(request.output_token_ids)
            previous = self._progress_last.get(request_id)
            self._progress_last[request_id] = produced
            if previous is None:
                continue
            grew = produced - previous
            if grew > 0:
                deltas.append([request_id, grew])
        for request_id in list(self._progress_last):
            if request_id not in live:
                del self._progress_last[request_id]
        return deltas

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
        progress = self._progress()
        if progress:
            events["progress"] = progress
        if self._admitted:
            events["admitted"] = list(self._admitted)
        if self._finished:
            events["finished"] = [list(entry) for entry in self._finished]
        if self._paused:
            events["paused"] = [list(entry) for entry in self._paused]
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
