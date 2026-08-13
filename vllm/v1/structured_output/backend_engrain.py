# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A structured-output backend backed by the engrain compiler.

engrain compiles a grammar to LALR(1) tables and a byte lexer, then groups
the vocabulary by what a token does to the lexer. A mask is the union of the
groups the parser admits, so per-step work is one table replay per group - a
few hundred, and independent of vocabulary size - rather than a walk over the
vocabulary. The compiler lives outside this tree and is imported lazily, so
vLLM only needs it when the backend is actually selected.

Upstreaming note: the constructor signature and the `StructuredOutputGrammar`
protocol are the only coupling. A backend registry would remove the need for
the `if/elif` edit in `vllm/v1/structured_output/__init__.py`.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger
from engrain.internals import StackTooDeep
from engrain.internals import ConfigurationsExceeded, WindowTooWide
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)

logger = init_logger(__name__)

# Rollback slots per matcher. Enough for the longest draft a speculative
# decoder proposes; the state it keeps is one snapshot per slot.
_MATCHER_ROLLBACK = 32


def _vocabulary(tokenizer) -> list[bytes]:
    """The tokenizer's byte strings, indexed by token id."""
    tokens: list[bytes] = []
    for token_id in range(len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        if piece is None:
            tokens.append(b"")
            continue
        try:
            tokens.append(tokenizer.convert_tokens_to_string([piece]).encode("utf-8"))
        except Exception:  # noqa: BLE001
            tokens.append(b"")
    return tokens


@dataclass
class EngrainGrammar(StructuredOutputGrammar):
    matcher: object
    words: int
    stop_token_ids: list[int]
    device: object = None
    grammar_id: int = 0
    # Which admission of that slot this is. A slot freed by an eviction is
    # reused, so the identifier alone would mask this request against whatever
    # took the slot - a wrong mask that looks like a working one.
    generation: int = 0
    _terminated: bool = field(default=False, repr=False)
    _processed: int = field(default=0, repr=False)
    _pinned: bool = field(default=False, repr=False)

    # A request holds its grammar in the pool for as long as it runs, and the
    # eviction policy only ever takes grammars nothing is running under - which
    # is what makes a budget safe rather than a source of wrong masks. The pin
    # is taken by the backend under the lock that resolved the identifier, not
    # here, because everything between the two is a window in which the pool can
    # evict what this request is about to depend on.

    def __del__(self) -> None:
        # vLLM gives a backend no "request finished" hook, so the request's own
        # lifetime is the signal: this object is per request and nothing else
        # holds it. Unpinning late would only delay an eviction; unpinning
        # early is what must not happen, and cannot here.
        if self._pinned and self.device is not None:
            try:
                self.device.unpin(self.grammar_id)
            except Exception:  # noqa: BLE001
                pass
            self._pinned = False

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        if self._terminated:
            return False
        for token in tokens:
            if token in self.stop_token_ids:
                if not self.matcher.can_terminate():
                    return False
                self.matcher.terminate()
                self._terminated = True
                return True
            if not self.matcher.accept_token(token):
                return False
            self._processed += 1
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """The longest prefix the grammar accepts, leaving the state untouched.

        This is the speculative-decoding hook: a draft is checked here and only
        the accepted prefix is committed afterwards.
        """
        accepted: list[int] = []
        for token in tokens:
            if token in self.stop_token_ids:
                if self.matcher.can_terminate():
                    accepted.append(token)
                break
            if not self.matcher.accept_token(token):
                break
            accepted.append(token)
        if accepted:
            self.matcher.rollback(len(accepted))
        return accepted

    def rollback(self, num_tokens: int) -> None:
        self.matcher.rollback(num_tokens)
        self._processed = max(0, self._processed - num_tokens)
        self._terminated = False

    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        """Fill one row. The batched path below is the one that matters."""
        row = bitmask[batch_index]
        row.zero_()
        self.matcher.fill_bitmask(row)
        self._allow_stops(row)

    def _allow_stops(self, row: torch.Tensor) -> None:
        # vLLM stops on a stop token, so it has to remain reachable once the
        # document is complete.
        if self.matcher.can_terminate():
            for stop in self.stop_token_ids:
                row[stop // 32] |= 1 << (stop % 32)

    def is_terminated(self) -> bool:
        return self._terminated

    def reset(self) -> None:
        self.matcher.reset()
        self._terminated = False
        self._processed = 0


@dataclass
class EngrainBackend(StructuredOutputBackend):
    def __post_init__(self) -> None:
        import engrain.internals

        self.compiler = engrain.internals.Compiler(_vocabulary(self.tokenizer))
        self.stop_token_ids = [
            token
            for token in [getattr(self.tokenizer, "eos_token_id", None)]
            if token is not None
        ]
        self.words = (self.vocab_size + 31) // 32
        self.compiled: dict[tuple, object] = {}
        # One arena for every schema the engine has seen, rather than one set of
        # tables per schema: a batch under many schemas is then one launch.
        self.pool: object = None
        self.grammar_ids: dict[tuple, tuple[int, int]] = {}
        # What the tables may occupy. They share the device with the model and
        # the KV cache, so the pool needs a ceiling rather than whatever the
        # allocator will still hand out. Past it a schema no request is running
        # under is evicted, and re-admitted from the cached artifact if it comes
        # back - a device copy, not a recompile.
        self.table_budget_bytes = int(
            os.environ.get("ENGRAIN_TABLE_BUDGET_MB", "1024")
        ) * (1 << 20)
        # Digits an unbounded number may run to. Off by default, because it
        # narrows the language the schema asks for and this engine will not do
        # that on its own. It is here because a model handed a mask that still
        # admits a digit emits one: on a schema whose last property is an
        # unbounded integer, 71.9% of requests ran to the token limit
        # mid-number - and XGrammar's 70.8% on the same schema says that is the
        # language's doing, not either engine's.
        digits = os.environ.get("ENGRAIN_MAX_DIGITS")
        self.max_digits = int(digits) if digits else None
        # The same argument for the other two unbounded constructs. Of 32
        # requests that ran to the token limit on a corpus of real schemas, 19
        # were inside a string and 13 between tokens - `{"xi":  \n\n\n` for two
        # hundred tokens, a position where the model has not decided and
        # whitespace is admitted, then admitted again.
        length = os.environ.get("ENGRAIN_MAX_STRING")
        self.max_string = int(length) if length else None
        space = os.environ.get("ENGRAIN_MAX_WHITESPACE")
        self.max_whitespace = int(space) if space else None
        # The replay window a grammar may want before it is kept on the host
        # instead. `cand_window` is `batch x configurations x readings x window`
        # and 95% of a batch's memory, and the window is the only factor in it
        # that one schema can raise for every row: over 116 corpus schemas it
        # is 20 at the median and 32 at the p90, then 349.
        #
        # Off by default, because it was measured and it is a bad trade.
        # Capping at the p90 takes a batch of 512 from 8.40 GiB to 1.32 and
        # sends the other tenth of schemas to the reference matcher - which
        # costs about 1.5 ms per row per step, so end to end over 409 corpus
        # schemas it is 9,050 tok/s against 2,503. Seven gigabytes for 3.6x of
        # the throughput is not a trade worth making by default; the memory
        # has to come from sizing the window by what candidates need rather
        # than from evicting the schemas that need it.
        # Where the configuration ceiling starts. It grows when a parse wants
        # more, so this is a floor rather than a guess: a row carries 2.5
        # configurations at the mean over real schemas, and every array a batch
        # holds is a multiple of this number.
        start = os.environ.get("ENGRAIN_START_CONFIGS", "8")
        self.start_configs = max(1, int(start))
        window = os.environ.get("ENGRAIN_WINDOW_CAP")
        self.window_cap = int(window) if window and window != "0" else None
        self.batch: object = None
        self._warned_about_ceiling = False
        self._warned_about_narrowing = False
        self._warned_about_bounds = False
        self._warned_about_window = False
        self._narrowed_rows = 0
        self._narrowed_steps = 0
        self._narrow_report = 1
        self._announced = False
        self._assigned: list[int] | None = None
        self._phases: dict[str, list] = {}
        if os.environ.get("ENGRAIN_TIMING"):
            import atexit

            atexit.register(self._report_phases)
        self.max_num_seqs = max(
            8, getattr(self.vllm_config.scheduler_config, "max_num_seqs", 8)
        )
        # vLLM compiles grammars on a thread pool, so two requests with
        # different schemas reach `compile_grammar` at once. Admission hands out
        # the next index and appends to shared arrays, and the cache is a
        # check-then-act, so without this two schemas take the same index and
        # one of them is masked against the other's tables.
        # Reentrant, because a fill holds it across `_batch_for`, which takes
        # it too. vLLM compiles grammars on a thread pool while decode steps
        # run, and admitting one can *move* the arena - a fill that checked the
        # revision a moment earlier then launches against freed memory. It is
        # rare and it is a wrong mask, which is the worst combination.
        self.lock = threading.RLock()
        try:
            from engrain import _engine as device_parser

            self.device_parser = device_parser
        except Exception:  # noqa: BLE001
            # Without the device kernels the CPU matcher still answers; the
            # difference is where the work happens, not whether it is correct.
            self.device_parser = None

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        key = (request_type, grammar_spec)
        # Compiling is the expensive part and needs no lock; only publishing the
        # result and taking an index do. A schema compiled twice under a race is
        # wasted work, not a wrong answer.
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = self._compile(request_type, grammar_spec)
        with self.lock:
            existing = self.compiled.get(key)
            if existing is not None:
                compiled = existing
            self.compiled[key] = compiled
            identifier, generation = 0, 0
            if self.device_parser is not None:
                if self.pool is None:
                    self.pool = self.device_parser.DeviceGrammar(
                        budget_bytes=self.table_budget_bytes,
                        window_cap=self.window_cap,
                        max_configs=self.start_configs,
                    )
                held = self.grammar_ids.get(key)
                # The cache outlives the pool's memory: a schema no request was
                # using may have been evicted since, and the compiled artifact
                # is still here to re-admit from. Re-admission is a device copy,
                # not a recompile, which is why the artifact is worth keeping
                # even when the tables are not.
                if held is None or not self.pool.holds(*held):
                    try:
                        identifier = self.pool.admit(compiled)
                    except WindowTooWide as refusal:
                        # This grammar would size every row of every batch for
                        # itself - the replay window is 95% of a batch's memory
                        # and one schema in ten costs the rest an eightfold
                        # buffer. It runs on the reference matcher instead:
                        # exact, slower for this request, and not larger for
                        # everyone else's.
                        if not self._warned_about_window:
                            self._warned_about_window = True
                            logger.warning(
                                "engrain: %s. That schema stays on the "
                                "reference matcher; the rest of the batch "
                                "keeps the smaller buffers.",
                                refusal,
                            )
                        return EngrainGrammar(
                            matcher=compiled.matcher(_MATCHER_ROLLBACK),
                            words=compiled.bitset_words,
                            stop_token_ids=self.stop_token_ids,
                        )
                    except (torch.cuda.OutOfMemoryError, MemoryError) as refusal:
                        # The arena shares the device with the model and its KV
                        # cache, and the eviction policy will go over budget
                        # rather than refuse a request - so the allocator is
                        # what says no, in the middle of admitting. `admit`
                        # unwinds, and this is the only place that can decide
                        # what to do instead: run the schema on the reference
                        # matcher. Killing the request, or the engine, because
                        # the *fast path* ran out of room is the wrong trade
                        # when a correct slower path is right here.
                        logger.warning(
                            "engrain: the grammar tables would not fit on the "
                            "device (%s). That schema runs on the reference "
                            "matcher; everything already resident is "
                            "untouched.",
                            refusal,
                        )
                        return EngrainGrammar(
                            matcher=compiled.matcher(_MATCHER_ROLLBACK),
                            words=compiled.bitset_words,
                            stop_token_ids=self.stop_token_ids,
                        )
                    self.grammar_ids[key] = (identifier, self.pool.generation(identifier))
                identifier, generation = self.grammar_ids[key]
                # Pinned here, under the lock that resolved it, rather than in
                # the grammar's constructor outside it. vLLM compiles on a
                # thread pool, so between admitting a grammar and pinning it
                # another request could admit one, evict this brand-new and
                # still unpinned grammar, and leave the request that asked for
                # it holding an identifier the pool no longer has - which is
                # exactly "grammar ids no longer in the pool: [33]", and only a
                # workload of hundreds of distinct schemas under a budget gets
                # near enough to the ceiling to show it.
                self.pool.pin(identifier)
            pool = self.pool
        if pool is not None:
            # Compiling happens when a request is admitted, which is before any
            # step it takes part in - so this is where the kernels get built.
            # Left to the first fill, Triton would compile inside a decode step
            # and vLLM would rightly call it a latency spike.
            self._batch_for(self.max_num_seqs)
        return EngrainGrammar(
            # How many steps the matcher can roll back, which is what
            # speculative decoding needs and nothing else does. Not a
            # configuration ceiling - that one is `ENGRAIN_MAX_CONFIGS` and is
            # shared with the device.
            matcher=compiled.matcher(_MATCHER_ROLLBACK),
            words=compiled.bitset_words,
            stop_token_ids=self.stop_token_ids,
            device=pool,
            grammar_id=identifier,
            generation=generation,
            _pinned=pool is not None,
        )

    def fill_bitmasks(
        self, batch: list[tuple[StructuredOutputGrammar, int]], bitmask: torch.Tensor
    ) -> None:  # noqa: D401
        """Fill every row of this step in one launch.

        The tables of every schema live in one arena and a sequence carries the
        index of the one it is under, so a batch under a dozen different schemas
        is a single launch rather than a dozen. That is the shape a serving
        batch has: requests bring their own schemas and the mixture changes
        every step.

        What comes back is copied into the bitmask vLLM expects, and that copy
        is the interface rather than the design - the mask is already where the
        sampler wants it, and a backend allowed to hand vLLM a device tensor
        would skip it.
        """
        rows = [(g, i) for g, i in batch if g.device is not None]
        for grammar, index in batch:
            if grammar.device is None:
                grammar.fill_bitmask(bitmask, index)
        if not rows:
            return

        # Held for the whole device section, not just the lookup: the pool must
        # not be admitted into between choosing the batch and launching against
        # it. Uncontended in steady state - admission happens when a request
        # arrives, not when a token is sampled.
        # A ceiling can be met at *run* time rather than at admission: the
        # stack a parse reaches is a property of the document, and a schema
        # with an unbounded array reaches any depth given a long enough one.
        # The first answer to that was to degrade the offending rows to the
        # reference matcher, which is exact - but the depth was found by
        # rescanning every row's configurations in Python, every step, because
        # the document keeps growing and the ceiling stays met. At batch 512 on
        # a schema with an unbounded array that was 4,981 us per step across
        # 64% of the steps: 58% of everything the backend spent, against 119 us
        # for the fill. So the ceiling is raised once instead, and degrading is
        # what happens only when it cannot be.
        try:
            host, narrowed = self._grow_or_fill(rows)
        except ValueError as refusal:
            import time

            degraded = time.perf_counter()
            if not self._warned_about_ceiling:
                logger.warning(
                    "engrain: %s. The rows past the ceiling fall back to the "
                    "reference matcher; the rest stay on the device. The mask "
                    "is the same either way.",
                    refusal,
                )
                self._warned_about_ceiling = True
            # Only the rows that actually exceed it. Sending the whole step to
            # the host because one request nests deeply would let a single
            # document decide how every other one is served.
            limit = self.batch.grammar.max_stack if self.batch is not None else 0
            deep = []
            shallow = []
            for grammar, index in rows:
                # `max_stack_depth` rather than a max over `configurations`.
                # The latter clones every stack of every row into Python to read
                # one integer from each, and this scan runs on every step for as
                # long as the document keeps growing - at batch 512 on a schema
                # with an unbounded array it was 4,981 us per step across 64% of
                # the steps, 58% of everything the backend spent against 119 us
                # for the fill it was protecting.
                if grammar.matcher.max_stack_depth() > limit:
                    deep.append((grammar, index))
                else:
                    shallow.append((grammar, index))
            self._phase("ceiling rescan", time.perf_counter() - degraded)
            fell_back = time.perf_counter()
            for grammar, index in deep:
                grammar.fill_bitmask(bitmask, index)
            self._phase("host fallback", time.perf_counter() - fell_back)
            self._phase("rows on the host", len(deep) / 1e6)
            if not shallow:
                return
            rows = shallow
            host, narrowed = self._fill(rows)

        # One vectorised copy, not a Python loop over rows. Profiled at batch
        # 256 the loop was 2,289 us of a 2,913 us step - 79% of everything the
        # backend spent, against 33 us for the fill it exists to deliver. The
        # destination rows are scattered, so the copy is an index_copy_ rather
        # than a slice assignment.
        import time

        started = time.perf_counter()
        width = host.shape[1]
        where = torch.tensor([index for _, index in rows], dtype=torch.long)
        bitmask[:, :width].index_copy_(0, where, host)
        if bitmask.shape[1] > width:
            bitmask[:, width:].index_fill_(0, where, 0)
        # The stop token is per row and depends on the matcher, so it stays a
        # loop - but it writes one word rather than copying a row.
        for grammar, index in rows:
            grammar._allow_stops(bitmask[index])
        self._phase("copy out", time.perf_counter() - started)
        # A row the device says it narrowed is filled again from the matcher,
        # which has no ceiling of that kind. Narrowing is the safe direction and
        # still the wrong answer: it forbids what the grammar allows, and the
        # model cannot route around a token that is not in the mask.
        if narrowed.any():
            for row, (grammar, index) in enumerate(rows):
                if narrowed[row]:
                    grammar.fill_bitmask(bitmask, index)
            self._warn_narrowed(int(narrowed.sum()))
        if os.environ.get("ENGRAIN_VERIFY"):
            self._verify(rows, host, narrowed)


    def _phase(self, name: str, elapsed: float) -> None:
        """Accumulate where a step goes, when asked. Off by default.

        A device fill that a host mask has to be read back from puts a
        synchronisation in the middle of a decode loop, and a synchronisation
        waits for everything already queued - including the model's forward.
        That does not show in a kernel timing, only here.
        """
        held = self._phases.setdefault(name, [0.0, 0])
        held[0] += elapsed
        held[1] += 1

    def _report_phases(self) -> None:
        if not self._phases:
            return
        steps = max(count for _, count in self._phases.values())
        print(f"ENGRAIN {steps} steps, per step:", flush=True)
        for name, (total, count) in sorted(
            self._phases.items(), key=lambda item: -item[1][0]
        ):
            print(
                f"ENGRAIN   {name:<16} {total / max(steps, 1) * 1e6:9.1f} us"
                f"   ({count} calls, {total:.2f} s total)",
                flush=True,
            )

    # The configuration ceiling a pool may grow to. The reference matcher
    # stops at 128 of its own, so past that the two would disagree about what
    # they carry and the device would be the wider - which is the direction a
    # mask may err in, and still not one to arrive at by accident.
    _CONFIG_CEILING = 128

    # Past this a parse is a runaway rather than a document, and the buffers
    # are `batch x configurations x depth`: at batch 512 and 128 configurations
    # every 256 of depth is 67 MB.
    _STACK_CEILING = 2048

    def _grow_or_fill(self, rows):
        """Fill, and if the batch is too shallow for the parse, deepen it once.

        The depth a parse reaches is a property of the document, so it cannot be
        settled when the pool is built. Degrading the offending rows to the
        reference matcher is exact but ruinous - measured at 4,577 us a step for
        three rows, because a host fill on a deep stack is ~1.5 ms - and it
        never stops, since the document keeps growing. Growing is once.

        Doubling rather than meeting the request: a document that passed the
        ceiling once will pass it again a token later, and each growth rebuilds
        the batch's buffers and re-records its graphs.
        """
        try:
            return self._fill(rows)
        except ConfigurationsExceeded as refusal:
            # Every per-configuration array is `batch x configurations x
            # something`, so this factor is the whole of what a batch costs:
            # 443 MiB at batch 512 with a ceiling of 128, against a measured
            # 2.5 configurations a row. Starting small and growing on demand
            # pays for what a workload turns out to need instead of what one
            # could imagine wanting.
            with self.lock:
                wanted = min(self._CONFIG_CEILING, max(refusal.needed * 2, 8))
                if self.pool is None or wanted <= self.pool.max_configs:
                    raise
                logger.warning(
                    "engrain: a parse carried %d configurations against the "
                    "batch's %d; rebuilding it %d wide. This happens a few "
                    "times as a workload reveals its widest schema.",
                    refusal.needed,
                    self.pool.max_configs,
                    wanted,
                )
                self.pool.max_configs = wanted
                self.batch = None
            return self._fill(rows)
        except StackTooDeep as refusal:
            with self.lock:
                wanted = min(self._STACK_CEILING, max(refusal.needed * 2, 512))
                if self.pool is None or wanted <= self.pool.max_stack:
                    raise
                logger.warning(
                    "engrain: a parse reached a stack of %d against the "
                    "batch's %d; rebuilding it %d deep. The rows stay on the "
                    "device, which is the point: sending them to the host "
                    "matcher instead costs about 1.5 ms each, every step, for "
                    "as long as the document keeps growing.",
                    refusal.needed,
                    self.pool.max_stack,
                    wanted,
                )
                self.pool.max_stack = wanted
                self.batch = None
            return self._fill(rows)

    def _fill(self, rows):
        """The device fill, and what it says about its own answer.

        Returns the mask rows on the host and, beside them, which of those rows
        the engine flagged as narrowed. Both come back on the copy the mask
        already makes, so reading the flags costs no extra synchronisation.
        """
        import time

        with self.lock:
            started = time.perf_counter()
            device = self._batch_for(len(rows))
            sized = time.perf_counter()
            self._phase("batch_for", sized - started)
            # Only when the assignment has actually changed. `set_grammars`
            # resets the whole batch to each grammar's start state - depth,
            # counts, lexer states, flags, one stack entry per configuration -
            # and `set_matchers` overwrites all of it a line later with the
            # state the request is really in. At batch 512 with a deep stack
            # that reset was 3,282 us a step against 421 for the load. The
            # assignment changes when requests join or leave, which is rare
            # next to how often a token is sampled.
            wanted = self._ids(rows, device.batch)
            if wanted != self._assigned:
                device.set_grammars(wanted)
                self._assigned = wanted
            named = time.perf_counter()
            self._phase("set_grammars", named - sized)
            # The matchers go straight to the packer rather than through a dict
            # of Python configuration lists: at batch 512 that is 352 us
            # against 1,373.
            device.set_matchers([grammar.matcher for grammar, _ in rows])
            seeded = time.perf_counter()
            self._phase("set_matchers", seeded - named)
            mask = device.fill_mask()
            filled = time.perf_counter()
            self._phase("fill launch", filled - seeded)
            host = mask[: len(rows)].to("cpu", non_blocking=False)
            copied = time.perf_counter()
            self._phase("mask to host", copied - filled)
            _, flags = device.problems()
            narrowed = flags[: len(rows)].to("cpu", non_blocking=False).bool()
            self._phase("flags to host", time.perf_counter() - copied)
            return host, narrowed

    def _warn_narrowed(self, count: int) -> None:
        """Say it once, then say it again when it stops being a curiosity.

        A row that narrows falls back to the reference matcher, and a host fill
        is ~1.5 ms - so one row is a footnote and a hundred a step is the whole
        serving loop. Warning once could not tell those apart, and on the exact
        fragment the difference was the whole result.
        """
        self._narrowed_rows += count
        self._narrowed_steps += 1
        if self._narrowed_steps < self._narrow_report:
            return
        self._narrow_report *= 10
        logger.warning(
            "engrain: %d row(s) met a ceiling mid-parse and were given a "
            "narrowed mask, over %d step(s) - %.1f rows a step. Those rows "
            "fall back to the reference matcher, so the mask is exact; the "
            "cost is that they leave the device.",
            self._narrowed_rows,
            self._narrowed_steps,
            self._narrowed_rows / self._narrowed_steps,
        )

    def _verify(self, rows, host, narrowed) -> None:
        """Compare what the device produced against the matcher, row by row.

        Only for finding out where a serving-path disagreement comes from; it is
        a device-to-host copy per row and has no place in a decode loop.
        """
        reference = torch.zeros(host.shape[1], dtype=torch.int32)
        for row, (grammar, index) in enumerate(rows):
            # A narrowed row does not reach the sampler - it is refilled from
            # the matcher - so comparing the device's answer for it would report
            # a disagreement the caller never sees.
            if narrowed[row]:
                continue
            reference.zero_()
            grammar.matcher.fill_bitmask(reference)
            if not torch.equal(host[row], reference):
                extra = int(((host[row] & ~reference) != 0).sum())
                missing = int(((reference & ~host[row]) != 0).sum())
                print(
                    f"ENGRAIN row {row} (bitmask {index}, grammar "
                    f"{grammar.grammar_id}): {extra} words with extra bits, "
                    f"{missing} with missing",
                    flush=True,
                )
                solo = self.pool.new_batch(1)
                solo.set_grammars([grammar.grammar_id])
                solo.set_batch_configurations({0: grammar.matcher.configurations()})
                alone = solo.fill_mask()[0].cpu()
                print(
                    "ENGRAIN   same row computed alone: "
                    + ("agrees with the batch" if torch.equal(alone, host[row])
                       else "DIFFERS from the batch")
                    + "; alone vs matcher: "
                    + ("agrees" if torch.equal(alone, reference) else "DIFFERS"),
                    flush=True,
                )
                print(
                    "ENGRAIN state "
                    + json.dumps(
                        [
                            {
                                "row": other,
                                "grammar": g.grammar_id,
                                "configurations": g.matcher.configurations(),
                            }
                            for other, (g, _) in enumerate(rows)
                        ]
                    ),
                    flush=True,
                )
                raise SystemExit(3)

    def _ids(self, rows, batch: int) -> list[int]:
        """The grammar id of every row, padded to the batch's width.

        The padding rows carry no matcher and produce nothing, but they still
        name a slot - and a slot that names a grammar the pool has evicted is
        refused, which would kill the step over rows nobody asked about. The
        first row's id is live by construction: its request pinned it. Padding
        with the literal 0 is what a workload of hundreds of distinct schemas
        under a table budget eventually breaks, because grammar 0 is simply the
        oldest and the first to be evicted.
        """
        ids = [grammar.grammar_id for grammar, _ in rows]
        return ids + [ids[0]] * (batch - len(ids))

    def _batch_for(self, size: int):
        """A device batch big enough for `size` sequences, reused across steps.

        Rebuilt when a grammar has raised a ceiling the buffers were sized
        from, and when it is too small. **Not** when the pool has merely moved
        its arrays, which is what admitting a grammar into a full array does.

        Nothing a batch owns points into the arena. Its buffers - the mask, the
        stacks, the lexer states, the memo - are sized from the pool's ceilings
        and allocated for itself, and `outgrown` is exactly the question of
        whether those ceilings have risen. What does point into the arena is a
        recorded CUDA graph, and a batch refuses to replay one it did not
        record against; this backend records none, because a mask that has to
        be read back on the host cannot be in a graph anyway.

        Rebuilding on a move was a whole batch of allocations and kernel
        warmup, paid every time a request arrived with a schema large enough to
        grow an array - which on a workload of hundreds of schemas is most of
        them, and which lands on the arriving request's time to first token.
        """
        with self.lock:
            if self.batch is None or self.batch.batch < size or self.batch.outgrown:
                width = max(size, self.max_num_seqs)
                # Said before it is spent. Everything a batch allocates is a
                # function of what the pool already knows, so an operator can
                # be told the number rather than discover it in an allocation
                # failure - which is the whole reason the ceilings grow on
                # demand instead of being guessed high.
                if not self._announced or self.batch is None:
                    self._announced = True
                    logger.info(
                        "engrain: a batch of %d over %d configurations takes "
                        "%.1f MiB of device memory, and the grammar tables "
                        "another %.1f.",
                        width,
                        self.pool.max_configs,
                        self.pool.footprint(width)["total"] / 2**20,
                        self.pool.resident_bytes() / 2**20,
                    )
                self.batch = self.pool.new_batch(width)
                # A new batch has been told nothing.
                self._assigned = None
                # Triton compiles on first use, and first use would otherwise
                # be in the middle of a step.
                self.batch.warmup()
            self.batch.config_count.fill_(1)
            return self.batch

    def _bounded(self, schema: str):
        """Compile with the caller's bounds, and without them if they do not fit.

        A bound is an optimisation - it stops a model running to the token
        limit inside a construct JSON leaves open - and an optimisation that
        makes a schema uncompilable is worse than one that is skipped. Bounding
        whitespace turns a star into a counted automaton, which costs lexer
        states, and one corpus schema in 409 goes past the budget for it.
        """
        try:
            return self.compiler.compile_json_schema(
                schema,
                max_digits=self.max_digits,
                max_string=self.max_string,
                max_whitespace=self.max_whitespace,
            )
        except Exception:  # noqa: BLE001
            if self.max_string is None and self.max_whitespace is None:
                raise
            if not self._warned_about_bounds:
                self._warned_about_bounds = True
                logger.warning(
                    "engrain: a schema does not fit with the string and "
                    "whitespace bounds applied; compiling it without them. It "
                    "keeps the schema at the cost of the runaway they prevent."
                )
            return self.compiler.compile_json_schema(
                schema, max_digits=self.max_digits
            )

    def _compile(self, request_type: StructuredOutputOptions, grammar_spec: str):
        if request_type == StructuredOutputOptions.JSON:
            return self._bounded(grammar_spec)
        if request_type == StructuredOutputOptions.JSON_OBJECT:
            return self.compiler.compile_json_schema(
                json.dumps({"type": "object"})
            )
        if request_type == StructuredOutputOptions.REGEX:
            return self.compiler.compile_regex(grammar_spec)
        if request_type == StructuredOutputOptions.GRAMMAR:
            return self.compiler.compile_ebnf(grammar_spec, "root")
        raise ValueError(f"engrain does not support {request_type}")

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        # Filled rows are written wholesale; unfilled rows must allow
        # everything, which is what all-ones means here.
        return torch.full(
            (max_num_seqs, self.words), -1, dtype=torch.int32, device="cpu"
        )

    def destroy(self) -> None:
        self.compiled.clear()
