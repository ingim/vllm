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

from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)


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

    def __post_init__(self) -> None:
        # A request holds its grammar in the pool for as long as it runs. The
        # eviction policy only ever takes grammars nothing is running under, so
        # this is what makes a budget safe rather than a source of wrong masks.
        if self.device is not None:
            self.device.pin(self.grammar_id)
            self._pinned = True

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
        self.batch: object = None
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
                        budget_bytes=self.table_budget_bytes
                    )
                held = self.grammar_ids.get(key)
                # The cache outlives the pool's memory: a schema no request was
                # using may have been evicted since, and the compiled artifact
                # is still here to re-admit from. Re-admission is a device copy,
                # not a recompile, which is why the artifact is worth keeping
                # even when the tables are not.
                if held is None or not self.pool.holds(*held):
                    identifier = self.pool.admit(compiled)
                    self.grammar_ids[key] = (identifier, self.pool.generation(identifier))
                identifier, generation = self.grammar_ids[key]
            pool = self.pool
        if pool is not None:
            # Compiling happens when a request is admitted, which is before any
            # step it takes part in - so this is where the kernels get built.
            # Left to the first fill, Triton would compile inside a decode step
            # and vLLM would rightly call it a latency spike.
            self._batch_for(self.max_num_seqs)
        return EngrainGrammar(
            matcher=compiled.matcher(32),
            words=compiled.bitset_words,
            stop_token_ids=self.stop_token_ids,
            device=pool,
            grammar_id=identifier,
            generation=generation,
        )

    def fill_bitmasks(
        self, batch: list[tuple[StructuredOutputGrammar, int]], bitmask: torch.Tensor
    ) -> None:
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
        with self.lock:
            device = self._batch_for(len(rows))
            device.set_grammars(
                [grammar.grammar_id for grammar, _ in rows]
                + [0] * (device.batch - len(rows))
            )
            # The matchers go straight to the packer rather than through a dict
            # of Python configuration lists: at batch 512 that is 352 us
            # against 1,373.
            device.set_matchers([grammar.matcher for grammar, _ in rows])
            host = device.fill_mask()[: len(rows)].to("cpu", non_blocking=False)
        # vLLM's bitmask spans the model's padded vocabulary and ours spans the
        # tokenizer's, so the rows differ in width. The tail is padding rather
        # than tokens, and a row is written wholesale, so it has to be cleared
        # or the padding would be left allowed from whatever was there before.
        width = host.shape[1]
        for row, (grammar, index) in enumerate(rows):
            bitmask[index, :width].copy_(host[row])
            bitmask[index, width:].zero_()
            grammar._allow_stops(bitmask[index])
        if os.environ.get("ENGRAIN_VERIFY"):
            self._verify(rows, host)

    def _verify(self, rows, host) -> None:
        """Compare what the device produced against the matcher, row by row.

        Only for finding out where a serving-path disagreement comes from; it is
        a device-to-host copy per row and has no place in a decode loop.
        """
        reference = torch.zeros(host.shape[1], dtype=torch.int32)
        for row, (grammar, index) in enumerate(rows):
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

    def _batch_for(self, size: int):
        """A device batch big enough for `size` sequences, reused across steps.

        Rebuilt when the pool has moved under it as well as when it is too
        small: admitting a grammar can raise a ceiling, and buffers sized
        against the old one are too small for the new.
        """
        with self.lock:
            stale = self.batch is not None and (
                self.batch.pool_revision != self.pool.revision
                # A grammar can raise a ceiling without moving an array, and
                # buffers sized against the old one are too small for the new.
                or self.batch.outgrown
            )
            if self.batch is None or self.batch.batch < size or stale:
                self.batch = self.pool.new_batch(max(size, self.max_num_seqs))
                # Triton compiles on first use, and first use would otherwise
                # be in the middle of a step.
                self.batch.warmup()
            self.batch.config_count.fill_(1)
            return self.batch

    def _compile(self, request_type: StructuredOutputOptions, grammar_spec: str):
        if request_type == StructuredOutputOptions.JSON:
            return self.compiler.compile_json_schema(
                grammar_spec, max_digits=self.max_digits
            )
        if request_type == StructuredOutputOptions.JSON_OBJECT:
            return self.compiler.compile_json_schema(
                json.dumps({"type": "object"}), max_digits=self.max_digits
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
