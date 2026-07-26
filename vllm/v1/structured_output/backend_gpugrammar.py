# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A structured-output backend backed by the gpugrammar compiler.

gpugrammar compiles a grammar to LALR(1) tables and a byte lexer, then groups
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
class GpuGrammarGrammar(StructuredOutputGrammar):
    matcher: object
    words: int
    stop_token_ids: list[int]
    device: object = None
    _terminated: bool = field(default=False, repr=False)
    _processed: int = field(default=0, repr=False)

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
class GpuGrammarBackend(StructuredOutputBackend):
    def __post_init__(self) -> None:
        import gpugrammar

        self.compiler = gpugrammar.Compiler(_vocabulary(self.tokenizer))
        self.stop_token_ids = [
            token
            for token in [getattr(self.tokenizer, "eos_token_id", None)]
            if token is not None
        ]
        self.words = (self.vocab_size + 31) // 32
        self.compiled: dict[tuple, object] = {}
        self.device: dict[tuple, object] = {}
        self.batches: dict[int, object] = {}
        try:
            from gpu_lr1 import device_parser

            self.device_parser = device_parser
        except Exception:  # noqa: BLE001
            # Without the device kernels the CPU matcher still answers; the
            # difference is where the work happens, not whether it is correct.
            self.device_parser = None

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        key = (request_type, grammar_spec)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = self._compile(request_type, grammar_spec)
            self.compiled[key] = compiled
            if self.device_parser is not None:
                self.device[key] = self.device_parser.DeviceGrammar(compiled)
        return GpuGrammarGrammar(
            matcher=compiled.matcher(32),
            words=compiled.bitset_words,
            stop_token_ids=self.stop_token_ids,
            device=self.device.get(key),
        )

    def fill_bitmasks(
        self, batch: list[tuple[StructuredOutputGrammar, int]], bitmask: torch.Tensor
    ) -> None:
        """Fill every row of this step, on the accelerator where possible.

        Requests sharing a grammar share its device tables, so they are grouped
        and each group is one kernel launch over its sequences. What comes back
        is copied into the bitmask vLLM expects; that copy is the interface, not
        the design — the mask is already where the sampler wants it, and a
        backend allowed to hand vLLM a device tensor would skip it.
        """
        import torch as _torch

        by_grammar: dict[int, list[tuple[GpuGrammarGrammar, int]]] = {}
        for grammar, index in batch:
            if grammar.device is None:
                grammar.fill_bitmask(bitmask, index)
                continue
            by_grammar.setdefault(id(grammar.device), []).append((grammar, index))

        for entries in by_grammar.values():
            device = entries[0][0].device
            rows = self._batch_for(device, len(entries))
            rows.set_batch_configurations(
                {
                    row: grammar.matcher.configurations()
                    for row, (grammar, _) in enumerate(entries)
                }
            )
            host = rows.fill_mask()[: len(entries)].to("cpu", non_blocking=False)
            for row, (grammar, index) in enumerate(entries):
                bitmask[index].copy_(host[row])
                grammar._allow_stops(bitmask[index])
        del _torch

    def _batch_for(self, device, size: int):
        """A device batch big enough for `size` sequences, reused across steps."""
        cached = self.batches.get(id(device))
        if cached is None or cached.batch < size:
            cached = device.new_batch(max(size, 8))
            self.batches[id(device)] = cached
        cached.config_count.fill_(1)
        return cached

    def _compile(self, request_type: StructuredOutputOptions, grammar_spec: str):
        if request_type == StructuredOutputOptions.JSON:
            return self.compiler.compile_json_schema(grammar_spec)
        if request_type == StructuredOutputOptions.JSON_OBJECT:
            return self.compiler.compile_json_schema(json.dumps({"type": "object"}))
        if request_type == StructuredOutputOptions.REGEX:
            return self.compiler.compile_regex(grammar_spec)
        if request_type == StructuredOutputOptions.GRAMMAR:
            return self.compiler.compile_ebnf(grammar_spec, "root")
        raise ValueError(f"gpugrammar does not support {request_type}")

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        # Filled rows are written wholesale; unfilled rows must allow
        # everything, which is what all-ones means here.
        return torch.full(
            (max_num_seqs, self.words), -1, dtype=torch.int32, device="cpu"
        )

    def destroy(self) -> None:
        self.compiled.clear()
