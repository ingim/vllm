"""The cache channel, enacted: a policy's eviction order, honoured.

Stage ② for the cache channel, the counterpart of `plex_queue.py` for the
schedule channel. Stage ① lets a policy *see* the pages the engine is
about to take; this lets it *choose among them*.

## Where the decision is

`BlockPool.get_new_blocks` takes blocks off the head of
`free_block_queue`, which vLLM keeps in eviction order — least recently
used first. That single `popleft_n` is the eviction decision, and it is
the only one: everything upstream is about which blocks are free, not
which free block goes next.

So this does not replace the queue, override the pool, or change what is
evictable. It reorders the front of a queue the engine already
maintains, immediately before the engine reads it, and only for the
pages a policy named. Pages the policy did not name keep their LRU
position relative to each other, which is what makes a partial order
safe: **a policy that names three pages has expressed a preference about
three pages, not about the pool.**

## The page identity, and why it has to be the observer's

A policy names a page by the id stage ① published, `p<hash>`, derived
from the block's own `block_hash`. Deriving it the same way here is not
a convenience — a different derivation would silently name different
blocks, and the failure would look like a policy making poor choices
rather than a bridge that lost the referent.

## What it deliberately does not do

**It does not free anything.** Freeing a block that a request still
references is a correctness bug, and no eviction order should be able to
cause one. This only reorders blocks that are *already free*, so the
worst a wrong order can do is evict a cache entry that would have been
useful — which is exactly the decision under measurement.

**It does not fall back.** If the bridge is silent the queue is left
alone and the engine evicts as it always would, in LRU order. Stage ①'s
principle carries forward: a policy that does not answer must not change
what the engine would have done.

Enabled by `VLLM_PLEX_EVICT=/path/to/evict.jsonl`, one installed order
per line, most recent wins.
"""

from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue

from vllm.v1.core.sched.plex_observer import _page_id


def page_id(block: Any) -> str | None:
    """The id stage ① published for this block, or None if it has none.

    Delegates to the observer's own derivation rather than repeating it.
    A second copy is how the referent gets lost: the ids agreed until
    someone changed one of them, and a policy naming a page that resolves
    to nothing is indistinguishable from a policy that agrees with LRU.
    That already happened once, when the id was `hash(block_hash)` and
    Python randomised it per process.
    """
    block_hash = getattr(block, "block_hash", None)
    if block_hash is None:
        return None
    return _page_id(block_hash)


class PlexEviction:
    """A standing eviction order, applied at the moment the pool reads.

    The order is a document replaced wholesale, for the same reason the
    schedule table is: a partially applied eviction order is not a
    thing a policy can have meant.
    """

    def __init__(self, source: str | None = None) -> None:
        self._source = source if source else os.environ.get("VLLM_PLEX_EVICT")
        self._source_stamp: tuple[int, int] | None = None
        self._order: list[str] = []
        self.installs = 0
        # Pages named by the policy that were found in the free queue and
        # moved, against pages named at all. Reported rather than
        # inferred: a policy whose every choice names a block that is not
        # free has decided nothing, and from outside that is
        # indistinguishable from a policy that agrees with LRU.
        self.named = 0
        self.applied = 0
        self.calls = 0

    @staticmethod
    def maybe() -> PlexEviction | None:
        """Attached only when asked for, like every other stage."""
        if not os.environ.get("VLLM_PLEX_EVICT"):
            return None
        return PlexEviction()

    def reload(self) -> None:
        """Re-read the order if the file changed.

        Stat-then-read rather than watch: the engine must not block on a
        runtime it does not own, and a stale order is a decision about an
        older pool rather than a failure.
        """
        if not self._source:
            return
        try:
            status = os.stat(self._source)
        except OSError:
            return
        stamp = (status.st_mtime_ns, status.st_size)
        if stamp == self._source_stamp:
            return
        self._source_stamp = stamp
        try:
            with open(self._source, encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            return
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                document = json.loads(line)
            except ValueError:
                continue
            order = document.get("evict")
            if isinstance(order, list):
                self._order = [str(page) for page in order]
                self.installs += 1
            return

    def prefer(self, queue: FreeKVCacheBlockQueue) -> int:
        """Move the policy's victims to the head of the free queue.

        Returns how many were moved. Called immediately before the pool
        reads, so nothing can happen between the reorder and the take —
        a read may happen mid-pass, a write may not, and this is a write.
        """
        if not self._order:
            return 0

        # Walk the queue once, building only the mapping the order needs.
        # Walking it per named page would be quadratic in the pool, and
        # the pool is the largest thing in the engine.
        wanted = set(self._order)
        found: dict[str, Any] = {}
        block = queue.fake_free_list_head.next_free_block
        tail = queue.fake_free_list_tail
        while block is not None and block is not tail and len(found) < len(wanted):
            identifier = page_id(block)
            if identifier is not None and identifier in wanted:
                found[identifier] = block
            block = block.next_free_block

        victims = [found[page] for page in self._order if page in found]
        self.named += len(self._order)
        self.calls += 1
        if not victims:
            self._report()
            return 0

        for victim in victims:
            queue.remove(victim)
        queue.prepend_n(victims)

        self.applied += len(victims)
        self._report()
        return len(victims)

    def _report(self) -> None:
        """Say what the order actually reached, periodically.

        Not optional and not silent. A policy whose every named page is
        absent from the free queue has decided nothing, and from the
        outside that is indistinguishable from a policy that agrees with
        LRU -- which is exactly the reading v0.7 published for its
        `continuum` arm. The counters separate the two, and they belong
        in the engine's own log because that is where a reader goes when
        an arm comes out flat.
        """
        if self.calls % 200:
            return
        print(
            f"[plex-cache] calls={self.calls} named={self.named} "
            f"applied={self.applied} order={len(self._order)} "
            f"installs={self.installs}",
            file=sys.stderr,
            flush=True,
        )
