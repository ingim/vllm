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


def first_cached(queue: FreeKVCacheBlockQueue) -> Any:
    """The first block in the queue that holds cached content.

    Everything before it has no hash and costs nothing to take, which is
    the invariant `free_blocks` maintains by prepending unhashed blocks
    and appending hashed ones. An eviction order is about the cached
    part; the free part is not a decision.
    """
    block = queue.fake_free_list_head.next_free_block
    tail = queue.fake_free_list_tail
    while block is not None and block is not tail:
        if getattr(block, "block_hash", None) is not None:
            return block
        block = block.next_free_block
    return None


def splice_before(queue: FreeKVCacheBlockQueue, anchor: Any, blocks: list) -> None:
    """Insert `blocks`, in order, immediately before `anchor`.

    The queue has no insert-at-position, and building one out of
    `remove` plus `prepend_n` is what put cached pages ahead of unused
    ones. The links are set directly and `num_free_blocks` is restored
    by hand, because `remove` decremented it for each block taken out.
    """
    if not blocks:
        return
    previous = anchor.prev_free_block
    for block in blocks:
        block.prev_free_block = previous
        previous.next_free_block = block
        previous = block
    previous.next_free_block = anchor
    anchor.prev_free_block = previous
    queue.num_free_blocks += len(blocks)


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
        self._rank: dict[str, int] = {}
        # How many named pages the last splice actually found. The
        # settled check compares against what was placeable, not against
        # the order's length: an order naming five hundred pages of
        # which twelve are free is satisfied by those twelve.
        # `None` until a splice has run, so "nothing placed yet" is not
        # confused with "nothing was placeable". The first call after an
        # install must always do the work: with a zero here it settled
        # immediately and `applied` stayed at zero forever.
        self._placeable: int | None = None
        self.installs = 0
        # Pages named by the policy that were found in the free queue and
        # moved, against pages named at all. Reported rather than
        # inferred: a policy whose every choice names a block that is not
        # free has decided nothing, and from outside that is
        # indistinguishable from a policy that agrees with LRU.
        self.named = 0
        self.applied = 0
        self.calls = 0
        # Allocations where the queue already matched the order. The
        # difference between this and `calls` is how much work the
        # attachment avoided.
        self.settled = 0

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
                # A new order invalidates the last splice's accounting.
                self._placeable = None
                self._order = [str(page) for page in order]
                # Position of each page, so `_settled` can start from
                # wherever the queue's cached region begins rather than
                # only from the front.
                self._rank = {page: i for i, page in enumerate(self._order)}
                self.installs += 1
            return

    def _settled(self, queue: FreeKVCacheBlockQueue) -> bool:
        """Are the named pages at the front, in order, allowing removals?

        Three versions, and the middle one is the lesson.

        **Exact match from `order[0]`** failed on every call after the
        first, because the engine takes blocks off the head: the fast
        path fired 8 to 36 times in 200 allocations.

        **Non-decreasing rank anywhere in the region** settled
        everything: `settled=1000, applied=0`, because a cached region
        with no named page in it trivially satisfies it. A relaxation
        that a *completely unordered* queue also satisfies is not a
        weaker check, it is no check.

        What the splice actually establishes is that the named pages sit
        **at the front of the cached region, in the order's order**, and
        that survives removals -- a page taken by the engine or pulled
        out by a cache hit does not make the rest wrong. So: walk from
        the first cached block while the pages are named, require their
        ranks to increase, and require the run to end only when the
        order is exhausted or an unnamed page appears. An unnamed page
        *before* a named one is the violation the splice exists to fix.
        """
        if self._placeable is None:
            return False
        block = first_cached(queue)
        tail = queue.fake_free_list_tail
        highest = -1
        seen_named = 0
        while block is not None and block is not tail:
            identifier = page_id(block)
            rank = self._rank.get(identifier) if identifier else None
            if rank is None:
                # An unnamed page here means everything after it is
                # behind something the order did not ask to be behind.
                # Settled only if the splice's own placements are all
                # accounted for.
                # Every named page still *present* is accounted for.
                #
                # Comparing against the splice's own count fails the
                # moment a cache hit takes one of them out: the run is
                # shorter and nothing is wrong. What has to hold is that
                # no named page sits behind an unnamed one, which is
                # checked by looking for any further along.
                return not self._named_beyond(block)
            if rank < highest:
                return False
            highest = rank
            seen_named += 1
            block = block.next_free_block
        return True

    def _named_beyond(self, block: Any) -> bool:
        """Is any page the order names still further down the queue?

        The splice put them all at the front, so one found after an
        unnamed page has been passed by something the order did not ask
        to be passed by -- a new block freed into the middle, or a
        reordering the engine did for its own reasons.
        """
        while block is not None:
            identifier = page_id(block)
            if identifier is not None and identifier in self._rank:
                return True
            block = block.next_free_block
        return False

    def prefer(self, queue: FreeKVCacheBlockQueue) -> int:
        """Move the policy's victims to the head of the free queue.

        Returns how many were moved. Called immediately before the pool
        reads, so nothing can happen between the reorder and the take —
        a read may happen mid-pass, a write may not, and this is a write.
        """
        if not self._order:
            return 0

        self.calls += 1
        if self._settled(queue):
            self.settled += 1
            # Reported here too. The counters used to print only on the
            # slow path, so once the fast path started firing the log
            # went silent -- and a silent counter is the same evidence
            # as an absent one, which is the confusion §50 cost a round
            # to.
            self._report()
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
        if not victims:
            self._report()
            return 0

        # Spliced in front of the first *cached* block, not in front of
        # the queue.
        #
        # vLLM's `free_blocks` prepends blocks with no hash and appends
        # blocks with one, so the queue is [never-used ... cached] and
        # the engine spends the never-used blocks first. Prepending to
        # the head puts cached pages ahead of blocks that cost nothing to
        # take, and the engine then evicts cache it did not have to.
        #
        # Measured, and it is the dominant term: an order naming exactly
        # what LRU was going to evict anyway -- the identity order, which
        # should be free -- cost 0.0097 of hit rate, against 0.0133 for a
        # deliberately hostile one. Two thirds of what looked like the
        # policies' bad judgement was this.
        # By object identity, not by membership. `KVCacheBlock` defines
        # `__eq__` without `__hash__`, so it is unhashable and `in set(...)`
        # raises -- which killed the engine on the first allocation. The
        # offline test used a stand-in class that was hashable and could
        # not have caught it.
        anchor = first_cached(queue)
        taken = {id(victim) for victim in victims}
        for victim in victims:
            queue.remove(victim)
        if anchor is None or id(anchor) in taken:
            anchor = first_cached(queue)
        if anchor is None:
            queue.append_n(victims)
        else:
            splice_before(queue, anchor, victims)

        self.applied += len(victims)
        self._placeable = len(victims)
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
            f"applied={self.applied} settled={self.settled} "
            f"order={len(self._order)} installs={self.installs}",
            file=sys.stderr,
            flush=True,
        )
