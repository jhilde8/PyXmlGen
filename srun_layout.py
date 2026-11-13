"""
Cutting a stream of noise vectors into per-srun work.

A job's work is one or more streams of hits, each hit hit_size noise vectors,
and every srun takes the next n vectors of a stream. Since a module computes a
contiguous range of ONE hit (initVec/nVec), each srun's share comes back as a
list of per-hit ranges.

Two ways to cut: cut() hands each srun a fixed vector count (config.CG_LAYOUT,
the hand-balanced h8 layout), and split() balances solver calls on
config.CG_COST (every other HitSet, and every repack).
"""
import math
from dataclasses import dataclass

import config


@dataclass(frozen=True)
class Segment:
    """Noise vectors [first, first + count) of one hit."""
    hit: int
    first: int
    count: int


@dataclass(frozen=True)
class Call:
    """One solver call: vectors [first, first + batch) of one hit."""
    exact: bool
    hit: int
    first: int


def cut(hits, hit_size, counts):
    """Consume `hits` in order, giving srun k exactly counts[k] vectors as
    contiguous per-hit segments. A zero count gives an empty list. The counts
    have to use the whole stream: a remainder would be a hit range that no srun
    computes."""
    total = len(hits) * hit_size
    if any(n < 0 for n in counts):
        raise ValueError(f"negative srun count in {counts}")
    if sum(counts) != total:
        raise ValueError(f"srun counts sum to {sum(counts)} but the stream is "
                         f"{len(hits)} hits x {hit_size} = {total} vectors")

    out = []
    h, pos = 0, 0
    for n in counts:
        segs = []
        while n > 0:
            take = min(n, hit_size - pos)
            segs.append(Segment(hits[h], pos, take))
            n -= take
            pos += take
            if pos == hit_size:
                h, pos = h + 1, 0
        out.append(segs)
    return out


def batch_of(flavor):
    return max(config.CG_BATCH_SIZE[flavor], 1)


def cost(flavor, call):
    return config.CG_COST[(flavor, "exact" if call.exact else "sloppy")]


def split(flavor, calls, n):
    """Contiguous runs of `calls` of near-equal total cost: a call goes to the
    srun whose share of the total contains its midpoint, so every srun is
    within one call of total / n."""
    total = sum(cost(flavor, c) for c in calls)
    sruns = [[] for _ in range(n)]
    done = 0.0
    for c in calls:
        w = cost(flavor, c)
        sruns[min(int((done + w / 2) * n / total), n - 1)].append(c)
        done += w
    return [s for s in sruns if s]


def segments(flavor, calls):
    """An srun's calls as (sloppy, exact) segment lists, adjacent calls of
    one hit and accuracy merged into one range."""
    batch = batch_of(flavor)
    out = {False: [], True: []}
    for c in calls:
        segs = out[c.exact]
        last = segs[-1] if segs else None
        if last and last.hit == c.hit and last.first + last.count == c.first:
            segs[-1] = Segment(c.hit, last.first, last.count + batch)
        else:
            segs.append(Segment(c.hit, c.first, batch))
    return out[False], out[True]


def all_calls(flavor, hits):
    """Every solver call of a configuration, hit by hit in the order of
    config.cg_hit_order, exact before sloppy within a hit, so each hit's noise
    is used by a contiguous run of calls."""
    sloppy, exact = config.cg_hit_order(flavor, hits)
    batch = batch_of(flavor)
    return [Call(is_exact, h, first)
            for h in dict.fromkeys(sloppy + exact)
            for is_exact, stream in ((True, exact), (False, sloppy))
            if h in stream
            for first in range(0, config.N_HIGH, batch)]


def cg_srun_ranges(flavor, hits=config.HitSet()):
    """The original CG job of one configuration, as one (sloppy, exact) pair
    of segment lists per srun: what gen_cg.py builds and the vector scan
    checks against. The default HitSet is config.CG_LAYOUT; any other is cut
    into sruns of at most config.CG_SRUN_HOURS each, startup included."""
    if hits == config.HitSet():
        layout = config.CG_LAYOUT[flavor]
        sloppy, exact = config.cg_hit_order(flavor, hits)
        return list(zip(cut(sloppy, config.N_HIGH, [n for n, _ in layout]),
                        cut(exact, config.N_HIGH, [n for _, n in layout])))

    calls = all_calls(flavor, hits)
    total = sum(cost(flavor, c) for c in calls)
    budget = config.CG_SRUN_HOURS * 3600 - config.CG_STARTUP[flavor]
    return [segments(flavor, s)
            for s in split(flavor, calls, math.ceil(total / budget))]
