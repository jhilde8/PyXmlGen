"""
Cutting a stream of noise vectors into per-srun work.

A job's work is one or more streams of hits, each hit hit_size noise vectors,
and every srun takes the next n vectors of a stream. Since a module computes a
contiguous range of ONE hit (initVec/nVec), each srun's share comes back as a
list of per-hit ranges.
"""
from dataclasses import dataclass

import config


@dataclass(frozen=True)
class Segment:
    """Noise vectors [first, first + count) of one hit."""
    hit: int
    first: int
    count: int


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


def cg_srun_ranges(flavor):
    """config.CG_LAYOUT[flavor] as one (sloppy, exact) pair of segment lists
    per srun: the original CG job, which gen_cg.py builds and the vector scan
    checks against."""
    layout = config.CG_LAYOUT[flavor]
    sloppy = cut(config.CG_SLOPPY_HIT_ORDER[flavor], config.N_HIGH,
                 [n for n, _ in layout])
    exact = cut(config.CG_EXACT_HIT_ORDER[flavor], config.N_HIGH,
                [n for _, n in layout])
    return list(zip(sloppy, exact))
