"""
Which high-mode V files of a CG layout are on disk, and which are not.

Hadrons' A2AVectorsIo::writeElement writes noise vector i of a hit to its own
file, <high_stem><extension>.<traj>/elem<i>.bin. A vector counts as done only
at exactly config.V_ELEM_BYTES: a node failure during a write leaves a short
file, and the scan reports it beside the missing ones.

The expected set is always the original layout (srun_layout.cg_srun_ranges),
never a repack's. A repack redoes only part of it, so scanning against the
original stays right however many repacks have run, and the scan of a finished
configuration doubles as its output check.

Each vector directory is listed once (os.scandir), not stat'ed file by file.
"""
import os
from dataclasses import dataclass

import config
from srun_layout import cg_srun_ranges


@dataclass(frozen=True)
class RangeScan:
    """One solver range of the original layout and what of it is on disk:
    indices with no file, and index -> size for files of the wrong size."""
    srun: int
    exact: bool
    hit: int
    first: int
    count: int
    missing: tuple
    wrong_size: dict

    def bad(self):
        """Every index that has to be solved again, sorted."""
        return sorted(self.missing + tuple(self.wrong_size))


def vector_dir(flavor, hit, exact, traj):
    return (config.high_stem(flavor, exact)
            + config.high_extension(flavor, hit, "v", exact) + f".{traj}")


def _sizes(path):
    try:
        with os.scandir(path) as entries:
            return {e.name: e.stat().st_size for e in entries if e.is_file()}
    except FileNotFoundError:
        return {}


def scan(flavor, traj):
    """Every range of the original layout of `flavor`, in srun order, checked
    against the files of configuration `traj`."""
    listings = {}
    out = []
    for k, (sloppy, exact) in enumerate(cg_srun_ranges(flavor)):
        for is_exact, segs in ((True, exact), (False, sloppy)):
            for seg in segs:
                path = vector_dir(flavor, seg.hit, is_exact, traj)
                if path not in listings:
                    listings[path] = _sizes(path)
                sizes = listings[path]
                missing, wrong = [], {}
                for i in range(seg.first, seg.first + seg.count):
                    size = sizes.get(f"elem{i}.bin")
                    if size is None:
                        missing.append(i)
                    elif size != config.V_ELEM_BYTES:
                        wrong[i] = size
                out.append(RangeScan(k, is_exact, seg.hit, seg.first,
                                     seg.count, tuple(missing), wrong))
    return out


def index_runs(indices):
    """Sorted indices as 'a-b c d-e', consecutive indices joined."""
    parts = []
    i = 0
    while i < len(indices):
        j = i
        while j + 1 < len(indices) and indices[j + 1] == indices[j] + 1:
            j += 1
        a, b = indices[i], indices[j]
        parts.append(f"{a}" if a == b else f"{a}-{b}")
        i = j + 1
    return " ".join(parts)
