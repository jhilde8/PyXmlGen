"""
Which sparsened V and W files are on disk, and which are not.

MUtilities::A2ACoarseGrid writes one block of a combined array through
A2AVectorsIo::writeBlock, mode i of the block to
<SPARSE_ROOT>/<block>.<traj>/elem<i>.bin, numbered from 0 within the block.
The write goes straight to Lustre as elem<i>.bin.tmp and is renamed when
complete, so a killed job leaves .tmp files, and a file counts as done only
at exactly config.SPARSE_ELEM_BYTES. A node lost before its pages were
flushed can still leave a full-size bad file, which only the checksum scan
(Hadrons/tests/Test_a2a_checksum_scan on the coarse grid) finds.

A block is the unit of a redo: the module always writes it from elem0.

Each block directory is listed once (os.scandir), not stat'ed file by file.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import config


@dataclass(frozen=True)
class BlockScan:
    """One block and what of it is on disk: indices with no file, index ->
    size for files of the wrong size, and the number of .tmp files."""
    stem: str
    size: int
    missing: tuple
    wrong_size: dict
    tmp: int

    def name(self):
        return Path(self.stem).name

    def bad(self):
        """Every index that is not on disk complete, sorted."""
        return sorted(self.missing + tuple(self.wrong_size))


def _sizes(path):
    try:
        with os.scandir(path) as entries:
            return {e.name: e.stat().st_size for e in entries if e.is_file()}
    except FileNotFoundError:
        return {}


def scan(blocks, traj):
    """Every block of `blocks` (output stem -> mode count), in order, checked
    against the files of configuration `traj`."""
    out = []
    for stem, size in blocks.items():
        sizes = _sizes(f"{stem}.{traj}")
        missing, wrong = [], {}
        for i in range(size):
            s = sizes.get(f"elem{i}.bin")
            if s is None:
                missing.append(i)
            elif s != config.SPARSE_ELEM_BYTES:
                wrong[i] = s
        tmp = sum(name.endswith(".tmp") for name in sizes)
        out.append(BlockScan(stem, size, tuple(missing), wrong, tmp))
    return out
