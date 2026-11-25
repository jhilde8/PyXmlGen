"""
Which meson-field files of a contraction job are on disk and complete, read
off the HDF5 files themselves, and whether two replicas of a job agree.

A contraction module with timeSliceIO writes one HDF5 file per (ioname,
timeslice), <output>.<traj>/<ioname>.t%04d.h5, to node-local storage, and the
submission script drains the directory to <base>/<field>.<traj>/. Each file
holds one dataset, <ioname>/a2aMatrix, of shape (1, left leg, right leg),
chunked (1, c, c) with Fletcher32 and no other filter (A2AMatrixIo::initFile).

Nothing about the chunking is expected from outside: the block parameters set
c, and they differ between configurations. A file is complete if its dataset
has the field's shape, every chunk is allocated, and every chunk is stored at
prod(chunk) * itemsize + 4 bytes. HDF5 stores a partial edge chunk at full
size, and a chunk never written is not allocated at all, so a module that died
mid-write leaves missing chunks; a copy cut short leaves a file HDF5 refuses
to open (its end-of-file address lies past the end). Only the chunk index is
read for this, never the data.

Replica comparison. Fletcher32 stores a 32-bit checksum as the last 4 bytes of
each chunk, so two replicas of a file are compared chunk by chunk on those 4
bytes: a 4-byte read per chunk, about 30M of them per configuration at 8 hits,
instead of the ~30 TB of data. Equal checksums in every chunk are bitwise
agreement up to a 2^-32 collision per chunk. This does not check a chunk
against its own checksum; any read through the filter does that, and the
replica kept is read in full before the other is deleted.

The files are inspected by a process pool, since the per-chunk reads are
latency bound.
"""
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import ceil, prod
from pathlib import Path

import h5py

TIMESLICE = re.compile(r"(.*)\.t(\d{4})\.h5$")

# HADRONS_A2AM_NAME, the dataset under the ioname group.
DATASET = "a2aMatrix"

# Fletcher32 appends its checksum to every stored chunk.
CHECKSUM_BYTES = 4


@dataclass(frozen=True)
class FieldSpec:
    """What one field should be: n_io ionames, each with n_t timeslice
    files, each a dataset of `shape`."""
    name: str
    n_io: int
    n_t: int
    shape: tuple

    def n_files(self):
        return self.n_io * self.n_t


@dataclass(frozen=True)
class FieldScan:
    """One field across the replicas scanned. Lists indexed by replica: the
    timeslices with a file missing and how many files that is, file -> reason
    for files that are not complete, and names that are not one of the
    field's timeslices (listed, but harmless). Across replicas: file -> what
    differs, and how many files are present, complete and in agreement
    everywhere."""
    spec: FieldSpec
    ionames: int
    missing_t: list
    n_missing: list
    bad: list
    extra: list
    mismatch: dict
    good: int

    def complete(self):
        return self.good == self.spec.n_files()


def field_dir(base, field, traj):
    return Path(base) / f"{field}.{traj}"


def _listing(path):
    try:
        with os.scandir(path) as entries:
            return [e.name for e in entries if e.is_file()]
    except FileNotFoundError:
        return []


def inspect(path, ioname, shape, checksums):
    """(problem, layout) for one file. problem is None for a complete file,
    else why it is not. layout is (chunk shape, dtype, {chunk offset: stored
    checksum}) if `checksums`, else None."""
    try:
        with h5py.File(path, "r") as f:
            dset = f.get(f"{ioname}/{DATASET}")
            if not isinstance(dset, h5py.Dataset):
                return f"no dataset {ioname}/{DATASET}", None
            if dset.shape != shape:
                return f"shape {dset.shape}, expected {shape}", None
            chunks = dset.chunks
            if chunks is None:
                return "not chunked", None
            if (dset.id.get_create_plist().get_nfilters() != 1
                    or not dset.fletcher32):
                return "filters other than Fletcher32 alone", None
            dtype = dset.dtype.str
            stored = prod(chunks) * dset.dtype.itemsize + CHECKSUM_BYTES
            info = []
            dset.id.chunk_iter(info.append)
    except OSError as e:
        return f"unreadable: {e}", None

    n_chunks = prod(ceil(s / c) for s, c in zip(shape, chunks))
    if len(info) != n_chunks:
        return f"{n_chunks - len(info)} of {n_chunks} chunks missing", None
    for s in info:
        if s.size != stored or s.filter_mask:
            return (f"chunk {s.chunk_offset} stored {s.size} bytes, "
                    f"expected {stored}"), None
    if not checksums:
        return None, None

    sums = {}
    fd = os.open(path, os.O_RDONLY)
    try:
        for s in sorted(info, key=lambda s: s.byte_offset):
            end = s.byte_offset + s.size
            sums[s.chunk_offset] = os.pread(fd, CHECKSUM_BYTES,
                                            end - CHECKSUM_BYTES)
            if len(sums[s.chunk_offset]) != CHECKSUM_BYTES:
                return f"short read at chunk {s.chunk_offset}", None
    finally:
        os.close(fd)
    return None, (chunks, dtype, sums)


def _disagreement(layouts):
    """None if every replica's layout is the first one's, else what
    differs from it first."""
    chunks, dtype, sums = layouts[0]
    for k, (c, d, s) in enumerate(layouts[1:], 1):
        if c != chunks or d != dtype:
            return f"replica {k}: chunks {c} {d}, against {chunks} {dtype}"
        differ = sorted(o for o in sums if s[o] != sums[o])
        if differ:
            return (f"{len(differ)} of {len(sums)} chunk checksums differ, "
                    f"first at {differ[0]}")
    return None


def _check(task):
    """One file in every replica that has it: (field, file, problems by
    replica index, disagreement). The comparison needs the file complete in
    every replica."""
    field, name, ioname, shape, paths = task
    compare = len(paths) > 1 and None not in paths
    problems = {}
    layouts = []
    for k, path in enumerate(paths):
        if path is None:
            continue
        problem, layout = inspect(path, ioname, shape, compare)
        if problem:
            problems[k] = problem
        layouts.append(layout)
    if not compare or problems:
        return field, name, problems, None
    return field, name, problems, _disagreement(layouts)


def scan(specs, bases, traj, workers=None):
    """Every field of `specs` under each base of `bases` (one per replica),
    for configuration `traj`, in spec order. With more than one base, a file
    complete in every replica is compared across them."""
    tasks = []
    listed = {}
    for spec in specs:
        present = []
        extra = []
        for base in bases:
            names = set()
            odd = []
            for name in _listing(field_dir(base, spec.name, traj)):
                m = TIMESLICE.match(name)
                if m and int(m.group(2)) < spec.n_t:
                    names.add(name)
                else:
                    odd.append(name)
            present.append(names)
            extra.append(tuple(sorted(odd)))
        listed[spec.name] = (present, extra)
        for name in sorted(set().union(*present)):
            ioname = TIMESLICE.match(name).group(1)
            paths = tuple(str(field_dir(b, spec.name, traj) / name)
                          if name in p else None
                          for b, p in zip(bases, present))
            tasks.append((spec.name, name, ioname, spec.shape, paths))

    bad = {spec.name: [{} for _ in bases] for spec in specs}
    mismatch = {spec.name: {} for spec in specs}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for field, name, problems, differ in pool.map(_check, tasks,
                                                      chunksize=16):
            for k, problem in problems.items():
                bad[field][k][name] = problem
            if differ:
                mismatch[field][name] = differ

    out = []
    for spec in specs:
        present, extra = listed[spec.name]
        union = set().union(*present)
        stems = {TIMESLICE.match(n).group(1) for n in union}
        missing_t, n_missing = [], []
        for names in present:
            ts = set()
            for stem in stems:
                ts |= {t for t in range(spec.n_t)
                       if f"{stem}.t{t:04d}.h5" not in names}
            if len(stems) < spec.n_io:
                ts = set(range(spec.n_t))
            missing_t.append(tuple(sorted(ts)))
            n_missing.append(max(spec.n_files() - len(names), 0))
        good = sum(1 for n in union
                   if all(n in p for p in present)
                   and not any(n in b for b in bad[spec.name])
                   and n not in mismatch[spec.name])
        out.append(FieldScan(spec, len(stems), missing_t, n_missing,
                             bad[spec.name], extra, mismatch[spec.name], good))
    return out


def runs(values):
    """Sorted integers as 'a-b c d-e', consecutive ones joined, so a lost
    node shows as the slab of timeslices it owned."""
    parts = []
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[j] + 1:
            j += 1
        a, b = values[i], values[j]
        parts.append(f"{a}" if a == b else f"{a}-{b}")
        i = j + 1
    return " ".join(parts)
