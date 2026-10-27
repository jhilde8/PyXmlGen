"""
Which meson-field files of a contraction job are on disk after the drain, and
which are not.

A contraction module with timeSliceIO writes one HDF5 file per (ioname,
timeslice), <output>.<traj>/<ioname>.t%04d.h5, to node-local NVMe, and the
submission script drains the directory to <base>/<field>.<traj>/. The files
are chunked with Fletcher32, so an interrupted file is short, and a lost node
takes the files it owned with it: a field that finished can still miss the
timeslices of one node.

The expected files of every field are config.MF_EXPECTED, tabulated from a
good configuration as {bytes per file: files}. A file is the right size if
its size is one of its field's; the sizes depend only on the job, never on
the data, and differ by tens of bytes where a missing chunk costs a megabyte,
so a short file cannot land on another valid size.

No file names are tabulated. A field's ionames are read off the files
present and each must have all N_T timeslices, so a missing timeslice is
reported by t, and an ioname with no file at all only by the count falling
short.

Each field directory is listed once (os.scandir), not stat'ed file by file.
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path

import config

TIMESLICE = re.compile(r"(.*)\.t(\d{4})\.h5$")


@dataclass(frozen=True)
class FieldScan:
    """One field of a job and what of it is on disk: how many files are
    missing and at which timeslices, file -> size for the wrong size, and
    files that are not one of its timeslices."""
    name: str
    expected: int
    n_missing: int
    missing_t: tuple
    wrong_size: dict
    extra: tuple

    def present(self):
        return self.expected - self.n_missing - len(self.wrong_size)

    def complete(self):
        return self.n_missing == 0 and not self.wrong_size


def field_dir(base, field, traj):
    return Path(base) / f"{field}.{traj}"


def _sizes(path):
    try:
        with os.scandir(path) as entries:
            return {e.name: e.stat().st_size for e in entries if e.is_file()}
    except FileNotFoundError:
        return {}


def scan_field(base, field, sizes, traj):
    """Field `field` of configuration `traj` against its {bytes: files}."""
    nt = config.LATT_SIZE[-1]
    n_files = sum(sizes.values())
    stems = {}
    extra = []
    wrong = {}
    for name, s in sorted(_sizes(field_dir(base, field, traj)).items()):
        m = TIMESLICE.match(name)
        if not (m and int(m.group(2)) < nt):
            extra.append(name)
            continue
        stems.setdefault(m.group(1), set()).add(int(m.group(2)))
        if s not in sizes:
            wrong[name] = s
    found = sum(len(ts) for ts in stems.values())
    missing_t = set()
    for ts in stems.values():
        missing_t |= set(range(nt)) - ts
    if len(stems) < n_files // nt:
        missing_t = set(range(nt))
    return FieldScan(field, n_files, max(n_files - found, 0),
                     tuple(sorted(missing_t)), wrong, tuple(extra))


def scan(tag, traj):
    """Every field of job `tag` in config.MF_EXPECTED, in table order, checked
    against the files of configuration `traj`."""
    base, fields = config.MF_EXPECTED[tag]
    return [scan_field(base, field, sizes, traj)
            for field, sizes in fields.items()]


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
