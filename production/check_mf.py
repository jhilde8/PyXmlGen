"""Check the drained meson fields of contraction jobs against config.

    check_mf.py h8 --conf 1540 1550

Read-only. For every job and configuration, mf_scan.scan checks each field
of config.MF_EXPECTED on disk: a field is complete only if it has its full
count of files, each at one of its tabulated sizes. The report has one line
per field, with any missing or wrong-size files as timeslice runs, so a lost
node shows as the slab it owned in every field it wrote. Files that are not
one of a field's timeslices are listed but do not make it incomplete.

A field is the unit of a rerun: a module that died mid-write leaves every
one of its files short. The last report line names the fields to rerun.

The report is printed and written to <out>/check.<tag>.<conf>.txt (default
config.MF_DIR). Exit status 1 if anything is missing or the wrong size.

Run it on Frontier, after the drain has ended: a file still being copied
counts as the wrong size.
"""
import argparse
import sys
import time
from pathlib import Path

# The toolkit (config, mf_scan) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from mf_scan import TIMESLICE, scan, runs


def report(tag, traj, scans):
    """Report lines for one job and configuration, and whether every file is
    on disk at a tabulated size."""
    lines = []
    expected = present = n_missing = n_wrong = 0
    rerun = []
    for f in scans:
        expected += f.expected
        present += f.present()
        n_missing += f.n_missing
        n_wrong += len(f.wrong_size)

        details = []
        if f.n_missing == f.expected:
            details.append("absent")
        elif f.n_missing:
            details.append(f"missing {f.n_missing} at t {runs(f.missing_t)}")
        if f.wrong_size:
            ts = sorted({int(TIMESLICE.match(n).group(2)) for n in f.wrong_size})
            details.append(f"size {len(f.wrong_size)} at t {runs(ts)}")
        if f.extra:
            details.append(f"extra {len(f.extra)}: {', '.join(f.extra[:3])}"
                           + (" ..." if len(f.extra) > 3 else ""))
        if not f.complete():
            rerun.append(f.name)
        lines.append(f"{f.name:<24} {f.present():>5}/{f.expected:<5}  "
                     + ("; ".join(details) if details else "ok"))

    complete = not rerun
    head = (f"# {tag} conf {traj}: {present}/{expected} files, "
            f"{n_missing} missing, {n_wrong} wrong size -> "
            f"{'complete' if complete else 'INCOMPLETE'}")
    tail = [f"# rerun: {' '.join(rerun)}"] if rerun else []
    return [head] + lines + tail, complete


def main():
    parser = argparse.ArgumentParser(
        description="Check drained meson fields against config.MF_EXPECTED.")
    parser.add_argument("tags", nargs="+", choices=sorted(config.MF_EXPECTED))
    parser.add_argument("--conf", nargs="+", type=int, required=True)
    parser.add_argument("--out", default=config.MF_DIR,
                        help="where check.<tag>.<conf>.txt goes")
    args = parser.parse_args()

    out_dir = Path(args.out)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    all_complete = True
    for tag in args.tags:
        for traj in args.conf:
            lines, complete = report(tag, traj, scan(tag, traj))
            all_complete &= complete
            print("\n".join(lines) + "\n")
            text = [f"# checked {stamp}, base "
                    f"{config.MF_EXPECTED[tag][0]}"] + lines
            (out_dir / f"check.{tag}.{traj}.txt").write_text(
                "\n".join(text) + "\n")
    sys.exit(0 if all_complete else 1)


if __name__ == "__main__":
    main()
