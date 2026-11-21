"""Check the high-mode V files of CG jobs against the original layout.

    check_cg.py s --conf 1540 1550
    check_cg.py l s c --conf 1540
    check_cg.py s h16 --conf 1200 1540

Read-only. For every flavour and configuration, vector_scan.scan checks each
range of the original layout on disk -- gen_cg.py's for the tag (default h8,
config.CG_LAYOUT), with the configuration's AMA hit (config.hit_set): a vector is done only if elem<i>.bin exists
at exactly config.V_ELEM_BYTES. The report has one line per hit and accuracy,
with any missing or wrong-size indices and the original srun (CG<f>.NN) that
owned them, so it serves both as the output check of a finished job and as
the diagnosis after a failure. gen_cg_repack.py scans again rather than read
this file, so a repack always starts from the disk as it is.

The report is printed and written to <out>/check.CG<f>.<conf>.txt (default
config.CG_DIR, beside the job logs). Exit status 1 if anything is missing.

Run it on Frontier, after the jobs writing these directories have ended: a
vector still being written counts as wrong-size.
"""
import argparse
import sys
import time
from pathlib import Path

# The toolkit (config, vector_scan) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from vector_scan import scan, index_runs
from gen_cg import add_replica_arg, flavors_and_n_hit, replica_of, srun_run_id


def report(flavor, traj, scans, hits=config.HitSet(), replica=None):
    """Report lines for one flavour and configuration, and whether every
    vector is on disk at full size."""
    groups = {}
    for r in scans:
        groups.setdefault((r.exact, r.hit), []).append(r)

    lines = []
    expected = present = n_missing = n_wrong = 0
    for (exact, hit), ranges in sorted(groups.items()):
        total = sum(r.count for r in ranges)
        missing = sum(len(r.missing) for r in ranges)
        wrong = sum(len(r.wrong_size) for r in ranges)
        expected += total
        present += total - missing - wrong
        n_missing += missing
        n_wrong += wrong

        label = config.high_extension(flavor, hit, "v", exact)
        acc = "exact" if exact else "sloppy"
        line = f"{label:<10} {acc:<6} {total - missing - wrong:>5}/{total}"
        details = []
        for r in ranges:
            owner = srun_run_id(flavor, r.srun, hits, replica)
            if r.missing:
                details.append(f"missing {index_runs(list(r.missing))} "
                               f"({owner})")
            for i, size in sorted(r.wrong_size.items()):
                details.append(f"size {i}:{size} ({owner})")
        lines.append(line + ("  " + "; ".join(details) if details else "  ok"))

    complete = n_missing == 0 and n_wrong == 0
    rep = "" if replica is None else f" replica {replica}"
    head = (f"# CG{flavor} {hits.tag}{rep} conf {traj}: {present}/{expected} vectors, "
            f"{n_missing} missing, {n_wrong} wrong size -> "
            f"{'complete' if complete else 'INCOMPLETE'}")
    return [head] + lines, complete


def main():
    parser = argparse.ArgumentParser(
        description="Check the CG output V files against the original layout.")
    parser.add_argument("targets", nargs="+", metavar="l|s|c|hN",
                        help="flavours, then optionally a tag hN "
                             f"(default h{config.N_HIT})")
    parser.add_argument("--conf", nargs="+", type=int, required=True)
    parser.add_argument("--out", default=config.CG_DIR,
                        help="where check.CG<f>[.hN][.<rep>].<conf>.txt goes")
    add_replica_arg(parser)
    args = parser.parse_args()
    flavors, n_hit = flavors_and_n_hit(parser, args.targets)
    replica = replica_of(parser, args)

    out_dir = Path(args.out)
    suffix = ("" if n_hit == config.N_HIT else f".h{n_hit}") + (
        "" if replica is None else f".{replica}")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    all_complete = True
    for flavor in flavors:
        for traj in args.conf:
            hits = config.hit_set(n_hit, traj)
            lines, complete = report(flavor, traj,
                                     scan(flavor, traj, hits, replica),
                                     hits, replica)
            all_complete &= complete
            print("\n".join(lines) + "\n")
            text = [f"# checked {stamp}, expected size "
                    f"{config.V_ELEM_BYTES} bytes"] + lines
            (out_dir / f"check.CG{flavor}{suffix}.{traj}.txt").write_text(
                "\n".join(text) + "\n")
    sys.exit(0 if all_complete else 1)


if __name__ == "__main__":
    main()
