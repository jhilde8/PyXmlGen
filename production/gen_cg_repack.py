"""Rebuild the unfinished part of a CG job as new, balanced sruns.

    gen_cg_repack.py s --conf 1540 1550
    gen_cg_repack.py l --conf 1540 --sruns 2
    gen_cg_repack.py s h16 --conf 1540     an h8 configuration to 16 hits

For each configuration, vector_scan.scan checks the original layout of the
tag (default h8) on disk, with the configuration's AMA hit (config.hit_set):
an h8 configuration in config.AMA8_CONFS keeps AMA hit 8, so at h16 its
missing work is sloppy hits 9..16 and nothing exact. It is the same scan
check_cg.py reports, and every missing or wrong-size vector is work again. A batched flavour (light, config.CG_BATCH_SIZE) redoes whole
batches: every original range starts on a multiple of the batch size, so the
batch holding index i starts at i - i % batch, and vectors of that batch
already on disk are solved again with the rest. It also redoes the batch
before the first bad vector of each run, when that batch is in the same range:
with no fsync, the last batch written before a node failure is suspect even
at full size. 1054 vectors on disk of a range from 0 redoes 992 onwards.

The work is cut into sruns of equal walltime, each solver call weighted by
config.CG_COST, sloppy and exact free to share an srun. Calls are ordered by
hit, exact before sloppy, and every srun takes a contiguous run of them, so a
hit's noise is loaded by as few sruns as possible. The srun count defaults to
the number of original sruns with anything left, which keeps the job's node
count no larger than the original's; --sruns overrides it.

Every srun is gen_cg.build_srun over its ranges, with the trajectory written
in: a repack is particular to one configuration, so there is no template. For
the same reason the light write layout is written in as config.CG_SOLVER_SPLIT,
which has to match the script's --batched-solver-split.
Written to <out> (default config.CG_XML_DIR, where the XMLs reference their
schedule files):

    par.CG<f>.repack.NN.<conf>.xml, schedule.CG<f>.repack.NN.<conf>.txt
    layout.CG<f>.repack.<conf>.txt   one line per srun, as gen_cg.py's

The previous repack of the same flavour and configuration is deleted first:
cmd-run-CG-repack.sh runs every repack XML it finds, and a leftover from a
larger earlier repack would solve and write the same vectors as a new one.

Run it on Frontier, after the jobs writing these directories have ended.
"""
import argparse
import sys
from pathlib import Path

# The toolkit (config, srun_layout, vector_scan) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from srun_layout import Call, batch_of, segments, split
from vector_scan import scan
from gen_cg import FLAVORS, build_srun, describe, n_hit_of, srun_hours
from check_cg import report


def calls_left(flavor, scans):
    """The solver calls that put back every bad vector, exact before sloppy
    within each hit. A batched flavour also redoes the batch before each run
    of bad vectors, inside the same range: writes are not fsync'ed, so the
    last batch written before a node failure can be on disk at full size and
    still be bad."""
    batch = batch_of(flavor)
    firsts = {}
    for r in scans:
        bad = r.bad()
        for i in bad:
            starts = firsts.setdefault((r.hit, not r.exact), set())
            starts.add(i - i % batch)
            prev = i - i % batch - batch
            if batch > 1 and (i - 1) not in bad and prev >= r.first:
                starts.add(prev)
    return [Call(not sloppy, hit, first)
            for (hit, sloppy), starts in sorted(firsts.items())
            for first in sorted(starts)]


def clear_previous(out_dir, flavor, traj):
    for pattern in (f"par.CG{flavor}.repack.*.{traj}.xml",
                    f"schedule.CG{flavor}.repack.*.{traj}.txt"):
        for path in out_dir.glob(pattern):
            path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Generate repack sruns for the CG work missing on disk.")
    parser.add_argument("flavor", choices=FLAVORS)
    parser.add_argument("tag", nargs="?",
                        help=f"hN, the target hit count (default h{config.N_HIT})")
    parser.add_argument("--conf", nargs="+", type=int, required=True)
    parser.add_argument("--sruns", type=int,
                        help="srun count (default: original sruns with work left)")
    parser.add_argument("--out", default=config.CG_XML_DIR)
    args = parser.parse_args()
    n_hit = n_hit_of(parser, args.tag)

    flavor = args.flavor
    out_dir = Path(args.out)
    nodes = config.CG_NODES_PER_SRUN[flavor]
    total_sruns = 0
    for traj in args.conf:
        hits = config.hit_set(n_hit, traj)
        scans = scan(flavor, traj, hits)
        lines, complete = report(flavor, traj, scans, hits)
        print("\n".join(lines))
        clear_previous(out_dir, flavor, traj)
        if complete:
            print("nothing to repack\n")
            continue

        calls = calls_left(flavor, scans)
        n = args.sruns or len({r.srun for r in scans if r.bad()})
        sruns = split(flavor, calls, min(n, len(calls)))

        layout = []
        hours = []
        for k, srun_calls in enumerate(sruns):
            run_id = f"CG{flavor}.repack.{k:02d}.{traj}"
            sloppy, exact = segments(flavor, srun_calls)
            job, _ = build_srun(flavor, run_id, sloppy, exact, traj=traj,
                                io_split=config.CG_SOLVER_SPLIT[flavor])
            job.write(out_dir / f"par.{run_id}.xml",
                      out_dir / f"schedule.{run_id}.txt")
            hours.append(srun_hours(flavor, sloppy, exact))
            ranges = " ".join(d for d in (describe(exact, flavor, True),
                                          describe(sloppy, flavor)) if d)
            layout.append(f"{run_id} nodes={nodes} "
                          f"sloppy={sum(g.count for g in sloppy)} "
                          f"exact={sum(g.count for g in exact)} "
                          f"est={hours[-1]:.2f}h | {ranges}")
        (out_dir / f"layout.CG{flavor}.repack.{traj}.txt").write_text(
            "\n".join(layout) + "\n")
        total_sruns += len(sruns)
        print(f"{len(sruns)} sruns x {nodes} nodes = {len(sruns) * nodes} "
              f"nodes, est. {min(hours):.2f}-{max(hours):.2f} h each\n")

    print(f"wrote {total_sruns} repack sruns for '{flavor}' "
          f"({total_sruns * nodes} nodes) to {out_dir}")


if __name__ == "__main__":
    main()
