"""Rebuild the unfinished part of a CG job as new, balanced sruns.

    gen_cg_repack.py s --conf 1540 1550
    gen_cg_repack.py l --conf 1540 --sruns 2

For each configuration, vector_scan.scan checks the original layout on disk
(the same scan check_cg.py reports), and every missing or wrong-size vector is
work again. A batched flavour (light, config.CG_BATCH_SIZE) redoes whole
batches: every original range starts on a multiple of the batch size, so the
batch holding index i starts at i - i % batch, and vectors of that batch
already on disk are solved again with the rest.

The work is cut into sruns of equal walltime, each solver call weighted by
config.CG_COST, sloppy and exact free to share an srun. Calls are ordered by
hit, exact before sloppy, and every srun takes a contiguous run of them, so a
hit's noise is loaded by as few sruns as possible. The srun count defaults to
the number of original sruns with anything left, which keeps the job's node
count no larger than the original's; --sruns overrides it.

Every srun is gen_cg.build_srun over its ranges, with the trajectory written
in: a repack is particular to one configuration, so there is no template.
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
from dataclasses import dataclass
from pathlib import Path

# The toolkit (config, srun_layout, vector_scan) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from srun_layout import Segment
from vector_scan import scan
from gen_cg import FLAVORS, build_srun, describe
from check_cg import report


@dataclass(frozen=True)
class Call:
    """One solver call: vectors [first, first + batch) of one hit."""
    exact: bool
    hit: int
    first: int


def batch_of(flavor):
    return max(config.CG_BATCH_SIZE[flavor], 1)


def cost(flavor, call):
    return config.CG_COST[(flavor, "exact" if call.exact else "sloppy")]


def calls_left(flavor, scans):
    """The solver calls that put back every bad vector, exact before sloppy
    within each hit."""
    batch = batch_of(flavor)
    firsts = {}
    for r in scans:
        for i in r.bad():
            firsts.setdefault((r.hit, not r.exact), set()).add(i - i % batch)
    return [Call(not sloppy, hit, first)
            for (hit, sloppy), starts in sorted(firsts.items())
            for first in sorted(starts)]


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


def clear_previous(out_dir, flavor, traj):
    for pattern in (f"par.CG{flavor}.repack.*.{traj}.xml",
                    f"schedule.CG{flavor}.repack.*.{traj}.txt"):
        for path in out_dir.glob(pattern):
            path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Generate repack sruns for the CG work missing on disk.")
    parser.add_argument("flavor", choices=FLAVORS)
    parser.add_argument("--conf", nargs="+", type=int, required=True)
    parser.add_argument("--sruns", type=int,
                        help="srun count (default: original sruns with work left)")
    parser.add_argument("--out", default=config.CG_XML_DIR)
    args = parser.parse_args()

    flavor = args.flavor
    out_dir = Path(args.out)
    nodes = config.CG_NODES_PER_SRUN[flavor]
    total_sruns = 0
    for traj in args.conf:
        scans = scan(flavor, traj)
        lines, complete = report(flavor, traj, scans)
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
            job, _ = build_srun(flavor, run_id, sloppy, exact, traj=traj)
            job.write(out_dir / f"par.{run_id}.xml",
                      out_dir / f"schedule.{run_id}.txt")
            seconds = (config.CG_STARTUP[flavor]
                       + sum(cost(flavor, c) for c in srun_calls))
            hours.append(seconds / 3600)
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
