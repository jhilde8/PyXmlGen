"""The pipi cross diagram C, (A10) of pions_A2A, at rest.

    gen_contract_C.py --timing                     one small par file to time C
    gen_contract_C.py --timing --tag ama --width w4p1_n45 --deltas 8

The contraction. With the D diagram's labels, the sink pions at ts+t (p) and
ts+t+Delta (-p), the source pions at ts-Delta (q) and ts (-q),

    C_pq(t) = tr[pi_p(ts+t) pi_q(ts-Delta) pi_-p(ts+t+Delta) pi_-q(ts)]

so X1 pairs with X3 and X2 with X4, as in the first term of D, and the I=2
combination 2D - 2C matches times and Delta. The trace starts at a source
for FlexibleContractorDense, dt = ts - Delta and s = t + Delta:

    tr[pi_q(dt) pi_-p(tLast+Delta) pi_-q(dt+Delta) pi_p(tLast)]

Roles alternate, so no product is needed. The sink sweeps s = Delta ..
Delta + max_t (t = 0..max_t) from every source time, one dataset per source
(_noavg); the translation average and the shell sums are analysis.

There is no exact (p,q) <-> (q,p) relation: reversing the trace needs
Pi^dagger = Pi(-p), which stochastic high modes satisfy only on average, so
all 27 x 27 ordered pairs are needed. How they are tiled into par files is
decided by the timing run, so only --timing exists for now.

Names. Source first, like the two-point functions,

    C_pi_w2p03_n11_d6_p001_0_pi_w2p03_n11_p010_noavg

for q = p001 at the source and p = p010 at the sink: C_<source stem>_d<Delta>
_<q>_0_<sink stem>_<p>, the stems those of gen_contract_2pt (pi, accuracy,
smearing). One file per contraction, <output>/<name>.<traj>.h5, in
contract_C[_<tag>].

--timing writes par.C_timing<suffix>.xml: one width, the first --deltas value,
and (000,000), (001,001), (001,010) on 5 fields. The log gives the Eigen
thread count, "Loaded N field(s) in X s", and "gathers X s, products Y s,
traces Z s" per contraction: the per-field load and per-contraction compute
that any tiling is built from.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from contractor_xml import FlexibleJob
from gen_contract_2pt import PRESETS, TRAJ, field_file, ioname, mom_tag, negate, smearings

# 6 is the 48I choice in lattice units, 8 the same physical separation on 64I.
DELTAS = [6, 8]
MAX_T = 48
TIMING_PAIRS = [([0, 0, 0], [0, 0, 0]), ([0, 0, 1], [0, 0, 1]), ([0, 0, 1], [0, 1, 0])]


def build(pairs, preset, smear, deltas, max_t, trajs, output):
    """One par file: C for every (source q, sink p) in pairs, every Delta and
    every accuracy of the preset, at one smearing state."""
    traj_start, traj_end, traj_step = trajs
    job = FlexibleJob(nt=config.N_T, output=output,
                      traj_start=traj_start, traj_end=traj_end, traj_step=traj_step)
    for acc in preset["accuracies"]:
        stem = f"pi{acc}{smear}"

        def matrix(mom):
            name = f"{stem}_{mom_tag(mom)}"
            if name not in job.matrices:
                row, col = config.field_axes("pi")
                job.add_matrix(name=name,
                               file=field_file("pi", config.GAMMA5, mom, preset, acc, smear),
                               dataset=ioname(config.GAMMA5, mom), row=row, col=col)
            return name

        for q, p in pairs:
            src, src_m = matrix(q), matrix(negate(q))
            snk, snk_m = matrix(p), matrix(negate(p))
            for delta in deltas:
                job.add_contraction(
                    name=f"C_{stem}_d{delta}_{mom_tag(q)}_0_{stem}_{mom_tag(p)}_noavg",
                    terms=[(src, "source", 0), (snk_m, "sink", delta),
                           (src_m, "source", delta), (snk, "sink", 0)],
                    translations=f"0..{config.N_T - 1}",
                    min_t=delta, max_t=delta + max_t, translation_average=False)
    return job


def main():
    widths = [s[1:] or "unsmeared" for s in smearings()]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--timing", action="store_true", required=True,
                        help="one par file with three contractions at the first Delta")
    parser.add_argument("--tag", default=PRESETS["production"]["tag"],
                        help="hit tag of the fields: hN for N sloppy hits, or "
                             "ama for the AMA hit, sloppy and exact")
    parser.add_argument("--width", choices=widths, default="w2p03_n11")
    parser.add_argument("--deltas", type=int, nargs="+", default=DELTAS)
    parser.add_argument("--max-t", type=int, default=MAX_T,
                        help="last sink-source separation t = s - Delta")
    parser.add_argument("--traj-start", type=int, default=TRAJ)
    parser.add_argument("--traj-end", type=int,
                        help="exclusive, as trajCounter (default: start + 1)")
    parser.add_argument("--traj-step", type=int, default=20)
    args = parser.parse_args()

    if max(args.deltas) + args.max_t >= config.N_T:
        parser.error("Delta + max_t must stay below nt")

    # The tag picks everything, as in gen_contract_2pt.
    ama = args.tag == PRESETS["ama"]["tag"]
    preset = dict(PRESETS["ama" if ama else "production"], tag=args.tag)
    suffix = "" if args.tag == PRESETS["production"]["tag"] else f"_{args.tag}"
    output = f"{config.CONTRACTION_ROOT}/contract_C{suffix}"
    trajs = (args.traj_start,
             args.traj_end if args.traj_end is not None else args.traj_start + 1,
             args.traj_step)
    smear = "" if args.width == "unsmeared" else f"_{args.width}"

    job = build(TIMING_PAIRS, preset, smear, args.deltas[:1], args.max_t, trajs, output)
    path = job.write(Path(config.OUTPUT_ROOT) / "contractor" / f"par.C_timing{suffix}.xml")
    print(f"wrote {path}: {len(job.contractions)} contractions, {len(job.matrices)} "
          f"fields ({args.width}, {preset['tag']}), Delta {args.deltas[0]}, "
          f"t = 0..{args.max_t}, trajectories {trajs[0]}..{trajs[1] - 1}")
    print(f"  correlators   {output}/")


if __name__ == "__main__":
    main()
