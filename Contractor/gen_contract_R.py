"""The pipi rectangle diagram R, at rest.

    gen_contract_R.py --timing                     one small par file to time R
    gen_contract_R.py --timing --tag ama --width w4p1_n45 --deltas 8

The contraction. With the labels of gen_contract_C, the sink pions at ts+t (p)
and ts+t+Delta (-p), the source pions at ts-Delta (q) and ts (-q), R joins the
two source pions and the two sink pions directly,

    R_pq(t) = tr[pi_q(ts-Delta) pi_-q(ts) pi_p(ts+t) pi_-p(ts+t+Delta)]

and for FlexibleContractorDense, dt = ts - Delta and s = t + Delta as for C,

    tr[pi_q(dt) pi_-q(dt+Delta) pi_p(tLast) pi_-p(tLast+Delta)]

The two source terms and the two sink terms are each a fixed product, built
once per time in the job and shared by every contraction with the same
factors and offsets: (001, 001) uses one product as both its source and its
sink. The other time and momentum orderings R needs are left for production.

Names follow C, R_<source stem>_d<Delta>_<q>_0_<sink stem>_<p>_noavg, in
contract_R[_<tag>].

--timing writes par.R_timing<suffix>.xml: one width, the first --deltas value,
and (001, 001), (001, 010) on 4 fields with 2 distinct products. The log gives
"Loaded N field(s) in X s", "Fixed products: N in X s", and "gathers X s,
products Y s, traces Z s" per contraction. Each product is 61 GB at 8 hits,
like a field; the fields are freed once the products are built, so the peak
is 4 fields plus 2 products, ~370 GB, inside a quarter node.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from contractor_xml import FlexibleJob
from gen_contract_2pt import PRESETS, TRAJ, field_file, ioname, mom_tag, negate, smearings
from gen_contract_C import DELTAS, MAX_T

TIMING_PAIRS = [([0, 0, 1], [0, 0, 1]), ([0, 0, 1], [0, 1, 0])]


def build(pairs, preset, smear, deltas, max_t, trajs, output):
    """One par file: R for every (source q, sink p) in pairs, every Delta and
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
                    name=f"R_{stem}_d{delta}_{mom_tag(q)}_0_{stem}_{mom_tag(p)}_noavg",
                    terms=[(src, "source", 0), (src_m, "source", delta),
                           (snk, "sink", 0), (snk_m, "sink", delta)],
                    translations=f"0..{config.N_T - 1}",
                    min_t=delta, max_t=delta + max_t, translation_average=False)
    return job


def main():
    widths = [s[1:] or "unsmeared" for s in smearings()]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--timing", action="store_true", required=True,
                        help="one par file with two contractions at the first Delta")
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
    output = f"{config.CONTRACTION_ROOT}/contract_R{suffix}"
    trajs = (args.traj_start,
             args.traj_end if args.traj_end is not None else args.traj_start + 1,
             args.traj_step)
    smear = "" if args.width == "unsmeared" else f"_{args.width}"

    job = build(TIMING_PAIRS, preset, smear, args.deltas[:1], args.max_t, trajs, output)
    path = job.write(Path(config.OUTPUT_ROOT) / "contractor" / f"par.R_timing{suffix}.xml")
    print(f"wrote {path}: {len(job.contractions)} contractions, {len(job.matrices)} "
          f"fields ({args.width}, {preset['tag']}), Delta {args.deltas[0]}, "
          f"t = 0..{args.max_t}, trajectories {trajs[0]}..{trajs[1] - 1}")
    print(f"  correlators   {output}/")


if __name__ == "__main__":
    main()
