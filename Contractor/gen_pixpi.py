"""Two-pion products M for the rest-frame pipi diagrams, built by A2APixPi.

    gen_pixpi.py                          every smearing state, one par file each
    gen_pixpi.py --width w2p03_n11        one state
    gen_pixpi.py --tag ama                the AMA hit, both accuracies
    gen_pixpi.py --timing                 four products on two fields

The products. A two-pion operator at rest pairs pi_p and pi_-p at times Delta
apart; R and the type 1, 2 and 3 K -> pipi diagrams multiply its two fields
into one fixed-time product, source with source and sink with sink. Both time
orderings are needed, the earlier pion on the left or on the right,

    dR:  M[t] = pi_p(t)       pi_-p(t + Delta)      A2APixPi deltaR = Delta
    dL:  M[t] = pi_p(t + Delta) pi_-p(t)            A2APixPi deltaL = Delta

for all 27 momenta p (p running over the shell covers pi_-p on the left) and
every Delta of gen_contract_C: 27 x 2 x 2 = 108 products per smearing state.
Off-diagonal and moving-frame pairs are not built.

Each product is A2APixPi's full product: rows the left pion's dense W, cols
the right pion's expanded V, so every external time's hit block is there for
whatever closes it later. One is 128 slices of 2096 x 14288 ComplexD, 61 GB
at 8 hits, ~6.6 TB per smearing state; one slice is one 2096^2 x 14288 GEMM.

Layout. Next to the meson fields of the same preset,

    <MF base>/M_pi_<tag><acc><smear>.<traj>/<p>_d{R,L}<Delta>.t%04d.h5

dataset <p>_d{R,L}<Delta>, so a later contractor declares the product by its
whole-field path as it does a base field. Lustre striping of that directory
is the caller's (the -c 16 of the meson fields; never wide from Riker).

Par files. A production file holds all 27 fields of one smearing state, 1.65
TB at 8 hits, so one whole Riker node; the AMA set is 1-hit and holds both
accuracies. --timing holds p001 and p00m1 at --width and their four products
(dR and dL at each Delta, p001 on the left). A2APixPi logs "Loaded N
field(s) in X s" and "gather+GEMM X s, write Y s" per product: the
per-product compute and write rate any production layout is built from.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from contractor_xml import PixPiJob
from gen_contract_2pt import PRESETS, TRAJ, field_file, ioname, mom_tag, negate, smearings
from gen_contract_C import DELTAS

TIMING_MOM = [[0, 0, 1]]


def product_tag(mom, order, delta):
    """p001_dR6: pi_p on the left, the later pion on the side `order` names."""
    return f"{mom_tag(mom)}_d{order}{delta}"


def build(moms, preset, smear, deltas, trajs):
    """One par file: dR and dL at every Delta for pi_p x pi_-p, every p in
    moms, every accuracy of the preset, at one smearing state."""
    traj_start, traj_end, traj_step = trajs
    job = PixPiJob(config.N_T, traj_start, traj_end, traj_step)
    for acc in preset["accuracies"]:
        stem = f"pi{acc}{smear}"
        out_dir = f"{preset['base']}/M_pi_{preset['tag']}{acc}{smear}.@traj@"

        def matrix(mom):
            name = f"{stem}_{mom_tag(mom)}"
            if name not in job.matrices:
                row, col = config.field_axes("pi")
                job.add_matrix(name=name,
                               file=field_file("pi", config.GAMMA5, mom, preset, acc, smear),
                               dataset=ioname(config.GAMMA5, mom), row=row, col=col)
            return name

        for mom in moms:
            left, right = matrix(mom), matrix(negate(mom))
            for delta in deltas:
                for order, dl, dr in (("R", 0, delta), ("L", delta, 0)):
                    tag = product_tag(mom, order, delta)
                    job.add_product(left, right, dl, dr, f"{out_dir}/{tag}.h5", tag)
    return job


def main():
    widths = [s[1:] or "unsmeared" for s in smearings()]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--timing", action="store_true",
                        help="one par file, p001 x p00m1 at --width, four products")
    parser.add_argument("--tag", default=PRESETS["production"]["tag"],
                        help="hit tag of the fields: hN for N sloppy hits, or "
                             "ama for the AMA hit, sloppy and exact")
    parser.add_argument("--width", choices=widths,
                        help="one smearing state (default: all; w2p03_n11 for --timing)")
    parser.add_argument("--deltas", type=int, nargs="+", default=DELTAS)
    parser.add_argument("--traj-start", type=int, default=TRAJ)
    parser.add_argument("--traj-end", type=int,
                        help="exclusive, as trajCounter (default: start + 1)")
    parser.add_argument("--traj-step", type=int, default=20)
    args = parser.parse_args()

    # The tag picks everything, as in gen_contract_2pt.
    ama = args.tag == PRESETS["ama"]["tag"]
    preset = dict(PRESETS["ama" if ama else "production"], tag=args.tag)
    suffix = "" if args.tag == PRESETS["production"]["tag"] else f"_{args.tag}"
    trajs = (args.traj_start,
             args.traj_end if args.traj_end is not None else args.traj_start + 1,
             args.traj_step)
    par_dir = Path(config.OUTPUT_ROOT) / "contractor"

    if args.timing:
        jobs = [("pixpi_timing", TIMING_MOM, args.width or "w2p03_n11")]
    else:
        jobs = [(f"pixpi_{w}", config.PION_MOM, w) for w in ([args.width] if args.width else widths)]

    for tag, moms, width in jobs:
        smear = "" if width == "unsmeared" else f"_{width}"
        job = build(moms, preset, smear, args.deltas, trajs)
        path = job.write(par_dir / f"par.{tag}{suffix}.xml")
        print(f"wrote {path}: {len(job.products)} products, {len(job.matrices)} fields "
              f"({width}, {preset['tag']}), Delta {' '.join(map(str, args.deltas))}, "
              f"trajectories {trajs[0]}..{trajs[1] - 1}")
    print(f"  products      {preset['base']}/M_pi_{preset['tag']}*.@traj@/")


if __name__ == "__main__":
    main()
