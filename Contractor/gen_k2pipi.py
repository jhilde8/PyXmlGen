"""K -> pipi contractions, by K2pipiContractorDense. Timing only for now.

    gen_k2pipi.py --timing                       type 4, s loop, EMF type 0
    gen_k2pipi.py --timing --loop l --emf-type 2 --width w2p03_n11
    gen_k2pipi.py --timing --diagram 3                  tsep 24
    gen_k2pipi.py --timing --diagram 3 4 --tseps 20 24  one shared sweep

Type 3. The two pions sit on the kaon's quark line, the operator's loop is in
the EMF,

    T3(t_K, t_Q, t_pi) = tr[M(t_pi) WW_K(t_K) VV_Q(t_Q)],

M = pi_p pi_-p the rest-frame two-pion product of gen_pixpi, dR or dL. M's
expanded col meets the kaon's light W at t_K, the kaon's strange W the EMF's
rows, the EMF's light V cols M's dense rows. The fixed chain is
[pi_p, pi_-p, kaon]: the kaon at dt = t_K, the pions at dt + tsep and
dt + tsep + Delta, M built in the job and every edge of the chain gathered,
so it costs ~nothing. The EMF is swept at tau = 0..tsep, between the kaon and
the nearer pion. Type 3 and type 4 in one par file share the EMF reads.

Type 4. The kaon and the operator close on their own,

    T4(t_K, t_Q) = tr[WW_K(t_K) VV_Q(t_Q)],

times the pipi vacuum bubble, which comes from the single-pion two-point
functions. WW_K is the at-rest kaon W_l x W_s field (mf_ls_ww: rows the light
dense W, cols the strange dense W), VV_Q an EMF (rows the strange expanded V,
cols the light expanded V). Both EMF legs meet the kaon's dense legs, so both
are gathered at t_K; the trace is O(N^2) and the cost is reading the EMFs.

In the contractor's terms the kaon is the one fixed term at dt = t_K, and
each EMF a swept field at dt + tau = t_Q, tau = 0..MAX_T, dt = 0..nt-1, one
dataset per kaon time (_noavg). Every EMF slice is read once for all the
contractions that sweep it, whatever the tau window.

The EMFs. One file per (type, gamma family, timeslice),

    <MF base>/emf_<f>loop_<tag><acc>.<traj>/type<T>_<g>_<g>.t%04d.h5

each one 12288 x 14288 ComplexD slice, 2.8 GB at 8 hits; one type is the five
families of config.EMF_GAMMA_FAMILIES, gamma1 and gamma2 paired family by
family. The loop flavour and type do not change the size.

--timing writes par.k2pipi_t<diagrams>_timing<suffix>.xml (t4, t3, t34): one
loop flavour, one EMF type, all its families and the subtraction operator
(mix, the first of config.MIX_GAMMAS) swept against the kaon at --width
(unsmeared by default), 6 x 128 slices, ~2.2 TB read at 8 hits. Type 3 adds
p001 x p00m1 at the same width, dR and dL at the first Delta of
gen_contract_C, for every --tseps (default 24). The kaon is ~0.4 GB; type 3 loads two pion
fields, 122 GB at 8 hits, freed once the fixed chains are built. The log gives
"Fixed chains in X s" and "'<emf>': read X s, traces Y s" per swept field.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from contractor_xml import K2pipiJob
from gen_contract_2pt import PRESETS, TRAJ, field_file, ioname, mom_tag, negate, smearings
from gen_contract_C import DELTAS
from gen_pixpi import TIMING_MOM, product_tag

REST = [0, 0, 0]
MIX_GAMMA = config.MIX_GAMMAS.split()[0]
TSEPS = [24]
# Last K-Q separation; the signal is gone well before 31.
MAX_T = 31


def emf_families():
    return config.EMF_GAMMA_FAMILIES.split()


def emf_ioname(emf_type, family):
    """A2AExtendedMesonField's ioname: gamma1 and gamma2 lists paired by index."""
    return f"type{emf_type}_{family}_{family}"


def build(diagrams, preset, smear, loop, emf_type, tseps, delta, output, trajs):
    """One par file: every family of one EMF type and loop flavour swept, for
    every accuracy of the preset, kaon and pions at one smearing. Type 4 is
    tr[WW_K(dt) VV(dt + tau)]; type 3 tr[M(dt + tsep) WW_K(dt) VV(dt + tau)]
    for M at TIMING_MOM, dR and dL at delta, every tsep in tseps."""
    traj_start, traj_end, traj_step = trajs
    job = K2pipiJob(config.N_T, output, traj_start, traj_end, traj_step)

    # The kaon WW field is W x W, so it carries no accuracy.
    kaon = f"ls_ww{smear}_{mom_tag(REST)}"
    row, col = config.field_axes("ls_ww")
    job.add_matrix(name=kaon,
                   file=field_file("ls_ww", config.IDENTITY, REST, preset, "", smear),
                   dataset=ioname(config.IDENTITY, REST), row=row, col=col)

    row, col = config.field_axes("emf")
    for acc in preset["accuracies"]:
        swept = []
        for family in emf_families():
            io = emf_ioname(emf_type, family)
            name = f"emf_{loop}loop{acc}_{io}"
            job.add_matrix(name=name,
                           file=f"{preset['base']}/emf_{loop}loop_{preset['tag']}{acc}.@traj@/{io}.h5",
                           dataset=io, row=row, col=col)
            swept.append(name)
        # The subtraction operator: one gamma, the same size as an EMF.
        io = ioname(MIX_GAMMA, REST)
        name = f"mix{acc}_{io}"
        mix_row, mix_col = config.field_axes("mix")
        job.add_matrix(name=name,
                       file=f"{preset['base']}/mix_{preset['tag']}{acc}.@traj@/{io}.h5",
                       dataset=io, row=mix_row, col=mix_col)
        swept.append(name)
        emf = f"emf_{loop}loop{acc}_type{emf_type}"
        if 4 in diagrams:
            job.add_contraction(name=f"T4_{kaon}_{emf}_noavg",
                                terms=[(kaon, 0)], swept=swept,
                                translations=f"0..{config.N_T - 1}", min_t=0,
                                max_t=MAX_T, translation_average=False)
        if 3 not in diagrams:
            continue
        stem = f"pi{acc}{smear}"
        pi_row, pi_col = config.field_axes("pi")
        for mom in TIMING_MOM:
            left, right = f"{stem}_{mom_tag(mom)}", f"{stem}_{mom_tag(negate(mom))}"
            for name, m in ((left, mom), (right, negate(mom))):
                if name not in job.matrices:
                    job.add_matrix(name=name,
                                   file=field_file("pi", config.GAMMA5, m, preset, acc, smear),
                                   dataset=ioname(config.GAMMA5, m), row=pi_row, col=pi_col)
            for tsep in tseps:
                for order, dl, dr in (("R", 0, delta), ("L", delta, 0)):
                    tag = product_tag(mom, order, delta)
                    job.add_contraction(
                        name=f"T3_{stem}_{tag}_t{tsep}_{kaon}_{emf}_noavg",
                        terms=[(left, tsep + dl), (right, tsep + dr), (kaon, 0)],
                        swept=swept, translations=f"0..{config.N_T - 1}",
                        min_t=0, max_t=tsep, translation_average=False)
    return job


def main():
    widths = [s[1:] or "unsmeared" for s in smearings()]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--timing", action="store_true", required=True,
                        help="one par file: one loop flavour and EMF type")
    parser.add_argument("--diagram", type=int, nargs="+", choices=(3, 4), default=[4],
                        help="K -> pipi types in the par file, sharing one sweep")
    parser.add_argument("--tseps", type=int, nargs="+", default=TSEPS,
                        help="kaon to nearer-pion separations, type 3 only")
    parser.add_argument("--tag", default=PRESETS["production"]["tag"],
                        help="hit tag of the fields: hN for N sloppy hits, or "
                             "ama for the AMA hit, sloppy and exact")
    parser.add_argument("--width", choices=widths, default="unsmeared",
                        help="smearing of the kaon; the EMFs are unsmeared")
    parser.add_argument("--loop", choices=("s", "l", "c"), default="s")
    parser.add_argument("--emf-type", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--traj-start", type=int, default=TRAJ)
    parser.add_argument("--traj-end", type=int,
                        help="exclusive, as trajCounter (default: start + 1)")
    parser.add_argument("--traj-step", type=int, default=20)
    args = parser.parse_args()

    diagrams = sorted(set(args.diagram))
    if 3 in diagrams and max(args.tseps) + DELTAS[0] >= config.N_T:
        parser.error("tsep + Delta must stay below nt")

    # The tag picks everything, as in gen_contract_2pt.
    ama = args.tag == PRESETS["ama"]["tag"]
    preset = dict(PRESETS["ama" if ama else "production"], tag=args.tag)
    suffix = "" if args.tag == PRESETS["production"]["tag"] else f"_{args.tag}"
    output = f"{config.CONTRACTION_ROOT}/contract_k2pipi{suffix}"
    trajs = (args.traj_start,
             args.traj_end if args.traj_end is not None else args.traj_start + 1,
             args.traj_step)
    smear = "" if args.width == "unsmeared" else f"_{args.width}"

    job = build(diagrams, preset, smear, args.loop, args.emf_type, args.tseps,
                DELTAS[0], output, trajs)
    tag = "t" + "".join(map(str, diagrams))
    path = job.write(Path(config.OUTPUT_ROOT) / "contractor" / f"par.k2pipi_{tag}_timing{suffix}.xml")
    n_swept = len({e for _, s, *_ in job.contractions.values() for e in s})
    print(f"wrote {path}: {len(job.contractions)} contraction(s), {n_swept} swept field(s) "
          f"({args.loop} loop, type {args.emf_type}, {preset['tag']}), types "
          f"{' '.join(map(str, diagrams))}, kaon and pions {args.width}, "
          f"trajectories {trajs[0]}..{trajs[1] - 1}")
    print(f"  correlators   {output}/")


if __name__ == "__main__":
    main()
