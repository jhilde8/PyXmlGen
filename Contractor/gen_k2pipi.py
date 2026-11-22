"""K -> pipi contractions, by K2pipiContractorDense. Type 4 timing only for now.

    gen_k2pipi.py --timing                       type 4, s loop, EMF type 0
    gen_k2pipi.py --timing --loop l --emf-type 2 --width w2p03_n11

Type 4. The kaon and the operator close on their own,

    T4(t_K, t_Q) = tr[WW_K(t_K) VV_Q(t_Q)],

times the pipi vacuum bubble, which comes from the single-pion two-point
functions. WW_K is the at-rest kaon W_l x W_s field (mf_ls_ww: rows the light
dense W, cols the strange dense W), VV_Q an EMF (rows the strange expanded V,
cols the light expanded V). Both EMF legs meet the kaon's dense legs, so both
are gathered at t_K; the trace is O(N^2) and the cost is reading the EMFs.

In the contractor's terms the kaon is the one fixed term at dt = t_K, and
each EMF a swept field at dt + tau = t_Q, tau = 0..nt-1, dt = 0..nt-1, one
dataset per kaon time (_noavg). Every EMF slice is read once for all the
contractions that sweep it.

The EMFs. One file per (type, gamma family, timeslice),

    <MF base>/emf_<f>loop_<tag><acc>.<traj>/type<T>_<g>_<g>.t%04d.h5

each one 12288 x 14288 ComplexD slice, 2.8 GB at 8 hits; one type is the five
families of config.EMF_GAMMA_FAMILIES, gamma1 and gamma2 paired family by
family. The loop flavour and type do not change the size.

--timing writes par.k2pipi_t4_timing<suffix>.xml: one loop flavour, one
type, all its families swept against the kaon at --width (unsmeared by
default), 5 x 128 slices, ~1.8 TB read at 8 hits. The kaon is ~0.4 GB, so the
job's memory is one EMF slice plus its gather buffers. The log gives
"'<emf>': read X s, traces Y s" per swept field: read rate against trace cost.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from contractor_xml import K2pipiJob
from gen_contract_2pt import PRESETS, TRAJ, field_file, ioname, mom_tag, smearings

REST = [0, 0, 0]


def emf_families():
    return config.EMF_GAMMA_FAMILIES.split()


def emf_ioname(emf_type, family):
    """A2AExtendedMesonField's ioname: gamma1 and gamma2 lists paired by index."""
    return f"type{emf_type}_{family}_{family}"


def build_type4(preset, smear, loop, emf_type, output, trajs):
    """One par file: tr[WW_K(dt) VV(dt + tau)] for every family of one EMF type
    and loop flavour, every accuracy of the preset, kaon at one smearing."""
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
        job.add_contraction(name=f"T4_{kaon}_emf_{loop}loop{acc}_type{emf_type}_noavg",
                            terms=[(kaon, 0)], swept=swept,
                            translations=f"0..{config.N_T - 1}", min_t=0,
                            max_t=config.N_T - 1, translation_average=False)
    return job


def main():
    widths = [s[1:] or "unsmeared" for s in smearings()]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--timing", action="store_true", required=True,
                        help="one par file: type 4, one loop flavour and EMF type")
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

    # The tag picks everything, as in gen_contract_2pt.
    ama = args.tag == PRESETS["ama"]["tag"]
    preset = dict(PRESETS["ama" if ama else "production"], tag=args.tag)
    suffix = "" if args.tag == PRESETS["production"]["tag"] else f"_{args.tag}"
    output = f"{config.CONTRACTION_ROOT}/contract_k2pipi{suffix}"
    trajs = (args.traj_start,
             args.traj_end if args.traj_end is not None else args.traj_start + 1,
             args.traj_step)
    smear = "" if args.width == "unsmeared" else f"_{args.width}"

    job = build_type4(preset, smear, args.loop, args.emf_type, output, trajs)
    path = job.write(Path(config.OUTPUT_ROOT) / "contractor" / f"par.k2pipi_t4_timing{suffix}.xml")
    n_swept = sum(len(s) for _, s, *_ in job.contractions.values())
    print(f"wrote {path}: {len(job.contractions)} contraction(s), {n_swept} swept EMF(s) "
          f"({args.loop} loop, type {args.emf_type}, {preset['tag']}), kaon {args.width}, "
          f"trajectories {trajs[0]}..{trajs[1] - 1}")
    print(f"  correlators   {output}/")


if __name__ == "__main__":
    main()
