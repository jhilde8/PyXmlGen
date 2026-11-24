"""K -> pipi types 3 and 4, by K2pipiContractorDense: one invocation per hit
tag, one par file per (loop flavour, EMF type), each a quarter-node task.

    gen_k2pipi.py                    h8: 8 par files, s and l loops x types 0..3
    gen_k2pipi.py h4                 the 4-hit fields, sliced or computed
    gen_k2pipi.py h16 --all-gammas   every EMF gamma family

Type 3. The two pions sit on the kaon's quark line, the operator's loop is in
the EMF,

    T3 = tr[pi_a(dt + o_a) pi_b(dt + o_b) WW_K(dt) VV_Q(dt + tau)],

pi_a closing against the operator, pi_b against the kaon. The pions sit at
tsep and tsep + Delta from the kaon: dR puts the later one on the right
(pi_b), dL on the left (pi_a), so dR + dL is the x1 <-> x2 sum of the paper's
expressions. Every p of config.PION_MOM is pi_a once, so each back-to-back
pair gives four contractions per tsep (p or -p on the left, dR or dL). The
EMF is swept at tau = 0..tsep.

Type 4. The kaon and the operator close on their own,

    T4 = tr[WW_K(dt) VV_Q(dt + tau)],  tau = 0..MAX_T,

times the pipi vacuum bubble, which comes from the single-pion two-point
functions.

WW_K is the at-rest kaon W_l x W_s field (mf_ls_ww), VV_Q an EMF (rows the
strange expanded V, cols the light expanded V), the pions the gen_contract_2pt
pi fields. One dataset per kaon time (_noavg).

The tiling. An EMF type is one trace and colour topology of the 2023 paper
(App. B): type 1 1Tr cdiag (B9, B10, B15, B16), type 3 1Tr cmix (B12, B13,
B18, B19), type 0 2Tr cdiag (B11, B17), type 2 2Tr cmix (B14, B20). The q-loop
2Tr expressions are needed for both loops, so the s and l loops times the
four types are the 8 tasks. A task sweeps its type's GammaMU and
GammaMUGamma5 files (--all-gammas: every config.EMF_GAMMA_FAMILIES family).
Counting the gamma5 of the strange line's hermiticity, these are the paper's
gamma_mu gamma5 x gamma_mu and gamma_mu x gamma_mu gamma5 terms; LL and LR
are their sum and difference, in analysis. Types 3 and 4 share a task, since
each swept slice is read once for every contraction listing it.

The subtraction operator. The mix field with Identity, config.MIX_GAMMAS'
first gamma, is sbar gamma5 d in the same counting (B79-B86). Its legs are an
EMF's, so it is one more swept field of the first task's contractions,
sharing their fixed chains; its datasets are <contraction>_mix_Identity_0_0_0.

Memory. The contractor loads each fixed field at its first use and frees it
after its last, and the contractions here are grouped by pion pair {p, -p},
so at most the kaon and two pion fields are resident, plus the fixed chains
(dense light W x dense strange W per dt).

Files go to <OUTPUT_ROOT>/contractor/k2pipi/par.k2pipi.<tag>[.<width>].<f>loop_type<T>.xml,
no width when unsmeared, with TRAJ_START / TRAJ_END left for the submission
script to substitute, as for the CG. Correlators to contract_k2pipi[_<tag>], one file per contraction.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from contractor_xml import K2pipiJob
from gen_contract_2pt import PRESETS, field_file, ioname, mom_tag, negate, smearings
from gen_pixpi import product_tag

REST = [0, 0, 0]
LOOPS = ("s", "l")
EMF_TYPES = (0, 1, 2, 3)
SM_GAMMAS = ["GammaMU", "GammaMUGamma5"]
MIX_GAMMA = config.MIX_GAMMAS.split()[0]
TSEPS = [16, 24, 32]
DELTA = 8
# Last K-Q separation; the signal is gone well before 31.
MAX_T = 31


def emf_ioname(emf_type, family):
    """A2AExtendedMesonField's ioname: gamma1 and gamma2 lists paired by index."""
    return f"type{emf_type}_{family}_{family}"


def pion_pairs():
    """config.PION_MOM as back-to-back pairs, each with its left-pion
    momenta: [p, -p], or [p] when p = -p."""
    pairs, seen = [], set()
    for p in config.PION_MOM:
        if tuple(p) in seen:
            continue
        seen.update({tuple(p), tuple(negate(p))})
        pairs.append([p] if p == negate(p) else [p, negate(p)])
    return pairs


def build(preset, smear, loop, emf_type, gammas, mix, tseps, delta, output):
    """One task: types 3 and 4 against the `gammas` families of one EMF type
    and loop flavour, plus the subtraction operator with `mix`, for every
    accuracy of the preset, kaon and pions at one smearing."""
    job = K2pipiJob(config.N_T, output, "TRAJ_START", "TRAJ_END")
    translations = f"0..{config.N_T - 1}"

    # The kaon WW field is W x W, so it carries no accuracy.
    kaon = f"ls_ww{smear}_{mom_tag(REST)}"
    row, col = config.field_axes("ls_ww")
    job.add_matrix(name=kaon,
                   file=field_file("ls_ww", config.IDENTITY, REST, preset, "", smear),
                   dataset=ioname(config.IDENTITY, REST), row=row, col=col)

    for acc in preset["accuracies"]:
        swept = []
        row, col = config.field_axes("emf")
        for family in gammas:
            io = emf_ioname(emf_type, family)
            name = f"emf_{loop}loop{acc}_{io}"
            job.add_matrix(name=name,
                           file=f"{preset['base']}/emf_{loop}loop_{preset['tag']}{acc}.@traj@/{io}.h5",
                           dataset=io, row=row, col=col)
            swept.append(name)
        if mix:
            io = ioname(MIX_GAMMA, REST)
            name = f"mix{acc}_{io}"
            row, col = config.field_axes("mix")
            job.add_matrix(name=name,
                           file=f"{preset['base']}/mix_{preset['tag']}{acc}.@traj@/{io}.h5",
                           dataset=io, row=row, col=col)
            swept.append(name)
        op = f"emf_{loop}loop{acc}_type{emf_type}"

        job.add_contraction(name=f"T4_{kaon}_{op}_noavg", terms=[(kaon, 0)],
                            swept=swept, translations=translations, min_t=0,
                            max_t=MAX_T, translation_average=False)

        stem = f"pi{acc}{smear}"
        row, col = config.field_axes("pi")
        for lefts in pion_pairs():
            for p in lefts:
                job.add_matrix(name=f"{stem}_{mom_tag(p)}",
                               file=field_file("pi", config.GAMMA5, p, preset, acc, smear),
                               dataset=ioname(config.GAMMA5, p), row=row, col=col)
            for p in lefts:
                left, right = f"{stem}_{mom_tag(p)}", f"{stem}_{mom_tag(negate(p))}"
                for order, dl, dr in (("R", 0, delta), ("L", delta, 0)):
                    tag = product_tag(p, order, delta)
                    for tsep in tseps:
                        job.add_contraction(
                            name=f"T3_{stem}_{tag}_t{tsep}_{kaon}_{op}_noavg",
                            terms=[(left, tsep + dl), (right, tsep + dr), (kaon, 0)],
                            swept=swept, translations=translations,
                            min_t=0, max_t=tsep, translation_average=False)
    return job


def main():
    widths = [s[1:] or "unsmeared" for s in smearings()]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tag", nargs="?", default=PRESETS["production"]["tag"],
                        help="hit tag of every field: hN for N sloppy hits, or "
                             "ama for the AMA hit, sloppy and exact")
    parser.add_argument("--all-gammas", action="store_true",
                        help="sweep every EMF gamma family, not only "
                             f"{' and '.join(SM_GAMMAS)}")
    parser.add_argument("--width", choices=widths, default="unsmeared",
                        help="smearing of the kaon and the pions; the EMFs are unsmeared")
    parser.add_argument("--tseps", type=int, nargs="+", default=TSEPS,
                        help="kaon to nearer-pion separations of type 3")
    parser.add_argument("--delta", type=int, default=DELTA,
                        help="pion separation of type 3")
    args = parser.parse_args()

    if max(args.tseps) + args.delta >= config.N_T:
        parser.error("tsep + Delta must stay below nt")

    # The tag picks everything, as in gen_contract_2pt.
    ama = args.tag == PRESETS["ama"]["tag"]
    preset = dict(PRESETS["ama" if ama else "production"], tag=args.tag)
    suffix = "" if args.tag == PRESETS["production"]["tag"] else f"_{args.tag}"
    output = f"{config.CONTRACTION_ROOT}/contract_k2pipi{suffix}"
    smear = "" if args.width == "unsmeared" else f"_{args.width}"
    gammas = config.EMF_GAMMA_FAMILIES.split() if args.all_gammas else SM_GAMMAS

    out_dir = Path(config.OUTPUT_ROOT) / "contractor" / "k2pipi"
    tasks = [(loop, t) for loop in LOOPS for t in EMF_TYPES]
    for k, (loop, emf_type) in enumerate(tasks):
        job = build(preset, smear, loop, emf_type, gammas, k == 0, args.tseps,
                    args.delta, output)
        path = job.write(out_dir / f"par.k2pipi.{args.tag}{smear.replace('_', '.', 1)}"
                                   f".{loop}loop_type{emf_type}.xml")
        n_swept = len({e for _, s, *_ in job.contractions.values() for e in s})
        print(f"  {path.name:36s} {len(job.contractions):4d} contractions, "
              f"{n_swept} swept field(s){' with the mix' if k == 0 else ''}")
    print(f"wrote {len(tasks)} tasks for {args.tag} to {out_dir}: {' '.join(gammas)}, "
          f"kaon and pions {args.width}, tseps {' '.join(map(str, args.tseps))}, "
          f"Delta {args.delta}")
    print(f"  correlators   {output}/")


if __name__ == "__main__":
    main()
