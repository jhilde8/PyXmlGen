"""Single-particle two-point functions from the dense-W meson fields.

    gen_contract_2pt.py                  8-hit production set
    gen_contract_2pt.py --ama            the AMA correction hit, both accuracies
    gen_contract_2pt.py --channel kaon   just the kaon, to try one first

    kaon   tr[mf_sl(t) mf_ls(0)]           zero momentum             1
    pion   tr[mf_pi^p(t) mf_pi^-p(0)]      27 momenta, back to back  27

times two smearing states = 56 par files either way. Momentum-shell averaging
is post-processing, so every momentum gets its own contraction.

One par file per contraction, because the contractor is serial OpenMP and the
unit of parallelism is a whole process. The AMA run puts sloppy and exact in
the SAME par file as two products: the correction is their difference, so
computing them in one job removes any chance of pairing correlators from
different runs, and the 1-hit fields are small enough that holding both costs
nothing (mf_pi is 14.6 GB at one hit against 61 GB at eight).

The sigma is deliberately absent. tr[mf_ll(t) mf_ll(0)] is only its CONNECTED
piece; being isoscalar it also has a disconnected contribution, which is a
product of traces and cannot be written as one <product>. That needs the dense
BubbleContractor, so emitting the connected half now would invite mistaking it
for the whole thing.

Naming. ioname() and field_file() reproduce A2AMesonField's ionameFn and
filenameFn exactly, so momenta render as raw integers there (Gamma5_1_-1_0).
The matrix NAMES are ours and reach the correlator filename through
saveCorrelator's stem, so those use a filename-safe momentum tag, and in the
AMA case carry the accuracy so the two products stay distinct.

nLow belongs to a field's DENSE (row) axis, its LEFT leg -- W_s for mf_sl, W_l
for mf_ls and mf_pi. It does not depend on the hit count. The contractor
cross-checks it against the other term's expanded axis, so a wrong value fails
on the first trace rather than quietly producing a number.
"""
import argparse
import sys
from pathlib import Path

# The toolkit (config, contractor_xml) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from contractor_xml import ContractorJob

TRAJ = 1520

# (suffix on the meson field stem, label for the output directory). One entry per
# ladder rung, since every rung writes its own meson fields.
SMEARINGS = [("", "unsmeared")] + \
            [(f"_{rung[0]}", rung[0]) for rung in config.SMEAR_LADDER]

# Left (dense W) leg of each field, which is what sets nLow.
LEFT_FLAVOR = {"mf_pi": "l", "mf_sl": "s", "mf_ls": "l"}

# The AMA job solves one hit twice (config.AMA_HIT), so its fields are 1-hit
# and nHit must say so -- gen_meson_fields passes hits=[AMA_HIT], and
# VectorPool normalizes by len(hits).
PRESETS = {
    "production": dict(tag="h8", accuracies=[""], base=config.MF_BASE,
                       n_hit=config.N_HIT, out="contract_2pt"),
    "ama": dict(tag="ama", accuracies=["_sloppy", "_exact"],
                base=config.MF_AMA_BASE, n_hit=1, out="contract_2pt_ama"),
}


def n_low(field):
    return config.N_LOW if config.FLAVOR_HAS_LOW[LEFT_FLAVOR[field]] else 0


def ioname(gamma, mom):
    """A2AMesonField::ionameFn -- the HDF5 group and the file basename. mom_ is
    vector<vector<Real>>, so integers render bare: Gamma5_1_-1_0."""
    return gamma + "_" + "_".join(str(int(p)) for p in mom)


def field_file(field, gamma, mom, preset, acc, smear):
    """A2AMesonField::filenameFn, with @traj@ left for the contractor.
    timeSliceIO appends .t%04d before the .h5, which the contractor does
    itself, so this is the whole-field path."""
    stem = f"{field}_{preset['tag']}{acc}{smear}"
    return f"{preset['base']}/{stem}.@traj@/{ioname(gamma, mom)}.h5"


def mom_tag(mom):
    """Filename-safe momentum: p000, p100, pm100, p1m10, pm1m1m1."""
    return "p" + "".join(("m" if p < 0 else "") + str(abs(int(p))) for p in mom)


def negate(mom):
    return [-p for p in mom]


def contractions(channel):
    """(name, legs, terms), where legs are (role, field, gamma, mom) and terms
    names roles in trace order. Roles gain the accuracy suffix at build time.

    The kaon's two legs have DIFFERENT mode spaces -- mf_sl has strange rows
    (nLow 0) and mf_ls light rows (nLow 2000) -- which is why the contractor
    takes one mode space per summed index rather than one per matrix.

    At p = 0 the pion's two legs are the same field, so one matrix is declared
    and named twice rather than loading identical data into two disk vectors."""
    if channel in ("kaon", "all"):
        yield ("kaon_p000",
               [("mf_sl", "mf_sl", config.GAMMA5, [0, 0, 0]),
                ("mf_ls", "mf_ls", config.GAMMA5, [0, 0, 0])],
               ["mf_sl", "mf_ls"])
    if channel in ("pion", "all"):
        for mom in config.PION_MOM:
            tag, anti = mom_tag(mom), mom_tag(negate(mom))
            src, snk = f"mf_pi_{tag}", f"mf_pi_{anti}"
            if tag == anti:
                yield (f"pion_{tag}",
                       [(src, "mf_pi", config.GAMMA5, mom)],
                       [src, src])
            else:
                yield (f"pion_{tag}",
                       [(src, "mf_pi", config.GAMMA5, mom),
                        (snk, "mf_pi", config.GAMMA5, negate(mom))],
                       [src, snk])


def build(name, legs, terms, preset, smear, translation_average=True):
    """One par file, with one product per accuracy. cacheSize = nt holds each
    field resident so the disk vector never spills -- Riker has no node-local
    disk, so a spill would be pure Lustre traffic.

    translations is always the full 0..nt-1: dt is the SOURCE timeslice (the
    first term sits at times[0] + dt) while the sink tLast sweeps all nt
    regardless, so this uses every source placement the all-to-all propagator
    affords. translation_average=False writes each source to its own
    _dt_<dt> file instead of collapsing them, RAW -- the averaged path divides
    by translations.size() and the per-source path does not, so re-blocking in
    analysis means summing and dividing yourself."""
    job = ContractorJob(
        nt=config.N_T,
        disk_vector_dir=f"{config.CONTRACTION_ROOT}/dv/{name}",
        output=f"{config.CONTRACTION_ROOT}/{name}",
        traj_start=TRAJ, traj_end=TRAJ + 1, traj_step=1,
        n_hit=preset["n_hit"])
    for acc in preset["accuracies"]:
        for role, field, gamma, mom in legs:
            job.add_matrix(file=field_file(field, gamma, mom, preset, acc, smear),
                           dataset=ioname(gamma, mom),
                           name=role + acc,
                           cache_size=config.N_T,
                           n_low=n_low(field),
                           time_slice_io=True)
        job.add_product(terms=[t + acc for t in terms], times=["0"],
                        translations=f"0..{config.N_T - 1}",
                        translation_average=translation_average)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--channel", choices=("kaon", "pion", "all"),
                        default="all")
    parser.add_argument("--ama", action="store_true",
                        help="the AMA correction hit: 1-hit fields under "
                             "MF_AMA_BASE, sloppy and exact in one par file")
    parser.add_argument("--no-average", action="store_true",
                        help="write one correlator per source timeslice "
                             "(_dt_<dt>) instead of their average, so the "
                             "blocking is chosen in analysis")
    args = parser.parse_args()

    preset = PRESETS["ama" if args.ama else "production"]
    suffix = "_bysource" if args.no_average else ""
    out_dir = Path(config.OUTPUT_ROOT) / (preset["out"] + suffix)

    names = []
    for smear, label in SMEARINGS:
        for name, legs, terms in contractions(args.channel):
            full = f"{name}_{label}{suffix}"
            build(full, legs, terms, preset, smear,
                  translation_average=not args.no_average).write(
                out_dir / f"par.{full}.xml")
            names.append(full)

    products = len(preset["accuracies"])
    per_product = config.N_T if args.no_average else 1
    print(f"wrote {len(names)} par files to {out_dir}"
          f"   ({products} product{'s' if products > 1 else ''} each, "
          f"nHit = {preset['n_hit']}, "
          f"{products * per_product} correlator file(s) per run)")
    print(f"  meson fields  {preset['base']}/<stem>.{TRAJ}/")
    print(f"  correlators   {config.CONTRACTION_ROOT}/<name>/")
    print("\nEvery <output> directory must exist before its run: saveCorrelator"
          "\ncalls makeFileDir(dir), which creates dirname(dir) -- the parent --"
          "\nand never the directory itself.")


if __name__ == "__main__":
    main()
