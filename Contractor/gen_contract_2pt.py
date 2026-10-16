"""Single-particle two-point functions from the dense-W meson fields.

    gen_contract_2pt.py                  8-hit production set
    gen_contract_2pt.py --ama            the AMA correction hit, both accuracies
    gen_contract_2pt.py --no-average     one correlator per source timeslice
    gen_contract_2pt.py --channel kaon   just the kaon

One par file per momentum, holding every two-point function at that momentum
for every smearing state: unsmeared plus each rung of config.SMEAR_LADDER.

    kaon   tr[sl(t) ls(0)]               at rest only, so only in p000
    pion   tr[pi^p(t) pi^-p(0)]          27 momenta, back to back

Every declared matrix is loaded and held before the first product runs, so a
par file's footprint is the sum of its fields: at 8 hits a non-zero pion
momentum is six 61 GB fields plus one 61 GB transpose cache, about 430 GB, and
p000 with the kaon about 410 GB -- both inside a quarter node's 544 GB. The AMA
set is 1-hit and a quarter of that even with both accuracies.

Each non-zero pion field is loaded by two par files, its own momentum's and its
partner's (p001 and p00m1 both need both fields). Grouping by shell or by +-p
pair would remove that; deliberately left for later.

Momentum-shell averaging, and averaging p against -p, is post-processing.

The sigma is deliberately absent: tr[ll(t) ll(0)] is only its connected piece,
and the disconnected one needs BubbleContractorDense.

Names. Matrix names reach the correlator filename through saveCorrelator's
stem, terms[0]_times[0]_terms.back(), so they are built to read as the
correlator: field, accuracy, smearing, then momentum,

    pi_w2p03_n11_p001_0_pi_w2p03_n11_p00m1
    sl_exact_w4p1_n45_p000_0_ls_exact_w4p1_n45_p000

with the full smearing rung tag, the same string the meson field directories
carry. Momentum tags are filename-safe (m for minus); ioname() renders the raw
integers A2AMesonField used for the HDF5 group. Every stem is distinct, so all
par files of a preset share one output directory.

nLow belongs to a field's DENSE (row) axis, its LEFT leg -- W_s for sl, W_l for
ls and pi. It does not depend on the hit count.
"""
import argparse
import sys
from pathlib import Path

# The toolkit (config, contractor_xml) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from contractor_xml import ContractorJob

TRAJ = 1520

# Left (dense W) leg of each field, which is what sets nLow.
LEFT_FLAVOR = {"pi": "l", "sl": "s", "ls": "l"}

# The AMA job solves one hit twice (config.AMA_HIT), so its fields are 1-hit
# and nHit must say so.
PRESETS = {
    "production": dict(tag="h8", accuracies=[""], base=config.MF_BASE,
                       n_hit=config.N_HIT, out="contract_2pt"),
    "ama": dict(tag="ama", accuracies=["_sloppy", "_exact"],
                base=config.MF_AMA_BASE, n_hit=1, out="contract_2pt_ama"),
}


def smearings():
    """Unsmeared, then every ladder rung by its full tag -- the suffix
    gen_meson_fields puts on the smeared field stems."""
    return [""] + [f"_{rung_tag}" for rung_tag, _, _ in config.SMEAR_LADDER]


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
    stem = f"mf_{field}_{preset['tag']}{acc}{smear}"
    return f"{preset['base']}/{stem}.@traj@/{ioname(gamma, mom)}.h5"


def mom_tag(mom):
    """Filename-safe momentum: p000, p001, p00m1, p1m10."""
    return "p" + "".join(("m" if p < 0 else "") + str(abs(int(p))) for p in mom)


def negate(mom):
    return [-p for p in mom]


def momenta(channel):
    """(momentum tag, products) per par file, where a product is the list of
    legs (field, gamma, mom) in trace order. The kaon is at rest only."""
    for mom in config.PION_MOM:
        products = []
        if channel in ("pion", "all"):
            products.append([("pi", config.GAMMA5, mom),
                             ("pi", config.GAMMA5, negate(mom))])
        if channel in ("kaon", "all") and not any(mom):
            products.append([("sl", config.GAMMA5, mom),
                             ("ls", config.GAMMA5, mom)])
        if products:
            yield mom_tag(mom), products


def build(tag, products, preset, out_dir, translation_average):
    """One par file: every product at this momentum, for every accuracy and
    smearing state. A matrix used twice -- the pion at p000, whose two legs are
    the same field -- is declared once.

    cacheSize = nt holds each field resident so the disk vector never spills;
    Riker has no node-local disk. translations is always 0..nt-1: dt is the
    SOURCE timeslice and the sink sweeps all nt regardless. With the average
    off each source is written raw to its own _dt_<dt> file -- the averaged path
    divides by translations.size(), the per-source path does not."""
    job = ContractorJob(
        nt=config.N_T,
        disk_vector_dir=f"{config.CONTRACTION_ROOT}/dv/{out_dir}_{tag}",
        output=f"{config.CONTRACTION_ROOT}/{out_dir}",
        traj_start=TRAJ, traj_end=TRAJ + 1, traj_step=1,
        n_hit=preset["n_hit"])
    declared = set()
    for acc in preset["accuracies"]:
        for smear in smearings():
            for legs in products:
                terms = []
                for field, gamma, mom in legs:
                    name = f"{field}{acc}{smear}_{mom_tag(mom)}"
                    if name not in declared:
                        job.add_matrix(
                            file=field_file(field, gamma, mom, preset, acc, smear),
                            dataset=ioname(gamma, mom), name=name,
                            cache_size=config.N_T, n_low=n_low(field),
                            time_slice_io=True)
                        declared.add(name)
                    terms.append(name)
                job.add_product(terms=terms, times=["0"],
                                translations=f"0..{config.N_T - 1}",
                                translation_average=translation_average)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--channel", choices=("kaon", "pion", "all"),
                        default="all")
    parser.add_argument("--ama", action="store_true",
                        help="the AMA correction hit: 1-hit fields under "
                             "MF_AMA_BASE, sloppy and exact in each par file")
    parser.add_argument("--no-average", action="store_true",
                        help="write one correlator per source timeslice "
                             "(_dt_<dt>) instead of their average")
    args = parser.parse_args()

    preset = PRESETS["ama" if args.ama else "production"]
    out_dir = preset["out"] + ("_bysource" if args.no_average else "")
    par_dir = Path(config.OUTPUT_ROOT) / out_dir

    n_files = 0
    for tag, products in momenta(args.channel):
        build(tag, products, preset, out_dir, not args.no_average).write(
            par_dir / f"par.2pt_{tag}.xml")
        n_files += 1

    print(f"wrote {n_files} par files to {par_dir}"
          f"   (nHit = {preset['n_hit']}, {len(smearings())} smearing states, "
          f"{len(preset['accuracies'])} accuracy(ies))")
    print(f"  meson fields  {preset['base']}/<stem>.{TRAJ}/")
    print(f"  correlators   {config.CONTRACTION_ROOT}/{out_dir}/")
    print("\nThe <output> directory must exist before the run: saveCorrelator"
          "\ncalls makeFileDir(dir), which creates dirname(dir) -- the parent --"
          "\nand never the directory itself.")


if __name__ == "__main__":
    main()
