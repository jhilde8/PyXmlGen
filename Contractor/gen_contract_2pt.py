"""Single-particle two-point functions from the dense-W meson fields.

    gen_contract_2pt.py                  8-hit production set
    gen_contract_2pt.py --ama            the AMA correction hit, both accuracies
    gen_contract_2pt.py --no-average     one correlator per source timeslice
    gen_contract_2pt.py --channel kaon   just the kaon

One par file per momentum, holding every two-point function at that momentum
for every smearing state: unsmeared plus each rung of config.SMEAR_LADDER.

    kaon   tr[sl(t) ls(0)]               at rest only, so only in p000
    pion   tr[pi^p(t) pi^-p(0)]          27 momenta, back to back

FlexibleContractorDense loads every field a contraction uses before the first
one runs, so a par file's footprint is the sum of its fields: at 8 hits a
non-zero pion momentum is six 61 GB fields, about 370 GB, inside a quarter
node's 544 GB. The AMA set is 1-hit and a quarter of that even with both
accuracies.

Each non-zero pion field is loaded by two par files, its own momentum's and its
partner's (p001 and p00m1 both need both fields). Grouping by shell or by +-p
pair would remove that; deliberately left for later.

Momentum-shell averaging, and averaging p against -p, is post-processing.

The sigma is deliberately absent: tr[ll(t) ll(0)] is only its connected piece,
and the disconnected one needs BubbleContractorDense.

Names. Each contraction is named source_0_sink, from matrix names built to
read as the correlator: field, accuracy, smearing, then momentum,

    pi_w2p03_n11_p001_0_pi_w2p03_n11_p00m1
    sl_exact_w4p1_n45_p000_0_ls_exact_w4p1_n45_p000

with the full smearing rung tag, the same string the meson field directories
carry; --no-average appends _noavg. Momentum tags are filename-safe (m for
minus); ioname() renders the raw integers A2AMesonField used for the HDF5
group. Every name is distinct, so all par files of a preset share one
correlator directory, <name>.<traj>.h5 each.

Par files all land in output/contractor/ as par.<tag>.xml, tag being
2pt_<momentum> plus trailing modifiers (_ama, _noavg), so tag, par file and
log line up.

Axes come from config.field_axes: the left leg is the row, dense for a W --
W_s for sl, W_l for ls and pi -- and the right leg the expanded V col.
"""
import argparse
import sys
from pathlib import Path

# The toolkit (config, contractor_xml) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from contractor_xml import FlexibleJob

TRAJ = 1520

# The AMA job solves one hit twice (config.AMA_HIT), so its fields are 1-hit;
# the contractor reads the hit count off each field's dimensions.
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


def build(tag, products, preset, translation_average):
    """One par file: every two-point function at this momentum, for every
    accuracy and smearing state. A matrix used twice -- the pion at p000,
    whose two legs are the same field -- is declared once.

    translations is always 0..nt-1 and the sink sweeps every separation. With
    the average off each source is its own dataset <name>_dt_<dt>, written
    raw; the averaged one divides by the number of translations."""
    job = FlexibleJob(nt=config.N_T,
                      output=f"{config.CONTRACTION_ROOT}/{preset['out']}",
                      traj_start=TRAJ, traj_end=TRAJ + 1)
    for acc in preset["accuracies"]:
        for smear in smearings():
            for legs in products:
                names = []
                for field, gamma, mom in legs:
                    name = f"{field}{acc}{smear}_{mom_tag(mom)}"
                    if name not in job.matrices:
                        row, col = config.field_axes(field)
                        job.add_matrix(name=name,
                                       file=field_file(field, gamma, mom, preset, acc, smear),
                                       dataset=ioname(gamma, mom), row=row, col=col)
                    names.append(name)
                source, sink = names
                job.add_contraction(
                    name=f"{source}_0_{sink}" + ("" if translation_average else "_noavg"),
                    terms=[(source, "source", 0), (sink, "sink", 0)],
                    translations=f"0..{config.N_T - 1}", min_t=0, max_t=config.N_T - 1,
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
    par_dir = Path(config.OUTPUT_ROOT) / "contractor"

    # The job tag names the par file, the job's diskVectorDir, and -- by the
    # submission scripts' convention -- the log. Modifiers trail the momentum.
    suffix = ("_ama" if args.ama else "") + ("_noavg" if args.no_average else "")

    n_files = 0
    for mom, products in momenta(args.channel):
        tag = f"2pt_{mom}{suffix}"
        build(tag, products, preset, not args.no_average).write(
            par_dir / f"par.{tag}.xml")
        n_files += 1

    print(f"wrote {n_files} par files to {par_dir}"
          f"   (nHit = {preset['n_hit']}, {len(smearings())} smearing states, "
          f"{len(preset['accuracies'])} accuracy(ies))")
    print(f"  meson fields  {preset['base']}/<stem>.{TRAJ}/")
    print(f"  correlators   {config.CONTRACTION_ROOT}/{preset['out']}/")


if __name__ == "__main__":
    main()
