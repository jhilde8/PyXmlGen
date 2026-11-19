"""Single-particle two-point functions from the dense-W meson fields.

    gen_contract_2pt.py                  8-hit production set
    gen_contract_2pt.py --tag ama        the AMA correction hit, both accuracies
    gen_contract_2pt.py --no-average     one correlator per source timeslice
    gen_contract_2pt.py --channel kaon   just the kaon
    gen_contract_2pt.py --pairs all --no-average --tag h8 \\
        --traj-start 1400 --traj-end 1700 --traj-step 20
                                         every pion momentum pair, per source

One par file per momentum, holding every two-point function at that momentum
for every smearing state: unsmeared plus each rung of config.SMEAR_LADDER.

All pairs (--pairs all). The pion only, for every unordered pair of the 27
momenta: 378 contractions per smearing state, one par file per state, the
same smearing at source and sink. The other 351 ordered pairs follow from
cyclicity of the trace (see pion_pairs) and are rebuilt in the analysis.
These are the inputs of the pipi D diagram. Its two traces exchange gluons,
so pairs whose momenta do not match are physical and feed the off-diagonal
momentum shells of the GEVP. Every par file holds all 27 fields of its
state, ~1.7 TB at 8 hits: one whole Riker node (2.2 TB) per file, so the
three states are three nodes. Each back-to-back pair of the default mode is
there in one order or the other, so a --pairs all run also gives the
single-pion two-point functions.

--tag picks the whole input. ama is the AMA preset: the 1-hit fields under
MF_AMA_BASE, sloppy and exact in each par file. Any hN is the production
preset with that tag (mf_pi_<tag>...). A tag other than h8 also goes on the
correlator directory, contract_2pt_<tag>,
because correlator names carry no hit count and an h8 configuration can also
have h16 fields.

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


def pion_pairs():
    """Every unordered pair of pion momenta, source first, the source no later
    than the sink in config.PION_MOM. The trace is cyclic and every source
    and sink time is computed, so the reversed pair is a relabelling,
        C_{q->p}(dt, s) = C_{p->q}((dt + s) % nt, (nt - s) % nt),
    rebuilt in the analysis."""
    moms = config.PION_MOM
    return [[("pi", config.GAMMA5, moms[i]), ("pi", config.GAMMA5, moms[j])]
            for i in range(len(moms)) for j in range(i, len(moms))]


def build(products, preset, translation_average, smears, trajs):
    """One par file: every two-point function of `products`, for every
    accuracy and every smearing state in `smears`. A matrix used more than
    once -- the pion at p000 in the default mode, every pion field with
    --pairs all -- is declared once.

    translations is always 0..nt-1 and the sink sweeps every separation. With
    the average off each source is its own dataset <name>_dt_<dt>, written
    raw; the averaged one divides by the number of translations."""
    traj_start, traj_end, traj_step = trajs
    job = FlexibleJob(nt=config.N_T,
                      output=f"{config.CONTRACTION_ROOT}/{preset['out']}",
                      traj_start=traj_start, traj_end=traj_end, traj_step=traj_step)
    for acc in preset["accuracies"]:
        for smear in smears:
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
    parser.add_argument("--no-average", action="store_true",
                        help="write one correlator per source timeslice "
                             "(_dt_<dt>) instead of their average")
    parser.add_argument("--pairs", choices=("back-to-back", "all"),
                        default="back-to-back",
                        help="all: the pion for every unordered momentum pair, "
                             "one par file per smearing state (--channel is "
                             "ignored)")
    parser.add_argument("--tag", default=PRESETS["production"]["tag"],
                        help="hit tag of the fields: hN for N sloppy hits, or "
                             "ama for the AMA hit, sloppy and exact")
    parser.add_argument("--traj-start", type=int, default=TRAJ)
    parser.add_argument("--traj-end", type=int,
                        help="exclusive, as trajCounter (default: start + 1)")
    parser.add_argument("--traj-step", type=int, default=20)
    args = parser.parse_args()

    # The tag picks everything: ama is the AMA preset, any hN the production
    # one with that tag.
    ama = args.tag == PRESETS["ama"]["tag"]
    preset = dict(PRESETS["ama" if ama else "production"])
    if args.tag != preset["tag"]:
        preset["tag"] = args.tag
        preset["out"] += f"_{args.tag}"
    trajs = (args.traj_start,
             args.traj_end if args.traj_end is not None else args.traj_start + 1,
             args.traj_step)
    par_dir = Path(config.OUTPUT_ROOT) / "contractor"

    # The job tag names the par file, the job's diskVectorDir, and -- by the
    # submission scripts' convention -- the log. Modifiers trail the momentum.
    suffix = ("" if args.tag == PRESETS["production"]["tag"] else f"_{args.tag}") \
        + ("_noavg" if args.no_average else "")

    if args.pairs == "all":
        jobs = [(f"2pt_pion_pairs_{smear[1:] or 'unsmeared'}", pion_pairs(), [smear])
                for smear in smearings()]
    else:
        jobs = [(f"2pt_{mom}", products, smearings())
                for mom, products in momenta(args.channel)]

    for tag, products, smears in jobs:
        build(products, preset, not args.no_average, smears, trajs).write(
            par_dir / f"par.{tag}{suffix}.xml")

    print(f"wrote {len(jobs)} par files to {par_dir}"
          f"   (fields {preset['tag']}, trajectories {trajs[0]}..{trajs[1] - 1} "
          f"step {trajs[2]}, {len(preset['accuracies'])} accuracy(ies))")
    print(f"  meson fields  {preset['base']}/<stem>.@traj@/")
    print(f"  correlators   {config.CONTRACTION_ROOT}/{preset['out']}/")


if __name__ == "__main__":
    main()
