"""Single-particle two-point functions from the dense-W meson fields.

    gen_contract_2pt.py                 all channels, both smearings
    gen_contract_2pt.py --channel kaon  just the kaon, to try one first

One par file per contraction, because the contractor is serial OpenMP and the
unit of parallelism is a whole process -- four of them fit on a Riker node, one
per NUMA domain. Grouping products into one par file would only serialise them.

    kaon   tr[mf_sl(t) mf_ls(0)]           zero momentum             1
    pion   tr[mf_pi^p(t) mf_pi^-p(0)]      27 momenta, back to back  27

times two smearing states = 56 files. Momentum-shell averaging is a
post-processing step, so every momentum gets its own contraction and the 27 are
not combined here.

The sigma is deliberately absent. tr[mf_ll(t) mf_ll(0)] is only its CONNECTED
piece; being isoscalar it also has a disconnected contribution, which is a
product of traces and so cannot be written as one <product> at all. That needs
BubbleContractor, which has no dense-W path yet -- emitting the connected half
now would invite mistaking it for the whole thing later.

Naming. The meson field path and HDF5 group have to match what A2AMesonField
wrote, so ioname() and field_file() reproduce its ionameFn/filenameFn exactly;
momenta render as raw integers there ("Gamma5_1_-1_0"). The matrix NAMES are
ours and end up in the correlator filename through saveCorrelator's stem
(terms[0]_times[0]_terms.back()), so those use a filename-safe momentum tag.
Smearing is carried by the output directory rather than the stem, which keeps
the stem short and stops the two variants colliding.

nLow belongs to a field's DENSE (row) axis, which is its LEFT leg -- W_s for
mf_sl, W_l for mf_ls and mf_pi. The contractor cross-checks it against the
other term's expanded axis, so a wrong value fails on the first trace rather
than quietly producing a number.
"""
import argparse
import sys
from pathlib import Path

# The toolkit (config, contractor_xml) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from contractor_xml import ContractorJob

TAG = "h8"
TRAJ = 1520

# (suffix on the meson field stem, label for the output directory)
SMEARINGS = [("", "unsmeared"),
             (f"_{config.SMEAR_WIDTHS[0][0]}", "smeared")]

# Left (dense W) leg of each field, which is what sets nLow.
LEFT_FLAVOR = {"mf_pi": "l", "mf_sl": "s", "mf_ls": "l"}


def n_low(field):
    return config.N_LOW if config.FLAVOR_HAS_LOW[LEFT_FLAVOR[field]] else 0


def ioname(gamma, mom):
    """A2AMesonField::ionameFn -- the HDF5 group, and the file's basename.
    mom_ is vector<vector<Real>>, so integers render bare: Gamma5_1_-1_0."""
    return gamma + "_" + "_".join(str(int(p)) for p in mom)


def field_file(field, smear, gamma, mom, base=None):
    """A2AMesonField::filenameFn, with @traj@ left for the contractor to
    substitute. timeSliceIO appends .t%04d before the .h5, which the contractor
    does itself -- this is the whole-field path."""
    stem = f"{field}_{TAG}{smear}"
    return f"{base or config.MF_BASE}/{stem}.@traj@/{ioname(gamma, mom)}.h5"


def mom_tag(mom):
    """Filename-safe momentum: p000, p100, pm100, p1m10, pm1m1m1."""
    return "p" + "".join(("m" if p < 0 else "") + str(abs(int(p))) for p in mom)


def negate(mom):
    return [-p for p in mom]


def build(name, matrices, terms, smear):
    """One par file. `matrices` is (matrix name, field, gamma, mom); `terms` is
    the trace in order. cacheSize = nt holds each field resident so the disk
    vector never spills -- there is no node-local disk on Riker, so a spill
    would be pure Lustre traffic."""
    job = ContractorJob(
        nt=config.N_T,
        disk_vector_dir=f"{config.CONTRACTION_ROOT}/dv/{name}",
        output=f"{config.CONTRACTION_ROOT}/{name}",
        traj_start=TRAJ, traj_end=TRAJ + 1, traj_step=1,
        n_hit=config.N_HIT)
    for matrix_name, field, gamma, mom in matrices:
        job.add_matrix(file=field_file(field, smear, gamma, mom),
                       dataset=ioname(gamma, mom),
                       name=matrix_name,
                       cache_size=config.N_T,
                       n_low=n_low(field),
                       time_slice_io=True)
    job.add_product(terms=terms, times=["0"],
                    translations=f"0..{config.N_T - 1}",
                    translation_average=True)
    return job


def kaon_jobs():
    """tr[mf_sl(t) mf_ls(0)] at rest.

    The two legs have DIFFERENT mode spaces: mf_sl has strange rows (nLow 0,
    dense 96) and mf_ls light rows (nLow 2000, dense 2096). i runs strange
    against strange, j light against light -- the permutation is fixed by the
    dimensions, and a wrong one fails the contractor's four shape checks rather
    than returning a number."""
    for smear, label in SMEARINGS:
        name = f"kaon_p000_{label}"
        matrices = [
            ("mf_sl", "mf_sl", config.GAMMA5, [0, 0, 0]),
            ("mf_ls", "mf_ls", config.GAMMA5, [0, 0, 0]),
        ]
        yield name, build(name, matrices, ["mf_sl", "mf_ls"], smear)


def pion_jobs():
    """tr[mf_pi^p(t) mf_pi^-p(0)], back to back. At p = 0 the two legs are the
    same field, so one matrix is declared and named twice rather than loading
    61 GB of identical data into two disk vectors."""
    for smear, label in SMEARINGS:
        for mom in config.PION_MOM:
            tag, anti = mom_tag(mom), mom_tag(negate(mom))
            name = f"pion_{tag}_{label}"
            src = f"mf_pi_{tag}"
            if tag == anti:
                matrices = [(src, "mf_pi", config.GAMMA5, mom)]
                terms = [src, src]
            else:
                snk = f"mf_pi_{anti}"
                matrices = [
                    (src, "mf_pi", config.GAMMA5, mom),
                    (snk, "mf_pi", config.GAMMA5, negate(mom)),
                ]
                terms = [src, snk]
            yield name, build(name, matrices, terms, smear)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--channel", choices=("kaon", "pion", "all"),
                        default="all")
    args = parser.parse_args()

    out_dir = Path(config.OUTPUT_ROOT) / "contract_2pt"
    builders = {"kaon": [kaon_jobs], "pion": [pion_jobs],
                "all": [kaon_jobs, pion_jobs]}[args.channel]

    names = []
    for builder in builders:
        for name, job in builder():
            job.write(out_dir / f"par.{name}.xml")
            names.append(name)

    print(f"wrote {len(names)} par files to {out_dir}")
    print(f"  meson fields  {config.MF_BASE}/<stem>.{TRAJ}/")
    print(f"  correlators   {config.CONTRACTION_ROOT}/<name>/")
    print(f"  scratch       {config.CONTRACTION_ROOT}/dv/<name>/"
          f"   (stays empty at cacheSize = {config.N_T})")
    print("\nEvery <output> directory must exist before its run: saveCorrelator"
          "\ncalls makeFileDir(dir), which creates dirname(dir) -- the parent --"
          "\nand never the directory itself. To create them all:\n")
    print(f"  mkdir -p {config.CONTRACTION_ROOT}/{{"
          + ",".join(names[:2]) + ",...}")


if __name__ == "__main__":
    main()
