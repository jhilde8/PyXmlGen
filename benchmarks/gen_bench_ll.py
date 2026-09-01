"""
Light-light meson field timing job (group 3: pion and sigma), smeared.

Purpose is a timing projection, not physics: run the same two fields at
h = 1 and h = 2 so the measured cost ratio can be checked against the
mode-count model, after which every larger hit count is predicted rather
than measured.

Both light-light fields are W_l x V_l, so with the dense W representation
their cost goes as

    N_l * N_r = (N_LOW + N_SC*h) * (N_LOW + N_HIGH*h)

which is nowhere near quadratic in h -- the W side barely grows at all
(12 modes per hit against a 2000-mode low block). Predicted numbers:

    h    N_l(W_l)   N_r(V_l)   N_l*N_r        vs h=1
    1      2012       3536     7.114e6         1.000
    2      2024       5072     1.027e7         1.443
    4      2048       8144     1.668e7         2.344
    8      2096      14288     2.995e7         4.210

So the acceptance criterion for this job is a h2/h1 ratio of 1.443 on the
contraction timers, against 4.0 if the cost were quadratic in hits. That
gap is wide enough to be unambiguous with one pair of runs.

Compare the A2AMesonField module's own timers, not job walltime. The 1.443
above is for the N_l*N_r term only, and the other phases in this job scale
differently: the loaders and the smearing both go as N_l+N_r, predicting
7096/5548 = 1.279, and "Momentum phases" is h-independent. Hadrons reports
per-module timings, so the loaders and the smearing modules separate out on
their own; within the meson field module everything except "Momentum
phases" carries the N_l*N_r scaling, "IO" included, since the output is
nt*N_l*N_r per momentum and gamma.

Smeared rather than unsmeared, since the pion field is only ever produced
at the smearing width -- this is the production-realistic shape, and it
doubles as a check that the contraction cost really is independent of
whether its input vectors were smeared (the smear leaves the array shapes
untouched, so the table above is unchanged either way).

The writes are deliberate: IO is part of what is being projected, and it
carries the same N_l*N_r scaling as the contraction. Budget scratch space
for roughly 495 GB (h=1) and 715 GB (h=2) at ComplexD output, dominated by
the pion's 27 momenta.

Memory note: A2ACovariantSmear smears in place -- setup() envCreates an
empty array and execute() takes the input's buffers by std::move, so the
smeared array costs nothing beyond the unsmeared one it consumes. Peak
footprint is therefore just W + V, 5548 fields at h=1 and 7096 at h=2
(~18 TB and ~23 TB at single precision), plus one scratch field. The flip
side is that the unsmeared array is left empty and must not be referenced
after its smear module runs; nothing here does, since both fields are
built from the smeared arrays only.

One job per hit count rather than both in one: keeping the runs separate
stops the h=1 measurement from sharing an allocation with h=2 arrays.
"""
import sys
from pathlib import Path

# The toolkit (config, modules, hadrons_xml, vector_pool) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import modules as M
from hadrons_xml import Job
from vector_pool import VectorPool

HIT_SETS = ([0], [0, 1])


def build_job(hits, width):
    hits = list(hits)
    width_tag, alpha, N = width
    tag = "".join(f"h{h}" for h in hits)
    run_id = f"bench.ll.{tag}.{width_tag}"
    job = Job(run_id, schedule_file=config.schedule_file(run_id),
              graph_file=config.graph_file(run_id))
    pool = VectorPool(job)

    job.add(M.load_nersc("gauge", config.GAUGE_FILE))
    job.add(M.ape_smear("gauge_APE", "gauge", config.APE_ALPHA, config.APE_N,
                        config.ORTHOG_AXIS))

    lw = pool.combined("l", "w", hits)
    lw_sm = f"{lw}_{width_tag}"
    job.add(M.a2a_covariant_smear(
        lw_sm, a2a_vectors=lw, gauge="gauge_APE", alpha=alpha, N=N,
        orthog_axis=config.ORTHOG_AXIS, output="", multi_file=False))

    lv = pool.combined("l", "v", hits)
    lv_sm = f"{lv}_{width_tag}"
    job.add(M.a2a_covariant_smear(
        lv_sm, a2a_vectors=lv, gauge="gauge_APE", alpha=alpha, N=N,
        orthog_axis=config.ORTHOG_AXIS, output="", multi_file=False))

    # cacheBlock = block: the GPU path is fastest with the SumRing reduction
    # untiled, one tile spanning the whole block (see config.py).
    block = config.BLOCK_LIGHT_LIGHT

    name_sigma = f"mf_ll_{tag}_{width_tag}"
    job.add(M.a2a_meson_field(
        name_sigma, block, block, lw_sm, lv_sm,
        f"{config.TMP_OUTPUT}/{name_sigma}",
        config.IDENTITY, config.SIGMA_MOM))

    name_pi = f"mf_pi_{tag}_{width_tag}"
    job.add(M.a2a_meson_field(
        name_pi, block, block, lw_sm, lv_sm,
        f"{config.TMP_OUTPUT}/{name_pi}",
        config.GAMMA5, config.PION_MOM))

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "bench_ll"
    n = 0
    for width in config.SMEAR_WIDTHS:
        for hits in HIT_SETS:
            job, run_id = build_job(hits, width)
            job.write(out_dir / f"par.{run_id}.xml",
                      out_dir / f"schedule.{run_id}.txt")
            n += 1
    print(f"wrote {n} light-light benchmark jobs to {out_dir}")


if __name__ == "__main__":
    main()
