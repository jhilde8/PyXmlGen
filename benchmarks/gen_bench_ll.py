"""
Light-leg meson field timing job: pion, sigma, mf_ls_ww and mf_sl, smeared.

This is the first of the two production jobs. It owns every field whose legs
are covered by W_l, V_l and W_s -- the second job (gen_bench_svlv.py) owns
everything needing V_s, and mf_ls goes there rather than here because it
would cost this job a V_s load (1536*h fields) against svlv's flat W_l load.
The old three-way ll/kaon/svlv split is gone: mf_ls_ww and mf_sl arrive here
for the price of W_s alone, 12*h fields, which is nothing next to V_l.

Purpose is a timing projection, not physics: run all four fields at h = 1 and
h = 2 so the measured cost ratios can be checked against the mode-count
model, after which every larger hit count is predicted rather than measured.

Cost model. With the dense W representation the leg sizes are

    W_l = N_LOW + N_SC*h    V_l = N_LOW + N_HIGH*h    W_s = N_SC*h

and contraction cost goes as N_l*N_r. Three of the four scaling classes in
the whole campaign show up in this one job:

    field      legs         h=1        h=2        h=8      h2/h1   h8/h1
    mf_pi      W_l x V_l  7.114e6    1.027e7    2.995e7   1.4430   4.210
    mf_ll      W_l x V_l  7.114e6    1.027e7    2.995e7   1.4430   4.210
    mf_ls_ww   W_l x W_s  2.414e4    4.858e4    2.012e5   2.0119   8.334
    mf_sl      W_s x V_l  4.243e4    1.217e5    1.372e6   2.8688  32.33

against 4.0 (h2/h1) and 64.0 (h8/h1) if the cost were quadratic in hits.
mf_sl matters out of all proportion to its runtime: W_s x V_l and svlv's
V_s x V_l are the same constant-plus-linear structure -- the 12 versus 1536
cancels in the ratio -- so mf_sl's 2.8688 validates the only genuinely
quadratic class in the campaign for a couple of seconds of contraction,
which is what lets gen_bench_svlv.py buy svlv's absolute cost at h=1 without
paying for an h=2 point as well.

Compare the module's own timers, not job walltime. The ratios above are for
the N_l*N_r term only, and the other phases scale differently: the loaders
and the smearing both go as the total field count, predicting 7120/5560 =
1.281, and "Momentum phases" is h-independent. Hadrons reports per-module
timings, so the loaders and the smearing modules separate out on their own;
within each meson field module everything except "Momentum phases" carries
the N_l*N_r scaling, "IO" included, since the output is nt*N_l*N_r per
momentum and gamma.

Schedule (module order == schedule order, and Hadrons is given the file):

    noise_l, a2a_l_w        W_l
    a2a_l_v                 V_l
    noise_s, a2a_s_w        W_s
    gauge, gauge_APE
    three A2ACovariantSmear
    mf_ls_ww, mf_sl, mf_ll, mf_pi

All three loads come before the gauge so the vector IO is contiguous in the
log and the APE smear is not sandwiched between two multi-TB loads. The
fields run cheap-first: mf_ls_ww and mf_sl are seconds apiece, so a walltime
overrun on the expensive pair does not also cost the two ratios that only
this job can supply. Nothing is freed anywhere in this order -- all three
arrays outlive the last contraction -- so it costs no memory to choose.

Smeared rather than unsmeared, since the pion field is only ever produced at
the smearing width -- this is the production-realistic shape, and it doubles
as a check that the contraction cost really is independent of whether its
input vectors were smeared (the smear leaves the array shapes untouched, so
the table above is unchanged either way).

blocks differ by field: config.BLOCK_LIGHT_LIGHT for the two light-light
fields, config.BLOCK_STRANGE_LEG for the two with a strange leg, matching
production. cacheBlock = block throughout (GPU path, untiled SumRing).

timeSliceIO is on for the light-light pair and off for the kaon pair. It
skips the temporal all-gather and writes one file per (momentum, gamma,
timeslice), which spreads the write over min(nt, nRank) ranks instead of
nmom*ngamma of them -- decisive for the pion at 393 GB, pointless for
mf_ls_ww at 0.35 GB, where 128x the file count is Lustre metadata bought
against a sub-second write.

The writes are deliberate: IO is part of what is being projected, and it
carries the same N_l*N_r scaling as the contraction. Budget scratch space
for roughly 496 GB (h=1) and 717 GB (h=2) at ComplexD output, dominated by
the pion's 27 momenta; the two kaon fields add 1.0 and 2.4 GB.

Memory note: A2ACovariantSmear smears in place -- setup() envCreates an
empty array and execute() takes the input's buffers by std::move, so the
smeared array costs nothing beyond the unsmeared one it consumes. Peak
footprint is therefore just W_l + V_l + W_s, 5560 fields at h=1 and 7120 at
h=2 (35.8 TB and 45.8 TB at 6.44 GB per ComplexD field), plus one scratch
field. The flip side is that the unsmeared array is left empty and must not
be referenced after its smear module runs; nothing here does, since all four
fields are built from the smeared arrays only.

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
              graph_file=config.GRAPH)
    pool = VectorPool(job)

    lw = pool.combined("l", "w", hits)
    lv = pool.combined("l", "v", hits)
    sw = pool.combined("s", "w", hits)

    job.add(M.load_nersc("gauge", config.GAUGE_FILE))
    job.add(M.ape_smear("gauge_APE", "gauge", config.APE_ALPHA, config.APE_N,
                        config.ORTHOG_AXIS))

    def smear(array):
        name = f"{array}_{width_tag}"
        job.add(M.a2a_covariant_smear(
            name, a2a_vectors=array, gauge="gauge_APE", alpha=alpha, N=N,
            orthog_axis=config.ORTHOG_AXIS, output="", multi_file=False))
        return name

    lw_sm = smear(lw)
    lv_sm = smear(lv)
    sw_sm = smear(sw)

    def mf(name, block, left, right, gammas, mom, time_slice_io):
        job.add(M.a2a_meson_field(
            name, block, block, left, right,
            f"{config.TMP_OUTPUT}/{name}", gammas, mom,
            time_slice_io=time_slice_io))

    mf(f"mf_ls_ww_{tag}_{width_tag}", config.BLOCK_STRANGE_LEG,
       lw_sm, sw_sm, config.IDENTITY, config.KAON_MOM, False)
    mf(f"mf_sl_{tag}_{width_tag}", config.BLOCK_STRANGE_LEG,
       sw_sm, lv_sm, config.GAMMA5, config.KAON_MOM, False)

    mf(f"mf_ll_{tag}_{width_tag}", config.BLOCK_LIGHT_LIGHT,
       lw_sm, lv_sm, config.IDENTITY, config.SIGMA_MOM, True)
    mf(f"mf_pi_{tag}_{width_tag}", config.BLOCK_LIGHT_LIGHT,
       lw_sm, lv_sm, config.GAMMA5, config.PION_MOM, True)

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
    print(f"wrote {n} light-leg benchmark jobs to {out_dir}")


if __name__ == "__main__":
    main()
