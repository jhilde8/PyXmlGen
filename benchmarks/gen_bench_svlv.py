"""
svlv timing job: the strange A2A loop, mf_ls, the extended meson field, the
chromo-magnetic operator field and the subtraction operator.

This is the second of the two production jobs. It owns every field that needs
V_s; gen_bench_ll.py owns everything covered by W_l, V_l and W_s. mf_ls
(W_l x V_s) lives here rather than there because it costs this job a flat W_l
load, 2000 + 12*h fields, against a V_s load of 1536*h fields in the other
direction.

Run at h = 1 and h = 2 so the measured ratios can be checked against the
mode-count model, after which larger hit counts are predicted rather than
measured.

Cost model. Leg sizes with the dense W representation:

    W_l = N_LOW + N_SC*h    V_l = N_LOW + N_HIGH*h    V_s = N_HIGH*h

    field       legs         h=1        h=2        h=8      h2/h1   h8/h1
    mf_ls       W_l x V_s  3.090e6    6.218e6    2.576e7   2.0119   8.334
    emf_sloop   V_s x V_l  5.431e6    1.558e7    1.756e8   2.8688  32.325
    cmf_ape     V_s x V_l  5.431e6    1.558e7    1.756e8   2.8688  32.325
    mix         V_s x V_l  5.431e6    1.558e7    1.756e8   2.8688  32.325

V_s x V_l is the only genuinely quadratic class in the campaign, and it is
the reason this job exists: gen_bench_ll.py's mf_sl (W_s x V_l) already
validates the 2.8688 law for a couple of seconds of contraction, because the
12-versus-1536 constant cancels in the ratio, but nothing else can supply
svlv's ABSOLUTE cost. mf_ls repeats the A x D class (2.0119 / 8.334) that
mf_ls_ww measures in the other job, so it is a cross-check rather than a new
law -- it is here because production needs the field, not for its ratio.

Only emf_sloop is emitted. Production runs three extended meson fields
(strange, light and charm loops); all three take the same legs, the same
types and the same gamma families, and consume the loop as a precomputed
PropagatorField, so their contraction cost is identical to within run-to-run
noise. Multiply emf_sloop by three rather than paying for it twice more. The
charm loop needs charm vectors that this ensemble has no valence action for,
so its production cost is an open question separate from the contraction.

Schedule (module order == schedule order, and Hadrons is given the file):

    a2a_s_v                 V_s
    noise_s, a2a_s_w        W_s
    gauge, gauge_APE
    loop_s                  strange A2A loop
    noise_l, a2a_l_w        W_l
    mf_ls
    a2a_l_v                 V_l
    emf_sloop, cmf_ape, mix

The two light loads are deliberately split around mf_ls, and it is worth
13 TB. VirtualMachine::makeGarbageSchedule frees an object at the schedule
slot of its LAST consumer, and mf_ls is the only consumer of W_l in this job,
so loading V_l afterwards means W_l is released before V_l allocates. Loading
both together instead would hold 13.0 TB of W_l alongside V_l for the whole
back half of the job for no reason. The same argument put W_s before the
strange loop: loop_s is its last consumer, so it is gone before W_l arrives.

Peak memory is therefore V_s + V_l + loop_s, reached at emf_sloop and held to
the end:

                V_s      W_s      W_l      V_l    loop_s     peak
    h=1        9.89     0.077    12.96    22.78     0.077    32.7 TB
    h=2       19.79     0.155    13.04    32.67     0.077    52.5 TB

at 6.44 GB per ComplexD field on 64^3 x 128. For comparison gen_bench_ll.py
peaks at 35.8 and 45.8 TB, so svlv at h=2 is the high-water mark of the whole
campaign -- and would be 65.6 TB without the split above.

The loop is a PropagatorField, 144 complex per site, so 77.3 GB flat and
independent of the hit count. loop_s takes nLow = 0 because the strange
flavour has no low modes; the light loop carries N_LOW = 2000 and so is a
genuinely different cost. It is NOT measured here, because timing it would
mean holding W_l and V_l simultaneously, which is exactly what the ordering
above avoids. gen_bench_ll.py already holds both for mf_ll and mf_pi, so that
is where a light-loop timing costs nothing.

timeSliceIO is on for all four fields, and here it is a memory switch as much
as an IO one. EMF and CMOF size their output buffer

    Vector<HADRONS_A2AM_IO_TYPE> mBuf; mBuf.resize(ntOut*N_i*N_j);

where ntOut is nt without it and nt/P_t with it. Per rank, for V_s x V_l:

                    no tsIO      tsIO at P_t = 8
    h=1             11.12 GB           1.39 GB
    h=2             31.91 GB           3.99 GB

which is the correction to the old note in this file claiming mBuf sits at
nt*N_i*N_j and is the tiling forcing function. It is not: with timeSliceIO
the vectors are, by three orders of magnitude.

The IO argument points the same way. Without timeSliceIO, EMF and CMOF write
every file from grid->ThisRank() == 0, so all of it goes through one rank on
one node; mix is worse still under A2AMesonField's own rule, since its
writers are spread over nmom*ngamma and it has one of each. Output per field:

                    files       h=1        h=2
    mf_ls           7           44.3 GB    89.1 GB     7 momenta, 1 gamma
    emf_sloop       20         222.5 GB   638.2 GB     4 types x 5 families
    cmf_ape         4           44.5 GB   127.6 GB     2 parities x 2 orthogs
    mix             1           11.1 GB    31.9 GB     1 momentum, 1 gamma
    total                      322.4 GB   886.8 GB

With timeSliceIO each of those becomes nt files of 1/nt the size, spread over
min(nt, nRank) writers. Budget scratch for roughly 325 GB at h=1 and 890 GB
at h=2, and note that emf_sloop alone is about 70 percent of it.

Unsmeared throughout. The smearing cost is a property of the array size and
is already measured in gen_bench_ll.py, which smears three arrays; repeating
it here would add a covariant-smear pass over V_l and V_s, the two largest
objects in the campaign, to measure a number already in hand.

Gamma families: config.EMF_GAMMA_FAMILIES is the five-family LEFT/WET BSM
basis. gammas1 and gammas2 are PAIRED by index inside the module, not crossed
-- nFiles = types * gammas1.size() -- so five families give 5 outputs per
type, not 25. par.postCG.xml used six, adding SigmaMUNUGamma5; if that comes
back, EMF cost and output both scale by 6/5.

Output paths carry the config.TMP_OUTPUT token for the submission script to
substitute with the real (NVMe) directory; the schedule path is absolute.
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

# Mode-sweep blocking inside A2ALoopNew: how many mode views are open on the
# device at once, independent of how many modes are resident on the host.
LOOP_BLOCK = 50

# Contraction types, as in par.postCG.xml.
EMF_TYPES = "0 1 2 3"


def build_job(hits):
    hits = list(hits)
    tag = "".join(f"h{h}" for h in hits)
    run_id = f"bench.svlv.{tag}"
    job = Job(run_id, schedule_file=config.schedule_file(run_id),
              graph_file=config.GRAPH)
    pool = VectorPool(job)

    # cacheBlock = block: the GPU path is fastest with the SumRing reduction
    # untiled, one tile spanning the whole block (see config.py).
    block = config.BLOCK_STRANGE_LEG

    sv = pool.combined("s", "v", hits)
    sw = pool.combined("s", "w", hits)

    job.add(M.load_nersc("gauge", config.GAUGE_FILE))
    job.add(M.ape_smear("gauge_APE", "gauge", config.APE_ALPHA, config.APE_N,
                        config.ORTHOG_AXIS))

    loop_s = f"loop_s_{tag}"
    job.add(M.a2a_loop_new(loop_s, left=sv, right=sw, n_low=0,
                           block=LOOP_BLOCK))
    # W_s is dead here.

    lw = pool.combined("l", "w", hits)

    mf_ls = f"mf_ls_{tag}"
    job.add(M.a2a_meson_field(
        mf_ls, block, block, lw, sv,
        f"{config.TMP_OUTPUT}/{mf_ls}", config.GAMMA5, config.KAON_MOM,
        time_slice_io=True))
    # W_l is dead here, and V_l is loaded next rather than above so that the
    # two never coexist -- see the schedule note in the docstring.

    lv = pool.combined("l", "v", hits)

    emf_name = f"emf_sloop_{tag}"
    job.add(M.a2a_extended_meson_field(
        emf_name, block, block, EMF_TYPES, left=sv, right=lv,
        output=f"{config.TMP_OUTPUT}/{emf_name}",
        gammas1=config.EMF_GAMMA_FAMILIES, gammas2=config.EMF_GAMMA_FAMILIES,
        loop=loop_s, time_slice_io=True))

    cmf_name = f"cmf_ape_{tag}"
    job.add(M.a2a_chromomagnetic_operator_field(
        cmf_name, block, block, config.CMO_PARITIES, sv, lv, "gauge_APE",
        f"{config.TMP_OUTPUT}/{cmf_name}", config.CMO_IF_ORTHOGS,
        time_slice_io=True))

    # Subtraction operator: the lower-dimensional operator that mixes under
    # renormalization. Same legs, one gamma, zero momentum.
    mix_name = f"mix_{tag}"
    job.add(M.a2a_meson_field(
        mix_name, block, block, sv, lv,
        f"{config.TMP_OUTPUT}/{mix_name}", config.IDENTITY, config.ZERO_MOM,
        time_slice_io=True))

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "bench_svlv"
    n = 0
    for hits in HIT_SETS:
        job, run_id = build_job(hits)
        job.write(out_dir / f"par.{run_id}.xml",
                  out_dir / f"schedule.{run_id}.txt")
        n += 1
    print(f"wrote {n} svlv benchmark jobs to {out_dir}")


if __name__ == "__main__":
    main()
