"""
svlv timing job (groups 4 and 5): the A2A self loops, the extended meson
field, the chromo-magnetic operator field and the subtraction operator, all
at h=1 only.

Everything here shares the same external legs, left = V_s and right = V_l,
taken from par.postCG.xml:

    emf_sloop   A2AExtendedMesonField          types "0 1 2 3", loop = s
    emf_lloop   A2AExtendedMesonField          types "0 1 2 3", loop = l
    cmf         A2AChromoMagneticOperatorField gauge
    cmfAPE      A2AChromoMagneticOperatorField gauge_APE
    mix         A2AMesonField, Identity, zero momentum (subtraction operator)

Only emf_sloop is emitted. emf_lloop has identical legs, types and gamma
families, and the loop it consumes is a precomputed PropagatorField either
way, so its cost is the same to within run-to-run noise -- double the
emf_sloop number rather than paying for it. Both A2ALoopNew modules ARE
emitted, because they differ: the light loop carries N_LOW low modes and the
strange loop has none, so their costs are genuinely different and production
needs both.

Why h=1 only. svlv is V_s x V_l, whose h2/h1 ratio is

    (N_HIGH*h) * (N_LOW + N_HIGH*h)  ->  2 * (5072/3536) = 2.8688

which is exactly mf_sl's law -- same constant-plus-linear structure, the 1536
versus 12 cancels in the ratio. gen_bench_kaon.py measures that class at h=1
and h=2 for a few seconds of contraction, so an h=2 svlv run would re-measure
a known law at the highest price in the campaign. The EMF's IO obeys the same
ratio, since it writes nt*N_i*N_j per (type, gamma family). What this job is
for is svlv's ABSOLUTE cost, which nothing else can supply.

Memory is why svlv is the constrained group, and it is not the vectors. Each
EMF and CMOF rank allocates the whole meson field:

    Vector<HADRONS_A2AM_IO_TYPE> mBuf; mBuf.resize(nt*N_i*N_j);

which is nt * (N_HIGH*h) * (N_LOW + N_HIGH*h) * 16 bytes PER RANK -- 11.1 GB
at h=1, 31.9 GB at h=2, 102 GB at h=4. That is the tiling threshold, and it
sits on top of the vector arrays rather than sharing with them. Worth
checking against the reported peak on this run before trusting the h=4 and
h=8 tiling plan.

IO note: EMF and CMOF call makeFileDir (boss rank only) and write under
grid->ThisRank() == 0, so unlike A2AMesonField all of their output goes
through a single rank onto a single node. In-job that is fine and is faster
on node-local NVMe than on Lustre; it is the stage-out afterwards that is
serial from one node.

Gamma families: config.EMF_GAMMA_FAMILIES is the current five-family set
(the LEFT/WET BSM basis, no tensor structures). par.postCG.xml used six,
adding SigmaMUNUGamma5 -- if that comes back, EMF cost scales by 6/5.

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

HIT_SETS = ([0],[0, 1])

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

    job.add(M.load_nersc("gauge", config.GAUGE_FILE))
    job.add(M.ape_smear("gauge_APE", "gauge", config.APE_ALPHA, config.APE_N,
                        config.ORTHOG_AXIS))

    # Loops first, so each flavour's W array is released before the
    # contractions start. V_s and V_l have to survive to the end, being the
    # external legs of everything below.
    sv = pool.combined("s", "v", hits)
    sw = pool.combined("s", "w", hits)
    loop_s = f"loop_s_{tag}"
    job.add(M.a2a_loop_new(loop_s, left=sv, right=sw, n_low=0,
                           block=LOOP_BLOCK))
    # W_s is dead here.

    lv = pool.combined("l", "v", hits)
    lw = pool.combined("l", "w", hits)
    loop_l = f"loop_l_{tag}"
    job.add(M.a2a_loop_new(loop_l, left=lv, right=lw, n_low=config.N_LOW,
                           block=LOOP_BLOCK))
    # W_l is dead here. loop_l has no consumer in this job -- it is emitted
    # for its own timing, since production runs emf_lloop as well.

    emf_name = f"emf_sloop_svlv_{tag}"
    job.add(M.a2a_extended_meson_field(
        emf_name, block, block, EMF_TYPES, left=sv, right=lv,
        output=f"{config.TMP_OUTPUT}/{emf_name}",
        gammas1=config.EMF_GAMMA_FAMILIES, gammas2=config.EMF_GAMMA_FAMILIES,
        loop=loop_s))

    name = f"cmf_{tag}"
    job.add(M.a2a_chromomagnetic_operator_field(
        name, block, block, config.CMO_PARITIES, sv, lv, "gauge_APE",
        f"{config.TMP_OUTPUT}/{name}", config.CMO_IF_ORTHOGS))

    # Subtraction operator: the lower-dimensional operator that mixes under
    # renormalization. Same legs, one gamma, zero momentum.
    mix_name = f"mix_{tag}"
    job.add(M.a2a_meson_field(
        mix_name, block, block, sv, lv,
        f"{config.TMP_OUTPUT}/{mix_name}", config.IDENTITY, config.ZERO_MOM))

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
