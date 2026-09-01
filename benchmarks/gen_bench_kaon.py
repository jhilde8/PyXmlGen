"""
Kaon meson field timing job (groups 1 and 2): mf_ls, mf_sl and mf_ls_ww in
one job, unsmeared.

All three fields in one job on purpose. Between them they need exactly four
arrays -- W_l, V_s, W_s, V_l -- and mf_ls_ww's two legs are both already
resident for the other two fields, so one set of loads buys three
contractions. The module timers separate the fields afterwards, so this
measures each component once regardless of how production ends up splitting
the work into jobs.

Field definitions are production's, from gen_kaon.py's KAON_TYPES:

    mf_ls      Gamma5     W_l x V_s    config.KAON_MOM
    mf_sl      Gamma5     W_s x V_l    config.KAON_MOM
    mf_ls_ww   Identity   W_l x W_s    config.KAON_MOM

Note ww is Identity, not Gamma5.

Cost model. With the dense W representation the leg sizes are

    W_l = N_LOW + N_SC*h        V_l = N_LOW + N_HIGH*h
    W_s = N_SC*h                V_s = N_HIGH*h

and contraction cost goes as N_l*N_r, which is nowhere near quadratic in h:

    field      legs       h=1        h=2        h=4        h=8     h2/h1
    mf_ls      A x D    3.090e6    6.218e6    1.258e7    2.575e7   2.0119
    mf_ls_ww   A x C    2.414e4    4.858e4    9.830e4    2.012e5   2.0119
    mf_sl      C x B    4.243e4    1.217e5    3.909e5    1.372e6   2.8688
      (A = W_l, B = V_l, C = W_s, D = V_s)

The h2/h1 column is the acceptance criterion, against 4.0 if the cost were
quadratic in hits. Two things are worth noticing:

  - mf_ls and mf_ls_ww share a scaling law to five digits, because both are
    W_l times something purely linear in h. mf_sl's 2.8688 is a different
    law, and it is the same one svlv obeys (V_s x V_l has the identical
    constant-plus-linear structure), which is why this job validates the
    class that gen_bench_svlv.py's single h=1 point cannot.
  - mf_ls_ww is ~130x cheaper to contract than mf_ls and ~300x cheaper than
    the pion. Its job cost is essentially the W_l load, which is flat in h,
    so "effectively free at 8 hits" should show up directly here.

Unsmeared. Production makes these fields both ways (gen_kaon.py emits an
unsmeared set and one per width in config.SMEAR_WIDTHS), the contraction is
smearing-independent, and gen_bench_ll.py already measures the smear cost as
a clean linear law -- 15.0 ms per field on 256 nodes, so adding smearing here
would cost 15.0 ms * (W_l + V_s + W_s + V_l) and measure nothing new. To turn
it on, wrap each array the way gen_bench_ll.py does and add the gauge/APE
pair back.

Module order is the memory order: Hadrons frees an environment object once
nothing later in the schedule references it, so V_s is released by the time
V_l loads. Peak is W_l + W_s + V_s = 3560 fields at h=1 and 5120 at h=2
(~5.6 TB and ~8.0 TB at single precision), well under the light-light job.

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


def build_job(hits):
    hits = list(hits)
    tag = "".join(f"h{h}" for h in hits)
    run_id = f"bench.kaon.{tag}"
    job = Job(run_id, schedule_file=config.schedule_file(run_id),
              graph_file=config.graph_file(run_id))
    pool = VectorPool(job)

    # cacheBlock = block: the GPU path is fastest with the SumRing reduction
    # untiled, one tile spanning the whole block (see config.py). Every field
    # here has a strange leg.
    block = config.BLOCK_STRANGE_LEG

    def mf(name, left, right, gammas):
        job.add(M.a2a_meson_field(
            name, block, block, left, right,
            f"{config.TMP_OUTPUT}/{name}", gammas, config.KAON_MOM))

    # W_l first: it is the left leg of both mf_ls and mf_ls_ww, so it has to
    # outlive V_s.
    lw = pool.combined("l", "w", hits)

    sv = pool.combined("s", "v", hits)
    mf(f"mf_ls_{tag}", lw, sv, config.GAMMA5)
    # V_s is dead here -- nothing below references it.

    sw = pool.combined("s", "w", hits)
    mf(f"mf_ls_ww_{tag}", lw, sw, config.IDENTITY)
    # W_l is dead here.

    lv = pool.combined("l", "v", hits)
    mf(f"mf_sl_{tag}", sw, lv, config.GAMMA5)

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "bench_kaon"
    n = 0
    for hits in HIT_SETS:
        job, run_id = build_job(hits)
        job.write(out_dir / f"par.{run_id}.xml",
                  out_dir / f"schedule.{run_id}.txt")
        n += 1
    print(f"wrote {n} kaon benchmark jobs to {out_dir}")


if __name__ == "__main__":
    main()
