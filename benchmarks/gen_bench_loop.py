"""Timing job for the loop-and-sparsening step, 1 and 2 hits.

Emits the exact structure production/gen_loop_sparsen.py builds -- strange and
charm loops from per-hit high-mode loads, chained through A2ALoopNew's
inputLoop, with strange V sparsened along the way -- at the two hit counts the
rest of the campaign is measured at. It imports that builder rather than
restating it so the thing being timed cannot drift from the thing being run.

What these two points buy: the job is LINEAR in mode count, so unlike the
contraction jobs there is no quadratic term to extrapolate and one measurement
would in principle do. h=1 and h=2 give the per-link cost and confirm the
linearity, after which higher hit counts are multiplication. h=1 is also the
only case with a single link per flavour and therefore an empty inputLoop
throughout, so the pair brackets the chain: h=1 without any seeding, h=2 with
one.

The cost here is essentially all IO -- the loop contraction against a dense W
is negligible, and the benchmark logs already show loop_s at tens of
milliseconds against hundreds of seconds of loading. So what these runs are
really measuring is the per-field load rate at this job's node count, which is
the number the scaling table is missing and cannot be borrowed from the
contraction jobs: every load measurement so far is at 256 nodes (35.14 ms per
field for l_v, 46.63 for s_v) and this job is sized for 32. If the rate turns
out near-linear in client count, the 256-node one-call-per-flavour variant costs
the same node-hours, finishes 8x sooner and needs no chain at all -- so this
measurement decides the structure, not just the row.

Outputs go to their own roots. The sparsened per-hit filenames carry no
hit-count tag, so a benchmark writing to the production directory would land
s0_v on top of the real thing.
"""
import sys
from pathlib import Path

# The toolkit (config, modules, hadrons_xml) lives one level up; the job
# structure itself lives in production/.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "production"))

import config
from gen_loop_sparsen import build_job

HIT_COUNTS = (1, 2)

# Separate from production's roots so benchmark output cannot overwrite it.
BENCH_SPARSE_ROOT = "/lustre/orion/phy157/scratch/jhilde/64I/bench/vw_sparse"
BENCH_LOOP_ROOT = "/lustre/orion/phy157/scratch/jhilde/64I/bench/loop"


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "bench_loop"
    n = 0
    for n_hit in HIT_COUNTS:
        job, run_id = build_job(n_hit=n_hit, run_id=f"bench.loop.h{n_hit}",
                                sparse_root=BENCH_SPARSE_ROOT,
                                loop_root=BENCH_LOOP_ROOT)
        job.write(out_dir / f"par.{run_id}.xml",
                  out_dir / f"schedule.{run_id}.txt")
        n += 1
    print(f"wrote {n} loop benchmark jobs to {out_dir}")


if __name__ == "__main__":
    main()
