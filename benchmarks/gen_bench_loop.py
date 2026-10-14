"""Timing job for the sparsening step, 1 and 2 hits.

Emits the exact structure production/gen_sparsen.py builds -- light and strange
dense W and V loaded per hit and sparsened through A2ACoarseGrid -- at the two
hit counts the rest of the campaign is measured at. It imports that builder
rather than restating it so the thing being timed cannot drift from the thing
being run. (The file keeps its old name; it used to time the loop-and-sparsening
job, before the loops moved to the meson-field job and gen_charm_loop.py.)

The AMA hit and the light low block are left out. Both are fixed costs that do
not scale with the hit count, so with them in, h=1 and h=2 would no longer
isolate the per-hit cost.

What these two points buy: the job is LINEAR in mode count, so unlike the
contraction jobs there is no quadratic term to extrapolate and one measurement
would in principle do. h=1 and h=2 give the per-hit cost and confirm the
linearity, after which higher hit counts are multiplication.

The cost is all IO: a fine-field load and a sparsened write per mode. The load
rate at this node count is the number the scaling table is missing -- every load
measurement so far is at 256 nodes (35.14 ms per field for l_v, 46.63 for s_v)
and this job is sized for 32. The sparsened write is the other half: ~100 MB
per file, ~390 KB per rank here, so per-file overhead should dominate it, and
this is the measurement that decides whether A2ACoarseGrid needs propagator
records.

Outputs go to their own root. The sparsened per-hit directory names carry no
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
from gen_sparsen import build_job

HIT_COUNTS = (1, 2)

# Under bench/ so this cannot land on top of production's output: the sparsened
# per-hit names (s0_v ... s7_v) deliberately mirror the unsparsened ones under
# config.VW_BASE and so carry no hit-count tag of their own.
BASE = str(Path(config.VW_BASE).parent)
BENCH_SPARSE_ROOT = f"{BASE}/bench/sparse"


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "bench_sparsen"
    n = 0
    for n_hit in HIT_COUNTS:
        job, run_id = build_job(n_hit=n_hit, ama=False, low=False,
                                run_id=f"bench.sparsen.h{n_hit}",
                                sparse_root=BENCH_SPARSE_ROOT)
        job.write(out_dir / f"par.{run_id}.xml",
                  out_dir / f"schedule.{run_id}.txt")
        n += 1
    print(f"wrote {n} sparsening benchmark jobs to {out_dir}")


if __name__ == "__main__":
    main()
