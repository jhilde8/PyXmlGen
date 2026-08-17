"""
Same batched multi-hit EMF structure as gen_emf_multihit.py -- external
legs (light V, strange V) fully batched over all N_HIT hits in a single
LoadCombinedA2AVecs array each -- but one job per (h_loop, loop_flavor)
instead of bundling all three loop flavors into one job. Use this
variant if a merged three-flavor job's runtime turns out to be too long
for one allocation.

Each job reloads sv/lv independently (jobs are separate Hadrons runs, so
there's no way to share that IO across them) -- that's the cost of
splitting by flavor, traded against shorter individual job runtimes.
"""
from pathlib import Path

import config
import modules as M
from hadrons_xml import Job
from vector_pool import VectorPool

LOOP_FLAVORS = ["l", "s", "c"]
HITS = list(range(config.N_HIT))


def build_job(h_loop, loop_flavor):
    run_id = f"emf-multihit-h{h_loop}-{loop_flavor}loop"
    job = Job(run_id)
    pool = VectorPool(job)

    sv = pool.combined("s", "v", HITS)
    lv = pool.combined("l", "v", HITS)

    loop_v = pool.loop_leg(loop_flavor, h_loop, "v")
    loop_w = pool.loop_leg(loop_flavor, h_loop, "w")

    name = f"{loop_flavor}loop{h_loop}_svlv"
    job.add(M.a2a_extended_meson_field(
        name, config.BLOCK_STRANGE_LEG, config.CACHE_BLOCK_EMF_CMOF,
        "0 1 2 3", sv, lv, loop_v, loop_w, f"emf/{name}",
        config.EMF_GAMMA_FAMILIES, config.EMF_GAMMA_FAMILIES))

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "emf"
    n = 0
    for h_loop in HITS:
        for loop_flavor in LOOP_FLAVORS:
            job, run_id = build_job(h_loop, loop_flavor)
            job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
            n += 1
    print(f"wrote {n} per-flavor multi-hit emf jobs to {out_dir}")


if __name__ == "__main__":
    main()
