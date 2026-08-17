"""
Batched multi-hit EMF generator: one job per loop-hit index h_loop, with
the external legs (light V, strange V) fully batched over all N_HIT hits
in a single LoadCombinedA2AVecs array each, producing the full N_HIT x
N_HIT meson field block in one GEMM instead of one job per (hl, hs) pair.

The quark loop's V leg is a MUtilities::VectorPackRefSlice view into the
batched external array when that flavor's V data is already resident
there (light, strange -- h_loop always falls within the batched hit
range), so no re-load. The loop's W leg (and the loop's V leg for
charm, which has no external-leg array to reference) is always a fresh
single-hit combined() load, referenced through the same pointer-pack
module for a uniform LoopPropagatorPtr call in every case -- see
VectorPool.loop_leg().

This is the batched sibling of gen_emf.py, not a replacement for it --
gen_emf.py's per-(hl,hs,h_loop) jobs stay available for doing an
individual hit pair or filling in a partial hit-count increase later.
"""
from pathlib import Path

import config
import modules as M
from hadrons_xml import Job
from vector_pool import VectorPool

LOOP_FLAVORS = ["l", "s", "c"]
HITS = list(range(config.N_HIT))


def build_job(h_loop):
    run_id = f"emf-multihit-h{h_loop}"
    job = Job(run_id)
    pool = VectorPool(job)

    sv = pool.combined("s", "v", HITS)
    lv = pool.combined("l", "v", HITS)

    for loop_flavor in LOOP_FLAVORS:
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
        job, run_id = build_job(h_loop)
        job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
        n += 1
    print(f"wrote {n} multi-hit emf jobs to {out_dir}")


if __name__ == "__main__":
    main()
