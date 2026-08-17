"""
Generate one XML job per (hl, hs, h_loop) combination, covering all three
loop flavors (light, strange, charm) in a single job: left = strange-v @ hs,
right = light-v @ hl (fixed external legs, matching Frontier/MF/par.postCG.xml's
emf_sloop/emf_lloop), each paired with a quark loop of flavor l/s/c built
from that flavor's v and w at h_loop.

The three loop flavors share the same external (hl, hs) legs, so bundling
them into one job means the external IO only happens once instead of once
per flavor. Module order interleaves IO and EMF per flavor rather than
front-loading all IO first: external leg IO, then for each flavor in turn,
that flavor's loop-vector IO immediately followed by its EMF module. There's
no smearing or multi-width chaining downstream of these vectors (unlike
kaon/pion-sigma), so each flavor's loop vectors are freed right after its
own EMF module runs, before the next flavor's are even loaded -- interleaving
is both the simplest structure and already memory-optimal here.

The vv subtraction/mix operator used to live in this job too, but since it
doesn't depend on h_loop at all, computing it here would redundantly
recompute the same mix field once per loop hit (N_HIT times over). It lives
in the kaon job instead (gen_kaon.py), which already loads exactly the
strange-v/light-v pair it needs for a given (hl, hs).

No smearing and no gauge field are used here, matching the example.
"""
from itertools import product
from pathlib import Path

import config
import modules as M
from hadrons_xml import Job
from vector_pool import VectorPool

LOOP_FLAVORS = ["l", "s", "c"]


def build_job(hl, hs, h_loop):
    run_id = f"emf-h{h_loop}-l{hl}s{hs}"
    job = Job(run_id)
    pool = VectorPool(job)

    # External leg IO -- shared by all three loop flavors below.
    sv = pool.base("s", hs, "v")
    lv = pool.base("l", hl, "v")

    for loop_flavor in LOOP_FLAVORS:
        loop_v = pool.base(loop_flavor, h_loop, "v")
        loop_w = pool.base(loop_flavor, h_loop, "w")

        name = f"{loop_flavor}loop{h_loop}_sv{hs}lv{hl}"
        job.add(M.a2a_extended_meson_field(
            name, config.BLOCK_STRANGE_LEG, config.CACHE_BLOCK_EMF_CMOF,
            "0 1 2 3", sv, lv, loop_v, loop_w, f"emf/{name}",
            config.EMF_GAMMA_FAMILIES, config.EMF_GAMMA_FAMILIES))

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "emf"
    n = 0
    for hl, hs, h_loop in product(range(config.N_HIT), range(config.N_HIT),
                                   range(config.N_HIT)):
        job, run_id = build_job(hl, hs, h_loop)
        job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
        n += 1
    print(f"wrote {n} emf jobs to {out_dir}")


if __name__ == "__main__":
    main()
