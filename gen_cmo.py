"""
Generate one XML job per (hl, hs) light/strange hit pair for the
chromo-magnetic operator field: unsmeared (cmf) and APE-smeared-gauge
(cmfAPE) versions together, both using left = strange-v @ hs,
right = light-v @ hl -- same hit convention as the kaon job (CMO is "just
another meson field with a particular operator" per project discussion).

block = 128 (strange leg present), cacheBlock = 32 (matches EMF).
"""
from itertools import product
from pathlib import Path

import config
import modules as M
from hadrons_xml import Job
from vector_pool import VectorPool


def build_job(hl, hs):
    run_id = f"cmo-l{hl}s{hs}"
    job = Job(run_id)
    pool = VectorPool(job)

    # 1. IO
    sv = pool.base("s", hs, "v")
    lv = pool.base("l", hl, "v")

    # 2. gauge (tiny; CMO doesn't smear A2A vectors at all, only the gauge
    # field, so there's no unsmeared-vector-lifetime concern here)
    job.add(M.load_nersc("gauge", config.GAUGE_FILE))
    job.add(M.ape_smear("gauge_APE", "gauge", config.APE_ALPHA, config.APE_N,
                         config.ORTHOG_AXIS))

    for tag, gauge_name in [("cmf", "gauge"), ("cmfAPE", "gauge_APE")]:
        name = f"{tag}_s{hs}l{hl}"
        job.add(M.a2a_chromomagnetic_operator_field(
            name, config.BLOCK_STRANGE_LEG, config.CACHE_BLOCK_EMF_CMOF,
            config.CMO_PARITIES, sv, lv, gauge_name, f"emf/{name}",
            config.CMO_IF_ORTHOGS))

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "cmo"
    for hl, hs in product(range(config.N_HIT), range(config.N_HIT)):
        job, run_id = build_job(hl, hs)
        job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
    print(f"wrote {config.N_HIT * config.N_HIT} cmo jobs to {out_dir}")


if __name__ == "__main__":
    main()
