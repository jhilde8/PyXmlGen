"""
Batched multi-hit CMOF generator: light/strange V vectors fully batched
over all N_HIT hits in a single LoadCombinedA2AVecs array each, producing
the full N_HIT x N_HIT block in one A2AChromoMagneticOperatorField call
per gauge variant (cmf: bare gauge, cmfAPE: APE-smeared gauge) -- a
single job covers every hit combination, since CMOF has no loop vector
and no per-hit indexing left once both legs are batched.

The vv subtraction/mix operator is included here too, rather than in its
own job -- it only needs sv/lv, the same two arrays CMOF already loads,
so bundling it in avoids a third full-hit-count sv/lv load elsewhere.

Sibling of gen_cmo.py, which stays available for an individual hit pair.
"""
from pathlib import Path

import config
import modules as M
from hadrons_xml import Job
from vector_pool import VectorPool

HITS = list(range(config.N_HIT))


def build_job():
    run_id = "cmo-multihit"
    job = Job(run_id)
    pool = VectorPool(job)

    sv = pool.combined("s", "v", HITS)
    lv = pool.combined("l", "v", HITS)

    job.add(M.load_nersc("gauge", config.GAUGE_FILE))
    job.add(M.ape_smear("gauge_APE", "gauge", config.APE_ALPHA, config.APE_N,
                         config.ORTHOG_AXIS))

    for tag, gauge_name in [("cmf", "gauge"), ("cmfAPE", "gauge_APE")]:
        name = f"{tag}_multihit"
        job.add(M.a2a_chromomagnetic_operator_field(
            name, config.BLOCK_STRANGE_LEG, config.CACHE_BLOCK_EMF_CMOF,
            config.CMO_PARITIES, sv, lv, gauge_name, f"emf/{name}",
            config.CMO_IF_ORTHOGS))

    mix_name = "mix_multihit"
    job.add(M.a2a_meson_field(
        mix_name, config.BLOCK_STRANGE_LEG, config.CACHE_BLOCK_MF,
        sv, lv, f"emf/{mix_name}", config.IDENTITY, config.ZERO_MOM))

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "cmo"
    job, run_id = build_job()
    job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
    print(f"wrote 1 multi-hit cmo job to {out_dir}")


if __name__ == "__main__":
    main()
