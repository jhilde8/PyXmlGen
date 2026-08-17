"""
Standalone batched multi-hit job for mf_ll (light-w, light-v, Identity,
config.SIGMA_MOM -- the sigma meson field) alone -- one of the three
fields bundled together in gen_mf_stable_multihit.py, provided here
individually for cases that only need this one. Defaults to 2 hits
(matching that bundle); pass a different `hits` list to build_job() to
regenerate at a different truncation.
"""
from pathlib import Path

import config
import modules as M
from hadrons_xml import Job
from vector_pool import VectorPool

DEFAULT_HITS = list(range(2))


def build_job(hits=None):
    hits = list(hits) if hits is not None else DEFAULT_HITS
    run_id = "mf-ll-multihit-" + "".join(f"h{h}" for h in hits)
    job = Job(run_id)
    pool = VectorPool(job)

    lw = pool.combined("l", "w", hits)
    lv = pool.combined("l", "v", hits)

    job.add(M.load_nersc("gauge", config.GAUGE_FILE))
    job.add(M.ape_smear("gauge_APE", "gauge", config.APE_ALPHA, config.APE_N,
                         config.ORTHOG_AXIS))

    name = "mf_ll_multihit"
    job.add(M.a2a_meson_field(
        name, config.BLOCK_LIGHT_LIGHT, config.CACHE_BLOCK_MF,
        lw, lv, f"mf/{name}", config.IDENTITY, config.SIGMA_MOM))

    for width_tag, alpha, N in config.SMEAR_WIDTHS:
        slw_name = f"a2a_l_w_multihit_{width_tag}"
        job.add(M.a2a_covariant_smear(
            slw_name, a2a_vectors=lw, gauge="gauge_APE", alpha=alpha, N=N,
            orthog_axis=config.ORTHOG_AXIS, output="", multi_file=False))
        slv_name = f"a2a_l_v_multihit_{width_tag}"
        job.add(M.a2a_covariant_smear(
            slv_name, a2a_vectors=lv, gauge="gauge_APE", alpha=alpha, N=N,
            orthog_axis=config.ORTHOG_AXIS, output="", multi_file=False))

        name = f"mf_ll_multihit_{width_tag}"
        job.add(M.a2a_meson_field(
            name, config.BLOCK_LIGHT_LIGHT, config.CACHE_BLOCK_MF,
            slw_name, slv_name, f"mf/{name}", config.IDENTITY, config.SIGMA_MOM))

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "pion_sigma"
    job, run_id = build_job()
    job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
    print(f"wrote 1 mf_ll multihit job ({len(DEFAULT_HITS)} hits) to {out_dir}")


if __name__ == "__main__":
    main()
