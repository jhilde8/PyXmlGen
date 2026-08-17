"""
Standalone batched multi-hit job for mf_ls_ww (light-w, strange-w,
Identity, config.KAON_MOM) -- kept separate from the well-behaved bundle
(gen_mf_stable_multihit.py) since it needs the full hit count. Defaults
to all N_HIT hits; pass a different `hits` list to build_job() to
regenerate at a different truncation.
"""
from pathlib import Path

import config
import modules as M
from hadrons_xml import Job
from vector_pool import VectorPool

DEFAULT_HITS = list(range(config.N_HIT))


def build_job(hits=None):
    hits = list(hits) if hits is not None else DEFAULT_HITS
    run_id = "mf-ls-ww-multihit-" + "".join(f"h{h}" for h in hits)
    job = Job(run_id)
    pool = VectorPool(job)

    lw = pool.combined("l", "w", hits)
    sw = pool.combined("s", "w", hits)

    job.add(M.load_nersc("gauge", config.GAUGE_FILE))
    job.add(M.ape_smear("gauge_APE", "gauge", config.APE_ALPHA, config.APE_N,
                         config.ORTHOG_AXIS))

    name = "mf_ls_ww_multihit"
    job.add(M.a2a_meson_field(
        name, config.BLOCK_STRANGE_LEG, config.CACHE_BLOCK_MF,
        lw, sw, f"mf/{name}", config.IDENTITY, config.KAON_MOM))

    for width_tag, alpha, N in config.SMEAR_WIDTHS:
        slw_name = f"a2a_l_w_multihit_{width_tag}"
        job.add(M.a2a_covariant_smear(
            slw_name, a2a_vectors=lw, gauge="gauge_APE", alpha=alpha, N=N,
            orthog_axis=config.ORTHOG_AXIS, output="", multi_file=False))
        ssw_name = f"a2a_s_w_multihit_{width_tag}"
        job.add(M.a2a_covariant_smear(
            ssw_name, a2a_vectors=sw, gauge="gauge_APE", alpha=alpha, N=N,
            orthog_axis=config.ORTHOG_AXIS, output="", multi_file=False))

        name = f"mf_ls_ww_multihit_{width_tag}"
        job.add(M.a2a_meson_field(
            name, config.BLOCK_STRANGE_LEG, config.CACHE_BLOCK_MF,
            slw_name, ssw_name, f"mf/{name}", config.IDENTITY, config.KAON_MOM))

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "kaon"
    job, run_id = build_job()
    job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
    print(f"wrote 1 mf_ls_ww multihit job ({len(DEFAULT_HITS)} hits) to {out_dir}")


if __name__ == "__main__":
    main()
