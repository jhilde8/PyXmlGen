"""
Generate one XML job per (hL, hR) light hit pair covering both sigma
(mf_ll, Identity, momenta per config.SIGMA_MOM, unsmeared + the single
smearing width) and pion (mf_pi, Gamma5, 27 momenta, at that same single
smearing width) -- merged into one job because both are light-light and
would otherwise load/smear the same light vectors twice.

left = light-w @ hL, right = light-v @ hR (matches mf_ll in
Frontier/MF/par.postCG.xml), block = 221 (light-light), cacheBlock = 16.

Module order: IO, then gauge, then the unsmeared sigma field, then one
block per smearing width (see gen_kaon.py for why -- same reasoning here,
just fewer vectors since both legs are light).
"""
from itertools import product
from pathlib import Path

import config
import modules as M
from hadrons_xml import Job
from vector_pool import VectorPool


def build_job(hL, hR):
    run_id = f"pionsigma-h{hL}h{hR}"
    job = Job(run_id)
    pool = VectorPool(job)

    # 1. IO
    lw = pool.base("l", hL, "w")
    lv = pool.base("l", hR, "v")

    # 2. gauge (tiny, right before it's needed for smearing)
    job.add(M.load_nersc("gauge", config.GAUGE_FILE))
    job.add(M.ape_smear("gauge_APE", "gauge", config.APE_ALPHA, config.APE_N,
                         config.ORTHOG_AXIS))

    def sigma(width_tag=None):
        if width_tag is None:
            left_name, right_name = lw, lv
            name = f"mf_l{hL}l{hR}"
        else:
            alpha, N = WIDTH_PARAMS[width_tag]
            left_name = pool.smeared("l", hL, "w", width_tag, alpha, N)
            right_name = pool.smeared("l", hR, "v", width_tag, alpha, N)
            name = f"mf_l{hL}l{hR}_{width_tag}"
        job.add(M.a2a_meson_field(
            name, config.BLOCK_LIGHT_LIGHT, config.CACHE_BLOCK_MF,
            left_name, right_name, f"mf/{name}", config.IDENTITY, config.SIGMA_MOM))

    def pion(width_tag):
        alpha, N = WIDTH_PARAMS[width_tag]
        left_name = pool.smeared("l", hL, "w", width_tag, alpha, N)
        right_name = pool.smeared("l", hR, "v", width_tag, alpha, N)
        name = f"mf_pi_l{hL}l{hR}_{width_tag}"
        job.add(M.a2a_meson_field(
            name, config.BLOCK_LIGHT_LIGHT, config.CACHE_BLOCK_MF,
            left_name, right_name, f"mf/{name}", config.GAMMA5, config.PION_MOM))

    WIDTH_PARAMS = {tag: (alpha, N) for tag, alpha, N in config.SMEAR_WIDTHS}

    sigma()  # unsmeared
    for width_tag, alpha, N in config.SMEAR_WIDTHS:
        sigma(width_tag)
        pion(width_tag)

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "pion_sigma"
    for hL, hR in product(range(config.N_HIT), range(config.N_HIT)):
        job, run_id = build_job(hL, hR)
        job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
    print(f"wrote {config.N_HIT * config.N_HIT} pion_sigma jobs to {out_dir}")


if __name__ == "__main__":
    main()
