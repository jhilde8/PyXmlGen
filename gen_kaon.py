"""
Generate one XML job per (hl, hs) light/strange hit pair for the kaon
sector: mf_ls, mf_sl, mf_ls_ww (unsmeared and at the single smearing width
in config.SMEAR_WIDTHS, momenta per config.KAON_MOM), plus the vv
subtraction/mix operator (zero momentum only, matching
Frontier/MF/par.postCG.xml's mf_sl_vv_mix).

Hit indices are per-flavor, not per-role: light's v and w both use hl,
strange's v and w both use hs (a "hit" is one stochastic source, shared by
its V and W). left/right/vw assignment per type is copied directly from
Frontier/MF/par.postCG.xml's mf_ls / mf_sl / mf_ls_ww modules:

    ls    (Gamma5):    left = light-w @ hl, right = strange-v @ hs
    sl    (Gamma5):    left = strange-w @ hs, right = light-v @ hl
    ls_ww (Identity):  left = light-w @ hl, right = strange-w @ hs

Module order matters for memory: Hadrons frees an environment object once
nothing later in the schedule references it. The unsmeared A2A vectors are
huge, so every module that touches them (IO, the unsmeared meson fields,
mix, and hit-(0,0)'s coarse-grid sparsening) is grouped up front, followed
by the smearing block, which is the last thing to reference them.
"""
from itertools import product
from pathlib import Path

import config
import modules as M
from hadrons_xml import Job
from vector_pool import VectorPool

# (name_suffix, gammas, (left_flavor, left_vw), (right_flavor, right_vw))
KAON_TYPES = [
    ("", config.GAMMA5, ("l", "w"), ("s", "v")),
    ("", config.GAMMA5, ("s", "w"), ("l", "v")),
    ("_ww", config.IDENTITY, ("l", "w"), ("s", "w")),
]


def _hit_for(flavor, hl, hs):
    return hl if flavor == "l" else hs


def build_job(hl, hs):
    run_id = f"kaon-l{hl}s{hs}"
    job = Job(run_id)
    pool = VectorPool(job)

    # 1. IO: every unsmeared vector load/combine up front, so this is the
    #    only place they're built and every later reference is a cache hit.
    lw = pool.base("l", hl, "w")
    lv = pool.base("l", hl, "v")
    sv = pool.base("s", hs, "v")
    sw = pool.base("s", hs, "w")

    # 2. gauge field (tiny -- placed here, right before it's needed for
    #    vector smearing below).
    job.add(M.load_nersc("gauge", config.GAUGE_FILE))
    job.add(M.ape_smear("gauge_APE", "gauge", config.APE_ALPHA, config.APE_N,
                         config.ORTHOG_AXIS))

    # 3. Everything that consumes the unsmeared vectors: the 3 kaon types,
    #    mix, and (for hit (0,0) only) coarse-grid sparsening. Once this
    #    block ends, lw/lv/sv/sw are only referenced by the width-1.5 and
    #    width-3 smearing modules below.
    vecs = {("l", "w"): lw, ("l", "v"): lv, ("s", "v"): sv, ("s", "w"): sw}

    def emit_unsmeared(suffix, gammas, left, right):
        (lf, lvw), (rf, rvw) = left, right
        lhit, rhit = _hit_for(lf, hl, hs), _hit_for(rf, hl, hs)
        name = f"mf_{lf}{lhit}{rf}{rhit}{suffix}"
        job.add(M.a2a_meson_field(
            name, config.BLOCK_STRANGE_LEG, config.CACHE_BLOCK_MF,
            vecs[(lf, lvw)], vecs[(rf, rvw)], f"mf/{name}", gammas, config.KAON_MOM))

    for suffix, gammas, left, right in KAON_TYPES:
        emit_unsmeared(suffix, gammas, left, right)

    mix_name = f"mix_s{hs}l{hl}"
    job.add(M.a2a_meson_field(
        mix_name, config.BLOCK_STRANGE_LEG, config.CACHE_BLOCK_MF,
        sv, lv, f"emf/{mix_name}", config.IDENTITY, config.ZERO_MOM))

    if hl == 0 and hs == 0:
        job.add(M.a2a_coarse_grid("a2a_l_v000", 221, lv,
                                   config.COARSE_BLOCK_SIZE, config.COARSE_OFFSETS,
                                   "coarse/a2a_l_v000"))
        job.add(M.a2a_coarse_grid("a2a_l_w000", 221, lw,
                                   config.COARSE_BLOCK_SIZE, config.COARSE_OFFSETS,
                                   "coarse/a2a_l_w000"))
        job.add(M.a2a_coarse_grid("a2a_s_v000", 128, sv,
                                   config.COARSE_BLOCK_SIZE, config.COARSE_OFFSETS,
                                   "coarse/a2a_s_v000"))
        job.add(M.a2a_coarse_grid("a2a_s_w000", 128, sw,
                                   config.COARSE_BLOCK_SIZE, config.COARSE_OFFSETS,
                                   "coarse/a2a_s_w000"))

    # 4/5. One block per smearing width: smear all 4 base vectors first,
    # then every meson field that depends on that width.
    for width_tag, alpha, N in config.SMEAR_WIDTHS:
        svecs = {
            (lf, lvw): pool.smeared(lf, _hit_for(lf, hl, hs), lvw, width_tag, alpha, N)
            for lf, lvw in [("l", "w"), ("l", "v"), ("s", "v"), ("s", "w")]
        }

        for suffix, gammas, left, right in KAON_TYPES:
            (lf, lvw), (rf, rvw) = left, right
            lhit, rhit = _hit_for(lf, hl, hs), _hit_for(rf, hl, hs)
            name = f"mf_{lf}{lhit}{rf}{rhit}{suffix}_{width_tag}"
            job.add(M.a2a_meson_field(
                name, config.BLOCK_STRANGE_LEG, config.CACHE_BLOCK_MF,
                svecs[(lf, lvw)], svecs[(rf, rvw)], f"mf/{name}", gammas, config.KAON_MOM))

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "kaon"
    for hl, hs in product(range(config.N_HIT), range(config.N_HIT)):
        job, run_id = build_job(hl, hs)
        job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
    print(f"wrote {config.N_HIT * config.N_HIT} kaon jobs to {out_dir}")


if __name__ == "__main__":
    main()
