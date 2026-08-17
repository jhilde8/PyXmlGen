"""
Batched multi-hit job for the three statistically well-behaved meson
fields -- mf_ls, mf_sl (kaon, non-ww), and mf_ll (sigma) -- bundled
together since they share the same 4 vector arrays (light V/W, strange
V/W) and converge well enough to run at a reduced hit count. Defaults to
2 hits; pass a different `hits` list to build_job() to regenerate at a
different truncation.

Each of the 4 batched vector arrays, and its smeared counterpart at the
single configured smearing width, is loaded/smeared once and shared
across whichever of the three meson field types need it -- mf_ls needs
lw+sv, mf_sl needs sw+lv, mf_ll needs lw+lv, so lw and lv are each reused
by two of the three.

mf_ls_ww is deliberately not included here (see gen_mf_ls_ww_multihit.py)
-- it needs the full hit count, so bundling it in here would force the
other three up to that count too.
"""
from pathlib import Path

import config
import modules as M
from hadrons_xml import Job
from vector_pool import VectorPool

DEFAULT_HITS = list(range(2))


def build_job(hits=None):
    hits = list(hits) if hits is not None else DEFAULT_HITS
    run_id = "mf-stable-multihit-" + "".join(f"h{h}" for h in hits)
    job = Job(run_id)
    pool = VectorPool(job)

    lw = pool.combined("l", "w", hits)
    lv = pool.combined("l", "v", hits)
    sv = pool.combined("s", "v", hits)
    sw = pool.combined("s", "w", hits)

    job.add(M.load_nersc("gauge", config.GAUGE_FILE))
    job.add(M.ape_smear("gauge_APE", "gauge", config.APE_ALPHA, config.APE_N,
                         config.ORTHOG_AXIS))

    def emit(name, gammas, mom, block, left, right):
        job.add(M.a2a_meson_field(
            name, block, config.CACHE_BLOCK_MF, left, right, f"mf/{name}", gammas, mom))

    emit("mf_ls_multihit", config.GAMMA5, config.KAON_MOM, config.BLOCK_STRANGE_LEG, lw, sv)
    emit("mf_sl_multihit", config.GAMMA5, config.KAON_MOM, config.BLOCK_STRANGE_LEG, sw, lv)
    emit("mf_ll_multihit", config.IDENTITY, config.SIGMA_MOM, config.BLOCK_LIGHT_LIGHT, lw, lv)

    for width_tag, alpha, N in config.SMEAR_WIDTHS:
        svecs = {}
        for (flavor, vw), base_name in [(("l", "w"), lw), (("l", "v"), lv),
                                         (("s", "v"), sv), (("s", "w"), sw)]:
            name = f"a2a_{flavor}_{vw}_multihit_{width_tag}"
            job.add(M.a2a_covariant_smear(
                name, a2a_vectors=base_name, gauge="gauge_APE", alpha=alpha, N=N,
                orthog_axis=config.ORTHOG_AXIS, output="", multi_file=False))
            svecs[(flavor, vw)] = name

        emit(f"mf_ls_multihit_{width_tag}", config.GAMMA5, config.KAON_MOM,
             config.BLOCK_STRANGE_LEG, svecs[("l", "w")], svecs[("s", "v")])
        emit(f"mf_sl_multihit_{width_tag}", config.GAMMA5, config.KAON_MOM,
             config.BLOCK_STRANGE_LEG, svecs[("s", "w")], svecs[("l", "v")])
        emit(f"mf_ll_multihit_{width_tag}", config.IDENTITY, config.SIGMA_MOM,
             config.BLOCK_LIGHT_LIGHT, svecs[("l", "w")], svecs[("l", "v")])

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "mf_stable"
    job, run_id = build_job()
    job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
    print(f"wrote 1 mf_stable multihit job ({len(DEFAULT_HITS)} hits) to {out_dir}")


if __name__ == "__main__":
    main()
