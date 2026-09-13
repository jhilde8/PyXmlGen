"""Single contraction job: every meson field for one configuration at N_HIT hits.

512 nodes, --mpi 8.8.8.8, local [8,8,8,16], ntOut 16, block = cacheBlock =
config.BLOCK.

Everything except the charm loop runs here. At 512 nodes a field is 1.5 MiB per
rank and all four legs fit at once, so nothing is tiled:

    V_l  N_LOW + N_HIGH*h   14288 fields   20.9 GiB
    V_s          N_HIGH*h   12288          18.0
    W_l  N_LOW + N_SC*h      2096           3.1
    W_s          N_SC*h        96           0.1
                                           ------
                                           42.1 GiB of 59.6 per rank

The loops, gauge fields and coarse-grid staging add well under a GiB. Charm
would push the vectors to 60.3 GiB, which is why its loop is built separately
(production/gen_charm_loop.py, 32 nodes) and only loaded here. That job has to
finish first.

Schedule (module order == schedule order):

    gauge, gauge_APE, loop_c         cheap; a bad path fails in the first minute
    noise_l, W_l, V_l
    loop_l, save, sparsen V_l        read the unsmeared V_l
    noise_s, W_s, V_s                vector peak from here on
    loop_s, save, sparsen V_s
    mf_ls_ww, mf_sl, mf_ls, mf_ll, mf_pi              unsmeared
    emf_sloop, emf_lloop, emf_cloop, cmf_ape, mix     unsmeared
    smear V_s, V_l, W_l, W_s
    mf_ls_ww, mf_sl, mf_ls, mf_ll, mf_pi, cmf_ape     smeared

A2ACovariantSmear moves its source and leaves it empty, so every consumer of
an unsmeared array -- loops, sparsening and the unsmeared contractions -- is
scheduled ahead of the smears. The loops and sparsened vectors are on Lustre
before the first contraction, so a failure later loses contractions only. The
two thin kaon fields go first because they exercise the TMP_OUTPUT path and
per-rank file creation at under a GB per file, before the EMFs.

Loops. loop_l is one A2ALoopNew call with nLow = N_LOW, so its low-mode phase
opens 2*LOOP_BLOCK fields at once; loop_s has no low modes and ignores block.
Both are normalized by the loader (nHit = N_HIT on V), matching loop_c.

Sparsening. A2ACoarseGrid takes the whole array, so each flavour comes out as
one file holding every hit -- 76 records of 188 for V_l, 48 of 256 for V_s --
not the one-file-per-hit layout gen_loop_sparsen.py wrote. Both bin sizes have
to be registered instantiations of A2ACoarseGrid.

Output. Meson fields go to TMP_OUTPUT (node-local NVMe, substituted by the
submission script), ~30 TB per configuration, ~22 TB of it the three EMFs.
timeSliceIO is off only for mf_ls_ww and mf_sl, where 128x the file count buys
nothing on a sub-GB write.

Device memory. Each contraction module frees its A2ASpatialSum buffers at the
end of execute(), so the resident device footprint is one module's worth, the
27-momentum pion at ~11 GiB per rank, plus the Grid view cache set by
--device-mem.
"""
import sys
from pathlib import Path

# The toolkit (config, modules, hadrons_xml, vector_pool) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import modules as M
from hadrons_xml import Job
from vector_pool import VectorPool

# A2ALoopNew low-mode sweep: fields per kernel call.
LOOP_BLOCK = 50

EMF_TYPES = "0 1 2 3"

# A2ACoarseGrid records per file; each must divide its array. 14288 = 76*188,
# 12288 = 48*256.
SPARSE_BIN_LIGHT = 188
SPARSE_BIN_STRANGE = 256


def build_job(n_hit=config.N_HIT, run_id=None):
    hits = list(range(n_hit))
    run_id = run_id or f"contraction.h{n_hit}"
    job = Job(run_id, schedule_file=config.schedule_file(run_id),
              graph_file=config.GRAPH)
    pool = VectorPool(job)
    (width,) = config.SMEAR_WIDTHS
    width_tag, alpha, N = width
    block = config.BLOCK

    def out(name):
        return f"{config.TMP_OUTPUT}/{name}"

    # --- gauge and the external charm loop ---------------------------------
    job.add(M.load_nersc("gauge", config.GAUGE_FILE))
    job.add(M.ape_smear("gauge_APE", "gauge", config.APE_ALPHA, config.APE_N,
                        config.ORTHOG_AXIS))
    loop_c = f"loop_c_h{n_hit}"
    job.add(M.load_prop(loop_c, f"{config.LOOP_ROOT}/{loop_c}",
                        format=config.PROP_IO_FORMAT))

    # --- light legs, then everything that needs the unsmeared V_l ----------
    lw = pool.combined("l", "w", hits)
    lv = pool.combined("l", "v", hits)

    loop_l = f"loop_l_h{n_hit}"
    job.add(M.a2a_loop_new(loop_l, left=lv, right=lw, n_low=config.N_LOW,
                           block=LOOP_BLOCK))
    job.add(M.write_prop(f"save_{loop_l}", prop=loop_l,
                         file=f"{config.LOOP_ROOT}/{loop_l}",
                         format=config.PROP_IO_FORMAT))
    job.add(M.a2a_coarse_grid(
        f"sp_l_v_h{n_hit}", SPARSE_BIN_LIGHT, lv,
        config.COARSE_BLOCK_SIZE, config.COARSE_OFFSETS,
        f"{config.SPARSE_ROOT}/l_v_h{n_hit}"))

    # --- strange legs, then everything that needs the unsmeared V_s --------
    sw = pool.combined("s", "w", hits)
    sv = pool.combined("s", "v", hits)

    loop_s = f"loop_s_h{n_hit}"
    job.add(M.a2a_loop_new(loop_s, left=sv, right=sw, n_low=0,
                           block=LOOP_BLOCK))
    job.add(M.write_prop(f"save_{loop_s}", prop=loop_s,
                         file=f"{config.LOOP_ROOT}/{loop_s}",
                         format=config.PROP_IO_FORMAT))
    job.add(M.a2a_coarse_grid(
        f"sp_s_v_h{n_hit}", SPARSE_BIN_STRANGE, sv,
        config.COARSE_BLOCK_SIZE, config.COARSE_OFFSETS,
        f"{config.SPARSE_ROOT}/s_v_h{n_hit}"))

    # --- contractions ------------------------------------------------------
    def meson_fields(suffix, lw, lv, sw, sv):
        def mf(field, left, right, gammas, mom, time_slice_io):
            name = f"{field}_h{n_hit}{suffix}"
            job.add(M.a2a_meson_field(name, block, block, left, right,
                                      out(name), gammas, mom,
                                      time_slice_io=time_slice_io))

        mf("mf_ls_ww", lw, sw, config.IDENTITY, config.KAON_MOM, False)
        mf("mf_sl", sw, lv, config.GAMMA5, config.KAON_MOM, False)
        mf("mf_ls", lw, sv, config.GAMMA5, config.KAON_MOM, True)
        mf("mf_ll", lw, lv, config.IDENTITY, config.SIGMA_MOM, True)
        mf("mf_pi", lw, lv, config.GAMMA5, config.PION_MOM, True)

    def cmf(suffix, sv, lv):
        name = f"cmf_ape_h{n_hit}{suffix}"
        job.add(M.a2a_chromomagnetic_operator_field(
            name, block, block, config.CMO_PARITIES, sv, lv, "gauge_APE",
            out(name), config.CMO_IF_ORTHOGS, time_slice_io=True))

    meson_fields("", lw, lv, sw, sv)

    for flavor, loop in (("s", loop_s), ("l", loop_l), ("c", loop_c)):
        name = f"emf_{flavor}loop_h{n_hit}"
        job.add(M.a2a_extended_meson_field(
            name, block, block, EMF_TYPES, left=sv, right=lv, output=out(name),
            gammas1=config.EMF_GAMMA_FAMILIES,
            gammas2=config.EMF_GAMMA_FAMILIES,
            loop=loop, time_slice_io=True))

    cmf("", sv, lv)

    mix = f"mix_h{n_hit}"
    job.add(M.a2a_meson_field(mix, block, block, sv, lv, out(mix),
                              config.IDENTITY, config.ZERO_MOM,
                              time_slice_io=True))

    # --- smear in place, then the smeared contractions ---------------------
    def smear(array):
        name = f"{array}_{width_tag}"
        job.add(M.a2a_covariant_smear(
            name, a2a_vectors=array, gauge="gauge_APE", alpha=alpha, N=N,
            orthog_axis=config.ORTHOG_AXIS, output="", multi_file=False))
        return name

    sv_sm = smear(sv)
    lv_sm = smear(lv)
    lw_sm = smear(lw)
    sw_sm = smear(sw)

    meson_fields(f"_{width_tag}", lw_sm, lv_sm, sw_sm, sv_sm)
    cmf(f"_{width_tag}", sv_sm, lv_sm)

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "production"
    job, run_id = build_job()
    job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
    print(f"wrote {run_id} to {out_dir}")


if __name__ == "__main__":
    main()
