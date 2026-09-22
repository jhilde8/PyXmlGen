"""AMA correction job: every meson field for the correction hit, sloppy and exact.

256 nodes, --mpi 4.8.8.8, local [16,8,8,16], ntOut 16. Same right blocking as
the production job and no blocking on the left: every field's leftBlock is its
left leg's full mode count.

One extra hit (config.AMA_HIT = 8) is solved twice, sloppy and exact, from the
SAME noise, and this job builds the two complete field sets whose difference is
the AMA correction. The production 8-hit job is untouched: hit 8 appears in
nothing else, so there are no cross terms between the correction and the sloppy
observable it corrects, and both sets here are single-hit estimators -- nHit = 1
comes from len(hits), not from config.N_HIT.

What the two sets share, and what is done about it:

    W_l, W_s, W_c       identical -- W is the raw noise and the eigen-derived
                        low-mode block, neither of which involves a solve. One
                        array each, used by both passes.
    V low modes         identical, but loaded once per pass: a contraction leg
                        has to be one contiguous array, and the modules take
                        std::vector<FermionField>, not the pointer packs
                        MUtilities::FermionVectorPackRefSlice produces. The
                        price is 2000 extra field reads and 5.9 GiB per rank.
    mf_ls_ww            W x W, so bit-identical between the passes. Emitted
                        once per smearing state rather than twice.
    everything else     recomputed. In the V x V fields (EMF, CMF, mix) NOTHING
                        is shared -- the left leg is V_s, which is pure high
                        mode, so even the high x low block differs -- and in the
                        thin fields only the 2000 low columns of a V_l right leg
                        repeat. Splitting those into low-column and high-column
                        files would save ~0.5 TB of the job's ~4 TB, at the cost
                        of a mode-index layout no other field in the campaign
                        has. Not worth a permanent special case downstream.

Every output file is therefore structurally a 1-hit copy of its production
counterpart, with the standard [low, high] mode ordering, and the contractor
needs no new index convention -- only the smaller leg sizes.

Loops are built here rather than loaded. At one hit charm V is 1536 fields, so
the reason the production job farms charm out to gen_charm_loop.py (its vectors
would not fit beside the 8-hit light and strange legs) does not apply. The six
loops are not written to Lustre either: rebuilding one costs a single hit's
load, so the insurance is not worth 464 GB of collective writes.

Per-rank host memory at 2048 ranks, where a FermionField is 3 MiB:

    V_l sloppy + exact   7072 fields   20.7 GiB
    V_s sloppy + exact   3072           9.0
    W_l                  2012           5.9
    W_s                    12           0.04
    six loops                           0.2    (144 cplx/site, 36 MiB each)
                                       ------
                                       35.9 GiB of 59.6

The charm phase runs first and holds one accuracy's V_c at a time (4.5 GiB),
and the VM releases each before the light and strange legs load, so it never
adds to the peak above.

Device memory. The largest packed left leg is W_l at 5.9 GiB per rank, well
under the --device-mem 16000 the other jobs already pass; the light loop's
low-mode sweep opens 2*LOOP_BLOCK_LIGHT fields, 1.2 GiB here.

Output. About 4 TB per configuration to TMP_OUTPUT against production's 30 --
the thin fields dominate, because W_l is nearly all low modes and so barely
shrinks with the hit count (2012 against 2096) while V_l shrinks fourfold.
timeSliceIO everywhere, as in production.

Schedule (module order == schedule order):

    gauge, gauge_APE
    charm: noise, W, then V and loop once per accuracy   V freed between them
    light: noise, W, then V and loop once per accuracy
    strange: same
    mf_ls_ww                                             once, both passes use it
    per accuracy: mf_sl, mf_ls, mf_ll, mf_pi
    per accuracy: emf_sloop, emf_lloop, emf_cloop, cmf, cmf_ape, mix
    smear W_l, W_s, and each accuracy's V_l, V_s
    mf_ls_ww, then per accuracy the four thin fields, smeared

A2ACovariantSmear moves its source and leaves it empty, so every consumer of an
unsmeared array is scheduled ahead of the smears -- which here includes all six
loops.
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
LOOP_BLOCK_LIGHT = 200

# Unused with nLow = 0 -- the high-mode phase contracts N_SC fields per hit --
# but the module rejects zero.
LOOP_BLOCK_STRANGE = config.N_SC

EMF_TYPES = "0 1 2 3"

# Name suffix and exact flag per solve accuracy, in schedule order.
ACCURACIES = (("sloppy", False), ("exact", True))


def build_job(hit=config.AMA_HIT, run_id=None):
    hits = [hit]
    tag = f"h{hit}"
    run_id = run_id or f"contraction.ama.{tag}"
    job = Job(run_id, schedule_file=config.schedule_file(run_id),
              graph_file=config.GRAPH)
    pool = VectorPool(job)
    (width,) = config.SMEAR_WIDTHS
    width_tag, alpha, N = width

    # leftBlock per left leg: the whole leg, i.e. no left blocking.
    rb = config.RIGHT_BLOCK
    n_lw = config.leg_size("l", "w", 1)
    n_sw = config.leg_size("s", "w", 1)
    n_sv = config.leg_size("s", "v", 1)

    def out(name):
        return f"{config.TMP_OUTPUT}/{name}"

    job.add(M.load_nersc("gauge", config.GAUGE_FILE))
    job.add(M.ape_smear("gauge_APE", "gauge", config.APE_ALPHA, config.APE_N,
                        config.ORTHOG_AXIS))

    # --- vectors and loops, one flavor at a time ---------------------------
    # Charm keeps its own noise module although the file is the light one
    # (config.NOISE_FLAVOR), so the whole charm phase is a closed dependency
    # group the VM can release before the light legs load.
    loops = {}
    v = {}

    def flavor_phase(flavor, n_low, block):
        w = pool.combined(flavor, "w", hits, tag=tag)
        for acc, exact in ACCURACIES:
            leg = pool.combined(flavor, "v", hits, tag=tag, exact=exact)
            name = f"loop_{flavor}_{tag}_{acc}"
            job.add(M.a2a_loop_new(name, left=leg, right=w, n_low=n_low,
                                   block=block))
            loops[(flavor, acc)] = name
            v[(flavor, acc)] = leg
        return w

    flavor_phase("c", 0, LOOP_BLOCK_STRANGE)
    w_l = flavor_phase("l", config.N_LOW, LOOP_BLOCK_LIGHT)
    w_s = flavor_phase("s", 0, LOOP_BLOCK_STRANGE)

    # --- contractions ------------------------------------------------------
    def ww_field(lw, sw, sm):
        name = f"mf_ls_ww_{tag}{sm}"
        job.add(M.a2a_meson_field(name, n_lw, rb, lw, sw, out(name),
                                  config.IDENTITY, config.KAON_MOM,
                                  time_slice_io=True))

    def thin_fields(acc, lw, lv, sw, sv, sm):
        def mf(field, left, n_left, right, gammas, mom):
            name = f"{field}_{tag}_{acc}{sm}"
            job.add(M.a2a_meson_field(name, n_left, rb, left, right,
                                      out(name), gammas, mom,
                                      time_slice_io=True))

        mf("mf_sl", sw, n_sw, lv, config.GAMMA5,   config.KAON_MOM)
        mf("mf_ls", lw, n_lw, sv, config.GAMMA5,   config.KAON_MOM)
        mf("mf_ll", lw, n_lw, lv, config.IDENTITY, config.SIGMA_MOM)
        mf("mf_pi", lw, n_lw, lv, config.GAMMA5,   config.PION_MOM)

    def heavy_fields(acc, lv, sv):
        for flavor in ("s", "l", "c"):
            name = f"emf_{flavor}loop_{tag}_{acc}"
            job.add(M.a2a_extended_meson_field(
                name, n_sv, rb, EMF_TYPES, left=sv, right=lv, output=out(name),
                gammas1=config.EMF_GAMMA_FAMILIES,
                gammas2=config.EMF_GAMMA_FAMILIES,
                loop=loops[(flavor, acc)], time_slice_io=True))

        # Both on the unsmeared legs; they differ only in the links.
        for field, gauge in (("cmf", "gauge"), ("cmf_ape", "gauge_APE")):
            name = f"{field}_{tag}_{acc}"
            job.add(M.a2a_chromomagnetic_operator_field(
                name, n_sv, rb, config.CMO_PARITIES, sv, lv, gauge,
                out(name), config.CMO_IF_ORTHOGS, time_slice_io=True))

        name = f"mix_{tag}_{acc}"
        job.add(M.a2a_meson_field(name, n_sv, rb, sv, lv, out(name),
                                  config.IDENTITY, config.ZERO_MOM,
                                  time_slice_io=True))

    ww_field(w_l, w_s, "")
    for acc, _ in ACCURACIES:
        thin_fields(acc, w_l, v[("l", acc)], w_s, v[("s", acc)], "")
    for acc, _ in ACCURACIES:
        heavy_fields(acc, v[("l", acc)], v[("s", acc)])

    # --- smear in place, then the smeared thin fields ----------------------
    def smear(array):
        name = f"{array}_{width_tag}"
        job.add(M.a2a_covariant_smear(
            name, a2a_vectors=array, gauge="gauge_APE", alpha=alpha, N=N,
            orthog_axis=config.ORTHOG_AXIS, output="", multi_file=False))
        return name

    w_l_sm = smear(w_l)
    w_s_sm = smear(w_s)
    v_sm = {key: smear(array) for key, array in v.items() if key[0] != "c"}

    sm = f"_{width_tag}"
    ww_field(w_l_sm, w_s_sm, sm)
    for acc, _ in ACCURACIES:
        thin_fields(acc, w_l_sm, v_sm[("l", acc)], w_s_sm, v_sm[("s", acc)], sm)

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "production"
    job, run_id = build_job()
    job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
    print(f"wrote {run_id} to {out_dir}")


if __name__ == "__main__":
    main()
