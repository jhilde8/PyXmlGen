"""Every meson field of one hit set, in one or both solve accuracies.

    gen_meson_fields.py h8                       hits 0-7, sloppy only
    gen_meson_fields.py h12                      hits 0-11, sloppy only
    gen_meson_fields.py ama                      the correction hit, both ways
    gen_meson_fields.py h8 --hits 0 1 2 4 5 6 7 8   the eight hits that exist
    gen_meson_fields.py ama --load-loops         restart: read the loops back

The tag names the job and carries the hit count, so the two cannot disagree.
Everything below is common to both kinds of job; the tag decides only which
hits are loaded and how many accuracies each V is loaded in.

--hits says WHICH hits, never how many: the list has to be as long as the tag's
count, so a configuration missing one solve is expressible while a partial
block is not. That keeps the 1/nHit factor right by construction, since
VectorPool normalizes by len(hits). Extending a finished hit set is a different
job -- it needs the off-diagonal new x old blocks, so independent left and
right hit lists and the global hit count as an explicit normalization -- and
belongs in its own generator, not in a flag here.

    tag     hits                accuracies       nodes  --mpi      local
    hN      0..N-1              sloppy           512    8.8.8.8    [8,8,8,16]
    ama     config.AMA_HIT      sloppy, exact    256    4.8.8.8    [16,8,8,16]

(The layouts are the measured ones for h8 and ama; a larger hit set needs its
own node count. Nothing here emits them -- they belong to the submission
script -- but they are what the memory tables below assume.)

AMA. One extra hit is solved twice, sloppy and exact, from the SAME noise, and
the 'ama' job builds the two complete field sets whose difference is the
correction. The production hit set is untouched: the correction hit appears in
nothing else, so there are no cross terms with the sloppy observable it
corrects, and each accuracy is a single-hit estimator -- nHit = 1 falls out of
len(hits).

Naming. A production field is tagged with the hits it averages (mf_pi_h8). The
correction fields average nothing, so they are tagged 'ama' plus the accuracy
(mf_pi_ama_sloppy, emf_cloop_ama_exact, loop_l_ama_sloppy). A sloppy-only job
adds no accuracy suffix at all, so its names are unchanged by this generator
existing. The hit index survives only where it picks files, in the vector
filestems and the noise stem.

What the two accuracies share, and what is done about it:

    W_l, W_s            identical -- W is the raw noise and the eigen-derived
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
                        files would save ~0.5 TB of the ama job's ~4 TB, at the
                        cost of a mode-index layout no other field in the
                        campaign has. Not worth a permanent special case.

Every ama output file is therefore structurally a 1-hit copy of its production
counterpart, with the standard [low, high] mode ordering, and the contractor
needs no new index convention -- only the smaller leg sizes.

Loops. The light and strange loops are built here, where their vectors are
already resident, and written to Lustre ahead of the contractions, so a failure
later loses contractions only. --load-loops reads them back instead, for
restarting a job that got past them. The charm loop is always loaded:
production/gen_charm_loop.py builds it on a node count set by one hit of V
rather than spending this job's nodes on a pure-IO step, and that job has to
finish first.

Per-rank host memory. At 512 nodes with 8 hits a FermionField is 1.5 MiB and
all four legs fit at once, so nothing is tiled:

    V_l  N_LOW + N_HIGH*h   14288 fields   20.9 GiB
    V_s          N_HIGH*h   12288          18.0
    W_l  N_LOW + N_SC*h      2096           3.1
    W_s          N_SC*h        96           0.1
                                           ------
                                           42.1 GiB of 59.6 per rank

Charm would push that to 60.3, which is the other reason its loop is external.
At 256 nodes the ama job holds two accuracies of a one-hit set, 3 MiB a field:

    V_l sloppy + exact   7072 fields   20.7 GiB
    V_s sloppy + exact   3072           9.0
    W_l                  2012           5.9
    W_s                    12           0.04
    six loops                           0.2    (144 cplx/site, 36 MiB each)
                                       ------
                                       35.9 GiB of 59.6

Device memory. Each contraction module frees its A2ASpatialSum buffers at the
end of execute(), so the resident device footprint is one module's worth plus
the Grid view cache set by --device-mem. With the whole left leg packed, the
largest at 8 hits are the V_s-left fields (EMF, CMF, mix) at ~20 GiB per rank
and the pion at ~20.5, so keep --device-mem near 16000. The ama job's largest
pack is W_l at 5.9 GiB, well inside the same setting.

Output. Meson fields go to TMP_OUTPUT (node-local NVMe, substituted by the
submission script): ~30 TB per configuration at 8 hits, ~22 TB of it the three
EMFs; ~4 TB for the ama job, where the thin fields dominate instead, because
W_l is nearly all low modes and so barely shrinks with the hit count (2012
against 2096) while V_l shrinks fourfold. timeSliceIO is on for every field, so
everything downstream reads one layout. That includes the thin kaon fields:
mf_ls and mf_sl are only ever used together in the kaon two-point function, and
mf_ls_ww goes with them. Their extra files cost node-local metadata, not
Lustre, until the drain.

No sparsening here. It needs only the unsmeared V on disk, so it runs as its own
job on a small node count, where vector loads cost the fewest node-hours.

Schedule (module order == schedule order):

    gauge, gauge_APE, the charm loop per accuracy   a bad path fails early
    light: noise, W, then V and loop per accuracy, each loop saved
    strange: same
    mf_ls_ww                                        once, all passes use it
    per accuracy: mf_sl, mf_ls, mf_ll, mf_pi
    per accuracy: emf_sloop, emf_lloop, emf_cloop, cmf, cmf_ape, mix
    smear each accuracy's V_s and V_l, then W_l, W_s
    mf_ls_ww, then per accuracy the four thin fields, smeared

A2ACovariantSmear moves its source and leaves it empty, so every consumer of an
unsmeared array -- the loops and the unsmeared contractions -- is scheduled
ahead of the smears. The two thin kaon fields go first because they exercise the
TMP_OUTPUT path and per-rank file creation at under a GB per file, before the
EMFs.
"""
import argparse
import re
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

# (name suffix, exact flag) per pass. A sloppy-only job tags nothing with an
# accuracy; the correction job runs the same structure twice.
SLOPPY_ONLY = (("", False),)
BOTH_ACCURACIES = (("sloppy", False), ("exact", True))


def parse_tag(tag):
    """Hits and accuracies a tag stands for: 'hN' is the N-hit sloppy average,
    'ama' the correction hit solved both ways."""
    if tag == "ama":
        return [config.AMA_HIT], BOTH_ACCURACIES
    match = re.fullmatch(r"h(\d+)", tag)
    if not match:
        raise ValueError(f"tag '{tag}' is neither 'ama' nor 'hN'")
    return list(range(int(match.group(1)))), SLOPPY_ONLY


def build_job(tag, hits=None, accuracies=None, load_loops=False, run_id=None,
              loop_root=config.LOOP_ROOT):
    tag_hits, tag_accuracies = parse_tag(tag)
    accuracies = accuracies or tag_accuracies
    if hits is None:
        hits = tag_hits
    else:
        # Same count as the tag, any indices: a gap in the hit set is fine, a
        # partial block is not (see --hits in the module docstring).
        hits = list(hits)
        if len(hits) != len(tag_hits):
            raise ValueError(
                f"tag '{tag}' averages {len(tag_hits)} hits but {len(hits)} "
                f"were given; a partial block needs its own job")
    n_hit = len(hits)

    run_id = run_id or f"contraction.{tag}"
    job = Job(run_id, schedule_file=config.schedule_file(run_id),
              graph_file=config.GRAPH)
    pool = VectorPool(job)
    (width,) = config.SMEAR_WIDTHS
    width_tag, alpha, N = width

    # leftBlock per left leg: the whole leg, i.e. no left blocking.
    rb = config.RIGHT_BLOCK
    n_lw = config.leg_size("l", "w", n_hit)
    n_sw = config.leg_size("s", "w", n_hit)
    n_sv = config.leg_size("s", "v", n_hit)

    def out(name):
        return f"{config.TMP_OUTPUT}/{name}"

    def named(field, acc="", smear=""):
        return f"{field}_{tag}" + (f"_{acc}" if acc else "") + smear

    # --- gauge and the external charm loop ---------------------------------
    job.add(M.load_nersc("gauge", config.GAUGE_FILE))
    job.add(M.ape_smear("gauge_APE", "gauge", config.APE_ALPHA, config.APE_N,
                        config.ORTHOG_AXIS))

    loops = {}
    v = {}
    for acc, _ in accuracies:
        name = named("loop_c", acc)
        job.add(M.load_prop(name, f"{loop_root}/{name}",
                            format=config.PROP_IO_FORMAT))
        loops[("c", acc)] = name

    # --- legs and loops, one flavor at a time ------------------------------
    def flavor_phase(flavor, n_low, block):
        w = pool.combined(flavor, "w", hits, tag=tag)
        for acc, exact in accuracies:
            leg = pool.combined(flavor, "v", hits, tag=tag, exact=exact)
            name = named(f"loop_{flavor}", acc)
            if load_loops:
                job.add(M.load_prop(name, f"{loop_root}/{name}",
                                    format=config.PROP_IO_FORMAT))
            else:
                job.add(M.a2a_loop_new(name, left=leg, right=w, n_low=n_low,
                                       block=block))
                job.add(M.write_prop(f"save_{name}", prop=name,
                                     file=f"{loop_root}/{name}",
                                     format=config.PROP_IO_FORMAT))
            loops[(flavor, acc)] = name
            v[(flavor, acc)] = leg
        return w

    w_l = flavor_phase("l", config.N_LOW, LOOP_BLOCK_LIGHT)
    w_s = flavor_phase("s", 0, LOOP_BLOCK_STRANGE)

    # --- contractions ------------------------------------------------------
    # W x W, so identical in every accuracy: one field, no accuracy suffix.
    def ww_field(lw, sw, smear):
        name = named("mf_ls_ww", smear=smear)
        job.add(M.a2a_meson_field(name, n_lw, rb, lw, sw, out(name),
                                  config.IDENTITY, config.KAON_MOM,
                                  time_slice_io=True))

    def thin_fields(acc, lw, lv, sw, sv, smear):
        def mf(field, left, n_left, right, gammas, mom):
            name = named(field, acc, smear)
            job.add(M.a2a_meson_field(name, n_left, rb, left, right,
                                      out(name), gammas, mom,
                                      time_slice_io=True))

        mf("mf_sl", sw, n_sw, lv, config.GAMMA5,   config.KAON_MOM)
        mf("mf_ls", lw, n_lw, sv, config.GAMMA5,   config.KAON_MOM)
        mf("mf_ll", lw, n_lw, lv, config.IDENTITY, config.SIGMA_MOM)
        mf("mf_pi", lw, n_lw, lv, config.GAMMA5,   config.PION_MOM)

    def heavy_fields(acc, lv, sv):
        for flavor in ("s", "l", "c"):
            name = named(f"emf_{flavor}loop", acc)
            job.add(M.a2a_extended_meson_field(
                name, n_sv, rb, EMF_TYPES, left=sv, right=lv, output=out(name),
                gammas1=config.EMF_GAMMA_FAMILIES,
                gammas2=config.EMF_GAMMA_FAMILIES,
                loop=loops[(flavor, acc)], time_slice_io=True))

        # Both on the unsmeared legs; they differ only in the links.
        for field, gauge in (("cmf", "gauge"), ("cmf_ape", "gauge_APE")):
            name = named(field, acc)
            job.add(M.a2a_chromomagnetic_operator_field(
                name, n_sv, rb, config.CMO_PARITIES, sv, lv, gauge,
                out(name), config.CMO_IF_ORTHOGS, time_slice_io=True))

        name = named("mix", acc)
        job.add(M.a2a_meson_field(name, n_sv, rb, sv, lv, out(name),
                                  config.IDENTITY, config.ZERO_MOM,
                                  time_slice_io=True))

    ww_field(w_l, w_s, "")
    for acc, _ in accuracies:
        thin_fields(acc, w_l, v[("l", acc)], w_s, v[("s", acc)], "")
    for acc, _ in accuracies:
        heavy_fields(acc, v[("l", acc)], v[("s", acc)])

    # --- smear in place, then the smeared thin fields ----------------------
    def smear(array):
        name = f"{array}_{width_tag}"
        job.add(M.a2a_covariant_smear(
            name, a2a_vectors=array, gauge="gauge_APE", alpha=alpha, N=N,
            orthog_axis=config.ORTHOG_AXIS, output="", multi_file=False))
        return name

    v_sm = {}
    for acc, _ in accuracies:
        v_sm[("s", acc)] = smear(v[("s", acc)])
        v_sm[("l", acc)] = smear(v[("l", acc)])
    w_l_sm = smear(w_l)
    w_s_sm = smear(w_s)

    sm = f"_{width_tag}"
    ww_field(w_l_sm, w_s_sm, sm)
    for acc, _ in accuracies:
        thin_fields(acc, w_l_sm, v_sm[("l", acc)], w_s_sm, v_sm[("s", acc)], sm)

    return job, run_id


def main():
    parser = argparse.ArgumentParser(
        description="Generate one meson-field contraction job.")
    parser.add_argument("tag", help="hN for the N-hit sloppy average, or ama")
    parser.add_argument("--hits", type=int, nargs="+", metavar="H",
                        help="which hits to average, as many as the tag names "
                             "(default: 0..N-1, or the correction hit)")
    parser.add_argument("--load-loops", action="store_true",
                        help="read the light and strange loops from LOOP_ROOT "
                             "instead of building them (restart)")
    args = parser.parse_args()

    out_dir = Path(config.OUTPUT_ROOT) / "production"
    job, run_id = build_job(args.tag, hits=args.hits,
                            load_loops=args.load_loops)
    job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
    print(f"wrote {run_id} to {out_dir}")


if __name__ == "__main__":
    main()
