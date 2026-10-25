"""Every meson field of one hit set, in one or both solve accuracies.

    gen_meson_fields.py h8                       hits 0-7, sloppy only
    gen_meson_fields.py h12                      hits 0-11, sloppy only
    gen_meson_fields.py ama                      the correction hit, both ways
    gen_meson_fields.py h8 --hits 0 1 2 4 5 6 7 8   the eight hits that exist
    gen_meson_fields.py ama --load-loops         restart: read the loops back
    gen_meson_fields.py h8 --charm-loop          build the charm loop here too
    gen_meson_fields.py h8 --sparsen             write the sparsened V and W too
    gen_meson_fields.py h8 --smeared-only         both widths, no svlv
    gen_meson_fields.py h8 --width w2p03_n11      stop at the narrow rung

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
restarting a job that got past them. The charm loop is loaded by default, from
production/gen_charm_loop.py, which builds it on a node count set by one hit
of V; that job has to finish first. --charm-loop builds it here instead, first
of all, and saves it like the others, under the same name gen_charm_loop.py
writes (loop_c_h8, loop_c_ama). With split-grid reads the charm vectors cost
minutes here, so the separate job saves about 1% of this job's node-hours
against an extra job and a dependency. Charm is exact on every hit, so one V
and one loop serve every accuracy.

Smearing. config.SMEAR_LADDER is a ladder of increments, not a catalogue of
widths: each rung smears the arrays the previous rung produced, and the job
climbs it, contracting at every rung. So one job yields every width at the cost
of one vector load, which is the whole point -- a second width as its own job
would pay that load again, and the load is the dominant cost once the loops and
the V_s x V_l family are gone. A2ACovariantSmear moves its source, so climbing
costs no extra memory: a rung's arrays are freed as the next rung consumes them.

--width names the highest rung to reach rather than picking one entry, since
reaching a rung means applying every increment below it. --smeared-only drops the
loops, the unsmeared contractions and the V_s x V_l family, leaving the vector
loads, the smears and five contractions per rung, under its own run_id
(contraction.<tag>.<rung>) so it sits beside the full job.

Every job must climb the SAME ladder. An AMA correction differences the exact
and sloppy field sets, so if the two were smeared differently the difference
carries a smearing mismatch on top of the accuracy one, and nothing downstream
would catch it.

Per-rank host memory. At 512 nodes with 8 hits a FermionField is 1.5 MiB and
all four legs fit at once, so nothing is tiled:

    V_l  N_LOW + N_HIGH*h   14288 fields   20.9 GiB
    V_s          N_HIGH*h   12288          18.0
    W_l  N_LOW + N_SC*h      2096           3.1
    W_s          N_SC*h        96           0.1
                                           ------
                                           42.1 GiB of 59.6 per rank

Charm alongside them would push that to 60.3, so --charm-loop builds it before
any other leg loads: the loop is the last consumer of V_c (18.0 GiB) and W_c, so
both are freed before the light phase.
At 256 nodes the ama job holds two accuracies of a one-hit set, 3 MiB a field:

    V_l sloppy + exact   7072 fields   20.7 GiB
    V_s sloppy + exact   3072           9.0
    W_l                  2012           5.9
    W_s                    12           0.04
    five loops                          0.2    (144 cplx/site, 36 MiB each)
                                       ------
                                       35.9 GiB of 59.6

Device memory. Each contraction module frees its A2ASpatialSum buffers at the
end of execute(), so the resident device footprint is one module's worth plus
the Grid view cache set by --device-mem. With the whole left leg packed, the
largest at 8 hits are the V_s-left fields (EMF, CMF, mix) at ~20 GiB per rank,
so keep --device-mem near 16000. The pion's right pack is phased by all 27
momenta and ran out of device memory at RIGHT_BLOCK, so the pion alone takes a
smaller right block, PION_RIGHT_BLOCK (PION_RIGHT_BLOCK_AMA in the ama job). The ama job's largest
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

Sparsening. --sparsen writes the light and strange V and W to SPARSE_ROOT in
production/gen_sparsen.py's layout, one A2ACoarseGrid per block of a combined
array: the light low modes, then each hit. Each array is sparsened as soon as it
is loaded, so the writes are done before the loops, the contractions and the
first smear (which moves its source). A V hit block is scaled by n_hit, undoing
the loader's 1/nHit, so the sparse tree stays raw whichever job wrote it. The
low block is written by hN jobs only: it is the same files in every job, and an
ama job running beside an hN one on the same configuration would otherwise
write it concurrently; the exact V's low block is skipped for the same reason.
Charm is not sparsened. A sparsened field is ~100 MB, ~25 KB per rank at 512
nodes, so the write needs --io-split: on 1-node partitions 316 fields took 1.4 s
against 26.9 s from the full grid at 64 nodes. Each coarse block is 1/64 of its
input and is freed as soon as it is written.

Schedule (module order == schedule order):

    gauge, gauge_APE, the charm loop                a bad path fails early
                                                    (--charm-loop: noise, W, V,
                                                    loop, saved)
    light: noise, W, then V and loop per accuracy, each loop saved
                                                    (--sparsen: each W and V
                                                    sparsened right after it
                                                    loads)
    strange: same
    mf_ls_ww                                        once, all passes use it
    per accuracy: mf_sl, mf_ls, mf_ll, mf_pi
    per accuracy: emf_sloop, emf_lloop, emf_cloop, cmf, cmf_ape, mix
    per ladder rung: smear each accuracy's V_s and V_l, then W_l, W_s,
                     then mf_ls_ww and per accuracy the four thin fields

--smeared-only keeps the first three lines without their loops, and the last.

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

# mf_pi rightBlock, every smearing state, a factor of the V_l mode count: the
# 27-momentum phased right pack at RIGHT_BLOCK ran the device out of memory.
# 128 is known to fit in both jobs; h8 14288 = 94 x 152, ama 3536 = 26 x 136.
PION_RIGHT_BLOCK = 152
PION_RIGHT_BLOCK_AMA = 136

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


def select_ladder(target=None):
    """The config.SMEAR_LADDER rungs up to and including `target`, or the whole
    ladder if None. A target names a prefix, not one entry: reaching a rung means
    applying every increment below it."""
    if target is None:
        return list(config.SMEAR_LADDER)
    rungs = []
    for rung in config.SMEAR_LADDER:
        rungs.append(rung)
        if rung[0] == target:
            return rungs
    known = ", ".join(r[0] for r in config.SMEAR_LADDER)
    raise ValueError(f"no smearing rung '{target}' in config.SMEAR_LADDER "
                     f"(have {known})")


def build_job(tag, hits=None, accuracies=None, load_loops=False, run_id=None,
              loop_root=config.LOOP_ROOT, width=None, smeared_only=False,
              io_split="", charm_loop=False, sparsen=False,
              sparse_root=config.SPARSE_ROOT):
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

    ladder = select_ladder(width)

    # A smeared-only job emits no unsmeared output, so it takes its own run_id,
    # named for the rung it climbs to, and sits alongside the full job's files.
    if run_id is None:
        run_id = f"contraction.{tag}"
        if smeared_only:
            run_id += f".{ladder[-1][0]}"
    job = Job(run_id, schedule_file=config.schedule_file(run_id),
              graph_file=config.GRAPH)
    pool = VectorPool(job, io_split)

    # leftBlock per left leg: the whole leg, i.e. no left blocking.
    rb = config.RIGHT_BLOCK
    n_lw = config.leg_size("l", "w", n_hit)
    n_sw = config.leg_size("s", "w", n_hit)
    n_sv = config.leg_size("s", "v", n_hit)
    pion_rb = PION_RIGHT_BLOCK_AMA if tag == "ama" else PION_RIGHT_BLOCK

    def out(name):
        return f"{config.TMP_OUTPUT}/{name}"

    def named(field, acc="", smear=""):
        return f"{field}_{tag}" + (f"_{acc}" if acc else "") + smear

    # --- gauge and the external charm loop ---------------------------------
    job.add(M.load_nersc("gauge", config.GAUGE_FILE))
    job.add(M.ape_smear("gauge_APE", "gauge", config.APE_ALPHA, config.APE_N,
                        config.ORTHOG_AXIS))

    # The loops feed heavy_fields only, so a smeared-only job builds none of
    # them and does not load the charm one either. Charm is exact on every hit,
    # so one loop serves every accuracy. A restart reads back whatever loop the
    # first run saved, built here or not.
    loops = {}
    v = {}
    if not smeared_only:
        name = named("loop_c")
        if charm_loop and not load_loops:
            w_c = pool.combined("c", "w", hits, tag=tag)
            v_c = pool.combined("c", "v", hits, tag=tag)
            job.add(M.a2a_loop_new(name, left=v_c, right=w_c, n_low=0,
                                   block=LOOP_BLOCK_STRANGE))
            job.add(M.write_prop(f"save_{name}", prop=name,
                                 file=f"{loop_root}/{name}",
                                 format=config.PROP_IO_FORMAT))
        else:
            job.add(M.load_prop(name, f"{loop_root}/{name}",
                                format=config.PROP_IO_FORMAT))
        for acc, _ in accuracies:
            loops[("c", acc)] = name

    # --- sparsening, one A2ACoarseGrid per block --------------------------
    # Block layout of a combined array: low modes (if any), then per hit
    # N_HIGH expanded V or N_SC dense W. See "Sparsening" in the docstring.
    def sparsen_array(array, flavor, vw, exact=False):
        offset = 0
        if config.FLAVOR_HAS_LOW[flavor]:
            if tag != "ama" and not exact:
                job.add(M.a2a_coarse_grid(
                    f"sp_{array}_lo", array, config.COARSE_BLOCK_SIZE,
                    config.COARSE_OFFSETS, f"{sparse_root}/{flavor}_lo_{vw}",
                    mode_offset=0, mode_size=config.N_LOW, io_split=io_split))
            offset = config.N_LOW
        per_hit = config.N_HIGH if vw == "v" else config.N_SC
        scale = float(n_hit) if vw == "v" else 1.0
        for k, hit in enumerate(hits):
            ext = config.high_extension(flavor, hit, vw, exact)
            job.add(M.a2a_coarse_grid(
                f"sp_{array}_h{hit}", array, config.COARSE_BLOCK_SIZE,
                config.COARSE_OFFSETS, f"{sparse_root}/{ext}",
                mode_offset=offset + k*per_hit, mode_size=per_hit,
                scale=scale, io_split=io_split))

    # --- legs and loops, one flavor at a time ------------------------------
    def flavor_phase(flavor, n_low, block):
        w = pool.combined(flavor, "w", hits, tag=tag)
        if sparsen:
            sparsen_array(w, flavor, "w")
        for acc, exact in accuracies:
            leg = pool.combined(flavor, "v", hits, tag=tag, exact=exact)
            if sparsen:
                sparsen_array(leg, flavor, "v", exact)
            v[(flavor, acc)] = leg
            if smeared_only:
                continue
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
        def mf(field, left, n_left, right, gammas, mom, right_block=rb):
            name = named(field, acc, smear)
            job.add(M.a2a_meson_field(name, n_left, right_block, left, right,
                                      out(name), gammas, mom,
                                      time_slice_io=True))

        mf("mf_sl", sw, n_sw, lv, config.GAMMA5,   config.KAON_MOM)
        mf("mf_ls", lw, n_lw, sv, config.GAMMA5,   config.KAON_MOM)
        mf("mf_ll", lw, n_lw, lv, config.IDENTITY, config.SIGMA_MOM)
        mf("mf_pi", lw, n_lw, lv, config.GAMMA5, config.PION_MOM,
           right_block=pion_rb)

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

    if not smeared_only:
        ww_field(w_l, w_s, "")
        for acc, _ in accuracies:
            thin_fields(acc, w_l, v[("l", acc)], w_s, v[("s", acc)], "")
        for acc, _ in accuracies:
            heavy_fields(acc, v[("l", acc)], v[("s", acc)])

    # --- climb the smearing ladder, contracting at every rung --------------
    # Each rung smears the arrays the previous one produced, so `src` is rebound
    # as we climb while the name is always built from the unsmeared base, keeping
    # a rung's arrays tagged by the width they reach rather than by the chain.
    # A2ACovariantSmear moves its source, so a rung's arrays must be contracted
    # before the next rung's smears are emitted -- module order is schedule
    # order, and the ordering is not expressed in the dependency graph.
    base = {("w", "l"): w_l, ("w", "s"): w_s}
    src = dict(base)
    for acc, _ in accuracies:
        for flavor in ("l", "s"):
            base[("v", flavor, acc)] = v[(flavor, acc)]
            src[("v", flavor, acc)] = v[(flavor, acc)]

    for rung_tag, rung_alpha, rung_N in ladder:
        def smear(key):
            name = f"{base[key]}_{rung_tag}"
            job.add(M.a2a_covariant_smear(
                name, a2a_vectors=src[key], gauge="gauge_APE",
                alpha=rung_alpha, N=rung_N, orthog_axis=config.ORTHOG_AXIS,
                output="", multi_file=False))
            src[key] = name
            return name

        for acc, _ in accuracies:
            smear(("v", "s", acc))
            smear(("v", "l", acc))
        smear(("w", "l"))
        smear(("w", "s"))

        sm = f"_{rung_tag}"
        ww_field(src[("w", "l")], src[("w", "s")], sm)
        for acc, _ in accuracies:
            thin_fields(acc, src[("w", "l")], src[("v", "l", acc)],
                        src[("w", "s")], src[("v", "s", acc)], sm)

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
    parser.add_argument("--width", metavar="TAG",
                        choices=[r[0] for r in config.SMEAR_LADDER],
                        help="highest rung of config.SMEAR_LADDER to climb to; "
                             "every rung below it is applied and contracted on "
                             "the way (default: the whole ladder)")
    parser.add_argument("--smeared-only", action="store_true",
                        help="emit only the smeared thin fields: no loops, no "
                             "unsmeared contractions, no EMF/CMF/mix")
    parser.add_argument("--io-split", metavar="LAYOUT", default="2 2 2 2",
                        type=lambda s: s.replace(".", " "),
                        help="split-grid layout for every vector load and "
                             "sparsened write (default: 2.2.2.2; '' for "
                             "full-grid IO)")
    parser.add_argument("--charm-loop", action="store_true",
                        help="build and save the charm loop here instead of "
                             "loading gen_charm_loop.py's")
    parser.add_argument("--sparsen", action="store_true",
                        help="sparsen the light and strange V and W to "
                             "SPARSE_ROOT as they load (use with --io-split)")
    args = parser.parse_args()

    out_dir = Path(config.OUTPUT_ROOT) / "production"
    job, run_id = build_job(args.tag, hits=args.hits,
                            load_loops=args.load_loops,
                            width=args.width,
                            smeared_only=args.smeared_only,
                            io_split=args.io_split,
                            charm_loop=args.charm_loop,
                            sparsen=args.sparsen)
    job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
    print(f"wrote {run_id} to {out_dir}")


if __name__ == "__main__":
    main()
