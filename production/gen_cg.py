"""High-mode V solves: one XML per srun, every srun of a flavour's configuration.

    gen_cg.py l              light: 3 sruns of 128 nodes
    gen_cg.py s c            strange and charm: 40 and 27 sruns of 4 nodes
    gen_cg.py l s c h16      a fresh 16-hit configuration, AMA hit 16

The optional tag hN (default h<config.N_HIT>) gives config.HitSet.of(N):
production hits 0..N-1 and the AMA hit N. Work
per configuration and flavour, N_HIGH noise vectors per hit: light and strange
solve every hit sloppy and the AMA hit exact; charm
(config.EXACT_ONLY_FLAVORS) solves every hit exact. config.cg_hit_order gives
the sloppy and exact hit streams.

The default HitSet is cut by config.CG_LAYOUT, an (n_sloppy, n_exact) pair per
srun balanced by hand on total time per source (CG plus write; plans/CG
section 13). Any other HitSet is cut automatically: its solver calls, hit by
hit, go to sruns of near-equal config.CG_COST, as few as fit in
config.CG_SRUN_HOURS with the startup (srun_layout.cg_srun_ranges). Either
way every per-hit range becomes one high-mode V module with initVec / nVec.

An h8 configuration taken to 16 hits keeps its AMA hit 8
(config.AMA8_CONFS), so it is not built here but repacked:
`gen_cg_repack.py s h16 --conf <traj>` scans against HitSet.of(16, 8), finds
hits 9..16 missing and solves only those.

Noise is loaded, never generated: a hit split across sruns, and the AMA hit's
sloppy and exact solves, must see the same sources, and sruns generating their
own would all draw the same noise from the same runId and trajectory. The noise
job runs before the CG and leaves every hit at config.noise_filestem. A hit's
noise is loaded once per srun, right before its first range.

Output goes to config.high_stem(...) + config.high_extension(...):
vw/sloppy/<f><h>_v and vw/exact/<f>8_exact, or vw/exact/c<h>_v for charm, the
names every reader of the V tree uses. A range writes elem<i>.bin with i the noise vector index, so a hit
cut across sruns still lands in one directory.

Light (config.FLAVOR_HAS_LOW) is deflated: the coarse eigenpack (coarse
vectors read on 8-rank partitions, config.CG_EIGENPACK io_split) and ONE
CoarseDeflationBLAS200F, used directly as the exact solve's inner guess and
through DeflationSrcCast as the outer guess of both accuracies. It runs the
batched split-grid solver (A2AHighModeBatchedV, config.CG_BATCH_SIZE sources per
call); the split layout is the srun's --batched-solver-split, not an XML
parameter. Each batch is then written on split-grid partitions of that same
layout, through the config.CG_IO_SPLIT placeholder the submission script
substitutes. Strange and charm run the single-RHS solver with no deflation.

TRAJ_START / TRAJ_END stay as placeholders for the submission script to
substitute, so one set of XMLs serves any number of configurations; so does
IO_SPLIT in the light XMLs (config.CG_IO_SPLIT).

Each srun's run_id is CG<f>.NN for the default HitSet (par.CGl.00.xml,
schedule.CGl.00.txt) and CG<f>.<tag>.NN otherwise (par.CGs.h16.00.xml), so
the template sets of different hit counts do not share a glob. Also written
per flavour: layout.CG<f>[.<tag>].txt, one line per srun with its node count,
estimated walltime and ranges, to check the cut and to drive the submission
script.

After a failure, check_cg.py reports what is missing on disk and
gen_cg_repack.py rebuilds the remainder with build_srun. Both take the same
tag and rebuild the same layout from config, per configuration
(config.hit_set).
"""
import argparse
import re
import sys
from pathlib import Path

# The toolkit (config, modules, hadrons_xml, srun_layout) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import modules as M
from hadrons_xml import Job
from srun_layout import batch_of, cg_srun_ranges

FLAVORS = ("l", "s", "c")


def srun_run_id(flavor, index, hits=config.HitSet()):
    tag = "" if hits == config.HitSet() else f"{hits.tag}."
    return f"CG{flavor}.{tag}{index:02d}"


def n_hit_of(parser, tag):
    """The hit count of a tag hN (None for the default)."""
    tag = tag or f"h{config.N_HIT}"
    match = re.fullmatch(r"h(\d+)", tag)
    if not match:
        parser.error(f"'{tag}' is neither a flavour nor a tag 'hN'")
    return int(match.group(1))


def flavors_and_n_hit(parser, targets):
    """Positional 'l s c [hN]': the flavours, then at most one tag."""
    flavors = [t for t in targets if t in FLAVORS]
    tags = [t for t in targets if t not in FLAVORS]
    if not flavors:
        parser.error("no flavour given")
    if len(tags) > 1:
        parser.error(f"more than one tag: {' '.join(tags)}")
    return flavors, n_hit_of(parser, tags[0] if tags else None)


def srun_hours(flavor, sloppy, exact):
    """Estimated walltime of one srun from config.CG_COST and CG_STARTUP."""
    seconds = config.CG_STARTUP[flavor] + sum(
        g.count // batch_of(flavor) * config.CG_COST[(flavor, acc)]
        for acc, segs in (("sloppy", sloppy), ("exact", exact)) for g in segs)
    return seconds / 3600


def add_deflation(job):
    """Eigenpack, its coarse vectors read on split-grid partitions, and the
    shared guesser. Returns (inner, outer) guesser names."""
    e = config.CG_EIGENPACK
    job.add(M.load_coarse_eigenpack_split_200f(
        "epack", e["filestem"], e["size_fine"], e["size_coarse"], True,
        config.CG_LS, e["block_size"], False, e["io_split"]))
    job.add(M.coarse_deflation_blas_200f("guesser", "epack",
                                         config.CG_DEFLATION_SIZE))
    job.add(M.deflation_src_cast("guesserD", "guesser"))
    return "guesser", "guesserD"


def add_actions(job, flavor):
    """Double- and single-precision actions. Returns (outer, inner) names."""
    p = config.CG_ACTIONS[flavor]
    for single, gauge in ((False, "gauge"), (True, "gaugef")):
        name = f"mdwf{'f' if single else ''}_{flavor}"
        common = dict(gauge=gauge, Ls=config.CG_LS, mass=p["mass"], M5=p["M5"],
                      boundary=config.CG_BOUNDARY, twist=config.CG_TWIST,
                      single=single)
        if p["kind"] == "mobius":
            job.add(M.mobius_dwf(name, b=p["b"], c=p["c"], **common))
        else:
            job.add(M.scaled_dwf(name, scale=p["scale"], **common))
    return f"mdwf_{flavor}", f"mdwff_{flavor}"


def add_solver(job, flavor, acc, outer, inner, guessers):
    p = config.CG_SOLVERS[(flavor, acc)]
    name = f"mcg_{flavor}_{acc}"
    inner_g = guessers[0] if guessers and p.get("inner_guess") else ""
    outer_g = guessers[1] if guessers else ""
    if config.CG_BATCH_SIZE[flavor] > 1:
        job.add(M.mixed_precision_rb_prec_cg_batched(
            name, inner, outer, p["max_inner"], p["max_outer"],
            p["max_patchup"], p["residual"], p["update_residual"], p["if_cgd"],
            inner_guesser=inner_g, outer_guesser=outer_g))
    else:
        job.add(M.mixed_precision_rb_prec_cg(
            name, inner, outer, p["max_inner"], p["max_outer"], p["residual"],
            p["if_cgd"], inner_guesser=inner_g, outer_guesser=outer_g))
    return name


def add_range(job, flavor, acc, seg, solver, action, io_split):
    exact = acc == "exact"
    batch = config.CG_BATCH_SIZE[flavor]
    noise = f"noise_{flavor}_h{seg.hit}"
    if not job.has(noise):
        job.add(M.load_time_diluted_noise(
            noise, [config.noise_filestem(flavor, seg.hit)],
            config.N_NOISE_PER_STEM))

    ext = config.high_extension(flavor, seg.hit, "v", exact)
    name = f"v_{ext}_{seg.first}"
    output = config.high_stem(flavor, exact) + ext
    if batch > 1:
        # The module checks this too, but only once the job is running.
        if seg.count % batch:
            raise ValueError(f"'{name}': {seg.count} vectors is not a multiple "
                             f"of the batch size {batch}")
        job.add(M.a2a_high_mode_batched_v(name, noise, action, solver, output,
                                          seg.first, seg.count, batch,
                                          io_split=io_split))
    else:
        job.add(M.a2a_high_mode_v(name, noise, action, solver, output,
                                  seg.first, seg.count))


def build_srun(flavor, run_id, sloppy, exact, traj=None,
               io_split=config.CG_IO_SPLIT):
    """One srun's job over the given sloppy and exact segments. With traj None
    the trajectory stays as the TRAJ_START / TRAJ_END placeholders. io_split
    is the light write layout, the IO_SPLIT placeholder by default."""
    bounds = {} if traj is None else dict(traj_start=traj, traj_end=traj + 1)
    job = Job(run_id, schedule_file=config.cg_schedule_file(run_id,flavor),
              graph_file=config.CG_GRAPH, **bounds)

    job.add(M.load_nersc("gauge", config.GAUGE_FILE))
    job.add(M.gauge_single_precision_cast("gaugef", "gauge"))
    guessers = add_deflation(job) if config.FLAVOR_HAS_LOW[flavor] else None
    outer, inner = add_actions(job, flavor)

    # Exact first: in the light srun that holds both, the AMA hit's noise is
    # then used by its exact and sloppy ranges back to back and freed.
    work = [("exact", s) for s in exact] + [("sloppy", s) for s in sloppy]
    solvers = {}
    for acc in ("exact", "sloppy"):
        if any(a == acc for a, _ in work):
            solvers[acc] = add_solver(job, flavor, acc, outer, inner, guessers)
    for acc, seg in work:
        add_range(job, flavor, acc, seg, solvers[acc], outer, io_split)

    return job, run_id


def build_flavor(flavor, hits=config.HitSet()):
    return [(build_srun(flavor, srun_run_id(flavor, k, hits), s, e), s, e)
            for k, (s, e) in enumerate(cg_srun_ranges(flavor, hits))]


def describe(segs, flavor, exact=False):
    return " ".join(f"{config.high_extension(flavor, g.hit, 'v', exact)}"
                    f"[{g.first},{g.first + g.count})" for g in segs)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the high-mode CG jobs, one XML per srun.")
    parser.add_argument("targets", nargs="+", metavar="l|s|c|hN",
                        help="flavours, then optionally a tag hN "
                             f"(default h{config.N_HIT})")
    args = parser.parse_args()
    flavors, n_hit = flavors_and_n_hit(parser, args.targets)
    hits = config.HitSet.of(n_hit)

    out_dir = Path(config.OUTPUT_ROOT) / "cg"
    suffix = "" if hits == config.HitSet() else f".{hits.tag}"
    for flavor in flavors:
        lines = []
        hours = []
        sruns = build_flavor(flavor, hits)
        for (job, run_id), sloppy, exact in sruns:
            job.write(out_dir / f"par.{run_id}.xml",
                      out_dir / f"schedule.{run_id}.txt")
            hours.append(srun_hours(flavor, sloppy, exact))
            ranges = " ".join(d for d in (describe(exact, flavor, True),
                                          describe(sloppy, flavor)) if d)
            lines.append(f"{run_id} nodes={config.CG_NODES_PER_SRUN[flavor]} "
                         f"sloppy={sum(g.count for g in sloppy)} "
                         f"exact={sum(g.count for g in exact)} "
                         f"est={hours[-1]:.2f}h | {ranges}")
        (out_dir / f"layout.CG{flavor}{suffix}.txt").write_text(
            "\n".join(lines) + "\n")
        nodes = len(sruns) * config.CG_NODES_PER_SRUN[flavor]
        print(f"wrote {len(sruns)} sruns for '{flavor}' {hits.tag} "
              f"(AMA {hits.ama}, {nodes} nodes, est. "
              f"{min(hours):.2f}-{max(hours):.2f} h each) to {out_dir}")


if __name__ == "__main__":
    main()
