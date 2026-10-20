"""High-mode V solves: one XML per srun, every srun of a flavour's configuration.

    gen_cg.py l              light: 3 sruns of 64 nodes
    gen_cg.py s c            strange and charm: 40 and 27 sruns of 4 nodes
    gen_cg.py l s c          all three

Work per configuration and flavour, N_HIGH noise vectors per hit: light and
strange solve the 9 hits sloppy (production hits 0..N_HIT-1 and the AMA hit)
and the AMA hit exact; charm (config.EXACT_ONLY_FLAVORS) solves all 9 exact.
config.CG_LAYOUT gives every srun an (n_sloppy, n_exact) pair;
srun_layout.cg_srun_ranges turns the sloppy stream (hits in config.CG_SLOPPY_HIT_ORDER) and the exact
stream (config.CG_EXACT_HIT_ORDER) into per-hit ranges, and every range becomes
one high-mode V module with initVec / nVec. The layout is balanced on total
time per source (CG plus write); see plans/CG section 13, where the light cut
also has a closed form for re-balancing from measured costs.

Noise is loaded, never generated: a hit split across sruns, and the AMA hit's
sloppy and exact solves, must see the same sources, and sruns generating their
own would all draw the same noise from the same runId and trajectory. The noise
job runs before the CG and leaves every hit at config.noise_filestem. A hit's
noise is loaded once per srun, right before its first range.

Output goes to config.high_stem(...) + config.high_extension(...):
vw/sloppy/<f><h>_v and vw/exact/<f>8_exact, or vw/exact/c<h>_v for charm, the
names every reader of the V tree uses. A range writes elem<i>.bin with i the noise vector index, so a hit
cut across sruns still lands in one directory.

Light (config.FLAVOR_HAS_LOW) is deflated: the coarse eigenpack and ONE
CoarseDeflationBLAS200F, used directly as the exact solve's inner guess and
through DeflationSrcCast as the outer guess of both accuracies. It runs the
batched split-grid solver (A2AHighModeBatchedV, config.CG_BATCH_SIZE sources per
call); the split layout is the srun's --batched-solver-split, not an XML
parameter. Strange and charm run the single-RHS solver with no deflation.

TRAJ_START / TRAJ_END stay as placeholders for the submission script to
substitute, so one set of XMLs serves any number of configurations.

Each srun's run_id is CG<f>.NN (par.CGl.00.xml, schedule.CGl.00.txt). Also
written per flavour: layout.CG<f>.txt, one line per srun with its node
count and ranges, to check the cut and to drive the submission script.

After a failure, check_cg.py reports what is missing on disk and
gen_cg_repack.py rebuilds the remainder with build_srun.
"""
import argparse
import sys
from pathlib import Path

# The toolkit (config, modules, hadrons_xml, srun_layout) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import modules as M
from hadrons_xml import Job
from srun_layout import cg_srun_ranges

FLAVORS = ("l", "s", "c")


def srun_run_id(flavor, index):
    return f"CG{flavor}.{index:02d}"


def add_deflation(job):
    """Eigenpack and the shared guesser. Returns (inner, outer) guesser names."""
    e = config.CG_EIGENPACK
    job.add(M.load_coarse_eigenpack_200f(
        "epack", e["filestem"], True, e["size_fine"], e["size_coarse"], True,
        config.CG_LS, e["block_size"], False))
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


def add_range(job, flavor, acc, seg, solver, action):
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
                                          seg.first, seg.count, batch))
    else:
        job.add(M.a2a_high_mode_v(name, noise, action, solver, output,
                                  seg.first, seg.count))


def build_srun(flavor, run_id, sloppy, exact, traj=None):
    """One srun's job over the given sloppy and exact segments. With traj None
    the trajectory stays as the TRAJ_START / TRAJ_END placeholders."""
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
        add_range(job, flavor, acc, seg, solvers[acc], outer)

    return job, run_id


def build_flavor(flavor):
    return [(build_srun(flavor, srun_run_id(flavor, k), s, e), s, e)
            for k, (s, e) in enumerate(cg_srun_ranges(flavor))]


def describe(segs, flavor, exact=False):
    return " ".join(f"{config.high_extension(flavor, g.hit, 'v', exact)}"
                    f"[{g.first},{g.first + g.count})" for g in segs)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the high-mode CG jobs, one XML per srun.")
    parser.add_argument("flavors", nargs="+", choices=FLAVORS)
    args = parser.parse_args()

    out_dir = Path(config.OUTPUT_ROOT) / "cg"
    for flavor in args.flavors:
        lines = []
        sruns = build_flavor(flavor)
        for (job, run_id), sloppy, exact in sruns:
            job.write(out_dir / f"par.{run_id}.xml",
                      out_dir / f"schedule.{run_id}.txt")
            ranges = " ".join(d for d in (describe(exact, flavor, True),
                                          describe(sloppy, flavor)) if d)
            lines.append(f"{run_id} nodes={config.CG_NODES_PER_SRUN[flavor]} "
                         f"sloppy={sum(g.count for g in sloppy)} "
                         f"exact={sum(g.count for g in exact)} | {ranges}")
        (out_dir / f"layout.CG{flavor}.txt").write_text("\n".join(lines) + "\n")
        nodes = len(sruns) * config.CG_NODES_PER_SRUN[flavor]
        print(f"wrote {len(sruns)} sruns for '{flavor}' ({nodes} nodes) "
              f"to {out_dir}")


if __name__ == "__main__":
    main()
