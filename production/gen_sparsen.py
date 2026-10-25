"""Sparsened V and W, light and strange, one directory per hit.

    gen_sparsen.py                          production hits, AMA hit, low block
    gen_sparsen.py --skip-h8                the AMA hit and the light low block
    gen_sparsen.py --skip-h8 --skip-low     the AMA hit only
    gen_sparsen.py --io-split 2.2.2.1       loads read on split-grid partitions

32 nodes, 256 ranks. Pure IO: every module is a load or an A2ACoarseGrid.

Production now sparsens inside the meson-field job (gen_meson_fields.py
--sparsen, with split-grid writes), into this same layout. This job remains
for configurations whose meson fields were made without it.

Why it was its own job. A sparsened field is 1/64 of a fine one under the [4,4,4,1]
spatial blocking, ~100 MB, and a Lustre write needs roughly 4 MB per rank per
file before per-file overhead stops dominating (measured with
Hadrons/tests/Test_a2a_io_width.cpp). Inside a contraction job that is ~49 KB
per rank at 256 nodes and ~25 KB at 512, and the ama job did not finish its
sparsening within an hour at 256 nodes. At 32 nodes it is ~390 KB per rank:
still under the target, but 8x better, and the node count is set by memory
anyway (below). If this is still too slow, the next step is to pack 12 modes per
file as a PropagatorField record in A2ACoarseGrid, ~4.7 MB per rank here, with
a PropToFerm unpack on the read side.

No loops. The strange loop is built in the meson-field job, where V_s is loaded
anyway, and the charm loop in production/gen_charm_loop.py. Charm is not
sparsened, so nothing here needs charm vectors.

Per-rank host memory at 24 MiB per field. Only one block is ever resident:
each A2ACoarseGrid is the last consumer of its input, so the VM frees a block
before the next one loads.

    light low V            2000 fields   46.9 GiB
    light low W            2000          46.9
    one hit of V           1536          36.0
    one hit of dense W       12           0.3
    coarse output                        1/64 of its input

against ~59.6 usable. That is HOST memory: Frontier builds with
--enable-unified=no, so lattice data is host-resident (64 GB per rank) and HBM
holds only the open views. 16 nodes is not possible: a field is 48 MiB there and
the low block alone would be 93.8 GiB, one hit of V 72.0 GiB.

Layout, under config.SPARSE_ROOT, one directory per block, named like the
unsparsened block it came from (config.high_extension):

    l_lo_v, l_lo_w          light low modes, once -- deterministic, so shared
                            by every hit and both accuracies
    {f}{h}_v, {f}{h}_w      production hit h, h < N_HIT
    {f}8_v, {f}8_w          AMA hit: sloppy V, and the W both accuracies share
    {f}8_exact              AMA hit: exact V

with 8 = config.AMA_HIT. Each is <stem>.<traj>/elem<i>.bin, one fermion field
per mode, so the standard loaders read it back on the coarse grid with
highStem = SPARSE_ROOT + "/" and the same extensions as the fine tree.

Raw normalization. Every load passes n_hit = 1, so the sparse tree is a 1/64
copy of the vw tree and carries no 1/nHit; a reader applies it at load exactly
as for the fine vectors. Folding it in here would double count on read, and is
wrong in any case for the AMA hit (a single-hit estimator) and the low modes
(deterministic, never hit-averaged).

The light low W is loaded through LoadCombinedA2AVecsV with no high extensions.
LoadCombinedA2AVecsW cannot do a low-only load -- it requires a noise input --
and the V loader's only V-specific behaviour, nHit, touches high blocks only.

The dense W sparsens with no special handling. COARSE_BLOCK_SIZE is spatial
only, with the time index left at 1, and a dense W slot is a whole noise field
whose expansion into a mode is a mask on the time index -- so the mask and the
spatial subsample commute, and sparsening the dense array then expanding is the
same thing as expanding then sparsening. Expanding first would only be a way of
writing a great many zeros.

Schedule (module order == schedule order): light low V, light low W, then per
flavour and hit: noise, W, sparsen W, V, sparsen V. The AMA hit sparsens its W
once, then each accuracy's V.
"""
import argparse
import sys
from pathlib import Path

# The toolkit (config, modules, hadrons_xml) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import modules as M
from hadrons_xml import Job

SPARSEN_FLAVORS = ("l", "s")


def sparsen(job, fine, stem):
    job.add(M.a2a_coarse_grid(f"sp_{fine}", fine, config.COARSE_BLOCK_SIZE,
                              config.COARSE_OFFSETS, stem))


def add_low(job, flavor, sparse_root, io_split=""):
    """A flavour's low-mode V and W, each loaded and sparsened on its own."""
    for vw in ("v", "w"):
        name = f"lo_{vw}_{flavor}"
        # The V loader for W too: see the module docstring.
        job.add(M.load_combined_a2a_vecs_v(
            name, low_filestem=config.low_filestem(flavor, vw),
            n_low=config.N_LOW, high_stem="", high_extensions=[],
            high_size=0, io_split=io_split))
        sparsen(job, name, f"{sparse_root}/{flavor}_lo_{vw}")


def add_hit(job, flavor, hit, sparse_root, accuracies=(False,), io_split=""):
    """One hit: its dense W, then its V once per accuracy (the exact flag)."""
    noise = f"noise_{flavor}_h{hit}"
    w = f"w_{flavor}_h{hit}"

    job.add(M.load_time_diluted_noise(
        noise, [config.noise_filestem(flavor, hit)], config.N_NOISE_PER_STEM))
    # n_low = 0: the noise expanded into N_SC fields and nothing more.
    job.add(M.load_combined_a2a_vecs_w(w, low_filestem="", n_low=0,
                                       noise=noise))
    sparsen(job, w, f"{sparse_root}/{config.high_extension(flavor, hit, 'w')}")

    for exact in accuracies:
        ext = config.high_extension(flavor, hit, "v", exact)
        v = f"v_{ext}"
        job.add(M.load_combined_a2a_vecs_v(
            v, low_filestem="", n_low=0, high_stem=config.high_stem(flavor, exact),
            high_extensions=[ext], high_size=config.N_HIGH, n_hit=1,
            io_split=io_split))
        sparsen(job, v, f"{sparse_root}/{ext}")


def build_job(n_hit=config.N_HIT, h8=True, ama=True, low=True,
              flavors=SPARSEN_FLAVORS, run_id=None,
              sparse_root=config.SPARSE_ROOT, io_split=""):
    # run_id and sparse_root are parameters so that benchmarks/ can emit this
    # exact structure without its output landing on top of production's -- the
    # per-hit directory names carry no hit-count tag.
    low = low and any(config.FLAVOR_HAS_LOW[f] for f in flavors)
    parts = ([f"h{n_hit}"] if h8 else []) + (["ama"] if ama else []) \
        + (["low"] if low else [])
    if not parts:
        raise ValueError("nothing to sparsen: the production hits, the AMA hit "
                         "and the low block are all skipped")
    if h8 and ama and config.AMA_HIT < n_hit:
        raise ValueError(f"AMA hit {config.AMA_HIT} falls inside production "
                         f"hits 0..{n_hit - 1}; their directories would collide")

    run_id = run_id or "sparsen." + ".".join(parts)
    job = Job(run_id, schedule_file=config.schedule_file(run_id),
              graph_file=config.GRAPH)

    # Low block first, while nothing else is resident: at 46.9 GiB per rank it
    # does not fit beside a hit of V.
    if low:
        for flavor in flavors:
            if config.FLAVOR_HAS_LOW[flavor]:
                add_low(job, flavor, sparse_root, io_split)

    for flavor in flavors:
        if h8:
            for hit in range(n_hit):
                add_hit(job, flavor, hit, sparse_root, io_split=io_split)
        if ama:
            add_hit(job, flavor, config.AMA_HIT, sparse_root,
                    accuracies=(False, True), io_split=io_split)

    return job, run_id


def main():
    parser = argparse.ArgumentParser(
        description="Generate the sparsening job for light and strange V and W.")
    parser.add_argument("--skip-h8", action="store_true",
                        help=f"skip the production hits 0..{config.N_HIT - 1}")
    parser.add_argument("--skip-ama", action="store_true",
                        help="skip the AMA hit (sloppy and exact V, shared W)")
    parser.add_argument("--skip-low", action="store_true",
                        help="skip the light low-mode V and W")
    parser.add_argument("--io-split", metavar="LAYOUT", default="",
                        type=lambda s: s.replace(".", " "),
                        help="split-grid read layout for the vector loads, e.g. "
                             "2.2.2.1 (default: full-grid reads)")
    args = parser.parse_args()

    try:
        job, run_id = build_job(h8=not args.skip_h8, ama=not args.skip_ama,
                                low=not args.skip_low, io_split=args.io_split)
    except ValueError as e:
        parser.error(str(e))

    out_dir = Path(config.OUTPUT_ROOT) / "production"
    job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
    print(f"wrote {run_id} to {out_dir}")


if __name__ == "__main__":
    main()
