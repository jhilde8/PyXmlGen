"""Charm A2A loop propagators: the production chain and the AMA hit.

    gen_charm_loop.py                 loop_c_h8 and loop_c_ama
    gen_charm_loop.py h4              a 4-hit chain and loop_c_ama
    gen_charm_loop.py --skip-ama      the chain only
    gen_charm_loop.py --io-split 2.2.2.1   V read on split-grid partitions

32 nodes, 256 ranks, high modes only, no sparsening.

Charm gets its own job because nothing else in the campaign loads charm
vectors: the only consumer is the charm-loop extended meson field, which reads
the finished loop back through MIO::LoadProp. Putting it in the 512-node
contraction job would spend that job's node-hours on a pure-IO step, so it runs
here on a node count set by one hit of V instead.

Charm reuses the light noise. The charm loop enters only through the GIM
subtraction against the light loop, so the two are built from the same sources
and there is no noise_c on disk -- the W expansion reads noise_ud.

Per hit the job loads the noise, expands it into the hit's 12 dense W fields,
reads that hit's expanded V block, and contracts one link of the loop:

    noise_c_h<h> -> w_c_h<h> --+
                               +--> loop_c_h<h>  (seeded from loop_c_h<h-1>)
                  v_c_h<h> ----+

Declaring inputLoop as a module input orders the chain, and since the link is
the last consumer of its V and W, the VM releases the whole hit before the next
one loads. Only one hit of vectors is ever resident.

Per-rank host memory at 256 ranks, where a FermionField is 24 MiB (6 GiB / 256)
and a PropagatorField is 12x that:

    V, one hit          1536 fields   36.0 GiB
    W dense               12 fields    0.3 GiB
    gather buffer         12 fields    0.3 GiB   (A2ALoopNew vtilde)
    loop, link + seed      2 props     0.6 GiB
                                      ---------
                                      37.2 GiB of 59.6

Frontier builds with --enable-unified=no, so all of that is host DDR. The
aggregate is ~10 TB of ~16 TB. 16 nodes does not fit: one hit's V alone is
72.0 GiB per rank there.

Device memory. A2ALoopNew with nLow = 0 and a dense W never enters its blocked
sweep: the gather opens two fields at a time, and the high-mode contraction
opens the 12 gathered fields, the 12 W slots and the loop together, about
864 MiB per rank. The binding constraint is the loop on its own -- MemoryManager
asserts on any single view at or above --device-mem, and the PropagatorField is
288 MiB per rank here, past the 128 MiB default. The CG submission scripts
already pass --device-mem 16000, which covers it.

Normalization: every V load passes nHit = config.N_HIT, the global hit count,
not the one extension it reads -- the 1/nHit factor belongs to the estimator,
not to how the load is chunked. A2ALoopNew applies nothing, so the chain adds
correctly normalized blocks. (VectorPool.combined sets n_hit = len(hits) and so
cannot be used for a per-hit load.)

The loop goes to Lustre, not node-local NVMe: MIO::WriteProp is a collective
single-file write.

The AMA hit rides along in the same job, since it is one more hit of the same
loads. It is not a link of the chain: the correction is a single-hit
estimator, so its V loads at nHit = 1 and its loop starts from zero. It runs
first, so the full noise -> W -> V -> loop -> write path is exercised on one
hit and its output lands before the chain commits to the other eight. Memory
is unchanged -- the AMA loop's write is its last consumer, so its hit is
released before the chain's first loads.

Charm is solved exact on every hit (config.EXACT_ONLY_FLAVORS), so the AMA hit
has one charm loop, which the contraction job uses in both its sloppy and its
exact pass. The correction then carries no charm solver error, though the
charm-loop EMF still enters it through its sloppy and exact V legs.
"""
import argparse
import re
import sys
from pathlib import Path

# The toolkit (config, modules, hadrons_xml) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import modules as M
from hadrons_xml import Job

FLAVOR = "c"


def add_noise_and_w(job, tag, hit):
    """This hit's raw noise and the N_SC dense W fields expanded from it.
    Charm reads the light noise stem (config.NOISE_FLAVOR)."""
    noise = f"noise_{FLAVOR}_{tag}"
    w = f"w_{FLAVOR}_{tag}"
    job.add(M.load_time_diluted_noise(
        noise, [config.noise_filestem(FLAVOR, hit)], config.N_NOISE_PER_STEM))
    job.add(M.load_combined_a2a_vecs_w(
        w, low_filestem="", n_low=0, noise=noise))
    return w


def add_v(job, name, hit, n_hit, io_split=""):
    """One hit's expanded high-mode V block. n_hit is the GLOBAL hit count of
    the estimator this block belongs to, not the number of extensions read
    here -- the 1/nHit factor belongs to the estimator, not to how the load is
    chunked -- which is why the loop applies none."""
    job.add(M.load_combined_a2a_vecs_v(
        name, low_filestem="", n_low=0,
        high_stem=config.high_stem(FLAVOR),
        high_extensions=[config.high_extension(FLAVOR, hit, "v")],
        high_size=config.N_HIGH, n_hit=n_hit, io_split=io_split))
    return name


def add_loop(job, name, v, w, input_loop=""):
    # block is unused with nLow = 0 and a dense W -- the high-mode phase always
    # contracts N_SC fields -- but the module rejects zero.
    job.add(M.a2a_loop_new(name, left=v, right=w, n_low=0,
                           block=config.N_SC, input_loop=input_loop))
    return name


def add_ama(job, loop_root, hit=config.AMA_HIT, io_split=""):
    """The charm loop of the AMA correction hit, at nHit = 1 and unchained.
    Named for the correction, not the hit index (loop_c_ama), since nothing
    else in the campaign uses hit config.AMA_HIT. The hit index still picks
    the files."""
    w = add_noise_and_w(job, "ama", hit)
    v = add_v(job, f"v_{FLAVOR}_ama", hit, 1, io_split)
    loop = add_loop(job, f"loop_{FLAVOR}_ama", v, w)
    job.add(M.write_prop(f"save_{loop}", prop=loop,
                         file=f"{loop_root}/{loop}",
                         format=config.PROP_IO_FORMAT))


def build_job(n_hit=config.N_HIT, ama=True, run_id=None,
              loop_root=config.LOOP_ROOT, io_split=""):
    # run_id and loop_root are parameters so a benchmark can emit this exact
    # structure at a lower hit count without landing on production's output.
    run_id = run_id or f"loop.charm.h{n_hit}" + (".ama" if ama else "")
    job = Job(run_id, schedule_file=config.schedule_file(run_id),
              graph_file=config.GRAPH)

    if ama:
        add_ama(job, loop_root, io_split=io_split)

    link = ""
    for h in range(n_hit):
        w = add_noise_and_w(job, f"h{h}", h)
        v = add_v(job, f"v_{FLAVOR}_h{h}", h, n_hit, io_split)
        link = add_loop(job, f"loop_{FLAVOR}_h{h}", v, w, link)

    # A PropagatorField is 144 complex per site: 77.3 GB whatever the hit count.
    job.add(M.write_prop(f"save_loop_{FLAVOR}", prop=link,
                         file=f"{loop_root}/loop_{FLAVOR}_h{n_hit}",
                         format=config.PROP_IO_FORMAT))

    return job, run_id


def main():
    parser = argparse.ArgumentParser(
        description="Generate the charm loop job.")
    parser.add_argument("tag", nargs="?", default=f"h{config.N_HIT}",
                        help="hN for the N-hit chain")
    parser.add_argument("--skip-ama", action="store_true",
                        help="leave out the AMA hit's loop")
    parser.add_argument("--io-split", metavar="LAYOUT", default="",
                        type=lambda s: s.replace(".", " "),
                        help="split-grid read layout for the V loads, e.g. "
                             "2.2.2.1 (default: full-grid reads)")
    args = parser.parse_args()

    match = re.fullmatch(r"h(\d+)", args.tag)
    if not match:
        parser.error(f"tag '{args.tag}' is not 'hN'")
    job, run_id = build_job(n_hit=int(match.group(1)), ama=not args.skip_ama,
                            io_split=args.io_split)

    out_dir = Path(config.OUTPUT_ROOT) / "production"
    job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
    print(f"wrote {run_id} to {out_dir}")


if __name__ == "__main__":
    main()
