"""Charm A2A loop propagator, chained one hit at a time.

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
    loader staging       128 fields    3.0 GiB   (one bin, transient)
    W dense               12 fields    0.3 GiB
    gather buffer         12 fields    0.3 GiB   (A2ALoopNew vtilde)
    loop, link + seed      2 props     0.6 GiB
                                      ---------
                                      40.2 GiB of 59.6

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

Two jobs come out of this file, picked by the same tag gen_meson_fields.py
takes:

    gen_charm_loop.py h8      the production chain above
    gen_charm_loop.py ama     loop_c_ama_sloppy and loop_c_ama_exact

The AMA pair is one hit each at nHit = 1, kept out of the AMA contraction job
for the same reason charm is kept out of the 512-node one: nothing else there
loads charm vectors, so the contraction job should not spend its nodes on a
pure-IO step.
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
        w, config.LOW_BIN_SIZE, low_filestem="", n_low=0, noise=noise))
    return w


def add_v(job, name, hit, n_hit, exact=False):
    """One hit's expanded high-mode V block. n_hit is the GLOBAL hit count of
    the estimator this block belongs to, not the number of extensions read
    here -- the 1/nHit factor belongs to the estimator, not to how the load is
    chunked -- which is why the loop applies none."""
    job.add(M.load_combined_a2a_vecs_v(
        name, low_filestem="", n_low=0,
        high_stem=config.high_stem(exact),
        high_extensions=[config.high_extension(FLAVOR, hit, "v", exact)],
        high_size=config.N_HIGH,
        low_bin_size=config.LOW_BIN_SIZE,
        high_bin_size=config.HIGH_BIN_SIZE, n_hit=n_hit))
    return name


def add_loop(job, name, v, w, input_loop=""):
    # block is unused with nLow = 0 and a dense W -- the high-mode phase always
    # contracts N_SC fields -- but the module rejects zero.
    job.add(M.a2a_loop_new(name, left=v, right=w, n_low=0,
                           block=config.N_SC, input_loop=input_loop))
    return name


def build_job(n_hit=config.N_HIT, run_id=None, loop_root=config.LOOP_ROOT):
    # run_id and loop_root are parameters so a benchmark can emit this exact
    # structure at a lower hit count without landing on production's output.
    run_id = run_id or f"loop.charm.h{n_hit}"
    job = Job(run_id, schedule_file=config.schedule_file(run_id),
              graph_file=config.GRAPH)

    link = ""
    for h in range(n_hit):
        w = add_noise_and_w(job, f"h{h}", h)
        v = add_v(job, f"v_{FLAVOR}_h{h}", h, n_hit)
        link = add_loop(job, f"loop_{FLAVOR}_h{h}", v, w, link)

    # A PropagatorField is 144 complex per site: 77.3 GB whatever the hit count.
    job.add(M.write_prop(f"save_loop_{FLAVOR}", prop=link,
                         file=f"{loop_root}/loop_{FLAVOR}_h{n_hit}",
                         format=config.PROP_IO_FORMAT))

    return job, run_id


def build_ama_job(hit=config.AMA_HIT, run_id=None, loop_root=config.LOOP_ROOT):
    """The two charm loops of the AMA correction hit, sloppy and exact.

    No chain: each accuracy is a single-hit estimator, so nHit = 1 and the two
    loops are independent. They share one noise and one dense W -- W is the raw
    noise, identical in both sets -- and the sloppy V is released before the
    exact one loads, so the footprint is the 32-node job's one hit either way.

    Named for the correction, not the hit index (loop_c_ama_sloppy), since
    nothing else in the campaign uses hit config.AMA_HIT. The hit index still
    picks the files.
    """
    run_id = run_id or "loop.charm.ama"
    job = Job(run_id, schedule_file=config.schedule_file(run_id),
              graph_file=config.GRAPH)

    w = add_noise_and_w(job, "ama", hit)
    for acc, exact in (("sloppy", False), ("exact", True)):
        suffix = "_exact" if exact else ""
        v = add_v(job, f"v_{FLAVOR}_ama{suffix}", hit, 1, exact=exact)
        loop = add_loop(job, f"loop_{FLAVOR}_ama_{acc}", v, w)
        job.add(M.write_prop(f"save_{loop}", prop=loop,
                             file=f"{loop_root}/{loop}",
                             format=config.PROP_IO_FORMAT))

    return job, run_id


def main():
    parser = argparse.ArgumentParser(
        description="Generate one charm loop job.")
    parser.add_argument("tag", nargs="?", default=f"h{config.N_HIT}",
                        help="hN for the N-hit chain, or ama for the "
                             "correction hit's sloppy and exact loops")
    args = parser.parse_args()

    if args.tag == "ama":
        job, run_id = build_ama_job()
    else:
        match = re.fullmatch(r"h(\d+)", args.tag)
        if not match:
            parser.error(f"tag '{args.tag}' is neither 'ama' nor 'hN'")
        job, run_id = build_job(n_hit=int(match.group(1)))

    out_dir = Path(config.OUTPUT_ROOT) / "production"
    job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
    print(f"wrote {run_id} to {out_dir}")


if __name__ == "__main__":
    main()
