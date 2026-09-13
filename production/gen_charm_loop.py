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
"""
import sys
from pathlib import Path

# The toolkit (config, modules, hadrons_xml) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import modules as M
from hadrons_xml import Job

FLAVOR = "c"


def build_job(n_hit=config.N_HIT, run_id=None, loop_root=config.LOOP_ROOT):
    # run_id and loop_root are parameters so a benchmark can emit this exact
    # structure at a lower hit count without landing on production's output.
    run_id = run_id or f"loop.charm.h{n_hit}"
    job = Job(run_id, schedule_file=config.schedule_file(run_id),
              graph_file=config.GRAPH)

    link = ""
    for h in range(n_hit):
        noise = f"noise_{FLAVOR}_h{h}"
        v = f"v_{FLAVOR}_h{h}"
        w = f"w_{FLAVOR}_h{h}"

        job.add(M.load_time_diluted_noise(
            noise, [config.noise_filestem(FLAVOR, h)],
            config.N_NOISE_PER_STEM))

        job.add(M.load_combined_a2a_vecs_w(
            w, config.LOW_BIN_SIZE, low_filestem="", n_low=0, noise=noise))

        job.add(M.load_combined_a2a_vecs_v(
            v, low_filestem="", n_low=0,
            high_stem=f"{config.VW_BASE}/",
            high_extensions=[f"{FLAVOR}{h}_v"],
            high_size=config.N_HIGH,
            low_bin_size=config.LOW_BIN_SIZE,
            high_bin_size=config.HIGH_BIN_SIZE, n_hit=n_hit))

        # block is unused with nLow = 0 and a dense W -- the high-mode phase
        # always contracts N_SC fields -- but the module rejects zero.
        nxt = f"loop_{FLAVOR}_h{h}"
        job.add(M.a2a_loop_new(nxt, left=v, right=w, n_low=0,
                               block=config.N_SC, input_loop=link))
        link = nxt

    # A PropagatorField is 144 complex per site: 77.3 GB whatever the hit count.
    job.add(M.write_prop(f"save_loop_{FLAVOR}", prop=link,
                         file=f"{loop_root}/loop_{FLAVOR}_h{n_hit}",
                         format=config.PROP_IO_FORMAT))

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "production"
    job, run_id = build_job()
    job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
    print(f"wrote {run_id} to {out_dir}")


if __name__ == "__main__":
    main()
