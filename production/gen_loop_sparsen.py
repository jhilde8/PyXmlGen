"""Strange and charm A2A loop propagators, and sparsened strange V vectors.

8 hits, 32 nodes, high modes only.

This job builds the loops that no other job produces as a by-product. The light
loop is NOT here: the light-light job already holds V_l and W_l at full hit
count, and the loop contraction against a dense W is free next to that load, so
it belongs there -- ahead of the smearing, since A2ACovariantSmear moves its
source, so the loop and any unsmeared-V sparsening have to precede it. Charm has
no such host; nothing else in the campaign loads charm vectors. Strange could
ride along in svlv, but at 6 and 8 hits svlv tiles V_s into blocks and never
holds the whole thing, so the loop would have to be chained there anyway.

Dropping the low modes is what makes this job simple. Strange and charm have
none (config.FLAVOR_HAS_LOW), so:

  - every load is one whole hit of high modes from its own directory, which the
    binned loaders already address without needing a bin offset;
  - V_s at 8 hits is 12288 modes, exactly 48 bins of 256, so the sparsening
    needs no A2ACoarseGrid size that is not already registered.

Both were blockers while the light low modes were in scope -- 2000 divides by
none of the registered bin sizes, and no loader can start reading at a bin other
than the first -- and both disappear with this split.

Per hit the job holds one expanded V block, its dense W partner, and the
loader's bin staging:

    V (unpacked)   1536 fields   36.0 GiB per rank   (nt*Nsc, one hit)
    loader staging  128 fields    3.0 GiB            (one bin, transient)
    W dense          12 fields    0.3 GiB            (N_SC per hit)
    coarse                        0.6 GiB            (1/64 of V, [4,4,4,1])
    A2ACoarseGrid bvec            0.6 GiB            (packed copy, pre-write)
    loop                          0.3 GiB            (144 cplx/site)
                                 ---------
                                 40.8 GiB of ~59.6 usable

at 24 MiB per field on 256 ranks. That is HOST memory: Frontier builds with
--enable-unified=no, so lattice data is host-resident (512 GB DDR per node, 64
GB per rank) and HBM holds only the open views, bounded by --device-mem.

32 nodes rather than 16 is set by the first line: at 16 nodes a field is 48 MiB
and one hit's V alone is 72.0 GiB, past the per-rank cap.

The staging line used to be 1536 fields rather than 128, because
LoadCombinedA2AVecsV read a whole extension's bins before unpacking. That put
one hit at 72.0 GiB per rank and would have forced 64 nodes. It now reads bin by
bin -- identical IO, since A2AVectorsIo::read opens and closes one file per
element regardless -- so the staging is a factor highBinSize smaller. If that
change is ever reverted, this job needs 64 nodes.

Nothing scales with hit count except the number of links.

The loop accumulates across hits through A2ALoopNew's inputLoop, each link
seeding from its predecessor instead of from zero. Declaring inputLoop as a
module input orders the chain and lets the VM free every link but the pair in
flight, so the accumulator costs one PropagatorField, not one per hit.

An alternative with no chain at all: at 256 nodes a field is 3 MiB and the whole
12288-mode V_s is 36.0 GiB, so one A2ALoopNew call per flavour would do it.
Node-hours are roughly invariant between the two -- this job is pure IO and the
measured Lustre rate is per-node-limited at 515 MiB/s per node, so 8x the nodes
finishes in 8x less wall time for the same cost -- and 256 nodes matches the
contraction jobs. Worth measuring the 32-node load rate before assuming the
chain is the cheaper structure.

Normalization is applied at load, not in the loop: A2ALoopNew adds whatever it
is handed. Every array here is high-mode, so every V load passes nHit = N_HIT
and the loop applies nothing. (The distinction bites only where low modes are
present -- those are deterministic and not hit-averaged, so one scalar on an
accumulated light loop would wrongly scale its 2000 low modes as well.)

Sparsening rides on loads the loop has already paid for, which is the whole
reason it lives here. Charm is not sparsened.

Sparsened file layout is one file per hit. In A2AVectorsIo the multiFile flag
decides FILE count and binSize decides RECORD count within the file:

    multiFile = true    <stem>.<traj>/elem<i>.bin, one file per record
    multiFile = false   <stem>.<traj>.bin, one file holding every record

A2ACoarseGrid hardcodes false, so calling it once per hit at binSize 256 gives
exactly one file per hit with 6 records inside, and no code change. Read it back
with multi_file=False, size=N_HIGH and bin size 256 -- the default
multi_file=True on the binned loaders would look for elem0.bin and miss, and the
read bin size is welded to 256 by the SiteSpinorSet template.

Ordering: this job and the light-light job must both finish before svlv, which
consumes all three loops through MIO::LoadProp. They do not depend on each
other.
"""
import sys
from pathlib import Path

# The toolkit (config, modules, hadrons_xml) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import modules as M
from hadrons_xml import Job

N_HIT = 8

# Flavours needing a loop built here, and which of those get their V sparsened.
LOOP_FLAVORS = ("s", "c")
SPARSEN_FLAVORS = ("s",)

# Records per sparsened file. Must divide N_HIGH and be a registered
# A2ACoarseGrid instantiation: 1536 is 6 bins of 256. On a non-divisor the
# module prints "Irrelevant binsize ... NOT saved" and then writes anyway,
# reading past the end of the array on the tail bin.
SPARSE_BIN_HIGH = 256

# A2ALoopNew mode-sweep blocking: how many field views are open on the device at
# once, independent of how many modes are host-resident. 2*block*24 MiB has to
# fit inside --device-mem, which defaults to 128 MiB and asserts if a single view
# exceeds it. The dense high-mode phase contracts only N_SC modes and ignores
# this, so it only binds the expanded path.
LOOP_BLOCK = 50

# Sparsened vectors and loops go to Lustre, not node-local NVMe: both are
# collective single-file writes. EDIT THESE to the real Frontier directories.
SPARSE_ROOT = "/lustre/orion/phy157/scratch/jhilde/64I/vw_sparse"
LOOP_ROOT = "/lustre/orion/phy157/scratch/jhilde/64I/loop"


def add_flavor(job, flavor, n_hit, sparsen, sparse_root=SPARSE_ROOT):
    """One load-and-contract per hit. Returns the name of the finished loop."""
    link = ""

    for h in range(n_hit):
        noise = f"noise_{flavor}_h{h}"
        v = f"hi_v_{flavor}_h{h}"
        w = f"hi_w_{flavor}_h{h}"

        job.add(M.load_time_diluted_noise(
            noise, [config.noise_filestem(flavor, h)],
            config.N_NOISE_PER_STEM))

        # n_low = 0 skips the low-mode read entirely (guarded in the module), so
        # this is one hit's expanded V block and nothing else. One extension per
        # call, so this loads exactly that hit. n_hit is the GLOBAL hit count,
        # not the number of extensions here -- the 1/nHit hit-average factor
        # belongs to the estimator, not to how the load is chunked -- which is
        # why the loop applies none. (VectorPool.combined hardcodes
        # n_hit=len(hits) and so cannot be used for a per-hit load.)
        job.add(M.load_combined_a2a_vecs_v(
            v, low_filestem="", n_low=0,
            high_stem=f"{config.VW_BASE}/",
            high_extensions=[f"{flavor}{h}_v"],
            high_size=config.N_HIGH,
            low_bin_size=config.LOW_BIN_SIZE,
            high_bin_size=config.HIGH_BIN_SIZE, n_hit=n_hit))

        # Dense W: n_low = 0, so this is the noise expanded into N_SC fields for
        # this hit and nothing more. A2ALoopNew sees 1536 V against 12 W, deduces
        # the dense representation from the ratio, and reads it as a one-hit
        # array with no low-mode phase to run.
        job.add(M.load_combined_a2a_vecs_w(
            w, config.LOW_BIN_SIZE, low_filestem="", n_low=0, noise=noise))

        nxt = f"loop_{flavor}_h{h}"
        job.add(M.a2a_loop_new(nxt, left=v, right=w, n_low=0, block=LOOP_BLOCK,
                               input_loop=link))
        link = nxt

        # After the loop, so V's last consumer is here and the whole hit is
        # released together before the next one loads.
        if sparsen:
            job.add(M.a2a_coarse_grid(
                f"sp_{flavor}_h{h}", SPARSE_BIN_HIGH, v,
                config.COARSE_BLOCK_SIZE, config.COARSE_OFFSETS,
                f"{sparse_root}/{flavor}{h}_v"))

    return link


def build_job(flavors=LOOP_FLAVORS, n_hit=N_HIT, run_id=None,
              sparse_root=SPARSE_ROOT, loop_root=LOOP_ROOT):
    # run_id and the two output roots are parameters so that benchmarks/ can
    # emit this exact structure at other hit counts without duplicating it, and
    # without its sparsened vectors landing on top of production's -- the
    # per-hit filenames carry no hit-count tag.
    run_id = run_id or f"loop.sparsen.h{n_hit}"
    job = Job(run_id, schedule_file=config.schedule_file(run_id),
              graph_file=config.GRAPH)

    for flavor in flavors:
        # A low-mode leg needs V and W resident together at full length and has
        # no path here; asking for one would silently drop it from the loop
        # rather than fail, so refuse at generation time.
        if config.FLAVOR_HAS_LOW[flavor]:
            raise ValueError(
                f"module '{run_id}': flavor '{flavor}' has low modes, which this "
                f"job has no phase for -- build that loop in the job that "
                f"already loads its V and W")

        loop = add_flavor(job, flavor, n_hit, flavor in SPARSEN_FLAVORS,
                          sparse_root)

        # The chain's last link holds the finished loop. A PropagatorField is 144
        # complex per site, so this is 77.3 GB whatever the hit count.
        job.add(M.write_prop(f"save_loop_{flavor}", prop=loop,
                             file=f"{loop_root}/loop_{flavor}_h{n_hit}",
                             format=config.PROP_IO_FORMAT))

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "production"
    job, run_id = build_job()
    job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
    print(f"wrote {run_id} to {out_dir}")


if __name__ == "__main__":
    main()
