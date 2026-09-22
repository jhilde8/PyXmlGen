"""Verification job for A2ALoopNew's inputLoop chaining.

One loop built in a single call is compared against the same loop built as a
chain of four calls, each seeded from its predecessor. Both are handed to an
otherwise identical A2AExtendedMesonField, and the two meson fields are diffed
with the existing HDF5 harness:

    python3 ../diffs/diff_a2am.py emf_full_out.<traj> emf_chain_out.<traj>

EXPECT EXACTLY ZERO, not machine precision. LoopPropagator accumulates strictly
sequentially per site, starting from whatever the loop already holds:

    auto res = coalescedRead(loopv[ss]);
    for (int k = 0; k < lNk; k++)
        res = res + outerProduct(coalescedRead(l1[k][ss]), coalescedRead(l2[k][ss]));
    coalescedWrite(loopv[ss], res);

There is no tree reduction and no parallelism over k, so one call over 512 modes
and four calls over 128 perform the identical sequence of floating-point
additions. The chain merely stores the running total to memory three extra
times, and a double round-trip is exact. The EMF then inherits that: it is
deterministic on identical input, so bit-identical loops give bit-identical
meson fields. Anything nonzero here means something reordered, which is a
finding rather than rounding.

What this tests is PLUMBING, not arithmetic. The new code is getInput() pushing
par().inputLoop -- which is what orders the chain and keeps each link alive
until its successor has read it -- and the execute() branch that seeds from it.
Neither is reachable without the VM, so a dedicated executable calling
LoopPropagator in a loop would prove the accumulation identity while never
touching the declaration whose absence would silently produce a wrong loop.
That is why this is an XML job.

Chunking. The four 128-mode pieces come from four separate hit directories
rather than from slices of one array, which avoids depending on anything that
does not exist: no loader can begin reading at a bin other than the first, so a
single array cannot be split by loading. LoadCombinedA2AVecsV takes
highExtensions as a list and fills out[offset ...] one extension at a time with
offset advancing by highSize, so the baseline's mode h*128 + j IS chunk h's
mode j, in that order -- which is precisely what makes the equivalence exact
rather than approximate. nHit is 1 on every load, so the loader's 1/nHit factor
is the identity and both paths see byte-identical vectors.

highSize is one bin (128), so each extension is a single file, elem0 of that
hit's directory.

Strange is used because it has no low modes, so nLow is simply each array's
size. That matters: with left.size() == right.size() A2ALoopNew takes the
expanded-W branch and checks (size - nLow) % (nt*Nsc) == 0, so nLow = 0 on a
128-mode array is a hard Size error, not a benign default.

The loop's own legs are the same array on both sides, making the loop a
physically meaningless sum of V (x) V. That is deliberate -- the test is about
accumulation, not physics -- and it means the job needs no W vectors at all,
so it does not depend on whether expanded high-mode W survived the dense-W
migration. The EMF's external legs reuse the same array for the same reason.

Only the expanded path is exercised, which is sufficient: the seeding happens
before either branch is chosen, and gen_verify_loop.py already covers the dense
path. That job also covers what this one cannot -- that an empty inputLoop still
zeroes -- since every existing caller passes it empty and is validated against
the loop_vw reference.

Not covered here, and better suited to a dedicated executable: the fine chain,
128 links of one mode each. 128 modules of XML to test an arithmetic identity is
the wrong shape, while the same thing in a test binary is a three-line loop over
random(pRNG, ...) inputs with an in-memory norm2 comparison and no differ.
"""
import sys
from pathlib import Path

# The toolkit (config, modules, hadrons_xml) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import modules as M
from hadrons_xml import Job

# No low modes, so nLow is just each array's size on every call.
TEST_FLAVOR = "s"

# Four chunks of one high-mode bin each, one per hit directory.
N_CHUNKS = 4
CHUNK = config.HIGH_BIN_SIZE
N_TOTAL = N_CHUNKS * CHUNK

# nHit = 1 makes the loader's 1/nHit factor the identity, so both paths see
# byte-identical vectors.
NO_NORM = 1

LOOP_BLOCK = 50

# One type and one gamma family: nFiles = types * gammas1, so this is a single
# output file per EMF and the smallest thing the differ can compare.
TYPES = "0"
GAMMAS = "Gamma5"

EMF_LEFT_BLOCK = config.LEFT_BLOCK
EMF_RIGHT_BLOCK = config.RIGHT_BLOCK

# Merged layout (one file per type/gamma pair), which is what diffs/diff_a2am.py
# reads. The per-timeslice layout would need diff_a2am_ts.py and buys nothing
# here, since both sides are the same code path.
EMF_TS_IO = False


def load_chunks(job, name, hits):
    """One array holding CHUNK modes from each hit in `hits`, in that order."""
    job.add(M.load_combined_a2a_vecs_v(
        name, low_filestem="", n_low=0,
        high_stem=config.high_stem(),
        high_extensions=[config.high_extension(TEST_FLAVOR, h, "v") for h in hits],
        high_size=CHUNK,
        low_bin_size=config.LOW_BIN_SIZE,
        high_bin_size=config.HIGH_BIN_SIZE, n_hit=NO_NORM))
    return name


def build_job():
    run_id = "verify.loop.chain"
    job = Job(run_id, schedule_file=config.schedule_file(run_id),
              graph_file=config.GRAPH)

    hits = list(range(N_CHUNKS))

    # Baseline: every chunk in one array, one call over all of it.
    v_all = load_chunks(job, "v_all", hits)
    job.add(M.a2a_loop_new("loop_full", left=v_all, right=v_all,
                           n_low=N_TOTAL, block=LOOP_BLOCK, input_loop=""))

    # Chain: the same vectors, same order, one call per chunk. Each chunk's
    # array dies at its own link, so only one is resident at a time.
    link = ""
    for k, h in enumerate(hits):
        v_k = load_chunks(job, f"v_c{k}", [h])
        nxt = f"loop_c{k}"
        job.add(M.a2a_loop_new(nxt, left=v_k, right=v_k, n_low=CHUNK,
                               block=LOOP_BLOCK, input_loop=link))
        link = nxt

    # Identical in every respect but the loop they are handed.
    for name, loop in (("emf_full", "loop_full"), ("emf_chain", link)):
        job.add(M.a2a_extended_meson_field(
            name, EMF_LEFT_BLOCK, EMF_RIGHT_BLOCK, TYPES, left=v_all, right=v_all,
            output=f"{config.TMP_OUTPUT}/{name}", gammas1=GAMMAS,
            gammas2=GAMMAS, loop=loop, time_slice_io=EMF_TS_IO))

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "verify_loop_chain"
    job, run_id = build_job()
    job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
    print(f"wrote {run_id} to {out_dir}")


if __name__ == "__main__":
    main()
