"""
A2ALoopNew verification job: build the same extended meson field three ways in
one job, from one shared V array, one shared noise object and one shared pair
of external legs, then diff the HDF5 outputs offline with Frontier/diff_mf.py.

    vec    EMF builds the loop in-module from (V, W expanded)
    exp    EMF takes a loop from A2ALoopNew(V, W expanded)
    dense  EMF takes a loop from A2ALoopNew(V, W dense)

Two diffs, each varying exactly one thing:

  vec vs exp    only where the propagator came from -- in-module versus a
                named environment object. Verifies the loop-input plumbing.
                Rung `vec` deliberately touches none of the new loop-building
                code (expanded W, plain index-for-index pairing, the original
                whole-array LoopPropagator wrapper), so it stays an
                independent reference.
  exp vs dense  only the W representation. Same module, same block size, so a
                failure here is the dense index map or the timeslice gather.

The EMF's in-module path is scaffolding with a known deletion date: it cannot
fit at production hit counts, since both of the loop's mode arrays must stay
resident alongside the external legs for the module's whole execution. It goes
away once these diffs pass, and rung `vec` goes with it.

Both W arrays are expanded from the SAME noise object rather than one being
read from the stored W files, so the comparison isolates the loop contraction.
Whether the stored W files agree with the noise is a separate question, already
covered by gen_verify_densew.py.

The loop's V array cannot be truncated -- the dense path needs a complete
nt*N_SC expanded block per hit to gather from, and A2ALoopNew asserts exactly
that. The low block CAN be, since low modes carry no dilution and pair
index-for-index in both representations, so main() writes an nLow=0 job (the
cheap one, still covering the gather) and an nLow=400 job that additionally
covers A2ALoopNew's nLow offset arithmetic.

The external legs are a separate, heavily truncated V read through the same
loader production uses -- one high bin, no low modes -- since nothing about
the loop test depends on how many external modes there are, and the meson
fields stay small.

Resident field count: V nLow+N_HIGH, W_exp nLow+N_HIGH, W_dense nLow+N_SC,
external legs 128, plus two loop propagators. That is what sets the node count.
"""
from pathlib import Path

import config
import modules as M
from hadrons_xml import Job

FLAVOR = "l"
HIT = 0

# Loop-sweep blocking inside A2ALoopNew: how many mode views are open on the
# device at once, independent of how many modes are resident on the host.
LOOP_BLOCK = 50

# Truncated low block for the nLow > 0 variant: 2 bins, kept a multiple of
# LOW_BIN_SIZE because both W loaders assert lowSize % binSize == 0.
LOW_TRUNC = 2 * config.LOW_BIN_SIZE

# External legs: one high bin, no low modes. 128x128 meson fields.
EXT_HIGH = config.HIGH_BIN_SIZE

# All four loop-contraction types, and two gamma families paired diagonally
# (gammas1[i] with gammas2[i]) so more than one index structure of the loop is
# probed -- a single type could be blind to, say, a colour-index error under
# its colour trace.
TYPES = "0 1 2 3"
GAMMAS = "GammaMUGamma5 GammaMU"

EMF_BLOCK = EXT_HIGH
EMF_CACHE_BLOCK = config.CACHE_BLOCK_EMF_CMOF


def build_job(flavor, hit, n_low, block=LOOP_BLOCK):
    tag = f"{flavor}{hit}_nlow{n_low}"
    run_id = f"verify.loop.{tag}"
    job = Job(run_id)

    low_v = config.low_filestem(flavor, "v") if n_low else ""
    low_w = config.low_filestem(flavor, "w") if n_low else ""

    # --- the loop's mode arrays ------------------------------------------
    # n_hit=0: no 1/nHit rescaling. A common factor would cancel in every
    # diff, and leaving it out keeps these numbers comparable to the norm2
    # each module logs.
    v = f"a2a_{flavor}_v_{tag}"
    job.add(M.load_combined_a2a_vecs_v(
        v, low_v, n_low, f"{config.VW_BASE}/",
        [f"{flavor}{hit}_v"], config.N_HIGH,
        config.LOW_BIN_SIZE, config.HIGH_BIN_SIZE, n_hit=0))

    noise = f"noise_{flavor}_{tag}"
    job.add(M.load_time_diluted_noise(
        noise, [config.noise_filestem(flavor, hit)], config.N_NOISE_PER_STEM))

    w_exp = f"a2a_{flavor}_w_exp_{tag}"
    job.add(M.load_binned_a2a_vecs_w(
        w_exp, config.LOW_BIN_SIZE, low_w, n_low, noise))

    w_dense = f"a2a_{flavor}_w_dense_{tag}"
    job.add(M.load_combined_a2a_vecs_w(
        w_dense, config.LOW_BIN_SIZE, low_w, n_low, noise))

    # --- external legs, shared by all three EMFs -------------------------
    ext = f"a2a_{flavor}_ext_{tag}"
    job.add(M.load_combined_a2a_vecs_v(
        ext, "", 0, f"{config.VW_BASE}/",
        [f"{flavor}{hit}_v"], EXT_HIGH,
        config.LOW_BIN_SIZE, config.HIGH_BIN_SIZE, n_hit=0))

    # --- the two precomputed loops ---------------------------------------
    loop_exp = f"loop_exp_{tag}"
    job.add(M.a2a_loop_new(loop_exp, left=v, right=w_exp,
                           n_low=n_low, block=block))

    loop_dense = f"loop_dense_{tag}"
    job.add(M.a2a_loop_new(loop_dense, left=v, right=w_dense,
                           n_low=n_low, block=block))

    # --- the three meson fields ------------------------------------------
    # Same types and gammas throughout, so the three output directories hold
    # identically named .h5 files and diff_mf.py can match them by basename.
    job.add(M.a2a_extended_meson_field(
        f"emf_vec_{tag}", EMF_BLOCK, EMF_CACHE_BLOCK, TYPES,
        left=ext, right=ext, output=f"emf_verify/{tag}/vec",
        gammas1=GAMMAS, gammas2=GAMMAS,
        loop_vw1=v, loop_vw2=w_exp))

    job.add(M.a2a_extended_meson_field(
        f"emf_exp_{tag}", EMF_BLOCK, EMF_CACHE_BLOCK, TYPES,
        left=ext, right=ext, output=f"emf_verify/{tag}/exp",
        gammas1=GAMMAS, gammas2=GAMMAS,
        loop=loop_exp))

    job.add(M.a2a_extended_meson_field(
        f"emf_dense_{tag}", EMF_BLOCK, EMF_CACHE_BLOCK, TYPES,
        left=ext, right=ext, output=f"emf_verify/{tag}/dense",
        gammas1=GAMMAS, gammas2=GAMMAS,
        loop=loop_dense))

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "verify_loop"
    n = 0
    for n_low in (0, LOW_TRUNC):
        job, run_id = build_job(FLAVOR, HIT, n_low)
        job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
        n += 1
    print(f"wrote {n} A2ALoopNew verification jobs to {out_dir}")
    print("diff with, for each job tag and trajectory:")
    print("  python3 Frontier/diff_mf.py emf_verify/<tag>/vec emf_verify/<tag>/exp   --traj N")
    print("  python3 Frontier/diff_mf.py emf_verify/<tag>/exp emf_verify/<tag>/dense --traj N")


if __name__ == "__main__":
    main()
