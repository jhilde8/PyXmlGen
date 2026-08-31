"""
EMF verification job

We construct the EMF in three ways

1. full V and W vectors from disk input as left and right loop vectors. This reproduces 
   the current EMF production module. depending on the loop flavor, the v vector present
   in the loop is the same object as the external vector of the same flavor. These are 
   built as references to the same environment object.

2. Full V and W vectors from disk, precomputed A2A loop using the specific A2ALoopNew
   module. EMF inputs have been expanded to take this loop as input, since the downstream 
   functions do not care whether the loop is constructed in the EMF module or loaded in

3. full V, dense W vectors through the A2ALoopNew -> EMF path. verifies that the dense W
   version of the loop computation regresses to the full W case, and that this fully new 
   dense W and loop pre-computation regress to the current production standard

Once the verification passes, we will remove the capability of the EMF module to take in
loop vectors. This change is beneficial no matter the hit count, and at worst is a linear 
split of two pieces of the full EMF computation. Allows us to save and reuse the loops. 

This particular test loads in 2 hits of strange and light V vectors, 2 hits of expanded W vectors
and constructs 2 hits of dense W vectors from the saved noise fields. One xml is produced per loop
flavor. 

"""
from pathlib import Path

import config
import modules as M
from hadrons_xml import Job

FLAVOR_L = "l"
FLAVOR_S = "s"
HIT = 0

LOOP_FLAVOR = ["l", "s"]

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
TYPES = "0"
GAMMAS = "GammaMUGamma5 GammaMU"

EMF_BLOCK = EXT_HIGH
EMF_CACHE_BLOCK = config.CACHE_BLOCK_EMF_CMOF


def build_job(flavor, hits, n_low, block=LOOP_BLOCK):
    tag = f"{flavor}loop."+"".join(f"h{h}" for h in hits)
    run_id = f"verify.{tag}"
    job = Job(run_id)

    low_v = config.low_filestem(flavor, "v") if n_low else ""
    low_w = config.low_filestem(flavor, "w") if n_low else ""

    # ---- V vector load ---- #
    lv_ext = f"a2a_{FLAVOR_L}_v_{tag}"
    job.add(M.load_combined_a2a_vecs_v(
        lv_ext, f"{config.LOW_VW}", 400, f"{config.VW_BASE}/",
        [f"{FLAVOR_L}{h}_v" for h in hits], EXT_HIGH,
        config.LOW_BIN_SIZE, config.HIGH_BIN_SIZE, n_hit=0))
    
    sv_ext = f"a2a_{FLAVOR_S}_v_{tag}"
    job.add(M.load_combined_a2a_vecs_v(
        sv_ext, "", 0, f"{config.VW_BASE}/",
        [f"{FLAVOR_S}{h}_v" for h in hits], EXT_HIGH,
        config.LOW_BIN_SIZE, config.HIGH_BIN_SIZE, n_hit=0))
     
    # ---- W vector load ---- #
    noise = f"noise_{flavor}_{tag}"
    job.add(M.load_time_diluted_noise(
        noise, [config.noise_filestem(flavor, h) for h in hits], config.N_NOISE_PER_STEM))

    # light W vectors get low modes, so we branch the modules here. 
    if flavor == "l":
        lw_exp = f"a2a_{flavor}_w_exp_{tag}"
        job.add(M.load_combined_a2a_vecs_v(
            lw_exp, f"{config.LOW_VW}", LOW_TRUNC, f"{config.VW_BASE}/",
            [f"{flavor}{h}_w" for h in hits], config.N_HIGH,
            config.LOW_BIN_SIZE, config.HIGH_BIN_SIZE))

        lw_dense = f"a2a_{flavor}_w_dense_{tag}"
        job.add(M.load_combined_a2a_vecs_w(
            lw_dense, config.LOW_BIN_SIZE, low_w, n_low, noise))
    
    elif flavor == "s":
        sw_exp = f"a2a_{flavor}_w_exp_{tag}"
        job.add(M.load_combined_a2a_vecs_v(
            sw_exp, "", 0, f"{config.VW_BASE}/",
            [f"{flavor}{h}_w" for h in hits], config.N_HIGH,
            config.LOW_BIN_SIZE, config.HIGH_BIN_SIZE))
	
        sw_dense = f"a2a_{flavor}_w_dense_{tag}"
        job.add(M.load_combined_a2a_vecs_w(
            sw_dense, config.LOW_BIN_SIZE, low_w, n_low, noise))

    # --- the two precomputed loops ---------------------------------------
    if flavor == "s":
        left_loop   = sv_ext
        right_dense = sw_dense
        right_exp   = sw_exp

    elif flavor == "l":
        left_loop   = lv_ext
        right_dense = lw_dense
        right_exp   = lw_exp

    loop_exp = f"loop_{flavor}_exp_{tag}"
    job.add(M.a2a_loop_new(loop_exp, left=left_loop, right=right_exp,
                           n_low=n_low, block=block))

    loop_dense = f"loop_{flavor}_dense_{tag}"
    job.add(M.a2a_loop_new(loop_dense, left=left_loop, right=right_dense,
                           n_low=n_low, block=block))

    # --- the three meson fields ------------------------------------------
    # Same types and gammas throughout, so the three output directories hold
    # identically named .h5 files and diff_emf.py can match them by basename.
    job.add(M.a2a_extended_meson_field(
        f"emf_vec_{tag}", EMF_BLOCK, EMF_CACHE_BLOCK, TYPES,
        left=sv_ext, right=lv_ext, output=f"emf_verify/{tag}/vec",
        gammas1=GAMMAS, gammas2=GAMMAS,
        loop_vw1=left_loop, loop_vw2=right_exp))

    job.add(M.a2a_extended_meson_field(
        f"emf_exp_{tag}", EMF_BLOCK, EMF_CACHE_BLOCK, TYPES,
        left=sv_ext, right=lv_ext, output=f"emf_verify/{tag}/exp",
        gammas1=GAMMAS, gammas2=GAMMAS,
        loop=loop_exp))

    job.add(M.a2a_extended_meson_field(
        f"emf_dense_{tag}", EMF_BLOCK, EMF_CACHE_BLOCK, TYPES,
        left=sv_ext, right=lv_ext, output=f"emf_verify/{tag}/dense",
        gammas1=GAMMAS, gammas2=GAMMAS,
        loop=loop_dense))

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "verify_loop"
    n = 0
    hits = [0, 1]    
    for flavor in LOOP_FLAVOR:
        if flavor == "l": n_low = True 
        else: n_low=False

        job, run_id = build_job(flavor, hits, n_low)
        job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
        n += 1
    print(f"wrote {n} A2ALoopNew verification jobs to {out_dir}")
    print("diff with, for each job tag and trajectory:")
    print("  python3 Frontier/diff_mf.py emf_verify/<tag>/vec emf_verify/<tag>/exp   --traj N")
    print("  python3 Frontier/diff_mf.py emf_verify/<tag>/exp emf_verify/<tag>/dense --traj N")


if __name__ == "__main__":
    main()
