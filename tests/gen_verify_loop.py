"""
EMF verification job

We construct the EMF in three ways

1. full V and W vectors from disk input as left and right loop vectors. This
   reproduces the current EMF production module.

2. Full V and W vectors from disk, precomputed A2A loop using the specific
   A2ALoopNew module. EMF inputs have been expanded to take this loop as input,
   since the downstream functions do not care whether the loop is constructed
   in the EMF module or loaded in.

3. full V, dense W vectors through the A2ALoopNew -> EMF path. verifies that
   the dense W version of the loop computation regresses to the full W case,
   and that this fully new dense W and loop pre-computation regress to the
   current production standard.

Once the verification passes, we will remove the capability of the EMF module
to take in loop vectors. This change is beneficial no matter the hit count, and
at worst is a linear split of two pieces of the full EMF computation. Allows us
to save and reuse the loops.

One xml is produced per loop flavor, each covering the hits in `hits`.

Sizing: the loop's own V and W legs must be full size. A2ALoopNew infers the W
representation from the two array sizes and then checks the V block against
nHit*nt*Nsc, so a truncated V leg is rejected outright - and the dense path
genuinely needs every one of a hit's nt*Nsc expanded V modes to gather from.
The EMF's external legs are under no such constraint, and nothing about the
loop test depends on how many external modes there are, so they are truncated
hard (EXT_LOW low modes, EXT_HIGH per hit) to keep the meson fields and the
node count down. That is why the external and loop arrays are separate objects
here even for the loop's own flavor.
"""
import sys
from pathlib import Path

# The toolkit (config, modules, hadrons_xml, vector_pool) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import modules as M
from hadrons_xml import Job

LOOP_FLAVOR = ["l", "s"]

# One job per (flavor, hit set). The single-hit case is the cheap smoke test;
# the two-hit case is the one that matches production, and the only thing it
# adds is A2ALoopNew's per-hit offset arithmetic -- the gather itself is
# already exercised by one hit.
HIT_SETS = [[0], [0, 1]]

# Loop-sweep blocking inside A2ALoopNew: how many mode views are open on the
# device at once, independent of how many modes are resident on the host.
LOOP_BLOCK = 50

# Low-mode truncation, applied to every array that has low modes at all (the
# loop legs included - low modes carry no dilution and pair index for index in
# both W representations, so truncating them is safe and keeps A2ALoopNew's
# nLow offset arithmetic under test). Multiple of LOW_BIN_SIZE because the
# loaders read whole bins.
LOW_TRUNC = 2 * config.LOW_BIN_SIZE

# External legs only: two low bins and two high bins per hit.
EXT_LOW = LOW_TRUNC
EXT_HIGH = 2 * config.HIGH_BIN_SIZE

# Contraction types and the gamma families, paired diagonally (gammas1[i] with
# gammas2[i]).
TYPES = "0"
GAMMAS = "GammaMUGamma5 GammaMU"

EMF_BLOCK = config.HIGH_BIN_SIZE
EMF_CACHE_BLOCK = config.CACHE_BLOCK_EMF_CMOF


def low_stem(flavor, vw):
    """Low-mode filestem for `flavor`, or "" for flavors that have none."""
    return config.low_filestem(flavor, vw) if config.FLAVOR_HAS_LOW[flavor] else ""


def low_size(flavor, n_low):
    """`n_low` for flavors that have low modes, 0 for those that do not."""
    return n_low if config.FLAVOR_HAS_LOW[flavor] else 0


def build_job(flavor, hits, block=LOOP_BLOCK):
    tag = f"{flavor}loop." + "".join(f"h{h}" for h in hits)
    run_id = f"verify.{tag}"
    job = Job(run_id)

    n_low = low_size(flavor, LOW_TRUNC)

    # --- external legs, shared by all three EMFs -------------------------
    # Truncated, and separate objects from the loop legs below even for the
    # loop's own flavor. n_hit=0: no 1/nHit rescaling. A common factor cancels
    # in every diff, and leaving it out keeps the numbers comparable to the
    # norm2 each module logs.
    ext = {}
    for f in ("l", "s"):
        ext[f] = f"a2a_{f}_v_ext_{tag}"
        job.add(M.load_combined_a2a_vecs_v(
            ext[f], low_stem(f, "v"), low_size(f, EXT_LOW), f"{config.VW_BASE}/",
            [f"{f}{h}_v" for h in hits], EXT_HIGH,
            config.LOW_BIN_SIZE, config.HIGH_BIN_SIZE, n_hit=0))

    # --- the loop's mode arrays, full size by necessity ------------------
    v_loop = f"a2a_{flavor}_v_loop_{tag}"
    job.add(M.load_combined_a2a_vecs_v(
        v_loop, low_stem(flavor, "v"), n_low, f"{config.VW_BASE}/",
        [f"{flavor}{h}_v" for h in hits], config.N_HIGH,
        config.LOW_BIN_SIZE, config.HIGH_BIN_SIZE, n_hit=0))

    # Expanded W read straight from the stored w files, so this leg shares no
    # code with the dense one. The cost is that an exp-vs-dense failure has two
    # possible causes: the dense index map, or the stored w files disagreeing
    # with the noise. gen_verify_dense_ww.py covers the latter on its own.
    w_exp = f"a2a_{flavor}_w_exp_{tag}"
    job.add(M.load_combined_a2a_vecs_v(
        w_exp, low_stem(flavor, "w"), n_low, f"{config.VW_BASE}/",
        [f"{flavor}{h}_w" for h in hits], config.N_HIGH,
        config.LOW_BIN_SIZE, config.HIGH_BIN_SIZE, n_hit=0))

    noise = f"noise_{flavor}_{tag}"
    job.add(M.load_time_diluted_noise(
        noise, [config.noise_filestem(flavor, h) for h in hits],
        config.N_NOISE_PER_STEM))

    w_dense = f"a2a_{flavor}_w_dense_{tag}"
    job.add(M.load_combined_a2a_vecs_w(
        w_dense, config.LOW_BIN_SIZE, low_stem(flavor, "w"), n_low, noise))

    # --- the two precomputed loops ---------------------------------------
    loop_exp = f"loop_{flavor}_exp_{tag}"
    job.add(M.a2a_loop_new(loop_exp, left=v_loop, right=w_exp,
                           n_low=n_low, block=block))

    loop_dense = f"loop_{flavor}_dense_{tag}"
    job.add(M.a2a_loop_new(loop_dense, left=v_loop, right=w_dense,
                           n_low=n_low, block=block))

    # --- the three meson fields ------------------------------------------
    # Same legs, types and gammas throughout, so the three output directories
    # hold identically named .h5 files and diff_mf.py can match them by
    # basename.
    for rung, kwargs in (
            ("vec",   dict(loop_vw1=v_loop, loop_vw2=w_exp)),
            ("exp",   dict(loop=loop_exp)),
            ("dense", dict(loop=loop_dense))):
        job.add(M.a2a_extended_meson_field(
            f"emf_{rung}_{tag}", EMF_BLOCK, EMF_CACHE_BLOCK, TYPES,
            left=ext["s"], right=ext["l"], output=f"emf_verify/{tag}/{rung}",
            gammas1=GAMMAS, gammas2=GAMMAS, **kwargs))

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "verify_loop"
    n = 0
    for flavor in LOOP_FLAVOR:
        for hits in HIT_SETS:
            job, run_id = build_job(flavor, hits)
            job.write(out_dir / f"par.{run_id}.xml",
                      out_dir / f"schedule.{run_id}.txt")
            n += 1
    print(f"wrote {n} A2ALoopNew verification jobs to {out_dir}")
    print("diff with, for each job tag and trajectory:")
    print("  python3 Frontier/diff_mf.py emf_verify/<tag>/vec emf_verify/<tag>/exp   --traj N")
    print("  python3 Frontier/diff_mf.py emf_verify/<tag>/exp emf_verify/<tag>/dense --traj N")


if __name__ == "__main__":
    main()
