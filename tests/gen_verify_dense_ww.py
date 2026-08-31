"""
Dense-W verification job: compute the same meson field twice in one job --
once from the expanded W vectors read from disk (truncated to a few bins),
once from the dense W fields expanded in-job from the raw noise -- sharing
one truncated V array so the right-hand operand is bit-identical, then diff
the two HDF5 outputs offline with Frontier/diff_densew_mf.py.

Truncation is 3 high-mode bins = 3*128 = 384 expanded W vectors per hit,
chosen because lcm(12, 128) = 384: the truncation boundary aligns exactly
with a timeslice boundary (384 = 12 spin-color columns x 32 timeslices), so
every covered timeslice t = 0..31 is complete and the diff needs no
partial-timeslice handling. The dense side is full-volume either way (12
fields per hit), so only meson times 0..31 have an expanded reference to
compare against; the rest of the expanded field is checked to be exactly
zero by the diff script.

No low modes on either side (nLow = 0): they would go through the identical
loader and files in both paths and verify nothing, and excluding them makes
the expanded row index pure high-mode (row = hit*384 + t*12 + sc vs dense
row hit*12 + sc). No nHit normalization on either side: any common factor
would cancel in the diff, so raw values keep the test transparent.

Requires the raw noise files (config.noise_filestem) to exist for the test
trajectory, i.e. the generation-side SaveSpinColorDiagonalNoise rerun for
the configuration whose expanded W vectors are already on disk.

The hit count generalizes by parameter alone: main() writes a single-hit
job and a two-hit job.
"""
import sys
from pathlib import Path

# The toolkit (config, modules, hadrons_xml, vector_pool) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import modules as M
from hadrons_xml import Job

FLAVOR_L = "l"
FLAVOR_S = "s"
N_BINS_TRUNC = 3
HIGH_TRUNC = N_BINS_TRUNC * config.HIGH_BIN_SIZE
LOW_TRUNC = 2 * config.LOW_BIN_SIZE

def build_job(hits):
    hits = list(hits)
    tag = "".join(f"h{h}" for h in hits)
    run_id = f"verify.dense.lsww.{tag}"
    job = Job(run_id)

    wl_exp = f"a2a_{FLAVOR_L}_w_exp_{tag}"
    job.add(M.load_combined_a2a_vecs_v(
        wl_exp, f"{config.LOW_VW}", LOW_TRUNC, f"{config.VW_BASE}/",
        [f"{FLAVOR_L}{h}_w" for h in hits], HIGH_TRUNC,
        config.LOW_BIN_SIZE, config.HIGH_BIN_SIZE))
    
    ws_exp = f"a2a_{FLAVOR_S}_w_exp_{tag}"
    job.add(M.load_combined_a2a_vecs_v(
        ws_exp, "", 0, f"{config.VW_BASE}/",
        [f"{FLAVOR_S}{h}_w" for h in hits], HIGH_TRUNC,
        0, config.HIGH_BIN_SIZE))

    noise_l = f"noise_{FLAVOR_L}_{tag}"
    job.add(M.load_time_diluted_noise(
        noise_l, [config.noise_filestem(FLAVOR_L, h) for h in hits],
        config.N_NOISE_PER_STEM))
    
    noise_s = f"noise_{FLAVOR_S}_{tag}"
    job.add(M.load_time_diluted_noise(
        noise_s, [config.noise_filestem(FLAVOR_S, h) for h in hits],
        config.N_NOISE_PER_STEM))

    wl_dense = f"a2a_{FLAVOR_L}_w_dense_{tag}"
    job.add(M.load_combined_a2a_vecs_w(
        wl_dense, config.LOW_BIN_SIZE, f"{config.LOW_VW}", LOW_TRUNC, noise_s))

    ws_dense = f"a2a_{FLAVOR_S}_w_dense_{tag}"
    job.add(M.load_combined_a2a_vecs_w(
        ws_dense, 0, f"{config.LOW_VW}", 0, noise_s))
    
    job.add(M.a2a_new_meson_field(
        f"mf_verify_exp_{tag}", config.BLOCK_STRANGE_LEG, config.CACHE_BLOCK_MF,
        wl_exp, ws_exp, f"mf_verify/exp_{tag}/mf",
        config.GAMMA5, config.PION_MOM))
    job.add(M.a2a_new_meson_field(
        f"mf_verify_dense_{tag}", config.BLOCK_STRANGE_LEG, config.CACHE_BLOCK_MF,
        wl_dense, ws_dense, f"mf_verify/dense_{tag}/mf",
        config.GAMMA5, config.PION_MOM))

    return job, run_id


def main():
    out_dir = Path(config.OUTPUT_ROOT) / "verify_densew"
    n = 0
    for hits in ([0], [0, 1]):
        job, run_id = build_job(hits)
        job.write(out_dir / f"par.{run_id}.xml", out_dir / f"schedule.{run_id}.txt")
        n += 1
    print(f"wrote {n} dense-W verification jobs to {out_dir}")


if __name__ == "__main__":
    main()
