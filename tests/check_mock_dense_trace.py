"""Compare the three mock contractor runs against the numpy oracle.

Run gen_mock_dense_trace.py, then the three binaries, then this:

    python3 tests/gen_mock_dense_trace.py
    cd output/mock_dense && mkdir -p corr.expanded corr.dense corr.densets
    HadronsContractor par.mock.expanded.xml
    ContractorDense   par.mock.dense.xml
    ContractorDense   par.mock.densets.xml
    python3 tests/check_mock_dense_trace.py | tee output/mock_dense/check.log

expanded vs dense isolates the index remap; dense vs densets isolates the
one-file-per-timeslice loading. If they disagree, which pair differs tells you
which of the two changes broke it.

Exits non-zero on any mismatch.
"""
import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gen_mock_dense_trace import OUT_DIR, TRAJ, N_T

TAGS = ("expanded", "dense", "densets")

# saveCorrelator builds the stem as terms[0]_times[0]_terms.back(), and the
# product is "mf mf" at time 0.
FILE_STEM = "mf_0_mf"

TOL = 1e-10


def read_correlator(tag):
    path = OUT_DIR / f"corr.{tag}" / f"{FILE_STEM}.{TRAJ}.h5"
    with h5py.File(path, "r") as f:
        raw = f[FILE_STEM]["correlator"][...]
    return raw["re"] + 1j * raw["im"]


def main():
    oracle_path = OUT_DIR / "correlator.npy"
    if not oracle_path.exists():
        print(f"no oracle at {oracle_path}; run gen_mock_dense_trace.py first")
        return 2
    oracle = np.load(oracle_path)

    values = {}
    for tag in TAGS:
        try:
            values[tag] = read_correlator(tag)
        except (OSError, KeyError) as e:
            print(f"could not read '{tag}': {e}")
            return 2

    width = 30
    print(f"{'t':>2}  {'numpy oracle':>{width}}  "
          + "  ".join(f"{tag:>{width}}" for tag in TAGS))
    for t in range(N_T):
        row = "  ".join(f"{values[tag][t].real:+.8e}{values[tag][t].imag:+.8e}j"
                        for tag in TAGS)
        print(f"{t:>2}  {oracle[t].real:+.8e}{oracle[t].imag:+.8e}j  {row}")

    scale = np.max(np.abs(oracle))
    print(f"\ncorrelator scale {scale:.3e}, tolerance {TOL:.0e} relative\n")

    worst = 0.0
    checks = [(f"{tag} vs oracle", values[tag], oracle) for tag in TAGS]
    checks.append(("dense vs expanded", values["dense"], values["expanded"]))
    checks.append(("densets vs dense", values["densets"], values["dense"]))
    for label, a, b in checks:
        err = np.max(np.abs(a - b))
        worst = max(worst, err)
        print(f"  max |{label:<20s}| = {err:.3e}  "
              + ("ok" if err <= TOL * scale else "MISMATCH"))

    if worst > TOL * scale:
        print(f"\nFAILED: worst deviation {worst:.3e} exceeds {TOL * scale:.3e}")
        return 1
    print(f"\nPASSED: worst deviation {worst:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
