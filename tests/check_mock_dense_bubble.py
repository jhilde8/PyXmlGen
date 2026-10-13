"""Compare the three mock bubble runs against the numpy oracles.

    python3 tests/check_mock_dense_bubble.py | tee output/mock_dense_bubble/check.log

expanded vs dense isolates the index remap (accTrDense for one term,
accTrMulDense for two); dense vs densets isolates the one-file-per-timeslice
loading. Which pair differs says which change broke it.

Exits non-zero on any mismatch.
"""
import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gen_mock_dense_bubble import OUT_DIR, TRAJ, STEM_1, STEM_2
from gen_mock_dense_trace import N_T

TAGS = ("expanded", "dense", "densets")

# (oracle file, saveCorrelator stem, label)
PRODUCTS = [("bubble1.npy", STEM_1, "tr[Pi(t)]"),
            ("bubble2.npy", STEM_2, "tr[Pi(t) Pi(t+1)]")]

TOL = 1e-10


def read_bubble(tag, stem):
    path = OUT_DIR / f"corr.{tag}" / f"{stem}.{TRAJ}.h5"
    with h5py.File(path, "r") as f:
        raw = f[stem]["correlator"][...]
    return raw["re"] + 1j * raw["im"]


def main():
    worst = 0.0
    for oracle_file, stem, label in PRODUCTS:
        oracle_path = OUT_DIR / oracle_file
        if not oracle_path.exists():
            print(f"no oracle at {oracle_path}; run gen_mock_dense_bubble.py first")
            return 2
        oracle = np.load(oracle_path)

        values = {}
        for tag in TAGS:
            try:
                values[tag] = read_bubble(tag, stem)
            except (OSError, KeyError) as e:
                print(f"could not read '{tag}' for {label}: {e}")
                return 2

        width = 30
        print(f"== {label}   group {stem}")
        print(f"{'t':>2}  {'numpy oracle':>{width}}  "
              + "  ".join(f"{tag:>{width}}" for tag in TAGS))
        for t in range(N_T):
            row = "  ".join(f"{values[tag][t].real:+.8e}{values[tag][t].imag:+.8e}j"
                            for tag in TAGS)
            print(f"{t:>2}  {oracle[t].real:+.8e}{oracle[t].imag:+.8e}j  {row}")

        scale = np.max(np.abs(oracle))
        checks = [(f"{tag} vs oracle", values[tag], oracle) for tag in TAGS]
        checks.append(("dense vs expanded", values["dense"], values["expanded"]))
        checks.append(("densets vs dense", values["densets"], values["dense"]))
        print()
        for name, a, b in checks:
            err = np.max(np.abs(a - b))
            worst = max(worst, err / scale)
            print(f"  max |{name:<20s}| = {err:.3e}  "
                  + ("ok" if err <= TOL * scale else "MISMATCH"))
        print()

    if worst > TOL:
        print(f"FAILED: worst relative deviation {worst:.3e}")
        return 1
    print(f"PASSED: worst relative deviation {worst:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
