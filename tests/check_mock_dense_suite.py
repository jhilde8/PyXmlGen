"""Compare FlexibleContractorDense correlators against FlexibleContractorDWC,
and BubbleContractorDense bubbles against BubbleContractorDWC.

    python3 check_mock_dense_suite.py <run dir> [--tol TOL] [--traj N]

Reads manifest.json written by gen_mock_dense_suite.py. Each of our
contractions is matched to the reference output named by Masaaki's term stem,
dataset by dataset (one per translation, or the single averaged one). Exact
agreement is not expected: both tools sum trace rows under an OpenMP critical
section, whose order varies from run to run, so the default tolerance is a
relative L2 of 1e-13 per dataset.

Bubbles are matched to Masaaki's output by his term stem, rolled by our first
term's offset. The manifest's identities check that tr of an A2APixPi product
equals the 2-term trace of its factors, in both tools' outputs.
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np


def correlator(path, dataset):
    with h5py.File(path, "r") as f:
        raw = f[dataset]["correlator"][()]
    return raw["re"] + 1j * raw["im"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--tol", type=float, default=1e-13)
    p.add_argument("--traj", type=int, default=0)
    args = p.parse_args()

    with open(os.path.join(args.run, "manifest.json")) as f:
        manifest = json.load(f)

    ok = True
    for c in manifest["contractions"]:
        ref = os.path.join(args.run, "corr_ref", f"{c['ref']}.{args.traj}.h5")
        test = os.path.join(args.run, "corr_test", f"{c['test']}.{args.traj}.h5")
        if c["average"]:
            pairs = [(c["ref"], c["test"])]
        else:
            pairs = [(f"{c['ref']}_dt_{dt}", f"{c['test']}_dt_{dt}") for dt in c["translations"]]

        worst, scale = 0.0, 0.0
        for rds, tds in pairs:
            r, t = correlator(ref, rds), correlator(test, tds)
            n = np.linalg.norm(r)
            worst = max(worst, np.linalg.norm(r - t) / n if n > 0 else np.linalg.norm(t))
            scale = max(scale, n)
        good = worst <= args.tol and scale > 0
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {c['test']:<10} vs {c['ref']:<34} "
              f"{len(pairs):2d} dataset(s)  worst rel L2 {worst:.2e}  |ref| {scale:.3e}")

    def bubble(kind, name):
        return correlator(os.path.join(args.run, f"bub_{kind}", f"{name}.{args.traj}.h5"), name)

    def report(label, r, t):
        n = np.linalg.norm(r)
        err = np.linalg.norm(r - t) / n if n > 0 else np.linalg.norm(t)
        good = err <= args.tol and n > 0
        print(f"  {'PASS' if good else 'FAIL'}  {label:<48} rel L2 {err:.2e}  |ref| {n:.3e}")
        return good

    print("== bubbles: BubbleContractorDense vs BubbleContractorDWC")
    refs = {}
    for b in manifest.get("bubbles", []):
        refs[b["test"]] = np.roll(bubble("ref", b["ref"]), -b["roll"])
        ok &= report(f"{b['test']} vs {b['ref']} (roll {b['roll']})",
                     refs[b["test"]], bubble("test", b["test"]))

    print("== bubbles: tr(product) vs 2-term trace of its factors")
    for x, y in manifest.get("identities", []):
        ok &= report(f"ours {x} vs {y}", bubble("test", x), bubble("test", y))
        ok &= report(f"ref  {x} vs {y}", refs[x], refs[y])

    print("OVERALL:", "ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
