#!/usr/bin/env python3
"""
diff_mf.py — compare meson field HDF5 outputs from Test_fmf_cpu (reference)
and Test_mf_gpu.

Files are discovered automatically from the FMF reference directory; every
.h5 file found there is looked up by the same name in the MF directory.
This covers all (gamma x momentum) combinations produced by the tests without
requiring the script to know the exact filenames in advance.

Usage:
    python3 diff_mf.py [fmf_dir] [mf_dir] [--traj N] [--tol TOL]

Defaults:
    fmf_dir = fmf_cpu_out.0   (FMF CPU reference)
    mf_dir  = mf_gpu_out.0    (MF GPU output)
    traj    = 0
    tol     = 1e-10  (relative L2 norm threshold for PASS)
"""

import sys
import os
import glob
import argparse
import numpy as np
import h5py


def load_matrix(h5path, ioname):
    with h5py.File(h5path, "r") as f:
        raw = f[ioname]["a2aMatrix"][()]
    return raw["re"].astype(np.float64) + 1j * raw["im"].astype(np.float64)


def compare(fmf_dir, mf_dir, tol):
    all_pass = True
    print(f"{'':=<72}")
    print(f"  MF dir (reference) : {fmf_dir}")
    print(f"  MF dir (test)      : {mf_dir}")
    print(f"  tol                 : {tol:.1e}")
    print(f"{'':=<72}")

    fmf_files = sorted(glob.glob(os.path.join(fmf_dir, "*.h5")))
    if not fmf_files:
        print(f"ERROR: no .h5 files found in {fmf_dir}")
        return 1

    for fmf_file in fmf_files:
        ioname  = os.path.splitext(os.path.basename(fmf_file))[0]
        mf_file = os.path.join(mf_dir, os.path.basename(fmf_file))

        if not os.path.exists(mf_file):
            print(f"\n[{ioname}]  MISSING in MF dir: {mf_file}")
            all_pass = False
            continue

        try:
            fmf = load_matrix(fmf_file, ioname)
            mf  = load_matrix(mf_file,  ioname)
        except Exception as e:
            print(f"\n[{ioname}]  READ ERROR: {e}")
            all_pass = False
            continue

        if fmf.shape != mf.shape:
            print(f"\n[{ioname}]  SHAPE MISMATCH: FMF={fmf.shape}  MF={mf.shape}")
            all_pass = False
            continue

        diff    = np.abs(fmf - mf)
        fmf_abs = np.abs(fmf)

        mean_abs   = diff.mean()
        norm2_fmf  = (fmf_abs ** 2).sum()
        norm2_diff = (diff ** 2).sum()
        rel_l2     = np.sqrt(norm2_diff / norm2_fmf) if norm2_fmf > 0 else 0.0

        idx     = np.unravel_index(diff.argmax(), diff.shape)
        fmf_val = fmf[idx]
        mf_val  = mf[idx]

        status = "PASS" if rel_l2 <= tol else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"\n[{ioname}]  →  {status}")
        print(f"  shape       : {fmf.shape}")
        print(f"  mean|Δ|     : {mean_abs:.6e}")
        print(f"  rel L2 |Δ|  : {rel_l2:.6e}   ({'≤' if rel_l2 <= tol else '>'} tol {tol:.1e})")
        print(f"  worst index : {idx}")
        print(f"    MF ref  value : {fmf_val.real:+.15e}  {fmf_val.imag:+.15e}i")
        print(f"    MF test value : {mf_val.real:+.15e}  {mf_val.imag:+.15e}i")

    print(f"\n{'':=<72}")
    print(f"  OVERALL : {'ALL PASS' if all_pass else 'FAIL'}")
    print(f"{'':=<72}")
    return 0 if all_pass else 1


def main():
    p = argparse.ArgumentParser(
        description="Diff MF (reference) vs MF HDF5 meson field outputs."
    )
    p.add_argument("fmf_dir", nargs="?", default="fmf_cpu_out",
                   help="FMF output directory (default: fmf_cpu_out)")
    p.add_argument("mf_dir",  nargs="?", default="mf_gpu_out",
                   help="MF output directory  (default: mf_gpu_out)")
    p.add_argument("--traj", type=int,   default=0,
                   help="Trajectory number appended to bare dir names (default: 0)")
    p.add_argument("--tol",  type=float, default=1e-10,
                   help="Relative L2 norm tolerance for PASS (default: 1e-10)")
    args = p.parse_args()

    def resolve(d, traj):
        if os.path.isdir(d):
            return d
        candidate = f"{d}.{traj}"
        if os.path.isdir(candidate):
            return candidate
        return d  # let the file-level checks report the error

    fmf_dir = resolve(args.fmf_dir, args.traj)
    mf_dir  = resolve(args.mf_dir,  args.traj)

    sys.exit(compare(fmf_dir, mf_dir, args.tol))


if __name__ == "__main__":
    main()
