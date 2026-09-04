#!/usr/bin/env python3
"""
diff_a2am.py — compare two directories of A2AMatrixIo HDF5 output, merged
layout (timeSliceIO false), reference against test.

Field-agnostic on purpose. A2AMesonField, A2AExtendedMesonField and
A2AChromoMagneticOperatorField all write through A2AMatrixIo, so every one of
them produces <ioname>.h5 holding /<ioname>/a2aMatrix of shape (nt, ni, nj)
with a compound {re, im} element type. Only the ioname alphabet differs:

    MF      Gamma5_0_0_1              gamma _ px _ py _ pz
    EMF     type0_GammaMU_GammaMU     type _ gamma1 _ gamma2
    CMOF    parity0_GijSij            parity _ orthogonality

Nothing below parses an ioname, so all three work unchanged:

    python3 diff_a2am.py fmf_cpu_out.0  mf_gpu_out.0
    python3 diff_a2am.py emf_mt_out.0   emf_gpu_out.0
    python3 diff_a2am.py cmof_mt_out.0  cmof_gpu_out.0

Files are DISCOVERED from the reference directory rather than constructed
from a hardcoded type/gamma/momentum list, and every .h5 found there is
looked up by the same name in the test directory. That is the property that
makes this general, and it is what stops a field being silently skipped when
the gamma set grows -- a script that builds names from a fixed list reports
ALL PASS on the subset it happens to know about.

For per-timeslice output (timeSliceIO true) use diff_a2am_ts.py, which
handles the .tNNNN infix this script's splitext would fold into the ioname.

Usage:
    python3 diff_a2am.py [ref_dir] [test_dir] [--traj N] [--tol TOL]

Defaults:
    ref_dir  = fmf_cpu_out.0
    test_dir = mf_gpu_out.0
    traj     = 0
    tol      = 1e-10  (relative L2 norm threshold for PASS)
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
        description="Diff reference vs test A2AMatrixIo output (MF, EMF or CMOF)."
    )
    p.add_argument("ref_dir",  nargs="?", default="fmf_cpu_out",
                   help="reference output directory (default: fmf_cpu_out)")
    p.add_argument("test_dir", nargs="?", default="mf_gpu_out",
                   help="test output directory      (default: mf_gpu_out)")
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

    fmf_dir = resolve(args.ref_dir,  args.traj)
    mf_dir  = resolve(args.test_dir, args.traj)

    sys.exit(compare(fmf_dir, mf_dir, args.tol))


if __name__ == "__main__":
    main()
