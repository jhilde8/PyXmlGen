#!/usr/bin/env python3
"""
diff_a2am_ts.py — compare per-timeslice A2AMatrixIo output (timeSliceIO true)
against a merged reference.

Field-agnostic, for the same reason diff_a2am.py is: A2AMesonField,
A2AExtendedMesonField and A2AChromoMagneticOperatorField all write through
A2AMatrixIo, and their merged oracles (A2AFewMesonField,
A2AExtendedMesonFieldMT, A2AChromoMagneticOperatorFieldMT) all write the
merged layout. Only the ioname alphabet differs:

    MF      Gamma5_0_0_1              gamma _ px _ py _ pz
    EMF     type0_GammaMU_GammaMU     type _ gamma1 _ gamma2
    CMOF    parity0_GijSij            parity _ orthogonality

    python3 diff_a2am_ts.py fmf_cpu_out.0  mf_gpu_out.0
    python3 diff_a2am_ts.py emf_mt_out.0   emf_gpu_out.0
    python3 diff_a2am_ts.py cmof_mt_out.0  cmof_gpu_out.0

Layout:

    reference   <ref_dir>/<ioname>.h5         dataset (nt, ni, nj)
    test        <test_dir>/<ioname>.t0007.h5  dataset ( 1, ni, nj)

Note the HDF5 group name does NOT carry the timeslice -- dataname_ is still
the ioname -- so a file named Gamma5_0_0_1.t0007.h5 contains
/Gamma5_0_0_1/a2aMatrix. Stripping the .tNNNN to recover the group name is
the one thing a reader has to get right that diff_a2am.py did not have to.
The ioname is matched with glob.escape plus an end-anchored regex, so one
group whose name is a prefix of another's (type0_GammaMU_GammaMU against
type0_GammaMU_GammaMUGamma5) does not bleed into it.

Two comparison modes, both worth running:

  merge   Place each file's slab at the index parsed from its name, then
          compare the assembled (nt, ni, nj) against the reference. Catches
          value errors, and reports gaps by name -- if ownerFn leaves some
          (m, g, t) unowned that file is never created, and index placement
          says "t=53 missing" instead of shifting every later timeslice.

  slice   For each t, CONSTRUCT the name <ioname>.t%04d.h5, open it, and
          compare against ref[t]. This is the contractor's read pattern, and
          it is the mode that catches a file which exists, has plausible
          contents, and is the wrong timeslice.

Neither mode is redundant: merge is the value gate, slice is the
access-pattern gate, and only slice addresses files the way production will.

Run the producing test with Pt >= 4 (e.g. --mpi 4.4.4.4, or 4.4.2.8 for
Pt=8). With one rank in time, timeSliceIO is a no-op and this script would
pass without the slab logic ever executing.

Usage:
    python3 diff_a2am_ts.py [ref_dir] [test_dir] [--traj N] [--tol TOL]
                            [--mode {merge,slice,both}] [--max-report N]
"""

import sys
import os
import re
import glob
import argparse
import numpy as np
import h5py

from diff_a2am import load_matrix

TS_RE = re.compile(r"^(?P<ioname>.+)\.t(?P<t>\d+)\.h5$")


def metrics(ref, test):
    diff = np.abs(ref - test)
    n2ref = (np.abs(ref) ** 2).sum()
    n2diff = (diff ** 2).sum()
    rel = np.sqrt(n2diff / n2ref) if n2ref > 0 else 0.0
    idx = np.unravel_index(diff.argmax(), diff.shape) if diff.size else ()
    return {
        "mean": diff.mean() if diff.size else 0.0,
        "rel": rel,
        "idx": idx,
        "ref": ref[idx] if diff.size else 0.0,
        "test": test[idx] if diff.size else 0.0,
    }


def scan_timeslices(mf_dir, ioname):
    """Map t -> path for <ioname>.tNNNN.h5. Returns (found, dupes, unparsed)."""
    found, dupes, unparsed = {}, [], []

    for path in glob.glob(os.path.join(mf_dir, glob.escape(ioname) + ".t*.h5")):
        m = TS_RE.match(os.path.basename(path))
        if m is None or m.group("ioname") != ioname:
            unparsed.append(path)
            continue
        t = int(m.group("t"))
        if t in found:
            # Only reachable if two paddings coexist, e.g. .t7.h5 and .t0007.h5.
            dupes.append((t, found[t], path))
            continue
        found[t] = path

    return found, dupes, unparsed


def check_one(ioname, fmf_file, mf_dir, tol, mode, max_report):
    print(f"\n[{ioname}]")

    ref = load_matrix(fmf_file, ioname)
    if ref.ndim != 3:
        print(f"  REFERENCE SHAPE {ref.shape} is not (nt, ni, nj)")
        return False
    nt, ni, nj = ref.shape
    print(f"  reference   : {ref.shape}")

    found, dupes, unparsed = scan_timeslices(mf_dir, ioname)
    ok = True

    for path in unparsed:
        print(f"  UNPARSEABLE : {os.path.basename(path)}")
        ok = False
    for t, a, b in dupes:
        print(f"  DUPLICATE t={t}: {os.path.basename(a)} and {os.path.basename(b)}")
        ok = False

    if not found:
        print(f"  NO TIMESLICE FILES matching {ioname}.t*.h5 in {mf_dir}")
        return False

    missing = [t for t in range(nt) if t not in found]
    extra = sorted(t for t in found if t < 0 or t >= nt)
    print(f"  timeslices  : {len(found)} files, expected {nt}")
    if missing:
        ok = False
        head = ", ".join(str(t) for t in missing[:max_report])
        tail = "" if len(missing) <= max_report else f", ... (+{len(missing)-max_report})"
        print(f"  MISSING t   : {head}{tail}")
    if extra:
        ok = False
        print(f"  OUT OF RANGE t : {extra[:max_report]}")

    # One read per file; both modes work off this.
    slabs = {}
    for t, path in sorted(found.items()):
        if t < 0 or t >= nt:
            continue
        try:
            a = load_matrix(path, ioname)
        except Exception as e:
            print(f"  READ ERROR t={t}: {e}")
            ok = False
            continue
        if a.shape != (1, ni, nj):
            print(f"  SHAPE t={t}: {a.shape}, expected {(1, ni, nj)}"
                  + ("  (looks like timeSliceIO did not reach the module)"
                     if a.shape == ref.shape else ""))
            ok = False
            continue
        slabs[t] = a[0]

    if mode in ("merge", "both"):
        ok &= compare_merged(ref, slabs, tol)
    if mode in ("slice", "both"):
        ok &= compare_slices(ioname, ref, slabs, found, mf_dir, tol, max_report)

    return ok


def compare_merged(ref, slabs, tol):
    nt = ref.shape[0]
    have = sorted(slabs)
    if len(have) != nt:
        print(f"  merge       : SKIPPED ({len(have)}/{nt} slabs readable)")
        return False

    merged = np.empty_like(ref)
    for t, a in slabs.items():
        merged[t] = a

    m = metrics(ref, merged)
    status = "PASS" if m["rel"] <= tol else "FAIL"
    print(f"  merge       : {status}   rel L2 {m['rel']:.6e}   mean|d| {m['mean']:.6e}")
    if status == "FAIL":
        print(f"    worst index : {m['idx']}")
        print(f"      ref  : {m['ref'].real:+.15e} {m['ref'].imag:+.15e}i")
        print(f"      test : {m['test'].real:+.15e} {m['test'].imag:+.15e}i")
    return status == "PASS"


def compare_slices(ioname, ref, slabs, found, mf_dir, tol, max_report):
    nt = ref.shape[0]
    fails, misnamed, worst_t, worst_rel = [], [], None, -1.0

    for t in range(nt):
        # Address by CONSTRUCTED name, the way the contractor will.
        expected = os.path.join(mf_dir, f"{ioname}.t{t:04d}.h5")
        if not os.path.exists(expected):
            if t in found:
                misnamed.append((t, os.path.basename(found[t])))
            else:
                fails.append((t, None))
            continue
        if t not in slabs:
            fails.append((t, "unreadable"))
            continue

        m = metrics(ref[t], slabs[t])
        if m["rel"] > worst_rel:
            worst_rel, worst_t = m["rel"], t
        if m["rel"] > tol:
            fails.append((t, m))

    npass = nt - len(fails) - len(misnamed)
    status = "PASS" if not fails and not misnamed else "FAIL"
    print(f"  slice       : {status}   {npass}/{nt} timeslices, "
          f"worst rel L2 {worst_rel:.6e} at t={worst_t}")

    for t, name in misnamed[:max_report]:
        print(f"    t={t}: expected {ioname}.t{t:04d}.h5, found {name}")
    for t, m in fails[:max_report]:
        if m is None:
            print(f"    t={t}: no file")
        elif m == "unreadable":
            print(f"    t={t}: file present but not readable as (1, ni, nj)")
        else:
            print(f"    t={t}: rel L2 {m['rel']:.6e}  worst index {m['idx']}")
    n_shown = min(max_report, len(fails)) + min(max_report, len(misnamed))
    if len(fails) + len(misnamed) > n_shown:
        print(f"    ... (+{len(fails) + len(misnamed) - n_shown} more)")

    return status == "PASS"


def compare(fmf_dir, mf_dir, tol, mode, max_report):
    print(f"{'':=<72}")
    print(f"  reference (merged)  : {fmf_dir}")
    print(f"  test (per-timeslice): {mf_dir}")
    print(f"  mode                : {mode}")
    print(f"  tol                 : {tol:.1e}")
    print(f"{'':=<72}")

    fmf_files = sorted(glob.glob(os.path.join(fmf_dir, "*.h5")))
    if not fmf_files:
        print(f"ERROR: no .h5 files found in {fmf_dir}")
        return 1

    all_pass = True
    expected_total = 0
    for fmf_file in fmf_files:
        ioname = os.path.splitext(os.path.basename(fmf_file))[0]
        try:
            with h5py.File(fmf_file, "r") as f:
                expected_total += f[ioname]["a2aMatrix"].shape[0]
        except Exception as e:
            print(f"\n[{ioname}]  REFERENCE READ ERROR: {e}")
            all_pass = False
            continue
        all_pass &= check_one(ioname, fmf_file, mf_dir, tol, mode, max_report)

    # A file count below nmom*ngamma*nt means some (m, g, t) had no owner --
    # an ownerFn bug the value comparison cannot distinguish from a bad write.
    actual_total = len(glob.glob(os.path.join(mf_dir, "*.h5")))
    print(f"\n  file count  : {actual_total} in test dir, expected {expected_total}")
    if actual_total != expected_total:
        print("  FILE COUNT MISMATCH (unowned (m,g,t), or stale files present)")
        all_pass = False

    print(f"\n{'':=<72}")
    print(f"  OVERALL : {'ALL PASS' if all_pass else 'FAIL'}")
    print(f"{'':=<72}")
    return 0 if all_pass else 1


def resolve(d, traj):
    if os.path.isdir(d):
        return d
    candidate = f"{d}.{traj}"
    return candidate if os.path.isdir(candidate) else d


def main():
    p = argparse.ArgumentParser(
        description="Diff per-timeslice A2AMatrixIo output (MF, EMF or CMOF) "
                    "against a merged reference."
    )
    p.add_argument("ref_dir",  nargs="?", default="fmf_cpu_out",
                   help="reference (merged) output directory")
    p.add_argument("test_dir", nargs="?", default="mf_ts_out",
                   help="per-timeslice output directory")
    p.add_argument("--traj", type=int, default=0,
                   help="trajectory appended to bare dir names (default: 0)")
    p.add_argument("--tol", type=float, default=1e-10,
                   help="relative L2 tolerance for PASS (default: 1e-10)")
    p.add_argument("--mode", choices=("merge", "slice", "both"), default="both",
                   help="which comparison to run (default: both)")
    p.add_argument("--max-report", type=int, default=8,
                   help="max failing timeslices to list per field (default: 8)")
    args = p.parse_args()

    sys.exit(compare(resolve(args.ref_dir, args.traj),
                     resolve(args.test_dir, args.traj),
                     args.tol, args.mode, args.max_report))


if __name__ == "__main__":
    main()
