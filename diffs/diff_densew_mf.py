#!/usr/bin/env python3
"""
diff_densew_mf.py — verify dense/combined-W meson fields against the expanded
(time-diluted) reference, for every meson-field shape in the non-EMF suite.

The row axis is always the W side under test (dense vs expanded). Both axes are
described the same way, by where each *dense* slot lives in the expanded array
at meson time t:

    low block     slot k          ->  expanded k            (every t; low modes
                                                             are not diluted)
    high block    slot nLow+h*12+sc
                                  ->  nLow + h*T*12 + t*12 + sc   (t < T only)

where T = t_cover is the number of timeslices the expanded truncation reaches.
An axis that is not compressed (a V side) maps identically at every t.

Rather than infer the layout from array shapes — which is genuinely ambiguous,
since several (nhits, nLow, t_cover) triples fit any given pair of extents —
the flavour structure is stated with --case and cross-checked against the
shapes:

    case      meson field         row (W) lows   col lows   col side
    ------------------------------------------------------------------
    hh        truncated verify         no          no        V
    ll        pion / sigma            yes         yes        V
    ls        kaon                    yes          no        V
    sl        kaon                     no         yes        V
    ll_ww     ww, lows both sides     yes         yes        W
    ls_ww     kaon 3pt                yes          no        W
    sl_ww     kaon 3pt, legs swapped   no         yes        W

The first letter is the row (left) side; "l" carries low modes and "s" does
not. If a run fails and you are unsure which case your output actually is,
--identify reports every (case, n-low, n-hits) combination consistent with the
dataset extents.

with --n-hits and --n-low supplying the two numbers the case cannot know. Only
the light flavour carries low modes, so one --n-low covers whichever side has
them. Everything else (t_cover, the per-axis block boundaries) is then derived
and checked against the dataset extents.

Three checks per (gamma x momentum) file:

1. Matched entries: at each meson time, the expanded sub-block selected by the
   two maps is compared entry-by-entry against the dense array (aggregate
   relative L2 over all times, PASS below --tol).

2. Structural zeros: everything in the expanded array outside that sub-block
   must vanish — the expanded W is identically zero off its dilution timeslice,
   and IEEE sums of exact zeros stay exact — so the maximum absolute value
   there is checked against --zero-tol (default 0.0). Matched entries plus
   structural zeros partition the expanded array exactly; the dense array is
   *not* fully covered when T < nt, since dense entries at meson times beyond
   the truncation have no expanded partner.

3. Two-point closure: C(t1,t2) = sum_ij MF[t1][i][j] MF[t2][j][i]. The expanded
   path is the plain square trace (structural zeros select the surviving terms
   "the old way"); the dense path states the selection explicitly — factor 1's
   column must carry the same expanded index as factor 2's row, and vice versa.
   Agreement validates the map exactly as the contractor-side modules will use
   it. It requires the row and column index spaces to coincide, so it runs for
   hh and ll and is reported as a visible SKIP for the cross-flavour cases,
   where Pi_ji is not defined. Not physics (truncated mode sum, same momentum
   on both factors), pure numerical closure.

   Note for W on both sides: the high sector of such a field is diagonal in
   time (factor 1 forces t_i = t_j = t1, factor 2 forces t2), so only the low
   sector — never time-diluted — contributes off the diagonal. That falls out
   of the maps; nothing special-cases it.

Usage:
    python3 diff_densew_mf.py [exp_dir] [dense_dir] --case CASE [--n-hits N]
                              [--n-low N] [--traj N] [--tol TOL]
                              [--zero-tol ZTOL] [--no-closure]
    python3 diff_densew_mf.py --self-test

Defaults:
    exp_dir   = exp_h0      (expanded-W meson field output)
    dense_dir = dense_h0    (dense-W meson field output)
    case      = hh
    n-low     = 0      (required non-zero for any case with low modes)
    n-hits    = inferred, but only for cases with no low modes on the rows
    traj      = 0
    tol       = 1e-10  (relative L2 norm threshold, checks 1 and 3)
    zero-tol  = 0.0    (max |entry| allowed on structural zeros)
"""

import sys
import os
import glob
import argparse
import numpy as np
import h5py

N_SC = 12

# case -> (row has low modes, col has low modes, col side is W/compressed)
# The row axis is always the W side under test, so it is always compressed.
# "l" carries low modes, "s" does not; the first letter is the row (left) side.
CASES = {
    "hh":    (False, False, False),
    "ll":    (True,  True,  False),
    "ls":    (True,  False, False),
    "sl":    (False, True,  False),
    "ll_ww": (True,  True,  True),
    "ls_ww": (True,  False, True),
    "sl_ww": (False, True,  True),
}


def load_matrix(h5path, ioname):
    with h5py.File(h5path, "r") as f:
        raw = f[ioname]["a2aMatrix"][()]
    return raw["re"].astype(np.float64) + 1j * raw["im"].astype(np.float64)


# ---------------------------------------------------------------------------
# index maps
# ---------------------------------------------------------------------------
class Axis:
    """One index axis, and where each of its dense slots lives in the expanded
    array at a given meson time."""

    def __init__(self, name, n_exp, n_dense, nhits, n_low, compressed):
        self.name = name
        self.n_exp = n_exp
        self.n_dense = n_dense
        self.nhits = nhits
        self.compressed = compressed

        if (n_exp == n_dense) != (not compressed):
            state = "compressed" if compressed else "uncompressed"
            raise ValueError(f"{name}: --case says this axis is {state}, but "
                             f"the expanded extent ({n_exp}) "
                             f"{'differs from' if n_exp != n_dense else 'equals'}"
                             f" the dense extent ({n_dense})")

        if not compressed:
            # a V side: identity map, nLow plays no role in it
            self.n_low = n_low
            self.t_cover = None
            return

        block = nhits * N_SC
        if n_low + block != n_dense:
            raise ValueError(f"{name}: nLow ({n_low}) + nhits*{N_SC} ({block}) "
                             f"= {n_low + block} does not match the dense "
                             f"extent ({n_dense}); check --case, --n-hits "
                             f"and --n-low")
        rest = n_exp - n_low
        if rest <= 0 or rest % block != 0:
            raise ValueError(f"{name}: expanded extent minus nLow ({rest}) is "
                             f"not a positive multiple of nhits*{N_SC} "
                             f"({block}); check --case, --n-hits and --n-low")
        self.n_low = n_low
        self.t_cover = rest // block

    def exp_of_dense(self, t):
        """Expanded index for each dense slot at meson time t; -1 where the
        slot has no expanded partner (high slots beyond the truncation)."""
        if not self.compressed:
            return np.arange(self.n_exp, dtype=np.int64)

        out = np.full(self.n_dense, -1, dtype=np.int64)
        out[:self.n_low] = np.arange(self.n_low, dtype=np.int64)
        if t < self.t_cover:
            for h in range(self.nhits):
                d0 = self.n_low + h * N_SC
                e0 = self.n_low + h * self.t_cover * N_SC + t * N_SC
                out[d0:d0 + N_SC] = e0 + np.arange(N_SC, dtype=np.int64)
        return out

    def describe(self):
        if not self.compressed:
            return f"{self.name}: V side, n={self.n_exp} (nLow={self.n_low})"
        return (f"{self.name}: W side, nLow={self.n_low} nhits={self.nhits} "
                f"t_cover={self.t_cover} "
                f"({self.n_dense} dense / {self.n_exp} expanded)")


class Geometry:
    def __init__(self, exp_shape, dense_shape, case, nhits=None, n_low=0):
        if case not in CASES:
            raise ValueError(f"unknown --case '{case}'; "
                             f"choose from {sorted(CASES)}")
        row_low, col_low, col_compressed = CASES[case]

        nt, ni_e, nj_e = exp_shape
        nt_d, ni_d, nj_d = dense_shape
        if nt != nt_d:
            raise ValueError(f"meson-time mismatch: expanded nt={nt}, "
                             f"dense nt={nt_d}")

        if (row_low or col_low) and n_low == 0:
            raise ValueError(f"case '{case}' has low modes; pass --n-low")

        n_low_row = n_low if row_low else 0
        n_low_col = n_low if col_low else 0

        if nhits is None:
            if row_low:
                raise ValueError(f"case '{case}' has low modes on the rows, so "
                                 f"nhits cannot be read off the dense extent; "
                                 f"pass --n-hits")
            if ni_d <= 0 or ni_d % N_SC != 0:
                raise ValueError(f"cannot infer nhits: dense rows ({ni_d}) is "
                                 f"not a positive multiple of {N_SC}; "
                                 f"pass --n-hits")
            nhits = ni_d // N_SC

        self.case = case
        self.nt = nt
        self.nhits = nhits
        self.row = Axis("rows", ni_e, ni_d, nhits, n_low_row, True)
        self.col = Axis("cols", nj_e, nj_d, nhits, n_low_col, col_compressed)

        covers = {a.t_cover for a in (self.row, self.col) if a.compressed}
        if len(covers) != 1:
            raise ValueError(f"axes disagree on t_cover: {sorted(covers)}")
        self.t_cover = covers.pop()

    def maps(self, t):
        return self.row.exp_of_dense(t), self.col.exp_of_dense(t)


def _match_slots(a, b):
    """Dense-slot positions (pa, pb) with a[pa] == b[pb], ignoring -1."""
    ia = np.flatnonzero(a >= 0)
    ib = np.flatnonzero(b >= 0)
    if ia.size == 0 or ib.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    _, xa, xb = np.intersect1d(a[ia], b[ib], return_indices=True)
    return ia[xa], ib[xb]


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def entry_checks(exp, dense, geom):
    """Checks 1 and 2, accumulated one meson time at a time so the structural-
    zero set is never materialized as a whole-array copy."""
    matched = 0
    sum_diff2 = 0.0
    sum_ref2 = 0.0
    max_zero = 0.0
    worst = {"diff": -1.0}

    for t in range(geom.nt):
        r, c = geom.maps(t)
        rv = np.flatnonzero(r >= 0)
        cv = np.flatnonzero(c >= 0)
        sel = None

        if rv.size and cv.size:
            sel = np.ix_(r[rv], c[cv])
            e = exp[t][sel]
            d = dense[t][np.ix_(rv, cv)]
            diff = np.abs(e - d)

            matched += e.size
            sum_diff2 += float((diff ** 2).sum())
            sum_ref2 += float((np.abs(e) ** 2).sum())

            k = np.unravel_index(int(diff.argmax()), diff.shape)
            if float(diff[k]) > worst["diff"]:
                worst = {"diff": float(diff[k]),
                         "exp": complex(e[k]),
                         "dense": complex(d[k]),
                         "at": (t, int(rv[k[0]]), int(cv[k[1]]))}

        a = np.abs(exp[t])
        if sel is not None:
            a[sel] = 0.0
        if a.size:
            max_zero = max(max_zero, float(a.max()))

    rel_l2 = np.sqrt(sum_diff2 / sum_ref2) if sum_ref2 > 0 else 0.0
    return matched, rel_l2, max_zero, worst


def closure(exp, dense, geom):
    """Check 3. Returns (c_exp, c_dense, skip_reason)."""
    if geom.row.n_exp != geom.col.n_exp:
        return None, None, (f"case '{geom.case}' pairs different mode sets "
                            f"({geom.row.n_exp} rows vs {geom.col.n_exp} cols); "
                            f"Pi_ji is not defined")

    tmax = min(geom.t_cover, geom.nt)

    # C[a,b] = sum_ij exp[a][i,j] exp[b][j,i]  as one GEMM
    flat = exp[:tmax].reshape(tmax, -1)
    flat_t = np.ascontiguousarray(exp[:tmax].transpose(0, 2, 1)).reshape(tmax, -1)
    c_exp = flat @ flat_t.T

    rmaps = [geom.row.exp_of_dense(t) for t in range(tmax)]
    cmaps = [geom.col.exp_of_dense(t) for t in range(tmax)]

    c_dense = np.zeros((tmax, tmax), dtype=complex)
    for t1 in range(tmax):
        for t2 in range(tmax):
            # i lives on factor 1's rows and factor 2's columns
            ra, cb = _match_slots(rmaps[t1], cmaps[t2])
            # j lives on factor 1's columns and factor 2's rows
            rb, ca = _match_slots(rmaps[t2], cmaps[t1])
            if ra.size == 0 or rb.size == 0:
                continue
            a = dense[t1][np.ix_(ra, ca)]
            b = dense[t2][np.ix_(rb, cb)]
            c_dense[t1, t2] = np.einsum("pq,qp->", a, b)

    return c_exp, c_dense, None


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def compare_one(exp, dense, geom, tol, zero_tol, want_closure):
    matched, rel_l2, max_zero, worst = entry_checks(exp, dense, geom)

    ok_match = rel_l2 <= tol
    ok_zero = max_zero <= zero_tol

    c_exp = c_dense = c_diff = c_rel_l2 = None
    skip = "disabled by --no-closure" if not want_closure else None
    if want_closure:
        c_exp, c_dense, skip = closure(exp, dense, geom)
    if c_exp is not None:
        c_diff = np.abs(c_exp - c_dense)
        c_norm2 = float((np.abs(c_exp) ** 2).sum())
        c_rel_l2 = float(np.sqrt((c_diff ** 2).sum() / c_norm2)) if c_norm2 > 0 else 0.0
        ok_corr = c_rel_l2 <= tol
    else:
        ok_corr = True

    return {
        "matched": matched, "rel_l2": rel_l2, "max_zero": max_zero,
        "worst": worst, "ok_match": ok_match, "ok_zero": ok_zero,
        "c_exp": c_exp, "c_dense": c_dense, "c_diff": c_diff,
        "c_rel_l2": c_rel_l2, "ok_corr": ok_corr, "skip": skip,
        "total_exp": int(exp.size),
        "passed": ok_match and ok_zero and ok_corr,
    }


def report(ioname, geom, res, tol, zero_tol):
    print(f"\n[{ioname}]  →  {'PASS' if res['passed'] else 'FAIL'}")
    print(f"  case={geom.case}  nt={geom.nt}")
    print(f"    {geom.row.describe()}")
    print(f"    {geom.col.describe()}")
    print(f"  matched entries : {res['matched']}  "
          f"(+ {res['total_exp'] - res['matched']} structural zeros "
          f"= {res['total_exp']} expanded)")
    print(f"  rel L2 |Δ|      : {res['rel_l2']:.6e}   "
          f"({'≤' if res['ok_match'] else '>'} tol {tol:.1e})")
    print(f"  max |zero rows| : {res['max_zero']:.6e}   "
          f"({'≤' if res['ok_zero'] else '>'} zero-tol {zero_tol:.1e})")
    w = res["worst"]
    if w["diff"] >= 0.0:
        print(f"    worst matched entry at (t, row, col) = {w['at']}:")
        print(f"      expanded : {w['exp'].real:+.15e}  {w['exp'].imag:+.15e}i")
        print(f"      dense    : {w['dense'].real:+.15e}  {w['dense'].imag:+.15e}i")
    if res["skip"] is not None:
        print(f"  2pt closure     : SKIPPED — {res['skip']}")
    else:
        c_exp = res["c_exp"]
        ci = np.unravel_index(int(res["c_diff"].argmax()), res["c_diff"].shape)
        print(f"  2pt closure C(t1,t2) over {c_exp.shape[0]}x{c_exp.shape[1]} times:")
        print(f"    rel L2 |Δ|    : {res['c_rel_l2']:.6e}   "
              f"({'≤' if res['ok_corr'] else '>'} tol {tol:.1e})")
        print(f"    worst (t1,t2) : {tuple(int(x) for x in ci)}")
        print(f"      expanded : {c_exp[ci].real:+.15e}  {c_exp[ci].imag:+.15e}i")
        print(f"      dense    : {res['c_dense'][ci].real:+.15e}  "
              f"{res['c_dense'][ci].imag:+.15e}i")
    return res["passed"]


def compare(exp_dir, dense_dir, case, nhits, n_low, tol, zero_tol, want_closure):
    all_pass = True
    print(f"{'':=<72}")
    print(f"  expanded-W dir (reference) : {exp_dir}")
    print(f"  dense-W dir    (test)      : {dense_dir}")
    print(f"  case                       : {case}")
    print(f"  n-hits / n-low             : "
          f"{nhits if nhits is not None else 'inferred'} / {n_low}")
    print(f"  tol / zero-tol             : {tol:.1e} / {zero_tol:.1e}")
    print(f"{'':=<72}")

    exp_files = sorted(glob.glob(os.path.join(exp_dir, "*.h5")))
    if not exp_files:
        print(f"ERROR: no .h5 files found in {exp_dir}")
        return 1

    for exp_file in exp_files:
        ioname = os.path.splitext(os.path.basename(exp_file))[0]
        dense_file = os.path.join(dense_dir, os.path.basename(exp_file))

        if not os.path.exists(dense_file):
            print(f"\n[{ioname}]  MISSING in dense dir: {dense_file}")
            all_pass = False
            continue

        try:
            exp = load_matrix(exp_file, ioname)
            dense = load_matrix(dense_file, ioname)
            geom = Geometry(exp.shape, dense.shape, case, nhits, n_low)
            res = compare_one(exp, dense, geom, tol, zero_tol, want_closure)
        except Exception as e:
            print(f"\n[{ioname}]  ERROR: {e}")
            all_pass = False
            continue

        if not report(ioname, geom, res, tol, zero_tol):
            all_pass = False

    print(f"\n{'':=<72}")
    print(f"  OVERALL : {'ALL PASS' if all_pass else 'FAIL'}")
    print(f"{'':=<72}")
    return 0 if all_pass else 1


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _synthetic(nt, nhits, t_cover, n_low, case, seed):
    """Random dense array plus the expanded array implied by the maps."""
    rng = np.random.default_rng(seed)
    row_low, col_low, col_compressed = CASES[case]
    nlr = n_low if row_low else 0
    nlc = n_low if col_low else 0

    n_dense_r = nlr + nhits * N_SC
    n_exp_r = nlr + nhits * t_cover * N_SC
    if col_compressed:
        n_dense_c = nlc + nhits * N_SC
        n_exp_c = nlc + nhits * t_cover * N_SC
    else:
        n_dense_c = n_exp_c = nlc + nhits * t_cover * N_SC

    shape = (nt, n_dense_r, n_dense_c)
    dense = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    exp = np.zeros((nt, n_exp_r, n_exp_c), dtype=complex)

    row = Axis("rows", n_exp_r, n_dense_r, nhits, nlr, True)
    col = Axis("cols", n_exp_c, n_dense_c, nhits, nlc, col_compressed)
    for t in range(nt):
        r = row.exp_of_dense(t)
        c = col.exp_of_dense(t)
        rv = np.flatnonzero(r >= 0)
        cv = np.flatnonzero(c >= 0)
        if rv.size and cv.size:
            exp[t][np.ix_(r[rv], c[cv])] = dense[t][np.ix_(rv, cv)]
    return exp, dense


def self_test():
    nt, t_cover, n_low = 16, 4, 24
    cases = [("hh", 1), ("hh", 2), ("ll", 1), ("ll", 2),
             ("ls", 1), ("sl", 1), ("sl", 2), ("ls_ww", 1), ("ls_ww", 2)]

    ok = True
    print(f"{'':=<72}")
    print("  self-test on synthetic arrays")
    print(f"{'':=<72}")

    for i, (case, nhits) in enumerate(cases):
        exp, dense = _synthetic(nt, nhits, t_cover, n_low, case, seed=1234 + i)
        geom = Geometry(exp.shape, dense.shape, case, nhits, n_low)
        res = compare_one(exp, dense, geom, 1e-12, 0.0, True)

        clean = (res["rel_l2"] == 0.0 and res["max_zero"] == 0.0
                 and (res["c_rel_l2"] is None or res["c_rel_l2"] < 1e-12))

        # corrupt one matched entry and one structural zero, expect detection
        r, c = geom.maps(0)
        rv = np.flatnonzero(r >= 0)
        cv = np.flatnonzero(c >= 0)
        bad = exp.copy()
        bad[0, r[rv[0]], c[cv[0]]] += 1.0
        caught_match = compare_one(bad, dense, geom, 1e-12, 0.0, False)["rel_l2"] > 1e-12

        caught_zero = True
        if res["total_exp"] > res["matched"]:
            a = np.abs(exp[0])
            a[np.ix_(r[rv], c[cv])] = np.inf
            zi = np.unravel_index(int(a.argmin()), a.shape)
            bad2 = exp.copy()
            bad2[0][zi] += 1.0
            caught_zero = compare_one(bad2, dense, geom, 1e-12, 0.0, False)["max_zero"] > 0.0

        good = clean and caught_match and caught_zero
        ok = ok and good
        cl = "skipped" if res["c_rel_l2"] is None else f"{res['c_rel_l2']:.1e}"
        print(f"\n  [{'PASS' if good else 'FAIL'}] case={case} nhits={nhits}")
        print(f"        shapes    : exp {exp.shape}  dense {dense.shape}")
        print(f"        matched   : {res['matched']} of {res['total_exp']} expanded")
        print(f"        rel L2    : {res['rel_l2']:.1e}   max|zero| : "
              f"{res['max_zero']:.1e}   closure : {cl}")
        print(f"        detects   : matched corruption {caught_match}, "
              f"zero corruption {caught_zero}")

    # a mis-stated --case must not report PASS: either it fails to construct,
    # or the maps select the wrong entries and the diff fails loudly
    print(f"\n  mis-stated --case (an 'll' array diffed as 'hh'):")
    for nhits in (1, 2):
        exp, dense = _synthetic(nt, nhits, t_cover, n_low, "ll", seed=77 + nhits)
        try:
            g = Geometry(exp.shape, dense.shape, "hh", None, 0)
            r = compare_one(exp, dense, g, 1e-12, 0.0, False)
            caught = not r["passed"]
            how = (f"constructed as nhits={g.nhits} t_cover={g.t_cover}, "
                   f"then rel L2 {r['rel_l2']:.1e} / max|zero| {r['max_zero']:.1e}")
        except ValueError as e:
            caught = True
            how = f"rejected at construction: {e}"
        ok = ok and caught
        print(f"    [{'PASS' if caught else 'FAIL'}] nhits={nhits}: {how}")

    print(f"\n{'':=<72}")
    print(f"  SELF-TEST : {'ALL PASS' if ok else 'FAIL'}")
    print(f"{'':=<72}")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(
        description="Diff expanded-W (reference) vs dense-W meson field outputs."
    )
    p.add_argument("exp_dir", nargs="?", default="exp_h0",
                   help="expanded-W output directory (default: exp_h0)")
    p.add_argument("dense_dir", nargs="?", default="dense_h0",
                   help="dense-W output directory (default: dense_h0)")
    p.add_argument("--case", default="hh", choices=sorted(CASES),
                   help="flavour structure of the meson field, which fixes "
                        "which side carries low modes and whether the column "
                        "side is W or V (default: hh)")
    p.add_argument("--n-hits", type=int, default=None,
                   help="Number of hits in the arrays. Required for any case "
                        "with low modes on the rows; otherwise inferred from "
                        "the dense row count.")
    p.add_argument("--n-low", type=int, default=0,
                   help="Number of low modes on whichever side the case says "
                        "has them (default: 0)")
    p.add_argument("--traj", type=int, default=0,
                   help="Trajectory number appended to bare dir names (default: 0)")
    p.add_argument("--tol", type=float, default=1e-10,
                   help="Relative L2 tolerance, matched entries and 2pt closure "
                        "(default: 1e-10)")
    p.add_argument("--zero-tol", type=float, default=0.0,
                   help="Max |entry| allowed on structural zeros (default: 0.0)")
    p.add_argument("--no-closure", action="store_true",
                   help="Skip the two-point closure check")
    p.add_argument("--self-test", action="store_true",
                   help="Run the synthetic self-test and exit")
    args = p.parse_args()

    if args.self_test:
        sys.exit(self_test())

    def resolve(d, traj):
        if os.path.isdir(d):
            return d
        candidate = f"{d}.{traj}"
        if os.path.isdir(candidate):
            return candidate
        return d  # let the file-level checks report the error

    exp_dir = resolve(args.exp_dir, args.traj)
    dense_dir = resolve(args.dense_dir, args.traj)

    sys.exit(compare(exp_dir, dense_dir, args.case, args.n_hits, args.n_low,
                     args.tol, args.zero_tol, not args.no_closure))


if __name__ == "__main__":
    main()
