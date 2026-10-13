"""Synthetic oracle for the dense-W bubble contractions.

Same idea as gen_mock_dense_trace.py and it reuses that script's field
construction, but the quantity is different: a bubble is indexed by WHERE it
sits, not by a separation, so there is no translation average and the answer is
B(t) rather than C(separation).

Two products, covering both branches:

    tr[Pi(t)]              one term, no times   -- the disconnected sigma loop
    tr[Pi(t) Pi(t+1)]      two terms, times=1   -- a two-pion bubble

The offset is 1 rather than 0 on purpose: with 0 the test would pass even if
the contractor ignored times[0] entirely.

Three par files, so the two changes are separable:

    par.bubble.expanded.xml   BubbleContractor      expanded W, one file
    par.bubble.dense.xml      BubbleContractorDense dense W, one file
    par.bubble.densets.xml    BubbleContractorDense dense W, file per timeslice

The dense field holds exactly the non-zero rows of the expanded one, so the
contractions are the same sums with the zeros dropped. As with the trace mock,
the script evaluates everything in numpy first and refuses to write if the two
disagree.
"""
import sys
from pathlib import Path

import numpy as np

# The toolkit (config, contractor_xml) lives one level up; the trace mock is
# beside this file and owns the field construction.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from contractor_xml import BubbleJob
from gen_mock_dense_trace import (N_LOW, N_SC, N_T, N_HIT, N_DENSE, N_EXPANDED,
                                  DATASET, expand, build_fields, write_field,
                                  write_field_by_timeslice)

TRAJ = 0
SEED = 20260929

# times[0] for the two-term bubble. Non-zero so that dropping it would fail.
DELTA = 1

OUT_DIR = Path(config.OUTPUT_ROOT) / "mock_dense_bubble"

# saveCorrelator's stem: terms[i]_times[i]_... then terms.back(). One term
# takes no times, so the stem is just the term.
STEM_1 = "mf"
STEM_2 = f"mf_{DELTA}_mf"


def bubble1_expanded(expanded):
    """tr[Pi(t)] for every t, from the expanded field."""
    return np.array([np.trace(expanded[t]) for t in range(N_T)])


def bubble1_dense(dense):
    """The same, as accTrDense computes it: the diagonal survives only where
    the expanded row index is the partner of a dense row at this timeslice."""
    return np.array([sum(dense[t][i, expand(i, t)] for i in range(N_DENSE))
                     for t in range(N_T)])


def bubble2_expanded(expanded):
    """tr[Pi(t) Pi(t+DELTA)] for every t."""
    return np.array([np.trace(expanded[t] @ expanded[(t + DELTA) % N_T])
                     for t in range(N_T)])


def bubble2_dense(dense):
    """The same through the remap: i is dense in the first factor and expanded
    in the second, j the reverse, so each index expands through the OTHER
    factor's timeslice."""
    out = np.zeros(N_T, dtype=complex)
    for t in range(N_T):
        tb = (t + DELTA) % N_T
        col_a = [expand(j, tb) for j in range(N_DENSE)]
        col_b = [expand(i, t) for i in range(N_DENSE)]
        acc = 0.0 + 0.0j
        for i in range(N_DENSE):
            for j in range(N_DENSE):
                acc += dense[t][i, col_a[j]] * dense[tb][j, col_b[i]]
        out[t] = acc
    return out


def build_par(tag, dense_schema, time_slice_io=False, field=None):
    job = BubbleJob(nt=N_T, disk_vector_dir=f"{OUT_DIR}/dv.{tag}",
                    output=f"{OUT_DIR}/corr.{tag}",
                    traj_start=TRAJ, traj_end=TRAJ + 1,
                    n_hit=N_HIT if dense_schema else None)
    job.add_matrix(file=f"{OUT_DIR}/{field or tag}.@traj@.h5", dataset=DATASET,
                   name="mf", cache_size=N_T,
                   n_low=N_LOW if dense_schema else None,
                   time_slice_io=time_slice_io)
    job.add_product(terms=["mf"], times=[])
    job.add_product(terms=["mf", "mf"], times=[str(DELTA)])
    return job.write(OUT_DIR / f"par.bubble.{tag}.xml")


def main():
    rng = np.random.default_rng(SEED)
    dense, expanded = build_fields(rng)

    b1e, b1d = bubble1_expanded(expanded), bubble1_dense(dense)
    b2e, b2d = bubble2_expanded(expanded), bubble2_dense(dense)
    err = max(np.max(np.abs(b1e - b1d)), np.max(np.abs(b2e - b2d)))
    scale = max(np.max(np.abs(b1e)), np.max(np.abs(b2e)))
    print(f"dense {N_DENSE} x {N_EXPANDED}, expanded {N_EXPANDED} x {N_EXPANDED}, "
          f"nt {N_T}, delta {DELTA}")
    print(f"numpy oracle: max |expanded - dense| = {err:.3e} (scale {scale:.3e})")
    if err > 1e-10 * scale:
        print("MISMATCH: the index map in this script disagrees with itself")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_field(OUT_DIR / f"expanded.{TRAJ}.h5", expanded)
    write_field(OUT_DIR / f"dense.{TRAJ}.h5", dense)
    write_field_by_timeslice(OUT_DIR / f"densets.{TRAJ}.h5", dense)
    pars = [build_par("expanded", dense_schema=False),
            build_par("dense", dense_schema=True),
            build_par("densets", dense_schema=True, time_slice_io=True)]
    np.save(OUT_DIR / "bubble1.npy", b1e)
    np.save(OUT_DIR / "bubble2.npy", b2e)

    print(f"\nwrote fields, par files and oracles to {OUT_DIR}")
    print(f"\n  BubbleContractor      {pars[0]}")
    for p in pars[1:]:
        print(f"  BubbleContractorDense {p}")
    print(f"\nall three must reproduce, in {STEM_1}.{TRAJ}.h5 and {STEM_2}.{TRAJ}.h5:")
    for t in range(N_T):
        print(f"  t={t}  tr[Pi]={b1e[t].real:+.6e}{b1e[t].imag:+.6e}j"
              f"   tr[PiPi]={b2e[t].real:+.6e}{b2e[t].imag:+.6e}j")
    return 0


if __name__ == "__main__":
    sys.exit(main())
