"""Synthetic oracle for the dense-W contractor trace.

Writes one small meson field twice -- once dense (rows = compact W slots) and
once expanded (rows = the full mode index, with every row whose dilution
timeslice does not match the slice left at zero) -- plus a par file for each
binary, so that

    HadronsContractor par.mock.expanded.xml
    ContractorDense   par.mock.dense.xml

must produce the same correlator. The dense field holds exactly the non-zero
rows of the expanded one, so the two contractions are the same sum with the
zeros dropped. That checks the index remap and nothing else, which is why the
entries are random rather than physical, and why the dimensions are chosen
small enough to eyeball rather than to resemble the lattice.

The script evaluates both contractions in numpy and refuses to write anything
if they disagree, so the oracle is self-checking before either binary exists.

Mode index convention (Hadrons/A2AMatrix.hpp, A2AModeSpace):

    dense row     d = nLow + h*nSc + sc             nLow + nHit*nSc
    expanded mode e = nLow + (h*nt + t)*nSc + sc    nLow + nHit*nt*nSc

Columns are stored expanded in both files; only the row axis differs.
"""
import sys
from pathlib import Path

import numpy as np
import h5py

# The toolkit (config, contractor_xml) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from contractor_xml import ContractorJob

# Deliberately tiny and unphysical: nLow non-zero so the low block is
# exercised, nHit > 1 so the hit stride is, nt > 2 so a wrong timeslice cannot
# coincide with a right one.
N_LOW = 4
N_SC = 3
N_T = 4
N_HIT = 2

TRAJ = 0
DATASET = "mock"
SEED = 20260924

OUT_DIR = Path(config.OUTPUT_ROOT) / "mock_dense"

N_DENSE = N_LOW + N_HIT * N_SC
N_EXPANDED = N_LOW + N_HIT * N_T * N_SC

# Grid writes std::complex<double> as a compound of two doubles named re/im
# (Grid/serialisation/Hdf5Type.h). The layout matches complex128 but the member
# names differ from h5py's default, so spell the dtype out.
CPLX = np.dtype([("re", "<f8"), ("im", "<f8")])


def expand(d, t):
    """Expanded partner of dense slot d for a field taken at timeslice t."""
    if d < N_LOW:
        return d
    h, sc = divmod(d - N_LOW, N_SC)
    return N_LOW + (h * N_T + t) * N_SC + sc


def build_fields(rng):
    """A random dense field and the expanded field it stands for."""
    dense = (rng.standard_normal((N_T, N_DENSE, N_EXPANDED))
             + 1j * rng.standard_normal((N_T, N_DENSE, N_EXPANDED)))
    expanded = np.zeros((N_T, N_EXPANDED, N_EXPANDED), dtype=complex)
    for t in range(N_T):
        for i in range(N_DENSE):
            expanded[t, expand(i, t), :] = dense[t, i, :]
    return dense, expanded


def correlator_expanded(expanded):
    """What stock Contractor computes: tr(E[ta] E[tb]), translation averaged."""
    c = np.zeros(N_T, dtype=complex)
    for dt in range(N_T):
        for t_last in range(N_T):
            c[(t_last - dt) % N_T] += np.trace(expanded[dt] @ expanded[t_last])
    return c / N_T


def correlator_dense(dense):
    """What accTrMulDense computes, written index by index.

    i is dense in a and expanded in b, j the reverse, so each index is
    expanded through the time of the OTHER factor: a's columns with tb,
    b's columns with ta.
    """
    c = np.zeros(N_T, dtype=complex)
    for dt in range(N_T):
        col_b = [expand(i, dt) for i in range(N_DENSE)]
        for t_last in range(N_T):
            col_a = [expand(j, t_last) for j in range(N_DENSE)]
            acc = 0.0 + 0.0j
            for i in range(N_DENSE):
                for j in range(N_DENSE):
                    acc += dense[dt][i, col_a[j]] * dense[t_last][j, col_b[i]]
            c[(t_last - dt) % N_T] += acc
    return c / N_T


def write_field(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = np.empty(array.shape, dtype=CPLX)
    rec["re"] = array.real
    rec["im"] = array.imag
    with h5py.File(path, "w") as f:
        f.create_group(DATASET).create_dataset("a2aMatrix", data=rec)


def build_par(tag, n_low=None, n_hit=None):
    job = ContractorJob(nt=N_T, disk_vector_dir=f"{OUT_DIR}/dv.{tag}",
                        output=f"{OUT_DIR}/corr.{tag}",
                        traj_start=TRAJ, traj_end=TRAJ + 1, n_hit=n_hit)
    job.add_matrix(file=f"{OUT_DIR}/{tag}.@traj@.h5", dataset=DATASET,
                   name="mf", cache_size=N_T, n_low=n_low)
    job.add_product(terms=["mf", "mf"], times=["0"],
                    translations=f"0..{N_T - 1}", translation_average=True)
    return job.write(OUT_DIR / f"par.mock.{tag}.xml")


def main():
    rng = np.random.default_rng(SEED)
    dense, expanded = build_fields(rng)

    c_exp = correlator_expanded(expanded)
    c_den = correlator_dense(dense)
    err = np.max(np.abs(c_exp - c_den))
    scale = np.max(np.abs(c_exp))
    print(f"dense {N_DENSE} x {N_EXPANDED}, expanded {N_EXPANDED} x {N_EXPANDED}, nt {N_T}")
    print(f"numpy oracle: max |expanded - dense| = {err:.3e} (scale {scale:.3e})")
    if err > 1e-10 * scale:
        print("MISMATCH: the index map in this script disagrees with itself")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_field(OUT_DIR / f"expanded.{TRAJ}.h5", expanded)
    write_field(OUT_DIR / f"dense.{TRAJ}.h5", dense)
    par_exp = build_par("expanded")
    par_den = build_par("dense", n_low=N_LOW, n_hit=N_HIT)
    np.save(OUT_DIR / "correlator.npy", c_exp)

    print(f"\nwrote fields, par files and correlator.npy to {OUT_DIR}")
    print(f"\n  HadronsContractor {par_exp}")
    print(f"  ContractorDense   {par_den}")
    print("\nboth must reproduce:")
    for t, v in enumerate(c_exp):
        print(f"  t={t}  {v.real:+.10e} {v.imag:+.10e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
