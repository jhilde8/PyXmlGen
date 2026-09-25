"""Effective mass of a contractor correlator -- a quick look, not a measurement.

    effective_mass.py mf_sl_0_mf_ls.1520.h5
    effective_mass.py mf_pi_p100_0_mf_pi_pm100.1520.h5 --tmin 15 --tmax 40

Prints C(t), m_eff(t) = log(C(t)/C(t+1)), and the mean over the window. With a
single trajectory there is no covariance to fit against, so the +- quoted is
the spread of m_eff ACROSS t -- how flat the plateau is, not a statistical
error. Compare the number against expectation yourself.

Two free checks come with it. A zero-momentum two-point function is real up to
noise, so a large Im/Re means something is wrong upstream; and a correlator
translation-averaged over all nt sources with periodic boundaries must satisfy
C(t) = C(nt-t), which usually breaks earlier and more obviously than the mass
does. m_eff is biased near t = nt/2 by the backward-propagating state, so read
the plateau well before the midpoint.
"""
import argparse
import sys

import numpy as np
import h5py


def read_correlator(path):
    """The correlator out of a saveCorrelator file. The group is the product's
    stem (terms[0]_times[0]_terms.back()), which differs per contraction, so
    take the sole top-level key rather than hardcoding it."""
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if len(keys) != 1:
            raise KeyError(f"expected one group in {path}, found {keys}")
        raw = f[keys[0]]["correlator"][...]
        return keys[0], raw["re"] + 1j * raw["im"]


def effective_mass(c):
    """log(C(t)/C(t+1)), nan where the ratio is not positive."""
    re = c.real
    m = np.full(len(re) - 1, np.nan)
    good = (re[:-1] > 0) & (re[1:] > 0)
    m[good] = np.log(re[:-1][good] / re[1:][good])
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("correlator")
    parser.add_argument("--tmin", type=int, default=10)
    parser.add_argument("--tmax", type=int, default=30)
    args = parser.parse_args()

    stem, c = read_correlator(args.correlator)
    nt = len(c)
    m = effective_mass(c)

    mirrored = np.array([c[(nt - t) % nt] for t in range(nt)])
    scale = np.max(np.abs(c))

    print(f"{args.correlator}")
    print(f"  group {stem}, nt = {nt}")
    print(f"  Im/Re      {np.max(np.abs(c.imag)) / np.max(np.abs(c.real)):.3e}")
    print(f"  symmetry   {np.max(np.abs(c - mirrored)) / scale:.3e}"
          f"   (C(t) vs C(nt-t), relative)")

    print(f"\n   t       C(t).re        m_eff(t)")
    for t in range(max(0, args.tmin - 5), min(nt - 1, args.tmax + 6)):
        mark = "  <" if args.tmin <= t <= args.tmax else ""
        print(f"  {t:3d}   {c[t].real:+.6e}   {m[t]:.6f}{mark}")

    window = m[args.tmin:args.tmax + 1]
    print(f"\n  mean over [{args.tmin}, {args.tmax}] = "
          f"{np.nanmean(window):.5f} +- {np.nanstd(window):.5f} (flatness)")


if __name__ == "__main__":
    sys.exit(main())
