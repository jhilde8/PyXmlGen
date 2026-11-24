"""A2AMesonFieldSelectHits on single-file and timesliced fields, 8 hits -> 4.

    check_select_hits_ts.py <utilities build dir> <run dir>

For each of pi (W x V), ls_ww (W x W) and mix (V x V) one random 8-hit
ComplexD field is written twice: one file holding all NT slices, and NT
.tNNNN files of one slice each, the layouts A2AMesonField writes without and
with timeSliceIO. gen_select_hits.py writes the par files for both, the
utility cuts each to hits 0..3, and every output is compared with a numpy cut
of the same array: the low block and the first 4 hit blocks of each axis, a V
axis's hit blocks times 8/4. A power-of-two scale is exact, so the comparison
is bit for bit, as is the single-file output against the stacked timesliced
ones. The run dir must not exist; the utility refuses existing outputs.
"""
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

NT, N_LOW, N_HIGH, N_SC = 4, 4, 6, 3
N_HIT_IN, HITS = 8, [0, 1, 2, 3]
TRAJ = 100
DATASETS = {"pi": "Gamma5_0_0_1", "ls_ww": "Identity_0_0_0", "mix": "Identity_0_0_0"}


def axis(leg):
    """(low size, modes per hit, is V) of one leg, as gen_select_hits sees it."""
    flavor, vw = leg
    return (N_LOW if config.FLAVOR_HAS_LOW[flavor] else 0,
            N_HIGH if vw == "v" else N_SC, vw == "v")


def cut(a, ax, dim):
    """numpy reference: the low block, then HITS' blocks, V blocks rescaled."""
    low, per_hit, is_v = ax
    scale = N_HIT_IN / len(HITS) if is_v else 1.0
    blocks = [np.take(a, range(low), axis=dim)]
    for h in HITS:
        block = np.take(a, range(low + h * per_hit, low + (h + 1) * per_hit), axis=dim)
        blocks.append(block * scale)
    return np.concatenate(blocks, axis=dim)


def write(path, dataset, a):
    c = np.empty(a.shape, dtype=np.dtype([("re", "<f8"), ("im", "<f8")]))
    c["re"], c["im"] = a.real, a.imag
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_group(dataset).create_dataset("a2aMatrix", data=c)


def read(path, dataset):
    with h5py.File(path, "r") as f:
        c = f[dataset]["a2aMatrix"][()]
    return c["re"] + 1j * c["im"]


def main():
    bin_dir, run = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    run.mkdir(parents=True)
    rng = np.random.default_rng(7)
    expected, shapes, fields = {}, {}, []

    for kind, dataset in DATASETS.items():
        left, right = (axis(leg) for leg in config.FIELD_LEGS[kind])
        shape = (NT,) + tuple(low + N_HIT_IN * n for low, n, _ in (left, right))
        a = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        expected[kind], shapes[kind] = cut(cut(a, left, 1), right, 2), shape
        write(run / "in" / f"{kind}_single_h8.{TRAJ}" / f"{dataset}.h5", dataset, a)
        for t in range(NT):
            write(run / "in" / f"{kind}_ts_h8.{TRAJ}" / f"{dataset}.t{t:04d}.h5",
                  dataset, a[t:t + 1])
        fields += [("--field", kind, f"{kind}_{layout}_h8", f"{kind}_{layout}_h4")
                   for layout in ("single", "ts")]

    gen = Path(__file__).resolve().parent.parent / "Contractor" / "gen_select_hits.py"
    subprocess.run([sys.executable, str(gen), "--in-root", str(run / "in"),
                    "--out-root", str(run / "out"), "--traj-start", str(TRAJ),
                    "--traj-end", str(TRAJ + 1), "--n-hit-in", str(N_HIT_IN),
                    "--hits", *map(str, HITS), "--n-low", str(N_LOW),
                    "--n-high", str(N_HIGH), "--n-sc", str(N_SC),
                    "--xml-dir", str(run / "xml")] + [x for f in fields for x in f],
                   check=True)
    for par in sorted((run / "xml").glob("par.select.*.xml")):
        with open(run / f"log.{par.stem}", "w") as log:
            subprocess.run([str(bin_dir / "A2AMesonFieldSelectHits"), str(par)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    ok = True
    for kind, dataset in DATASETS.items():
        ref = expected[kind]
        single = read(run / "out" / f"{kind}_single_h4.{TRAJ}" / f"{dataset}.h5", dataset)
        ts_files = sorted((run / "out" / f"{kind}_ts_h4.{TRAJ}").glob("*.h5"))
        slices = [read(f, dataset) for f in ts_files]
        ts = np.concatenate(slices, axis=0)
        checks = {
            "single vs numpy": single.shape == ref.shape and np.array_equal(single, ref),
            "timesliced vs numpy": ts.shape == ref.shape and np.array_equal(ts, ref),
            "single vs timesliced": np.array_equal(single, ts),
            f"{NT} files of 1 slice": (len(ts_files) == NT
                                       and all(s.shape[0] == 1 for s in slices)),
        }
        for what, passed in checks.items():
            ok &= passed
            print(f"  {'PASS' if passed else 'FAIL'}  {kind:6s} {what:24s} "
                  f"{tuple(ref.shape)} from {shapes[kind]}")
    print("OVERALL:", "ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
