"""Hit-selection par files for A2AMesonFieldSelectHits: cut meson fields
computed over many hits down to a subset, for comparison against fields
computed over that subset alone.

    gen_select_hits.py --in-root R --out-root O --traj-start T --traj-end T1 \\
        --n-hit-in 8 --hits 0 1 \\
        --field pi  mf_pi_h8          mf_pi_h2 \\
        --field emf emf_lloop_h8      emf_lloop_h2

    gen_select_hits.py ... --n-low 2000 --n-high 1152    another ensemble

Each --field is KIND IN_STEM OUT_STEM. KIND is a config.FIELD_LEGS key and
fixes the two legs; the stems are field directories under --in-root and
--out-root, before the ".<traj>". The stems are taken as given, so the same
generator slices this toolkit's fields or a collaborator's, whatever their
naming. One par file per --field: the utility is serial and single-process,
so a field directory is the unit of parallelism.

Files are DISCOVERED in each input directory at --traj-start rather than built
from gamma/momentum/type lists, for the reason diff_a2am.py gives: a fixed list
silently skips whatever it does not know about. Either layout works -- the
dataset is the file stem with any .tNNNN timeslice infix removed, and the
output keeps the input's file name.

Normalization. On a V axis the utility rescales the high blocks by
nHitIn/len(hits); a W axis and every low block are left alone. That is the
1/nHit convention VectorPool applies at load, so a sliced field matches a
freshly computed --hits field with no further factor. --n-hit-in applies to
every --field of an invocation, so a reference whose fields carry different
hit counts takes one invocation per count.

The EMF's internal loop is not sliced -- the utility only touches the two mode
axes. A comparison EMF must be built from the reference's own loop.

Mode counts default to config; --n-low, --n-high and --n-sc override them for
an ensemble config does not describe. The utility checks
lowSize + nHitIn*modesPerHit against every dataset, so a wrong count fails on
the first file. It also refuses to overwrite an existing output.
"""
import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# The toolkit (config, contractor_xml) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from contractor_xml import SelectHitsJob, select_axis

TIMESLICE = re.compile(r"\.t\d{4}$")


@dataclass
class ModeCounts:
    n_low: int
    n_high: int
    n_sc: int


def axis(leg, n_hit_in, hits, modes):
    flavor, vw = leg
    return select_axis(
        low_size=modes.n_low if config.FLAVOR_HAS_LOW[flavor] else 0,
        modes_per_hit=modes.n_high if vw == "v" else modes.n_sc,
        n_hit_in=n_hit_in, is_v=(vw == "v"), hits=hits)


def select_field(job, kind, in_stem, out_stem, n_hit_in, hits, modes,
                 probe_traj):
    """One selection per .h5 file in <in_stem>.<probe_traj>/, written to the
    same file name under <out_stem>.@traj@/. Returns the file count."""
    left_leg, right_leg = config.FIELD_LEGS[kind]
    left = axis(left_leg, n_hit_in, hits, modes)
    right = axis(right_leg, n_hit_in, hits, modes)

    files = sorted(Path(f"{in_stem}.{probe_traj}").glob("*.h5"))
    if not files:
        raise SystemExit(f"no .h5 files in {in_stem}.{probe_traj}")
    for f in files:
        job.add(file=f"{in_stem}.@traj@/{f.name}",
                dataset=TIMESLICE.sub("", f.stem),
                output=f"{out_stem}.@traj@/{f.name}",
                left=left, right=right)
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--in-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--traj-start", type=int, required=True,
                        help="first trajectory, and the one probed for files")
    parser.add_argument("--traj-end", type=int, required=True,
                        help="one past the last trajectory")
    parser.add_argument("--traj-step", type=int, default=1)
    parser.add_argument("--n-hit-in", type=int, required=True,
                        help="hit count of every input field")
    parser.add_argument("--hits", type=int, nargs="+", default=[0, 1],
                        metavar="H", help="hits to keep, in output order")
    parser.add_argument("--field", nargs=3, action="append", required=True,
                        metavar=("KIND", "IN_STEM", "OUT_STEM"),
                        help=f"KIND is one of {', '.join(config.FIELD_LEGS)}")
    parser.add_argument("--n-low", type=int, default=config.N_LOW)
    parser.add_argument("--n-high", type=int, default=config.N_HIGH)
    parser.add_argument("--n-sc", type=int, default=config.N_SC)
    parser.add_argument("--xml-dir", default=str(Path(config.OUTPUT_ROOT)
                                                 / "select_hits"))
    args = parser.parse_args()

    for kind, _, _ in args.field:
        if kind not in config.FIELD_LEGS:
            parser.error(f"unknown field kind '{kind}' "
                         f"(have {', '.join(config.FIELD_LEGS)})")

    modes = ModeCounts(args.n_low, args.n_high, args.n_sc)
    for kind, in_stem, out_stem in args.field:
        job = SelectHitsJob(args.traj_start, args.traj_end, args.traj_step)
        n = select_field(job, kind, f"{args.in_root}/{in_stem}",
                         f"{args.out_root}/{out_stem}", args.n_hit_in,
                         args.hits, modes, args.traj_start)
        path = job.write(Path(args.xml_dir) / f"par.select.{out_stem}.xml")
        print(f"  {out_stem:32s} {n:5d} files  -> {path}")


if __name__ == "__main__":
    main()
