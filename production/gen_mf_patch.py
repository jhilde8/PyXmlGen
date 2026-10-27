"""Redo the meson fields a contraction job left incomplete, as one job.

    gen_mf_patch.py h8 --conf 1540
    gen_mf_patch.py ama --conf 1540 1550
    gen_mf_patch.py h8 --conf 1540 --fields cmf_ape_h8 mix_h8

For a job that failed in its contractions. For each configuration,
mf_scan.scan checks the job's fields on disk (the same scan check_mf.py
reports), and every incomplete field is contracted again in full: a module
that died mid-write leaves every one of its files short, so a field is the
smallest unit there is. --fields names the fields instead of scanning.

The patch is gen_meson_fields.build_job for the same tag, cut down to the
contractions to redo and everything they depend on, in the original order.
A module's inputs are the option values naming another module, so only the
vector loads, smears and loops those contractions use are kept. The smear
ladder survives as a chain: a field on the top rung keeps every smear below
it, but none of the lower rungs' contractions. Keeping the original order
keeps every consumer of an unsmeared array ahead of the smear that moves it.

The loops are always read back from LOOP_ROOT and nothing is sparsened: a
job that reached its contractions has saved its loops and written its sparse
vectors, all of which come before the first contraction.

The patch runs on the full job's node count and layout, and loads every
vector array its contractions need, so redoing one EMF still pays the whole
V_s and V_l load. The summary lists what is loaded.

Written to OUTPUT_ROOT/production with the trajectory written in, since a
patch is particular to one configuration:

    par.contraction.<tag>.patch.<conf>.xml, schedule.contraction.<tag>.patch.<conf>.txt

The previous patch of the same tag and configuration is deleted first, so a
configuration that is complete leaves no stale patch behind.

Run it on Frontier, after the drain has ended.
"""
import argparse
import sys
from pathlib import Path

# The toolkit (config, hadrons_xml, mf_scan) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from hadrons_xml import Job
from mf_scan import scan
from gen_meson_fields import build_job
from check_mf import report


def name(el):
    return el.findtext("id/name")


def output_field(el):
    out = el.findtext("options/output") or ""
    prefix = config.TMP_OUTPUT + "/"
    return out[len(prefix):] if out.startswith(prefix) else None


def prune(job, fields):
    """The modules of `job`, in order, that the contractions writing
    `fields` need: those contractions and every module they depend on."""
    modules = list(job.modules_el)
    names = {name(el) for el in modules}
    keep = {name(el) for el in modules if output_field(el) in fields}
    for el in reversed(modules):
        if name(el) in keep:
            keep |= {tok for o in el.find("options").iter()
                     for tok in (o.text or "").split()} & names
    return [el for el in modules if name(el) in keep]


def clear_previous(out_dir, run_id):
    for path in (out_dir / f"par.{run_id}.xml",
                 out_dir / f"schedule.{run_id}.txt"):
        path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a patch job for the meson fields missing on disk.")
    parser.add_argument("tag", choices=sorted(config.MF_EXPECTED))
    parser.add_argument("--conf", nargs="+", type=int, required=True)
    parser.add_argument("--fields", nargs="+", metavar="FIELD",
                        help="fields to redo (default: every incomplete one)")
    parser.add_argument("--hits", type=int, nargs="+", metavar="H",
                        help="as gen_meson_fields.py --hits, if the job used it")
    parser.add_argument("--io-split", metavar="LAYOUT", default="2 2 2 2",
                        type=lambda s: s.replace(".", " "),
                        help="as gen_meson_fields.py --io-split (default: 2.2.2.2)")
    args = parser.parse_args()

    out_dir = Path(config.OUTPUT_ROOT) / "production"
    known = set(config.MF_EXPECTED[args.tag][1])
    if args.fields and set(args.fields) - known:
        parser.error(f"not fields of '{args.tag}': "
                     f"{' '.join(sorted(set(args.fields) - known))}")

    for traj in args.conf:
        run_id = f"contraction.{args.tag}.patch.{traj}"
        clear_previous(out_dir, run_id)
        fields = args.fields
        if not fields:
            scans = scan(args.tag, traj)
            lines, complete = report(args.tag, traj, scans)
            print("\n".join(lines))
            if complete:
                print("nothing to patch\n")
                continue
            fields = [f.name for f in scans if not f.complete()]

        full, _ = build_job(args.tag, hits=args.hits, io_split=args.io_split,
                            load_loops=True)
        kept = prune(full, set(fields))
        job = Job(run_id, traj_start=traj, traj_end=traj + 1,
                  schedule_file=config.schedule_file(run_id),
                  graph_file=config.GRAPH)
        for el in kept:
            job.add(el)
        job.write(out_dir / f"par.{run_id}.xml",
                  out_dir / f"schedule.{run_id}.txt")

        def of_type(*types):
            return [name(el) for el in kept if el.findtext("id/type") in types]

        print(f"{run_id}: {len(fields)} fields, {len(kept)} modules")
        print("  loads  " + " ".join(of_type("MIO::LoadCombinedA2AVecsV",
                                             "MIO::LoadCombinedA2AVecsW")))
        print("  loops  " + (" ".join(of_type("MIO::LoadProp")) or "-"))
        print(f"  smears {len(of_type('MUtilities::A2ACovariantSmear'))}")
        print(f"wrote {out_dir / f'par.{run_id}.xml'}\n")


if __name__ == "__main__":
    main()
