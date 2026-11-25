"""Check the drained meson fields of contraction jobs, and compare replicas.

    check_mf.py h8 --conf 1540 1550
    check_mf.py ama --conf 1540 --replica a
    check_mf.py h8 --conf 1540 --compare a,b

Read-only. The fields a job writes are read off gen_meson_fields.build_job
for the tag, the whole job: every contraction module's output, its ioname
count (gammas x momenta, EMF types x gamma families, CMO parities x
orthogonal directions), and its dataset shape (1, left leg, right leg) from
config.FIELD_LEGS at the tag's hit count. A field has N_T files per ioname.

mf_scan.scan then checks each file on disk: it has the expected shape and
every HDF5 chunk is allocated at its full stored size. The chunk shape is
read from the file, so a configuration run with other block sizes needs
nothing here. Only the chunk index is read, never the data. --compare also
reads each chunk's stored Fletcher32 checksum in every replica and requires
them all equal: bitwise agreement, up to a 2^-32 collision per chunk.

The report has one line per field, missing, bad and mismatched files as
timeslice runs (a lost node shows as the slab of timeslices it owned), then
the first few bad and mismatched files with the reason. It is printed and
written to <out>/check.<tag>[.<rep>|.<a>-<b>].<conf>.txt (default
config.MF_DIR). Exit status 1 if any field is not complete everywhere.

Trees. On a machine with config.REPLICAS, --replica r checks mf.r (mf_ama.r
for ama), --compare a,b compares mf.a with mf.b, and neither checks the
verified tree. The kept replica still needs a full read through the filter
before the other is deleted: this does not check a chunk against its own
checksum.

Run it after the drain has ended, as a one-node batch job: the checksum reads
are about 30M small reads per replica per configuration at 8 hits, spread
over --workers processes (default: every core).
"""
import argparse
import sys
import time
from pathlib import Path

# The toolkit (config, mf_scan) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from mf_scan import FieldSpec, TIMESLICE, runs, scan
from gen_meson_fields import build_job, parse_tag

# Module type -> the option pairs whose product is its ioname count.
IONAME_AXES = {
    "MContraction::A2AMesonField": ("gammas", "mom"),
    "MContraction::A2AExtendedMesonField": ("types", "gammas1"),
    "MContraction::A2AChromoMagneticOperatorField": ("parities", "ifOrthogs"),
}

# Field name prefix -> config.FIELD_LEGS kind; mf_ls_ww before mf_ls.
KIND_PREFIXES = (("mf_ls_ww_", "ls_ww"), ("mf_ls_", "ls"), ("mf_sl_", "sl"),
                 ("mf_ll_", "ll"), ("mf_pi_", "pi"), ("emf_", "emf"),
                 ("cmf_", "cmf"), ("mix_", "mix"))

# Bad and mismatched files listed per field, after the summary line.
DETAIL_LINES = 5


def option_count(el, key):
    """Entries of one option: <elem> children (mom), else words."""
    opt = el.find(f"options/{key}")
    elems = opt.findall("elem")
    return len(elems) if elems else len((opt.text or "").split())


def field_kind(field):
    for prefix, kind in KIND_PREFIXES:
        if field.startswith(prefix):
            return kind
    raise ValueError(f"field '{field}' has no config.FIELD_LEGS kind")


def field_specs(tag):
    """FieldSpec of every field the full job of `tag` writes, in job order."""
    n_hit = len(parse_tag(tag)[0])
    job, _ = build_job(tag)
    prefix = config.TMP_OUTPUT + "/"
    specs = []
    for el in job.modules_el:
        axes = IONAME_AXES.get(el.findtext("id/type"))
        output = el.findtext("options/output") or ""
        if axes is None or not output.startswith(prefix):
            continue
        field = output[len(prefix):]
        left, right = config.FIELD_LEGS[field_kind(field)]
        shape = (1, config.leg_size(*left, n_hit),
                 config.leg_size(*right, n_hit))
        n_io = option_count(el, axes[0]) * option_count(el, axes[1])
        specs.append(FieldSpec(field, n_io, config.LATT_SIZE[-1], shape))
    return specs


def timeslices(names):
    return runs(sorted({int(TIMESLICE.match(n).group(2)) for n in names}))


def report(tag, traj, scans, replicas):
    """Report lines for one job and configuration, and whether every field
    is complete (and in agreement) in every replica."""
    multi = len(replicas) > 1
    lines = []
    details = []
    expected = good = 0
    incomplete = []
    for f in scans:
        n = f.spec.n_files()
        expected += n
        good += f.good
        parts = []
        for k, rep in enumerate(replicas):
            at = f"[{rep}] " if multi else ""
            if f.n_missing[k] == n:
                parts.append(f"{at}absent")
            elif f.n_missing[k]:
                parts.append(f"{at}missing {f.n_missing[k]} at t "
                             f"{runs(list(f.missing_t[k]))}")
            if f.bad[k]:
                parts.append(f"{at}bad {len(f.bad[k])} at t "
                             f"{timeslices(f.bad[k])}")
            if f.extra[k]:
                parts.append(f"{at}extra {len(f.extra[k])}: "
                             + ", ".join(f.extra[k][:3])
                             + (" ..." if len(f.extra[k]) > 3 else ""))
            for name, reason in sorted(f.bad[k].items())[:DETAIL_LINES]:
                details.append(f"    {f.spec.name} {at}bad {name}: {reason}")
        if 0 < f.ionames < f.spec.n_io:
            parts.append(f"{f.ionames}/{f.spec.n_io} ionames")
        if f.mismatch:
            parts.append(f"mismatch {len(f.mismatch)} at t "
                         f"{timeslices(f.mismatch)}")
            for name, what in sorted(f.mismatch.items())[:DETAIL_LINES]:
                details.append(f"    {f.spec.name} MISMATCH {name}: {what}")
        if not f.complete():
            incomplete.append(f.spec.name)
        lines.append(f"{f.spec.name:<28} {f.good:>5}/{n:<5}  "
                     + ("; ".join(parts) if parts else "ok"))

    complete = not incomplete
    label = "" if replicas == [None] else (
        f" replicas {','.join(replicas)}" if multi else f" replica {replicas[0]}")
    head = (f"# {tag}{label} conf {traj}: {good}/{expected} files "
            + ("complete and equal" if multi else "complete") + " -> "
            + ("complete" if complete else "INCOMPLETE"))
    tail = [f"# incomplete: {' '.join(incomplete)}"] if incomplete else []
    return [head] + lines + details + tail, complete


def replicas_of(parser, args):
    """The replica tags to scan: [None] for the verified tree."""
    if (args.replica or args.compare) and not config.REPLICAS:
        parser.error(f"{config.MACHINE} has no replicas")
    if args.replica:
        return [args.replica]
    if args.compare:
        reps = args.compare.split(",")
        unknown = [r for r in reps if r not in config.REPLICAS]
        if unknown or len(set(reps)) != len(reps) or len(reps) < 2:
            parser.error(f"--compare takes two or more distinct replicas of "
                         f"{','.join(config.REPLICAS)}, got {args.compare}")
        return reps
    return [None]


def main():
    parser = argparse.ArgumentParser(
        description="Check drained meson fields, and compare replicas.")
    parser.add_argument("tags", nargs="+", metavar="hN|ama")
    parser.add_argument("--conf", nargs="+", type=int, required=True)
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--replica", choices=config.REPLICAS or None,
                       help="check one replica's tree")
    which.add_argument("--compare", metavar="A,B",
                       help="check these replicas' trees and compare them")
    parser.add_argument("--workers", type=int,
                        help="inspecting processes (default: every core)")
    parser.add_argument("--out", default=config.MF_DIR,
                        help="where the check.*.txt reports go")
    args = parser.parse_args()
    for tag in args.tags:
        try:
            parse_tag(tag)
        except ValueError as e:
            parser.error(str(e))
    replicas = replicas_of(parser, args)
    suffix = "" if replicas == [None] else "." + "-".join(replicas)

    out_dir = Path(args.out)
    all_complete = True
    for tag in args.tags:
        specs = field_specs(tag)
        bases = [config.mf_base(tag == "ama", r) for r in replicas]
        for traj in args.conf:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            scans = scan(specs, bases, traj, args.workers)
            lines, complete = report(tag, traj, scans, replicas)
            all_complete &= complete
            print("\n".join(lines) + "\n")
            text = [f"# checked {stamp}, base {', '.join(bases)}"] + lines
            (out_dir / f"check.{tag}{suffix}.{traj}.txt").write_text(
                "\n".join(text) + "\n")
    sys.exit(0 if all_complete else 1)


if __name__ == "__main__":
    main()
