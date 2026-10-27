"""Check the sparsened V and W files against what the contraction jobs write.

    check_sparse.py --conf 1540 1550
    check_sparse.py --conf 1540 --tags ama

Read-only. The expected blocks are the A2ACoarseGrid modules of
gen_meson_fields.build_job(tag, sparsen=True) for each tag (default h8 and
ama, which share SPARSE_ROOT: the light low modes come from h8 only, the AMA
hit and its exact V from ama). For every configuration, sparse_scan.scan
checks each block on disk: a mode is done only if elem<i>.bin exists at
exactly config.SPARSE_ELEM_BYTES. The report has one line per block, with any
missing or wrong-size indices and the .tmp files a killed write left behind,
and a last line naming the blocks to redo, for the one-hit sparsen job.

This finds what is missing or short, not a full-size file whose pages were
lost with a node. After any abnormal end, checksum-scan everything the job
wrote with Hadrons/tests/Test_a2a_checksum_scan on the coarse grid, e.g.

    srun ... ./Test_a2a_checksum_scan --grid 16.16.16.128 --mpi ... \\
        --traj 1540 --size 1536 --stems <SPARSE_ROOT>/l0_v,<SPARSE_ROOT>/s0_v

The report is printed and written to <out>/check.sparse.<conf>.txt (default
config.MF_DIR). Exit status 1 if anything is missing or the wrong size.

Run it on Frontier, after the jobs writing SPARSE_ROOT have ended: a file
still being written counts as missing.
"""
import argparse
import sys
import time
from pathlib import Path

# The toolkit (config, sparse_scan, vector_scan) lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from sparse_scan import scan
from vector_scan import index_runs
from gen_meson_fields import build_job


def expected_blocks(tags):
    """Output stem -> mode count of every A2ACoarseGrid the jobs `tags`
    emit, in module order."""
    blocks = {}
    for tag in tags:
        job, _ = build_job(tag, sparsen=True)
        for el in job.modules_el:
            if el.findtext("id/type") == "MUtilities::A2ACoarseGrid":
                blocks[el.findtext("options/output")] = int(
                    el.findtext("options/modeSize"))
    return blocks


def report(traj, scans):
    """Report lines for one configuration, and whether every block is on
    disk at full size."""
    lines = []
    expected = present = n_missing = n_wrong = n_tmp = 0
    redo = []
    for b in scans:
        expected += b.size
        present += b.size - len(b.bad())
        n_missing += len(b.missing)
        n_wrong += len(b.wrong_size)
        n_tmp += b.tmp
        details = []
        if b.missing:
            details.append(f"missing {index_runs(list(b.missing))}")
        for i, size in sorted(b.wrong_size.items()):
            details.append(f"size {i}:{size}")
        if b.tmp:
            details.append(f"{b.tmp} .tmp")
        if b.bad():
            redo.append(b.name())
        lines.append(f"{b.name():<10} {b.size - len(b.bad()):>5}/{b.size:<5} "
                     + ("; ".join(details) if details else "ok"))

    complete = not redo
    head = (f"# sparse conf {traj}: {present}/{expected} vectors, "
            f"{n_missing} missing, {n_wrong} wrong size, {n_tmp} .tmp -> "
            f"{'complete' if complete else 'INCOMPLETE'}")
    tail = [f"# redo: {' '.join(redo)}"] if redo else []
    return [head] + lines + tail, complete


def main():
    parser = argparse.ArgumentParser(
        description="Check the sparsened V and W files on disk.")
    parser.add_argument("--conf", nargs="+", type=int, required=True)
    parser.add_argument("--tags", nargs="+", default=["h8", "ama"],
                        help="jobs whose sparsening to expect (default: h8 ama)")
    parser.add_argument("--out", default=config.MF_DIR,
                        help="where check.sparse.<conf>.txt goes")
    args = parser.parse_args()

    blocks = expected_blocks(args.tags)
    out_dir = Path(args.out)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    all_complete = True
    for traj in args.conf:
        lines, complete = report(traj, scan(blocks, traj))
        all_complete &= complete
        print("\n".join(lines) + "\n")
        text = [f"# checked {stamp}, expected size "
                f"{config.SPARSE_ELEM_BYTES} bytes"] + lines
        (out_dir / f"check.sparse.{traj}.txt").write_text("\n".join(text) + "\n")
    sys.exit(0 if all_complete else 1)


if __name__ == "__main__":
    main()
