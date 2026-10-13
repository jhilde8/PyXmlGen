#!/usr/bin/env bash
#
# End-to-end dense-W bubble regression: generate the mock fields and par files,
# run all three contractions, and diff them against the oracles.
#
#   tests/run_mock_dense_bubble.sh
#   BIN=/path/to/hadrons/build/utilities tests/run_mock_dense_bubble.sh
#
# Covers both branches: one term (tr[Pi(t)], the disconnected sigma loop) and
# two (tr[Pi(t) Pi(t+1)], a two-pion bubble). See run_mock_dense.sh for the
# directory rules -- <output> must exist, <diskVectorDir>/<name> must not --
# which is why the whole tree is wiped and rebuilt each run.

set -euo pipefail

BIN="${BIN:-/home/quarkonium/gpubuild/Hadrons/build_cpu/utilities}"

TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYXMLGEN="$(dirname "$TESTS")"
OUT="$PYXMLGEN/output/mock_dense_bubble"

echo "== generating mock fields and par files =="
rm -rf "$OUT"
python3 "$TESTS/gen_mock_dense_bubble.py" | head -3

echo
echo "== contracting =="
mkdir -p "$OUT/corr.expanded" "$OUT/corr.dense" "$OUT/corr.densets"
cd "$OUT"
"$BIN/BubbleContractor"      par.bubble.expanded.xml > run.expanded.log 2>&1
"$BIN/BubbleContractorDense" par.bubble.dense.xml    > run.dense.log    2>&1
"$BIN/BubbleContractorDense" par.bubble.densets.xml  > run.densets.log  2>&1
echo "expanded, dense, densets: done (logs in $OUT/run.*.log)"

echo
echo "== checking =="
cd "$PYXMLGEN"
python3 "$TESTS/check_mock_dense_bubble.py" | tee "$OUT/check.log"
