#!/usr/bin/env bash
#
# End-to-end dense-W contractor regression: generate the mock fields and par
# files, run all three contractions, and diff them against the oracle.
#
#   tests/run_mock_dense.sh
#   BIN=/path/to/hadrons/build/utilities tests/run_mock_dense.sh
#
# BIN is the directory holding HadronsContractor and ContractorDense. It
# defaults to the local CPU build. Pass it explicitly on Riker, and note that
# invoking the binaries by name instead would pick up whatever is first on
# PATH -- an older install without HDF5 fails deep inside A2AMatrixIo::load
# with a message that reads like a build misconfiguration.
#
# The output tree is wiped and rebuilt each run, which is not just tidiness:
# the two directories the contractor uses have opposite requirements.
#
#   corr.<tag>       must EXIST. saveCorrelator calls makeFileDir(dir), and
#                    makeFileDir creates dirname() of its argument -- handed
#                    the output directory it creates the PARENT and never the
#                    directory itself, so ResultWriter then cannot create the
#                    file. Shared by all four contractor utilities.
#
#   dv.<tag>/mf      must NOT exist. DiskVectorBase hard-errors on a
#                    pre-existing directory before creating its own, since it
#                    wipes what it owns. A successful run removes it, but any
#                    crash leaves it behind and blocks the next attempt --
#                    which is how one failure turns into two.

set -euo pipefail

BIN="${BIN:-/home/quarkonium/gpubuild/Hadrons/build_cpu/utilities}"

TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYXMLGEN="$(dirname "$TESTS")"
OUT="$PYXMLGEN/output/mock_dense"

echo "== generating mock fields and par files =="
rm -rf "$OUT"
python3 "$TESTS/gen_mock_dense_trace.py"

echo
echo "== contracting =="
mkdir -p "$OUT/corr.expanded" "$OUT/corr.dense" "$OUT/corr.densets"
cd "$OUT"
"$BIN/HadronsContractor" par.mock.expanded.xml > run.expanded.log 2>&1
"$BIN/ContractorDense"   par.mock.dense.xml    > run.dense.log    2>&1
"$BIN/ContractorDense"   par.mock.densets.xml  > run.densets.log  2>&1
echo "expanded, dense, densets: done (logs in $OUT/run.*.log)"

echo
echo "== checking =="
cd "$PYXMLGEN"
python3 "$TESTS/check_mock_dense_trace.py" | tee "$OUT/check.log"
