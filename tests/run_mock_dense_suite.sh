#!/bin/bash
# Regression of our dense-W tools against Masaaki's, on mock data.
#
#     run_mock_dense_suite.sh <utilities build dir> <run dir>
#
# e.g. Hadrons/build_cpu/utilities and a scratch directory. Build first with
#
#     make A2APixPi FlexibleContractorDense BubbleContractorDense \
#          A2AMesonFieldProduct FlexibleContractorDWC BubbleContractorDWC
#
# in the utilities build dir (all are EXTRA_PROGRAMS). The run dir must not
# exist: Masaaki's contractor refuses existing disk-vector directories.
#
#   1. gen_mock_dense_suite.py writes the fields and every par file.
#   2. A2AMesonFieldProduct (ref/) vs A2APixPi (test/): diff_a2am_ts.py at
#      tolerance 0, merged and per-slice. Expected bit for bit.
#   3. FlexibleContractorDWC (corr_ref/) vs FlexibleContractorDense
#      (corr_test/), each on its own author's products:
#      check_mock_dense_suite.py.
#   4. BubbleContractorDWC (bub_ref/) vs BubbleContractorDense (bub_test/),
#      plus tr(product) against the 2-term trace of its factors, in the same
#      check script.
set -e

bin=$(realpath "$1")
run=$(realpath -m "$2")
here=$(dirname "$(realpath "$0")")

python3 "$here/gen_mock_dense_suite.py" "$run"
cd "$run"

"$bin/A2AMesonFieldProduct" par.product.ref.xml     > log.A2AMesonFieldProduct 2>&1
"$bin/A2APixPi"             par.product.test.xml    > log.A2APixPi 2>&1
echo "== products: A2APixPi vs A2AMesonFieldProduct"
python3 "$here/../diffs/diff_a2am_ts.py" ref test --tol 0 --mode both | grep -E "^\[|merge  |slice  |OVERALL"

"$bin/FlexibleContractorDWC"   par.flexible.ref.xml  > log.FlexibleContractorDWC 2>&1
"$bin/FlexibleContractorDense" par.flexible.test.xml > log.FlexibleContractorDense 2>&1
"$bin/BubbleContractorDWC"     par.bubble.ref.xml    > log.BubbleContractorDWC 2>&1
"$bin/BubbleContractorDense"   par.bubble.test.xml   > log.BubbleContractorDense 2>&1
echo "== correlators: FlexibleContractorDense vs FlexibleContractorDWC"
python3 "$here/check_mock_dense_suite.py" "$run"
