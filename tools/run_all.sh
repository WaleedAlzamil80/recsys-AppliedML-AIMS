#!/usr/bin/env bash
# Run every experiment and rebuild the report assets.
#
#   bash tools/run_all.sh [SMALL_DIR] [BIG_DIR]
#
# Practicals 1-4 are the original scripts; 5-10 are the remaining items of
# todo.md. Expect ~30-60 minutes on a laptop for the small data set (the
# tuning sweep dominates); the 32M steps need the large data set and Numba.
set -euo pipefail
cd "$(dirname "$0")/.."
SMALL="${1:-datasets/ml-latest-small}"
BIG="${2:-datasets/ml-32m}"
PY="${PYTHON:-python}"

"$PY" tools/sanity_checks.py --data-dir "$SMALL"
"$PY" practice_01.py
"$PY" dataset_analysis.py --data-dir "$SMALL"
"$PY" practice_02.py
"$PY" practice_03.py
"$PY" practice_08.py --data-dir "$SMALL"     # tuning first: others reuse it
"$PY" practice_05.py --data-dir "$SMALL"
"$PY" practice_06.py --data-dir "$SMALL"
"$PY" practice_07.py --data-dir "$SMALL"
"$PY" practice_09.py --data-dir "$SMALL"
"$PY" practice_10.py --data-dir "$SMALL"
if [ -f "$BIG/ratings.csv" ]; then
  "$PY" practice_04.py --data-dir "$BIG"
  "$PY" benchmark_scaling.py --data-dir "$BIG" --small-dir "$SMALL"
else
  echo "skipping 32M steps: $BIG/ratings.csv not found"
fi
"$PY" tools/make_report_assets.py
echo "Done. Upload the whole report/ folder to Overleaf and compile main.tex."
