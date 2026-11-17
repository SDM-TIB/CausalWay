#!/usr/bin/env bash
# Regenerate every number in the paper's evaluation from scratch.
#
#   bash experiments/run_all.sh            # full run, ~25 min on 8 cores
#   bash experiments/run_all.sh --quick    # smoke test, ~3 min
#
# Each experiment appends one JSON line per run to experiments/out/*.jsonl and
# skips runs that are already there, so an interrupted run resumes. Delete the
# file to start that experiment over. report.py turns the outputs into
# experiments/out/REPORT.md and summary.json — the tables the paper quotes.
set -euo pipefail
cd "$(dirname "$0")/.."

# Hill climbing breaks score ties in set-iteration order; fix the hash seed so a
# rerun is bit-identical.
export PYTHONHASHSEED=0
PY="${PY:-python}"
QUICK="${1:-}"

$PY experiments/e1_ges_correctness.py $QUICK
$PY experiments/e2_constraint.py $QUICK
if [ -z "$QUICK" ]; then
  $PY experiments/e3_ges_prior.py
  $PY experiments/e4_interventional.py
  $PY experiments/e7_violation.py
  $PY experiments/e5_efficiency.py
else
  $PY experiments/e4_interventional.py --quick
fi
$PY experiments/report.py $QUICK
