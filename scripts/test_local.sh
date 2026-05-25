#!/bin/bash
# Local pipeline runner using solution/ (our code).
# Usage: ./scripts/test_local.sh <env> <num_instances>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOLUTION="$ROOT/solution"
BASELINE="$ROOT/baseline"
PY="$ROOT/.venv/Scripts/python.exe"
if [ ! -x "$PY" ]; then PY="$ROOT/.venv/bin/python"; fi

TASK="${1:-toggle_lights}"
N="${2:-10}"
export ENV_ID="$TASK"

WORK="$ROOT/_work/${TASK}"
rm -rf "$WORK"
mkdir -p "$WORK"

cp "$BASELINE/gym.py" "$WORK/gym.py"
# copy our solution files into the work dir so train/solve can find them via sys.path
cp "$SOLUTION"/*.py "$WORK/"

echo "=== generate ==="
PYTHONPATH="$WORK" "$PY" "$BASELINE/generate_states.py" --num_instances "$N" --public_output "$WORK/input_states.jsonl"

echo
echo "=== train (limit ${TRAIN_TIME_LIMIT:-60}s) ==="
cd "$WORK"
PYTHONPATH="$WORK" TRAIN_TIME_LIMIT="${TRAIN_TIME_LIMIT:-60}" "$PY" train.py

echo
echo "=== solve (limit ${SOLVE_TIME_LIMIT:-60}s) ==="
PYTHONPATH="$WORK" SOLVE_TIME_LIMIT="${SOLVE_TIME_LIMIT:-60}" "$PY" solve.py

echo
echo "=== check ==="
CHECK="$WORK/check"
mkdir -p "$CHECK"
cp "$WORK/gym.py" "$WORK/input_states.jsonl" "$WORK/output_actions.csv" "$CHECK/"
cd "$CHECK"
PYTHONPATH="$CHECK" "$PY" "$BASELINE/check.py" --input input_states.jsonl --submission output_actions.csv

echo
echo "=== result ==="
cat verdict.txt; echo
cat score.json; echo
