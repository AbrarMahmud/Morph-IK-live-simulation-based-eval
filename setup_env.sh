#!/usr/bin/env bash
# Creates the dedicated virtual environment (.venv) and installs the demo.
#   ./setup_env.sh          # default PyPI torch
#   ./setup_env.sh --cpu    # smaller CPU-only torch wheel (Linux / Windows)
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.11 python3.10 python3.12 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
[ -n "$PY" ] || { echo "Python 3.10-3.12 not found (set PYTHON=/path/to/python)"; exit 1; }
"$PY" - <<'PYCHECK'
import sys
v = sys.version_info[:2]
if not ((3, 10) <= v <= (3, 12)):
    sys.exit(f"Python {v[0]}.{v[1]} found; please use 3.10-3.12 (pybullet has no wheels for newer versions). "
             "Example: PYTHON=python3.11 ./setup_env.sh")
PYCHECK

echo "[setup] creating .venv with $("$PY" --version)"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel
if [ "${1:-}" = "--cpu" ]; then
  pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install -r requirements.txt
echo "[setup] running the offline self-test"
python -m graphik_demo selftest
echo
echo "[setup] done. Activate with:  source .venv/bin/activate"
