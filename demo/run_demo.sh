#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/.venv/lib/python3.14/site-packages:$PROJECT_ROOT/venv/lib/python3.14/site-packages:${PYTHONPATH:-}"

exec python3 "$SCRIPT_DIR/app.py" "$@"
