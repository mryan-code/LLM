#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
exec "$PYTHON_BIN" -m storm_zero_llm "$@"
