#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "$ROOT_DIR/.venv/bin/python3" ]]; then
  echo "Missing .venv. Run scripts/install.sh runtime first."
  exit 1
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/.venv/bin/activate"

python -m pip install --upgrade pip

if [[ "$(uname -s)" == "Darwin" ]]; then
  CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 python -m pip install llama-cpp-python --no-cache-dir
else
  python -m pip install llama-cpp-python --no-cache-dir
fi

python -c "from llama_cpp import Llama; print('llama-cpp-python ready')"
