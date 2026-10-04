#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

install_llama_cpp() {
  bash "$ROOT_DIR/scripts/install_llama_cpp.sh"
}

install_coreml_if_needed() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    return 0
  fi
  bash "$ROOT_DIR/scripts/install_coreml.sh"
  bash "$ROOT_DIR/scripts/ensure_coreml_model.sh"
}

case "${1:-runtime}" in
  runtime)
    install_llama_cpp
    # realtime adds websockets and faster-whisper. Without websockets the camera socket never listens.
    python -m pip install -e ".[runtime,image,realtime]"
    install_coreml_if_needed
    ;;
  text)
    install_llama_cpp
    python -m pip install -e ".[runtime,realtime]"
    ;;
  image)
    python -m pip install -e ".[image]"
    install_coreml_if_needed
    ;;
  test|dev)
    python -m pip install -e ".[dev]"
    ;;
  *)
    echo "Usage: scripts/install.sh [runtime|text|image|test]"
    exit 1
    ;;
esac

mkdir -p models data
