#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Skipping Core ML install (macOS only)."
  exit 0
fi

if [[ ! -f "$ROOT_DIR/.venv/bin/python3" ]]; then
  echo "Missing .venv. Run scripts/install.sh runtime first."
  exit 1
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/.venv/bin/activate"

python -m pip install --upgrade pip

# ml-stable-diffusion pins numpy<1.24 which does not build on Python 3.13.
python -m pip install "git+https://github.com/apple/ml-stable-diffusion.git" --no-deps
python -m pip install "coremltools>=8.0" scipy invisible-watermark

python <<'PY'
from pathlib import Path
import python_coreml_stable_diffusion

pipeline = Path(python_coreml_stable_diffusion.__file__).parent / "pipeline.py"
text = pipeline.read_text(encoding="utf-8")
old = "from transformers import CLIPFeatureExtractor, CLIPTokenizer"
new = """from transformers import CLIPTokenizer
try:
    from transformers import CLIPFeatureExtractor
except ImportError:
    from transformers.models.clip.image_processing_clip import CLIPImageProcessor as CLIPFeatureExtractor"""
if old in text:
    pipeline.write_text(text.replace(old, new), encoding="utf-8")
PY

python -c "from python_coreml_stable_diffusion.pipeline import get_coreml_pipe; print('Core ML Stable Diffusion ready')"
