#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Darwin" ]]; then
  exit 0
fi

read_env_value() {
  local key="$1"
  local env_file="$ROOT_DIR/.env"
  [[ -f "$env_file" ]] || return 1

  awk -F= -v key="$key" '
    $0 ~ "^[[:space:]]*" key "[[:space:]]*=" {
      value = $0
      sub(/^[^=]*=/, "", value)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      gsub(/^["\047]|["\047]$/, "", value)
      print value
      exit
    }
  ' "$env_file"
}

coreml_path="$(read_env_value IMAGE_COREML_PATH || true)"
variant="${IMAGE_COREML_VARIANT:-urpm-v13-split-einsum}"

if [[ -z "${coreml_path// }" ]]; then
  exit 0
fi

if [[ -d "$coreml_path/TextEncoder.mlmodelc" || -d "$coreml_path/Unet.mlmodelc" ]]; then
  echo "Core ML model already present at $coreml_path"
  exit 0
fi

echo "Core ML model missing at $coreml_path; downloading $variant"
bash "$ROOT_DIR/scripts/download_coreml_sd.sh" --variant "$variant" --output-dir "$(dirname "$coreml_path")"
