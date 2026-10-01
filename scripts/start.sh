#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEGACY_PID_FILE="$ROOT_DIR/.storm-zero-llm.pid"
MODE="foreground"
SEED_FOUNDATION="false"

cd "$ROOT_DIR"

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

stop_legacy_pid_file_server() {
  [[ -f "$LEGACY_PID_FILE" ]] || return 0

  local legacy_pid
  legacy_pid="$(<"$LEGACY_PID_FILE")"

  if [[ -n "$legacy_pid" ]] && kill -0 "$legacy_pid" 2>/dev/null; then
    echo "Stopping legacy PID-file server with PID $legacy_pid"
    kill "$legacy_pid"

    for _ in {1..30}; do
      if ! kill -0 "$legacy_pid" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
  else
    echo "Removing stale legacy PID file."
  fi

  rm -f "$LEGACY_PID_FILE"
}

for arg in "$@"; do
  case "$arg" in
    foreground|"")
      ;;
    --background|-b|background)
      MODE="background"
      ;;
    --seed-foundation)
      SEED_FOUNDATION="true"
      ;;
    --help|-h|help)
      echo "Usage: scripts/start.sh [--background|-b] [--seed-foundation]"
      echo
      echo "Starts or restarts the PM2 app named by PM2_APP in .env."
      exit 0
      ;;
    *)
      echo "Usage: scripts/start.sh [--background|-b] [--seed-foundation]"
      exit 1
      ;;
  esac
done

PM2_APP="${PM2_APP:-$(read_env_value PM2_APP || true)}"
LLM_HOST="${LLM_HOST:-$(read_env_value LLM_HOST || true)}"
LLM_PORT="${LLM_PORT:-${SERVER_PORT:-${PORT:-$(read_env_value LLM_PORT || true)}}}"

if [[ -z "${PM2_APP// }" ]]; then
  echo "Missing PM2_APP. Add PM2_APP to .env or export it before starting."
  exit 1
fi

if ! command -v pm2 >/dev/null 2>&1; then
  echo "Missing pm2. Install pm2 before starting the server."
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Run scripts/install.sh first."
  exit 1
fi

stop_legacy_pid_file_server

source .venv/bin/activate

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
LLM_HOST="${LLM_HOST:-127.0.0.1}"
LLM_PORT="${LLM_PORT:-8765}"
STORM_ZERO_SEED_FOUNDATION="false"
if [[ "$SEED_FOUNDATION" == "true" ]]; then
  STORM_ZERO_SEED_FOUNDATION="true"
fi
export PM2_APP LLM_HOST LLM_PORT STORM_ZERO_SEED_FOUNDATION

START_ARGS=("--host" "$LLM_HOST" "--port" "$LLM_PORT")
if [[ "$SEED_FOUNDATION" == "true" ]]; then
  START_ARGS+=("--seed-foundation")
fi

if pm2 describe "$PM2_APP" >/dev/null 2>&1; then
  echo "Restarting PM2 app $PM2_APP"
  export NODE_OPTIONS="--max-old-space-size=8192 ${NODE_OPTIONS:-}"
  pm2 restart "$PM2_APP" --update-env
else
  echo "Starting PM2 app $PM2_APP"
  export NODE_OPTIONS="--max-old-space-size=8192 ${NODE_OPTIONS:-}"
  pm2 start ecosystem.config.js
fi

if [[ "$MODE" != "background" ]]; then
  echo "PM2 keeps the server running in the background. Use 'pm2 logs $PM2_APP' to follow logs."
fi
