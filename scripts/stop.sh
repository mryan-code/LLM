#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEGACY_PID_FILE="$ROOT_DIR/.storm-zero-llm.pid"

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
  else
    echo "Removing stale legacy PID file."
  fi

  rm -f "$LEGACY_PID_FILE"
}

PM2_APP="${PM2_APP:-$(read_env_value PM2_APP || true)}"

if ! command -v pm2 >/dev/null 2>&1; then
  echo "pm2 is not installed; skipping PM2 stop."
  stop_legacy_pid_file_server
  exit 0
fi

if [[ -z "${PM2_APP// }" ]]; then
  echo "Missing PM2_APP. Add PM2_APP to .env or export it before stopping."
  exit 1
fi

if pm2 describe "$PM2_APP" >/dev/null 2>&1; then
  pm2 stop "$PM2_APP"
  echo "Stopped PM2 app $PM2_APP"
else
  echo "PM2 app $PM2_APP is not running."
fi

stop_legacy_pid_file_server
