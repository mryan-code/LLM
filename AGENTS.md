# Storm Zero LLM Agent Notes

This repository contains the local Python LLM service for Storm Zero.

## Canonical Documentation

- `README.md` — server setup, HTTP API, environment variables, testing, deployment
- `flow.md` — step-by-step `/chat` and `/use-model` request flow
- `foundation.md` — product-direction notes for training and memory layering

Historical implementation prompts live in `prompt.md`. When docs disagree, prefer `README.md` and `flow.md`.

## Deployment Files

Deployment files for the target host are located here:

`smb://Matt's Mac mini._smb._tcp.local/inkt/Documents/Personal/storm_zero/llm`

## Runtime Summary

- Python package: `src/storm_zero_llm`
- HTTP server: `python -m storm_zero_llm`
- Process manager: PM2 via `scripts/start.sh`
- Models: GGUF via `llama-cpp-python`
- Training source: MySQL (`tblglobal_rule`, user tables, conversation tables)
- TTS: Kokoro assets from `TTS_PATH`
- Per-request debug flags: request body `env.GLOBAL_DEBUG_LEVEL` and `env.DEBUG_USER`

- always assume I am talking about the deployment
- always fix any tests that are related to the files modified
