"""CLI entrypoint for Storm Zero LLM."""

from __future__ import annotations

import argparse
import os
import sys

from storm_zero_llm.agent import StormZeroAgent
from storm_zero_llm.config import StormZeroConfig
from storm_zero_llm.server import run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Storm Zero LLM local server")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--seed-foundation", action="store_true")
    parser.add_argument("--project-root", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = StormZeroConfig.load(project_root=args.project_root)
    host = args.host or config.llm_host
    port = args.port or config.llm_port

    agent = StormZeroAgent(config)
    seed_from_env = os.environ.get("STORM_ZERO_SEED_FOUNDATION", "false").lower() in {"1", "true", "yes", "on"}
    # if args.seed_foundation or seed_from_env:
    #     _sync_global_training(agent)

    try:
        run_server(agent, host=host, port=port)
    finally:
        agent.close()


def _sync_global_training(agent: StormZeroAgent) -> None:
    try:
        count = agent.sync_global_training()
    except Exception as exc:
        print(f"Warning: failed to sync global training from MySQL: {exc}", file=sys.stderr)
        return
    print(f"Synced {count} global training records.", file=sys.stderr)


if __name__ == "__main__":
    main()
