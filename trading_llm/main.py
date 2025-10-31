"""Entry point for the Lean 4-Agent LLM trading system."""
from __future__ import annotations

import argparse
import json
from typing import List

from .core import config
from .core.orchestrator import run_trading_cycle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Lean 4-Agent trading pipeline")
    parser.add_argument(
        "symbols",
        nargs="*",
        default=["RELIANCE.NS"],
        help="Symbols to analyze",
    )
    parser.add_argument(
        "--timeframes",
        default=",".join(config.TIMEFRAMES),
        help="Comma-separated list of timeframes to evaluate",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Do not append to trade_log.jsonl",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tfs: List[str] = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()] or config.TIMEFRAMES

    for symbol in args.symbols:
        result = run_trading_cycle(symbol, timeframes=tfs, save_log=not args.no_log)
        if args.pretty:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
