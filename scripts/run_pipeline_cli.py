from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv


def main() -> None:
    # Populate os.environ (not just a Settings object) BEFORE any import that might
    # reach the Claude Agent SDK, which spawns a subprocess needing ANTHROPIC_API_KEY
    # via env inheritance. See config/settings.py for why this can't just rely on
    # pydantic-settings' own .env loading.
    load_dotenv()

    parser = argparse.ArgumentParser(prog="run_pipeline_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Run the Scanner agent against config/universe.yaml")
    scan.add_argument("--universe-config", default="config/universe.yaml")
    scan.add_argument("--top-n", type=int, default=None, help="override top_n from the config file")

    args = parser.parse_args()
    if args.command == "scan":
        asyncio.run(_run_scan(args))


async def _run_scan(args: argparse.Namespace) -> None:
    # Deferred import: keeps --help / argument-parsing errors fast and free of any
    # dependency on ANTHROPIC_API_KEY / network access.
    from agents.scanner import load_universe_config, run_scanner

    watchlist, filters = load_universe_config(args.universe_config)
    if args.top_n:
        filters["top_n"] = args.top_n

    setups = await run_scanner(watchlist, filters)
    _print_table(setups)


def _print_table(setups: list) -> None:
    if not setups:
        print("No setups found.")
        return
    ranked = sorted(setups, key=lambda s: s.rank_score, reverse=True)
    print(f"{'RANK':<5}{'TICKER':<8}{'SCORE':<7}{'TYPE':<14}{'PRICE':<10}{'CHG%':<8}{'RVOL':<7}REASON")
    for i, s in enumerate(ranked, start=1):
        print(
            f"{i:<5}{s.ticker:<8}{s.rank_score:<7.1f}{s.setup_type:<14}"
            f"{s.price:<10.2f}{s.change_pct:<8.2f}{s.relative_volume:<7.2f}{s.reason}"
        )


if __name__ == "__main__":
    main()
