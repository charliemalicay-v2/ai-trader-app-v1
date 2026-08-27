from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from agents.base import run_structured_agent
from models.setup import ScannerOutput, ShortlistedSetup
from tools.market_data import screen_universe

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "scanner_system.md"


def load_universe_config(path: str = "config/universe.yaml") -> tuple[list[str], dict]:
    """Reads the YAML file, returns (watchlist, filters)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data["watchlist"], data["filters"]


def _build_user_prompt(candidates_df: pd.DataFrame, top_n: int) -> str:
    records = (
        candidates_df.reset_index()
        .rename(columns={"latest_price": "price", "latest_volume": "volume"})
        .to_dict(orient="records")
    )
    candidates_json = json.dumps(records, default=str)
    return (
        f"Here are {len(records)} candidate tickers with pre-computed screening metrics. "
        f"Return the top {top_n} as JSON matching the required schema.\n\n{candidates_json}"
    )


async def run_scanner(
    universe: list[str] | None = None, filters: dict | None = None
) -> list[ShortlistedSetup]:
    if universe is None or filters is None:
        cfg_universe, cfg_filters = load_universe_config()
        universe = universe if universe is not None else cfg_universe
        filters = filters if filters is not None else cfg_filters

    scan_time = datetime.now(timezone.utc)
    candidates_df = screen_universe(universe, filters)
    if candidates_df.empty:
        return []

    top_n = filters.get("top_n", 10)
    # Send up to 3x top_n candidates so Claude has real choices to rank, not just
    # the pre-sorted top N handed straight through.
    candidates = candidates_df.head(top_n * 3)

    system_prompt = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    user_prompt = _build_user_prompt(candidates, top_n)

    result: ScannerOutput = await run_structured_agent(system_prompt, user_prompt, ScannerOutput)

    # LLMs are unreliable clocks — stamp the actual scan time ourselves rather than
    # trusting whatever the model wrote for scanned_at. Matters for later run-history/audit.
    for setup in result.setups:
        setup.scanned_at = scan_time
    return result.setups
