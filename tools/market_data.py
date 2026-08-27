from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings

logger = logging.getLogger(__name__)

_METRIC_COLUMNS = [
    "latest_price",
    "latest_volume",
    "change_pct",
    "avg_volume",
    "relative_volume",
    "volatility_pct",
    "is_breakout",
]


def get_client() -> StockHistoricalDataClient:
    settings = get_settings()
    return StockHistoricalDataClient(
        settings.alpaca_api_key.get_secret_value(),
        settings.alpaca_secret_key.get_secret_value(),
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def fetch_recent_bars(
    tickers: list[str],
    days: int = 90,
    client: StockHistoricalDataClient | None = None,
) -> pd.DataFrame:
    """Fetch daily bars for `tickers` over the trailing `days` calendar days in one
    batched Alpaca request. Returns alpaca-py's `.df` (MultiIndex: symbol, timestamp;
    columns include open/high/low/close/volume). Logs a warning (does not raise) for
    any requested ticker missing from the response — e.g. a bad symbol or one with no
    data in the window; downstream code must tolerate a result missing some tickers.
    """
    if not tickers:
        raise ValueError("tickers must not be empty")

    client = client or get_client()
    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Day),
        start=datetime.now(timezone.utc) - timedelta(days=days),
    )
    bars = client.get_stock_bars(request).df

    if bars is None or bars.empty:
        logger.warning("No bar data returned for any of: %s", tickers)
        return bars if bars is not None else pd.DataFrame()

    present = set(bars.index.get_level_values(0).unique())
    missing = set(tickers) - present
    if missing:
        logger.warning("No bar data returned for tickers: %s", sorted(missing))
    return bars


def compute_screening_metrics(
    bars_df: pd.DataFrame, breakout_lookback_days: int = 20
) -> pd.DataFrame:
    """Pure, deterministic, network-free. Given the MultiIndex (symbol, timestamp)
    `bars_df` from `fetch_recent_bars`, computes per-ticker metrics from the latest row:

      - latest_price / latest_volume: the most recent close / volume
      - change_pct: % change of close vs. the prior day
      - avg_volume: rolling mean volume over `breakout_lookback_days`, EXCLUDING the
        latest day (via shift(1) before rolling) so a ticker's own volume spike can't
        inflate its own baseline
      - relative_volume: latest_volume / avg_volume
      - volatility_pct: stdev of daily % returns over the lookback window, as a percent
      - is_breakout: latest_price >= 98% of the highest `high` over the
        `breakout_lookback_days` PRIOR to today (also shifted, for the same reason)

    Returns a flat DataFrame indexed by ticker, sorted by ticker. Any ticker with fewer
    than `breakout_lookback_days + 1` rows of history is dropped (insufficient data to
    compute rolling metrics) and logged as a warning.
    """
    min_rows = breakout_lookback_days + 1
    rows: list[dict] = []
    dropped: list[str] = []

    if not bars_df.empty:
        for ticker, group in bars_df.groupby(level=0):
            g = group.sort_index(level=1)
            if len(g) < min_rows:
                dropped.append(ticker)
                continue

            close = g["close"]
            high = g["high"]
            volume = g["volume"]

            latest_price = float(close.iloc[-1])
            latest_volume = int(volume.iloc[-1])
            change_pct = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)

            avg_volume = float(
                volume.shift(1).rolling(breakout_lookback_days).mean().iloc[-1]
            )
            relative_volume = float(latest_volume / avg_volume) if avg_volume > 0 else 0.0

            returns = close.pct_change()
            volatility_pct = float(
                returns.rolling(breakout_lookback_days).std().iloc[-1] * 100
            )

            prior_high = float(high.shift(1).rolling(breakout_lookback_days).max().iloc[-1])
            is_breakout = bool(latest_price >= 0.98 * prior_high) if prior_high > 0 else False

            rows.append(
                {
                    "ticker": ticker,
                    "latest_price": latest_price,
                    "latest_volume": latest_volume,
                    "change_pct": change_pct,
                    "avg_volume": avg_volume,
                    "relative_volume": relative_volume,
                    "volatility_pct": volatility_pct,
                    "is_breakout": is_breakout,
                }
            )

    if dropped:
        logger.warning(
            "Dropped tickers with insufficient history (<%d rows): %s", min_rows, sorted(dropped)
        )

    if not rows:
        empty = pd.DataFrame(columns=_METRIC_COLUMNS)
        empty.index.name = "ticker"
        return empty

    return pd.DataFrame(rows).set_index("ticker").sort_index()


def screen_universe(tickers: list[str], filters: dict, days: int = 90) -> pd.DataFrame:
    """Orchestrates fetch -> compute -> filter -> cheap deterministic pre-sort:

      - drop rows where latest_price < filters['min_price']
      - drop rows where avg_volume < filters['min_volume']
      - drop rows where relative_volume < filters['min_relative_volume']
      - pre-sort remaining candidates by relative_volume * abs(change_pct) descending

    This pre-sort is a cheap heuristic, not the final ranking — final ranking/selection
    is Claude's job (see agents/scanner.py), per the "heavy numeric work stays in
    deterministic Python, Claude interprets/ranks" design principle. Returns the
    filtered/sorted DataFrame, which may be empty if nothing passes the filters.
    """
    if not tickers:
        raise ValueError("tickers must not be empty")

    bars_df = fetch_recent_bars(tickers, days=days)
    metrics = compute_screening_metrics(bars_df, filters.get("breakout_lookback_days", 20))
    if metrics.empty:
        return metrics

    filtered = metrics[
        (metrics["latest_price"] >= filters.get("min_price", 0))
        & (metrics["avg_volume"] >= filters.get("min_volume", 0))
        & (metrics["relative_volume"] >= filters.get("min_relative_volume", 0))
    ]
    if filtered.empty:
        return filtered

    pre_rank = filtered["relative_volume"] * filtered["change_pct"].abs()
    return (
        filtered.assign(_pre_rank=pre_rank)
        .sort_values("_pre_rank", ascending=False)
        .drop(columns="_pre_rank")
    )
