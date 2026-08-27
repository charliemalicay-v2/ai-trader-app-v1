from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import tools.market_data as market_data
from tools.market_data import compute_screening_metrics, screen_universe


def _ticker_frame(ticker, closes, highs, lows, volumes) -> pd.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    n = len(closes)
    idx = pd.MultiIndex.from_arrays(
        [[ticker] * n, [start + timedelta(days=i) for i in range(n)]],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _make_bars_df() -> pd.DataFrame:
    # AAA: 20 flat days then a breakout on price + relative volume (21 rows total,
    # exactly meeting the breakout_lookback_days=20 -> min_rows=21 threshold).
    aaa = _ticker_frame(
        "AAA",
        closes=[100.0] * 20 + [110.0],
        highs=[101.0] * 20 + [111.0],
        lows=[99.0] * 20 + [109.0],
        volumes=[1_000_000] * 20 + [3_000_000],
    )

    # BBB: flat throughout, no breakout, relative_volume ~1.0. High is kept well above
    # close (not just ~1%) so latest_price >= 0.98 * prior_high is unambiguously False.
    bbb = _ticker_frame(
        "BBB",
        closes=[50.0] * 21,
        highs=[55.0] * 21,
        lows=[45.0] * 21,
        volumes=[1_000_000] * 21,
    )

    # SHORT: insufficient history (5 rows < min_rows=21) -> must be dropped.
    short = _ticker_frame(
        "SHORT",
        closes=[10.0] * 5,
        highs=[10.5] * 5,
        lows=[9.5] * 5,
        volumes=[500_000] * 5,
    )

    return pd.concat([aaa, bbb, short])


def test_compute_screening_metrics_drops_insufficient_history():
    metrics = compute_screening_metrics(_make_bars_df(), breakout_lookback_days=20)
    assert set(metrics.index) == {"AAA", "BBB"}


def test_compute_screening_metrics_breakout_and_relative_volume():
    metrics = compute_screening_metrics(_make_bars_df(), breakout_lookback_days=20)

    aaa = metrics.loc["AAA"]
    assert aaa["latest_price"] == pytest.approx(110.0)
    assert aaa["change_pct"] == pytest.approx(10.0)
    # avg_volume/prior-high exclude the latest (breakout) day via shift(1), so they
    # reflect only the 20 flat days before it.
    assert aaa["avg_volume"] == pytest.approx(1_000_000.0)
    assert aaa["relative_volume"] == pytest.approx(3.0)
    assert bool(aaa["is_breakout"]) is True


def test_compute_screening_metrics_flat_ticker_no_breakout():
    metrics = compute_screening_metrics(_make_bars_df(), breakout_lookback_days=20)

    bbb = metrics.loc["BBB"]
    assert bbb["change_pct"] == pytest.approx(0.0)
    assert bbb["relative_volume"] == pytest.approx(1.0)
    assert bool(bbb["is_breakout"]) is False


def test_compute_screening_metrics_empty_input():
    empty = pd.DataFrame()
    metrics = compute_screening_metrics(empty, breakout_lookback_days=20)
    assert metrics.empty
    assert metrics.index.name == "ticker"


def test_screen_universe_filters_and_ranks(monkeypatch):
    bars = _make_bars_df()
    monkeypatch.setattr(market_data, "fetch_recent_bars", lambda tickers, days=90, client=None: bars)

    filters = {
        "min_price": 5.0,
        "min_volume": 500_000,
        "min_relative_volume": 1.2,
        "breakout_lookback_days": 20,
    }
    result = screen_universe(["AAA", "BBB", "SHORT"], filters)

    # BBB filtered out (relative_volume 1.0 < 1.2 threshold); SHORT was already
    # dropped upstream in compute_screening_metrics for insufficient history.
    assert list(result.index) == ["AAA"]


def test_screen_universe_raises_on_empty_tickers():
    with pytest.raises(ValueError):
        screen_universe([], {})
