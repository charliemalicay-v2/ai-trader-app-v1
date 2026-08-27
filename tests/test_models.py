from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from models.setup import ScannerOutput, ShortlistedSetup


def _make_setup(**overrides) -> dict:
    base = dict(
        ticker="nvda",
        scanned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        price=123.45,
        change_pct=2.5,
        volume=1_000_000,
        relative_volume=1.5,
        setup_type="breakout",
        rank_score=82.0,
        reason="Breaking out above 20-day high on 1.8x average volume",
    )
    base.update(overrides)
    return base


def test_shortlisted_setup_round_trip():
    setup = ShortlistedSetup(**_make_setup())
    restored = ShortlistedSetup.model_validate_json(setup.model_dump_json())
    assert restored == setup


def test_ticker_uppercased():
    setup = ShortlistedSetup(**_make_setup(ticker="nvda"))
    assert setup.ticker == "NVDA"


def test_rejects_negative_price():
    with pytest.raises(ValidationError):
        ShortlistedSetup(**_make_setup(price=-1))


def test_rejects_rank_score_out_of_range():
    with pytest.raises(ValidationError):
        ShortlistedSetup(**_make_setup(rank_score=150))


def test_scanner_output_wraps_list():
    setups = [ShortlistedSetup(**_make_setup()), ShortlistedSetup(**_make_setup(ticker="MSFT"))]
    output = ScannerOutput(setups=setups)
    assert output.setups == setups
