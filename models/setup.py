from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class ShortlistedSetup(BaseModel):
    ticker: str
    scanned_at: datetime
    price: float = Field(gt=0)
    change_pct: float
    volume: int = Field(ge=0)
    relative_volume: float = Field(ge=0)
    setup_type: str
    rank_score: float = Field(ge=0, le=100)
    reason: str

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, v: str) -> str:
        return v.strip().upper()


class ScannerOutput(BaseModel):
    """Wrapper so the structured-output JSON schema has a single top-level object
    (Claude Agent SDK's output_format needs one schema, not a bare list)."""

    setups: list[ShortlistedSetup]
