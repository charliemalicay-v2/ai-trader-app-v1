# DESIGN.md — AI Trading System (ai-trader-app-v1)

> Source: derived from 8 marketing-carousel screenshots in `artifacts/screenshot/` describing
> "I turned Claude Code into an AI Trader — one system, six trading agents." This document
> translates that concept into a concrete, buildable architecture.

## 0. Overview

A personal, human-in-the-loop AI trading assistant built as a **Python + Streamlit** app. Six
specialized **Claude Agent SDK** agents run a sequential pipeline — **Scanner → Researcher →
Technical Analyst → Quant Engine → Risk Manager → Trader** — that turns a universe of tickers
into a small number of vetted, structured **Final Trade Briefs**. **Alpaca** (`alpaca-py`)
supplies market data and, later, paper/live order execution. The user always makes the final
call — the Trader agent never auto-executes without explicit opt-in.

Guiding principle for a solo dev: **thin orchestration, fat data models, swappable tools**.
Each agent is a pure function of `(structured input) -> (structured output)` backed by a Claude
call with tool use; Streamlit is just a viewer/trigger over a pipeline that can also run
headless (CLI/cron) later.

---

## 1. High-Level Architecture

```
                     ┌─────────────────────────────────────────────────────────┐
                     │                     Streamlit Dashboard                   │
                     │   (triggers pipeline, renders briefs, config UI, confirm) │
                     └───────────────┬─────────────────────────▲────────────────┘
                                     │ run_pipeline()            │ reads
                                     ▼                            │
                     ┌─────────────────────────────────────────────────────────┐
                     │                      Orchestrator                        │
                     │        (pipeline.py — sequential, early-exit on reject)  │
                     └───────────────┬─────────────────────────────────────────┘
                                     │
   Universe of tickers               │
        │                            ▼
        │                 ┌───────────────────┐
        │                 │ 1. Scanner Agent   │  -> ShortlistedSetup[]
        │                 └─────────┬──────────┘
        │                           ▼
        │                 ┌───────────────────┐
        │                 │ 2. Researcher Agent│  -> ResearchBrief
        │                 └─────────┬──────────┘
        │                           ▼
        │                 ┌───────────────────┐
        │                 │ 3. Technical Agent │  -> TechnicalBrief
        │                 └─────────┬──────────┘
        │                           ▼
        │                 ┌───────────────────┐
        │                 │ 4. Quant Agent     │  -> BacktestResult
        │                 └─────────┬──────────┘
        │                           ▼
        │                 ┌───────────────────┐
        │                 │ 5. Risk Agent      │  -> RiskVerdict (PASS/REJECT) ──► early exit if REJECT
        │                 └─────────┬──────────┘
        │                           ▼
        │                 ┌───────────────────┐
        │                 │ 6. Trader Agent    │  -> FinalTradeBrief
        │                 └─────────┬──────────┘
        │                           ▼
        │                 (persisted to SQLite + surfaced in Streamlit)
        │                           │
        └── Alpaca Market Data ─────┴── Alpaca Trading API (paper by default; live opt-in, manual confirm)
```

**Data flow contract:** each agent's output Pydantic model is the next agent's input (plus the
original `ShortlistedSetup` for ticker context, which is threaded through the whole pipeline).
The **Risk Manager is a gate**: if `RiskVerdict.status == "REJECT"`, the orchestrator stops
before calling the Trader agent for that ticker — matching the slides' "no setup skips risk"
framing.

Streamlit does not call Claude directly — it only calls orchestrator functions and renders the
resulting Pydantic objects (already persisted). This keeps the UI layer thin and lets the same
pipeline run from a CLI/cron script for unattended scanning later.

---

## 2. Project Structure

```
ai-trader-app-v1/
├── README.md
├── DESIGN.md
├── pyproject.toml                 # deps: streamlit, claude-agent-sdk, alpaca-py, pandas,
│                                   #   pandas-ta, vectorbt (or backtesting.py), pydantic,
│                                   #   pydantic-settings, pyyaml, python-dotenv, sqlmodel,
│                                   #   tenacity, plotly, yfinance
├── .env.example                   # ANTHROPIC_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY,
│                                   #   ALPACA_PAPER=true, ALLOW_LIVE_TRADING=false
├── .streamlit/
│   └── config.toml
├── config/
│   ├── settings.py                 # pydantic Settings (env-driven)
│   ├── risk_rules.yaml             # user-editable "Your Rules" (max position size, etc.)
│   └── universe.yaml               # default ticker universe / watchlist / scan filters
│
├── models/                         # shared Pydantic data models — the "contracts" between agents
│   ├── __init__.py
│   ├── setup.py                    # ShortlistedSetup
│   ├── research.py                 # ResearchBrief, Financials, Sentiment
│   ├── technical.py                # TechnicalBrief, KeyLevels
│   ├── backtest.py                 # BacktestResult, StrategyRules
│   ├── risk.py                     # RiskVerdict, RiskCheck, RiskRuleConfig
│   └── trade.py                    # FinalTradeBrief
│
├── agents/                         # one module per Claude Agent SDK agent
│   ├── __init__.py
│   ├── base.py                     # shared agent runner: client, structured-output parsing, retries
│   ├── scanner.py                  # ScannerAgent
│   ├── researcher.py               # ResearcherAgent
│   ├── technical.py                # TechnicalAgent
│   ├── quant.py                    # QuantAgent
│   ├── risk.py                     # RiskAgent
│   ├── trader.py                   # TraderAgent
│   └── prompts/
│       ├── scanner_system.md
│       ├── researcher_system.md
│       ├── technical_system.md
│       ├── quant_system.md
│       ├── risk_system.md
│       └── trader_system.md
│
├── tools/                          # tool implementations bound to Claude Agent SDK agents
│   ├── __init__.py
│   ├── market_data.py               # Alpaca bars/quotes/snapshot wrappers
│   ├── news.py                      # Alpaca News API wrapper
│   ├── fundamentals.py              # yfinance-backed financials/earnings lookups
│   ├── ta_indicators.py             # pandas-ta wrappers (RSI, MACD, support/resistance, trend)
│   ├── backtester.py                # vectorbt/backtesting.py engine wrapper
│   └── risk_rules_engine.py         # deterministic rule evaluation, called by RiskAgent
│
├── orchestration/
│   ├── __init__.py
│   ├── pipeline.py                  # run_pipeline(), run_for_ticker(), early-exit logic
│   └── scheduler.py                 # optional: cron/loop entrypoint for unattended scans (Phase 6)
│
├── execution/
│   ├── __init__.py
│   ├── alpaca_client.py             # thin wrapper: TradingClient, paper/live switch
│   └── order_manager.py             # place_order(), guarded by explicit user confirmation
│
├── storage/
│   ├── __init__.py
│   ├── db.py                        # SQLite engine + session
│   └── repository.py                # CRUD for setups/briefs/trades (persist pipeline runs)
│
├── dashboard/
│   ├── Home.py                      # Streamlit entrypoint (multipage app)
│   └── pages/
│       ├── 1_Scanner.py             # live scanner results table
│       ├── 2_Ticker_Drilldown.py    # per-ticker: research/technical/quant/risk tabs
│       ├── 3_Final_Trade_Brief.py   # FinalTradeBrief + "you decide" confirm/dismiss
│       ├── 4_Backtest_Lab.py        # run/inspect backtests standalone
│       └── 5_Risk_Rules.py          # edit risk_rules.yaml via form
│
├── scripts/
│   ├── run_pipeline_cli.py          # headless run: python scripts/run_pipeline_cli.py --tickers NVDA,TSLA
│   └── init_db.py
│
└── tests/
    ├── test_models.py
    ├── test_pipeline.py
    ├── test_risk_rules_engine.py
    └── fixtures/
```

---

## 3. Per-Agent Design

All six agents follow the same construction pattern via `agents/base.py`: a Claude Agent SDK
agent is defined with a **system prompt** (role + strict output contract), a set of **bound
tools** (Python functions registered per the SDK's tool interface), and a **structured output**
requirement — the agent's final message must validate against the corresponding Pydantic model
(validate-and-retry loop, up to ~2 retries on `ValidationError`).

**Key design principle:** heavy numeric work (screening thousands of tickers, computing
indicators, running backtests, evaluating risk rules) runs in deterministic Python. Claude's job
is to interpret, rank, and synthesize already-computed data — not to crunch numbers itself. This
keeps results reproducible and auditable, which matters for anything touching real money.

### 3.1 Scanner Agent (`agents/scanner.py`)
- **Responsibility:** screen a broad universe of tickers for the best current setups (volume,
  momentum, volatility, breakouts), rank them, return a shortlist.
- **Input:** `universe: list[str]` (from `config/universe.yaml`) + scan filter thresholds.
- **Output:** `list[ShortlistedSetup]`.
- **Tools:** `tools/market_data.py` — Alpaca `get_snapshots` / `get_bars` for the universe; a
  local Python screening function computes % change, relative volume, ATR/volatility, and
  breakout flags *before* handing a pre-filtered candidate list to Claude.
- **System prompt role:** "You are a market scanner. Given pre-computed screening metrics for
  candidate tickers, rank them by setup quality and return the top N as structured JSON, each
  with a one-line reason."
- **Note:** scanning the full market (12,000+ tickers) on every run is impractical for a solo
  dev's API budget. Default `universe.yaml` should be a curated liquid-equity watchlist (e.g.
  S&P 500 / Nasdaq 100 constituents), with an optional "full market" mode gated behind Alpaca's
  asset list endpoint for later.

### 3.2 Researcher Agent (`agents/researcher.py`)
- **Responsibility:** given a `ShortlistedSetup`, gather news, financials, earnings timing,
  sentiment, and catalysts; synthesize a `ResearchBrief` with a confidence score.
- **Input:** `ShortlistedSetup`.
- **Output:** `ResearchBrief`.
- **Tools:** `tools/news.py` (Alpaca News API), `tools/fundamentals.py` (yfinance-backed
  financials/earnings).
- **System prompt role:** "You are a fundamental/news researcher. Use the provided tools to
  gather recent news, financials, sentiment and catalysts for the ticker, then produce a
  structured research brief with a checklist and a 0-100 confidence score."

### 3.3 Technical Analyst Agent (`agents/technical.py`)
- **Responsibility:** analyze price action — trend, structure, momentum, volume, key
  support/resistance levels.
- **Input:** `ShortlistedSetup`.
- **Output:** `TechnicalBrief`.
- **Tools:** `tools/market_data.py` (Alpaca daily bars, ~6-12 months), `tools/ta_indicators.py`
  — pandas-ta computed RSI(14), MACD(12,26,9), moving averages, ATR, and a simple
  local-extrema-based support/resistance finder. Indicator math runs in Python; Claude
  interprets the computed indicators into a bias/brief.
- **System prompt role:** "You are a technical analyst. Given computed indicators and recent
  price structure, determine trend, bias, momentum strength, key levels, and volume trend;
  return structured JSON. Status is COMPLETE only once all fields are populated."

### 3.4 Quant Engine Agent (`agents/quant.py`)
- **Responsibility:** turn a trading idea (seeded from the technical setup) into explicit
  strategy rules, backtest them against historical data, and report performance stats.
- **Input:** `ShortlistedSetup`, `TechnicalBrief`.
- **Output:** `BacktestResult` (embeds the `StrategyRules` used).
- **Tools:** `tools/backtester.py`; `tools/market_data.py` for historical bars.
- **System prompt role:** "You are a quant strategist. Given a technical setup, define explicit
  entry/exit/risk rules as structured `StrategyRules`, call the backtest tool, then summarize
  the results (win rate, drawdown, expectancy, profit factor, sample size) as a
  `BacktestResult`, always including the historical-performance disclaimer."
- **Design note:** the LLM *chooses and parameterizes* rules (e.g. "breakout above 20-day high
  with volume > 1.5x avg, stop at prior swing low"); the actual simulation loop is deterministic
  Python in `tools/backtester.py`.
- **Known limitation:** Alpaca's free IEX feed does not guarantee "10+ years" of daily history
  for every symbol the way the marketing slide implies — `BacktestResult.lookback_years`
  reports whatever history was actually available, rather than promising a fixed depth.

### 3.5 Risk Manager Agent (`agents/risk.py`)
- **Responsibility:** evaluate the setup against volatility, exposure, liquidity, downside, and
  user-configured rules; PASS or REJECT.
- **Input:** `ShortlistedSetup`, `ResearchBrief`, `TechnicalBrief`, `BacktestResult`, current
  portfolio state (positions/exposure from Alpaca account), `RiskRuleConfig`.
- **Output:** `RiskVerdict`.
- **Tools:** `tools/risk_rules_engine.py` — a **deterministic, non-LLM** rule evaluator (max
  position size %, max portfolio exposure, max correlated exposure, min liquidity/avg volume,
  max drawdown tolerance, blacklist/whitelist). The agent calls this tool and explains the
  resulting risk score in natural language; it **cannot override** the tool's PASS/REJECT
  verdict.
- **System prompt role:** "You are a risk manager. You must call the risk rule evaluation tool
  and cannot override its PASS/REJECT verdict — your job is to explain the resulting risk score
  and which rules were checked."
- **This is the pipeline's early-exit gate** — see §5.

### 3.6 Trader Agent (`agents/trader.py`)
- **Responsibility:** synthesize all upstream briefs into one `FinalTradeBrief` with entry zone,
  target, stop loss, R:R, timeframe, and an overall confidence score. Does **not** place orders.
- **Input:** `ShortlistedSetup`, `ResearchBrief`, `TechnicalBrief`, `BacktestResult`,
  `RiskVerdict` (must be PASS to reach this agent).
- **Output:** `FinalTradeBrief`.
- **Tools:** none required (pure synthesis); optionally a final live-price sanity check via
  `tools/market_data.py`.
- **System prompt role:** "You are the decision engine. Combine research, technical, quant, and
  risk outputs into one final trade brief with a concrete entry zone, target, stop loss, and
  risk/reward. You do not place trades — the human makes the final decision."
- **Human-in-the-loop:** `FinalTradeBrief` is rendered in Streamlit with explicit **Confirm &
  Submit Paper Order** / **Dismiss** actions. Order placement is a separate, explicit call into
  `execution/order_manager.py`, never automatic.

---

## 4. Data Models (`models/`)

All models are Pydantic `BaseModel`s so they validate agent output, serialize to JSON for
storage, and render directly in Streamlit.

```python
# models/setup.py
class ShortlistedSetup(BaseModel):
    ticker: str
    scanned_at: datetime
    price: float
    change_pct: float
    volume: int
    relative_volume: float
    setup_type: str            # e.g. "breakout", "momentum"
    rank_score: float          # 0-100
    reason: str                # one-line why it was shortlisted


# models/research.py
class Financials(BaseModel):
    revenue_yoy_pct: float | None
    eps_yoy_pct: float | None
    gross_margin_pct: float | None

class Sentiment(BaseModel):
    bullish_pct: float
    bearish_pct: float

class ResearchBrief(BaseModel):
    ticker: str
    headline: str
    financials: Financials
    days_to_earnings: int | None
    sentiment: Sentiment
    catalysts: list[str]
    checklist: list[str]
    verdict: str                # e.g. "STRONG SETUP"
    confidence_score: int       # 0-100
    generated_at: datetime


# models/technical.py
class KeyLevels(BaseModel):
    support: float
    resistance: float

class TechnicalBrief(BaseModel):
    ticker: str
    bias: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    trend: Literal["UPTREND", "DOWNTREND", "SIDEWAYS"]
    momentum: Literal["STRONG", "MODERATE", "WEAK"]
    key_levels: KeyLevels
    volume_trend: Literal["INCREASING", "DECREASING", "FLAT"]
    rsi_14: float
    macd_signal: str
    status: Literal["COMPLETE", "INCOMPLETE"]
    generated_at: datetime


# models/backtest.py
class StrategyRules(BaseModel):
    idea: str
    entry_conditions: list[str]
    exit_conditions: list[str]
    risk_management: list[str]

class BacktestResult(BaseModel):
    ticker: str
    strategy: StrategyRules
    win_rate_pct: float
    max_drawdown_pct: float
    expectancy_r: float
    profit_factor: float
    sample_size: int
    lookback_years: float
    disclaimer: str = "Historical results don't guarantee future performance."
    generated_at: datetime


# models/risk.py
class RiskRuleConfig(BaseModel):
    max_position_pct: float
    max_portfolio_exposure_pct: float
    max_drawdown_tolerance_pct: float
    min_avg_volume: int
    max_correlated_positions: int
    blacklist: list[str] = []

class RiskCheck(BaseModel):
    name: str                   # "volatility", "exposure", "liquidity", "downside", "custom_rules"
    passed: bool
    detail: str

class RiskVerdict(BaseModel):
    ticker: str
    checks: list[RiskCheck]
    risk_score: int             # 0-100, lower = safer
    status: Literal["PASS", "REJECT"]
    generated_at: datetime


# models/trade.py
class FinalTradeBrief(BaseModel):
    ticker: str
    research_status: Literal["COMPLETE"]
    technical_bias: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    quant_status: Literal["PASSED", "FAILED"]
    risk_status: Literal["APPROVED", "REJECTED"]
    confidence_score: int       # 0-100
    entry_zone_low: float
    entry_zone_high: float
    target: float
    stop_loss: float
    risk_reward_ratio: float
    timeframe: Literal["SCALP", "DAY", "SWING", "POSITION"]
    generated_at: datetime
    # linkage back to upstream briefs, for the drilldown UI
    research_brief: ResearchBrief
    technical_brief: TechnicalBrief
    backtest_result: BacktestResult
    risk_verdict: RiskVerdict
```

---

## 5. Orchestration Layer (`orchestration/pipeline.py`)

**Sequential pipeline per ticker**, run for each `ShortlistedSetup` returned by the Scanner:

1. `scanner_agent.run(universe) -> list[ShortlistedSetup]`
2. For each setup (capped to top N, e.g. top 5-10 by `rank_score`, to control Claude API cost):
   - `research_brief = researcher_agent.run(setup)`
   - `technical_brief = technical_agent.run(setup)`
   - `backtest_result = quant_agent.run(setup, technical_brief)`
   - `risk_verdict = risk_agent.run(setup, research_brief, technical_brief, backtest_result, risk_rules)`
   - if `risk_verdict.status == "REJECT"`: persist and skip to the next ticker (no Trader call)
   - else: `final_brief = trader_agent.run(setup, research_brief, technical_brief, backtest_result, risk_verdict)`
3. Persist every intermediate artifact (not just the final brief) via `storage/repository.py`,
   keyed by `(ticker, run_id, timestamp)` — this is what powers the per-agent drilldown tabs in
   Streamlit even for rejected setups.

**Claude Agent SDK coordination model:** a single lightweight Python orchestrator function (not
a meta-agent) calls each subagent in sequence and passes typed Pydantic objects between them.
This is deliberately **not** "one big autonomous agent deciding when to call the others" — for a
trading system, deterministic sequencing plus an explicit risk gate is safer and easier to debug
than letting an LLM control the flow. Each `agents/*.py` module exposes a
`run(...) -> PydanticModel` function wrapping a single Claude Agent SDK agent invocation (its own
system prompt + bound tools) and validates the structured output before returning.

**Concurrency (later optimization):** Researcher and Technical Analyst calls for a given ticker
are independent (both only need `ShortlistedSetup`) and could run concurrently via
`asyncio.gather`; Quant depends on Technical's output so must run after. Keep it synchronous
through Phase 4 to keep debugging simple.

**Idempotency/run IDs:** each pipeline invocation gets a `run_id` (UUID) so the dashboard can
show history across scans, not just the latest.

---

## 6. Streamlit Dashboard (`dashboard/`)

Multipage Streamlit app (`Home.py` + `pages/`):

1. **Home.py** — system status header (mirrors the marketing slides' terminal: agent status,
   last scan time), a "Run Scan" button that triggers `orchestration.pipeline.run_pipeline()`,
   and a summary table of the latest shortlisted setups with rank score.
2. **1_Scanner.py** — full scanner results table (`ShortlistedSetup` list), sortable/filterable
   by setup type, rank score, volume; click-through to drilldown.
3. **2_Ticker_Drilldown.py** — tabs per ticker: Research / Technical (plotly candlestick chart +
   RSI/MACD subplots) / Quant (backtest stats + equity curve) / Risk (checks table + PASS/REJECT
   badge). Reads from `storage/repository.py`, not live agent calls, so it's fast to browse.
4. **3_Final_Trade_Brief.py** — renders `FinalTradeBrief` (entry zone, target, stop, R:R,
   confidence) with a prominent **"YOU MAKE THE FINAL DECISION"** banner and two explicit
   actions: **Confirm → Submit Paper Order** (calls `execution/order_manager.py` with
   `paper=True` by default) and **Dismiss**. A separate, clearly-labeled toggle gated by
   `ALLOW_LIVE_TRADING` (default `false`) is required before any live-account order path is even
   selectable.
5. **4_Backtest_Lab.py** — standalone: pick a ticker + manually define/edit `StrategyRules`, run
   `tools/backtester.py` directly, inspect results — useful for iterating on strategies outside
   the full agent pipeline.
6. **5_Risk_Rules.py** — form-based editor for `risk_rules.yaml` (`RiskRuleConfig` fields), with
   validation and a "test against last N setups" preview.

**State/session:** Streamlit session state holds the current `run_id`; all pipeline runs and
agent outputs are persisted to SQLite so the dashboard survives restarts and can show run
history (not just an in-memory cache).

---

## 7. External Integrations

| Concern | Choice | Justification |
|---|---|---|
| Market data (bars/quotes/snapshots) | **Alpaca Market Data API** via `alpaca-py` | Already the confirmed execution provider; reusing it for data avoids a second data vendor/auth setup. Free IEX feed is sufficient for daily-bar screening and TA. |
| News / catalysts | **Alpaca News API** (`alpaca-py`'s `NewsClient`) | Same SDK/auth as everything else, free with an Alpaca account — no second API key to manage. |
| Fundamentals/earnings | **`yfinance`** (free, no key) behind `tools/fundamentals.py` | Alpaca doesn't expose fundamentals/earnings-calendar data; yfinance is free and zero-setup. Isolated behind an interface so it can be swapped for a paid provider (e.g. Financial Modeling Prep, Polygon) later without touching agent code. |
| Technical analysis | **`pandas-ta`** | Pure Python, pandas-native, simpler to install than TA-Lib (no C dependency headaches on Windows), covers RSI/MACD/ATR/moving averages. |
| Backtesting | **`vectorbt`** (or `backtesting.py` as a lighter fallback) | `vectorbt` is fast (vectorized) for iterating on rule-based strategies and integrates cleanly with pandas OHLCV from Alpaca. `tools/backtester.py` exposes a narrow interface — `run_backtest(bars, rules) -> BacktestResult` — so the library choice is swappable if `vectorbt`'s learning curve proves too heavy. |
| Trading execution | **Alpaca Trading API** (`alpaca-py` `TradingClient`) | Confirmed choice; paper trading (`ALPACA_PAPER=true`) is the default and only mode enabled until the user explicitly flips a live-trading config flag. |
| LLM/agents | **Claude Agent SDK** (Anthropic) | Confirmed choice; matches the "Claude Code" branding in the source material. |

---

## 8. Config & Secrets

- **`.env`** (gitignored, `.env.example` committed): `ANTHROPIC_API_KEY`, `ALPACA_API_KEY`,
  `ALPACA_SECRET_KEY`, `ALPACA_PAPER=true`, `ALLOW_LIVE_TRADING=false`.
- **`config/settings.py`** — a pydantic `BaseSettings` class loading from `.env`, validated at
  startup (fail fast if keys are missing when a real agent/data call is attempted; the dashboard
  can still load in a degraded "no keys configured yet" state for first-run UX).
- **`config/risk_rules.yaml`** — the user-editable "Your Rules" from the Risk Manager slide: max
  position size %, max portfolio exposure, max drawdown tolerance, min liquidity, correlated-
  position caps, ticker blacklist. Loaded into `RiskRuleConfig`, editable via the Streamlit Risk
  Rules page, which writes back to this YAML (not the DB) so it stays version-controllable/diffable.
- **`config/universe.yaml`** — default scan universe (curated watchlist) and scan filter
  thresholds (min volume, min price, momentum threshold).
- **Streamlit secrets** — `.streamlit/secrets.toml` is already gitignored, so it can optionally
  mirror `.env` for a future Streamlit Cloud deployment; local dev uses `.env` +
  `python-dotenv` as the single source of truth to avoid duplicated secret files.
- **Live trading safety** — `ALLOW_LIVE_TRADING` must be `true` in env **and** the user must
  check an explicit confirmation box in the dashboard before `execution/order_manager.py` will
  accept `paper=False`. Every documented workflow defaults to paper trading.

---

## 9. Build Phases / Roadmap

**Phase 1 — Data layer + Scanner**
Project skeleton, `pyproject.toml`, `config/settings.py`, Alpaca client wrapper
(`tools/market_data.py`), `models/setup.py`, and the Scanner agent (Python pre-filtering +
Claude ranking). Verify against a small hardcoded watchlist. CLI smoke test.

**Phase 2 — Researcher + Technical Analyst**
Add `tools/news.py`, `tools/fundamentals.py`, `models/research.py`, Researcher agent. Add
`tools/ta_indicators.py`, `models/technical.py`, Technical agent. Both consume
`ShortlistedSetup` independently — good phase to validate the base agent runner
(`agents/base.py`) and structured-output validation/retry pattern.

**Phase 3 — Quant backtesting**
Build `tools/backtester.py` around the chosen library, `models/backtest.py`, Quant agent.
Validate against one known historical setup manually before trusting agent-generated rules.

**Phase 4 — Risk + Trader decision engine**
Build `tools/risk_rules_engine.py` (deterministic), `config/risk_rules.yaml`, `models/risk.py`,
Risk agent. Build Trader agent + `models/trade.py`. Wire `orchestration/pipeline.py` end-to-end
with the early-exit gate. First phase where the full pipeline runs ticker-to-brief.

**Phase 5 — Streamlit dashboard**
Add `storage/db.py` + `repository.py` for persistence, then build dashboard pages in order: Home
→ Scanner → Ticker Drilldown → Final Trade Brief (confirm/dismiss, still no order placement) →
Backtest Lab → Risk Rules editor.

**Phase 6 — Paper trading loop / scheduling**
Add `execution/alpaca_client.py` + `order_manager.py` (paper-only initially), wire the "Confirm
& Submit Paper Order" button. Add `orchestration/scheduler.py` for unattended periodic scans
(e.g. APScheduler or a cron-invoked script) if desired. The live-trading opt-in path
(`ALLOW_LIVE_TRADING`) is the last thing enabled, after paper-trading behavior has been observed
over time.

---

## 10. Critical Files to Start With

- `models/*.py` — shared Pydantic contracts every agent and the dashboard depend on; define
  these first.
- `agents/base.py` — the common Claude Agent SDK runner (system prompt + tools +
  structured-output validation) reused by all six agents.
- `orchestration/pipeline.py` — the sequential pipeline with the Risk-gate early exit; the core
  control flow of the whole system.
- `tools/risk_rules_engine.py` — the deterministic rule evaluator the Risk agent must defer to
  for PASS/REJECT.
- `execution/order_manager.py` — the human-in-the-loop gate between `FinalTradeBrief` and any
  real order placement (paper-default, live opt-in).
