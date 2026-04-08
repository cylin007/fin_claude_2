# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Taiwan semiconductor stock quantitative trading & backtesting system. Implements a momentum-based strategy with dynamic entry/exit logic, market state detection (bull/neutral/weak/bear with panic detection), multi-stage position management, and full backtesting engine (2022-present) with real Taiwan broker commission/tax calculations.

**Language:** Pure Python (no build system). Run scripts directly with `python3`.

## Common Commands

```bash
# Daily trading operations
python3 run_daily_trading.py                    # Today's signal (default mode)
python3 run_daily_trading.py signal [date]      # Signal for specific date
python3 run_daily_trading.py status             # Current position status
python3 run_daily_trading.py add                # Interactive trade recording
python3 run_daily_trading.py history            # Trade history
python3 run_daily_trading.py verify [start] [end]  # Backtest alignment check
python3 run_daily_trading.py reset              # Reset engine state

# Full backtest with equity curve + 0050 benchmark
python3 run_full_backtest.py

# Ablation scripts (each tests a specific parameter dimension across 7 market regimes)
python3 run_7030_backtest.py          # 7030 ETF benchmark comparison
python3 run_direction_ablation.py     # Market filter modes (strict/moderate/relaxed)
python3 run_ewt_ablation.py           # EWT overnight sentiment filter
python3 run_mdd_ablation.py           # Max drawdown parameters
python3 run_pullback_ablation.py      # Strategy B (pullback) vs A (momentum)
python3 run_rsi_ablation.py           # RSI indicator optimization
python3 run_trailing_ablation.py      # Profit trailing stop optimization
python3 run_conv_regime_ablation.py   # Convergence/regime testing
python3 run_peer_ablation.py          # Peer group Z-score relative valuation
python3 run_round4_ablation.py        # Round 4: exit/risk + stock selection refinement

# Strategy C (midterm trend-following) backtests
python3 run_midterm_backtest.py       # Strategy C vs A comparison
python3 run_midterm_backtest_r2.py    # Strategy C round 2 refinement
python3 run_midterm_backtest_r3.py    # Strategy C round 3 refinement
```

## Dependencies

`yfinance`, `pandas`, `numpy`, `twstock` (optional), `requests`, `anthropic` (for signal filtering). API key stored in `.env` (`ANTHROPIC_API_KEY`). Data is cached as MD5-keyed pickles in `.cache/`.

## Architecture

### Core Modules

- **`strategy.py`** — Strategy logic engine. Defines `check_strategy_signal()` (buy B1-B7, sell S1-S4), `calculate_net_pnl()`, `calculate_fee()`, `calculate_tax()`, and `DEFAULT_CONFIG` with 40+ tunable parameters. This is the canonical source — `group_backtest.py` imports and re-exports these functions.
- **`group_backtest.py`** (~8K+ lines) — Monolithic backtest/simulation engine. `run_group_backtest()` is the main entry point. Also contains `INDUSTRY_CONFIGS` / `INDUSTRY_CONFIGS_V5` with per-industry parameter overrides, `PRESET_PERIODS` (7 market regimes), `reconstruct_market_history()` for market state pre-computation, and multiple operational modes (scan, verify, daily-vs-backtest comparison).
- **`run_daily_trading.py`** — Live daily execution via `DailyEngine` class. `step_day()` mirrors backtest logic exactly. State persisted in `daily_engine_state.json`. Imports strategy functions from `group_backtest.py` (which re-exports from `strategy.py`).
- **`run_full_backtest.py`** — Full backtest runner with equity curve plotting and 0050 DCA/signal-mirror benchmarks.
- **`stock_utils.py`** — Data fetching (`get_stock_data()`, `batch_download_stocks()`), market analysis (`get_market_status()`), `build_info_dict()` for per-stock indicator computation, and MD5-keyed pickle cache in `.cache/`.
- **`theme_config.py`** — Sub-theme momentum boost system. Maps semiconductor stocks to sub-themes (AI_HPC設計, CoWoS封裝, etc.) via `THEME_MAP`. Calculates theme heat scores to boost buy-signal ranking for hot sub-themes.
- **`strategy_pullback.py`** — Alternative "Strategy B" focused on buying pullbacks (vs momentum-chasing Strategy A). Complete signal function with its own `PULLBACK_CONFIG`. Tested via `run_pullback_ablation.py`.
- **`strategy_midterm.py`** — "Strategy C" midterm trend-following. Buys pullbacks to MA20 support in uptrends, holds longer (90-day zombie vs 15-day), concentrated 4-position portfolio at 15% NAV each. Own `MIDTERM_CONFIG`. Tested via `run_midterm_backtest.py` / `run_midterm_backtest_r2.py` / `run_midterm_backtest_r3.py`.
- **`industry_manager.py` / `industry_lookup.py`** — Stock list management for 上市/上櫃 markets.

### Import Chain

```
strategy.py (defines check_strategy_signal, calculate_*)
  ↑ imported by
group_backtest.py (re-exports + adds backtest engine, INDUSTRY_CONFIGS)
  ↑ imported by
run_daily_trading.py (DailyEngine, uses re-exported functions)
run_full_backtest.py (orchestrator)
```

When modifying strategy logic, edit `strategy.py` — the functions propagate through the import chain. When modifying backtest behavior (position management, daily loop), edit `group_backtest.py` and mirror changes in `run_daily_trading.py`.

### Strategy Signal Logic

**Buy conditions (B1-B7):** Trend confirmation (MA20>MA60), volume surge, price above prev close, bias limit (market-context-aware), fish tail momentum (5-day consecutive signals), breakout (10-day high), shooting star filter.

**Sell conditions (S1-S4):** Tiered stop-profit (2 tiers at 45%/10% net), break below MA60, hard stop-loss (-15% net, -12% in panic), zombie cleanup (>15 days held, ±5% net).

### Market State Classification

Evaluated on TWII/OTC indices: Bull → Neutral → Weak → Bear, with crash (1-day drop >-2.5%) and panic (1-day drop >-3% or 3-day cumulative >-3.5%) overlays.

### Position Sizing (V7 NAV-based)

Each trade = NAV × 2.8%. Cash reserve: 10% of initial capital. Key constants in `group_backtest.py`:

- `DAILY_MAX_POSITIONS = 12` — max concurrent holdings
- `DAILY_MAX_NEW_BUY = 4` — max new buys per day
- `DAILY_MAX_SWAP = 1` — max sell-then-buy swaps per day
- `DAILY_BUDGET = 25,000` NTD per trade
- `DAILY_DEFAULT_INDUSTRIES = ['半導體業']`

### Three-tier Config System

1. `DEFAULT_CONFIG` in `strategy.py` — base parameters
2. `INDUSTRY_CONFIGS` in `group_backtest.py` — industry-specific overrides (currently only 半導體業 active)
3. Runtime overrides — command-line or script-level modifications

### Data Flow

```
YFinance OHLCV → reconstruct_market_history() → per-day step_day()
  → evaluate sell signals (S1-S4) for positions
  → evaluate buy signals (B1-B7) for candidates
  → apply cash/position limits → update NAV → export CSV/JSON snapshots
```

### Output Directories

- `output/backtest_csv/` — all_trades.csv, equity_summary.csv, daily position/state snapshots
- `output/daily_csv/` — daily positions, trades, state JSONs

## Critical Design Constraint

**Daily engine must produce 100% identical results to backtest engine.** Any strategy logic change must be reflected in both `group_backtest.py` and `run_daily_trading.py`. Use `python3 run_daily_trading.py verify` to confirm alignment after changes.

### Execution Modes

The backtest supports multiple execution timing modes via `EXEC_MODE`:
- `'next_open'` — execute at next day's open (default, most realistic)
- `'same_close'` — execute at same-day close
- `'close_open'` — execute at next day's open with previous close price

### Stock Pre-screening Filters

Before strategy evaluation, stocks must pass: `MIN_VOLUME_SHARES = 500,000`, `MIN_TURNOVER = 50,000,000` NTD, `MIN_PRICE = 10` NTD, `MIN_DATA_DAYS = 60`. Tickers ending in `.TW`/`.TWO` are normalized; tickers starting with `00` are treated as ETFs (different tax rate).

## Ablation Testing

The ablation framework tests parameter changes across 7 market regimes defined in `PRESET_PERIODS`: full cycle, pure bear, pure bull, correction, crash (Trump tariff), V-recovery, and recent 6 months. A change is only accepted if it improves or maintains performance across all periods. The framework uses an explicit train/validation split (training: 2022-01–2025-06, validation: 2025-07+) to prevent overfitting.

## Fee Calculations (Taiwan Market)

- Broker commission: 0.1425% × 0.6 discount (min 1-20 NTD) — applied on both buy and sell
- Tax: ETF 0.1%, stocks 0.3% — sell only
- All P&L figures are net of fees/tax via `calculate_net_pnl()` in `strategy.py`
