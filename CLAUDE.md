# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Taiwan semiconductor stock quantitative trading & backtesting system. Implements a momentum-based strategy (Strategy A+) with dynamic entry/exit logic, market state detection (bull/neutral/weak/bear with panic detection), multi-stage position management, and full backtesting engine (2021-present) with real Taiwan broker commission/tax calculations.

**Current best performance (半導體業 V19b A+):** Sharpe 0.92, Calmar 0.89, Return +138%, MDD 24.1% (Training: 2021-01 ~ 2025-06)

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

# Strategy variants
python3 run_strategy_a_plus.py        # Strategy A+ (asymmetric zombie + reduced frequency)
python3 run_midterm_backtest.py       # Strategy C vs A comparison
python3 run_midterm_backtest_r2.py    # Strategy C round 2 refinement
python3 run_midterm_backtest_r3.py    # Strategy C round 3 refinement

# Strategy D (Trend Persistence) backtests
python3 run_trend_persistence_backtest.py   # Strategy D baseline (5-pos concentrated)
python3 run_trend_persistence_r2.py         # Strategy D Round 2 refinement
python3 run_trend_persistence_r2b.py        # Strategy D R2b validation completion

# Research ablation series (_r*.py)
python3 _r11_theme_limit.py           # Sub-theme concentration limits
python3 _r12_factor.py                # Revenue + RS factor model (V30)
python3 _r13_sizing.py                # Budget, trailing stop, score sizing
python3 _r14_reduce.py                # Partial sell, panic thresholds
python3 _r15_analysis.py              # MTF impact trade-by-trade analysis
python3 _r15_mtf_corr.py              # Multi-timeframe + correlation filter (V31/V32)
python3 _r16_new_directions.py        # EWT filter, gradual re-entry, idle 0050 (V33-V35)
python3 _r17_breadth_accel_fi.py      # Breadth, momentum accel, foreign investor (V33-V35)
python3 _r18_midlong.py               # Trend protect v2, ladder zombie, winner upgrade (V36-V38)
python3 _r20_vix_fx.py                # VIX panic filter + USD/TWD FX filter (V39/V40)
python3 _r21_crossmarket.py           # Cross-market signals: SOX, US10Y, TSM ADR, Korea (R21)
python3 _r22_execution.py             # Limit orders, entry patterns, volume patterns (V41-V43)
```

## Dependencies

`yfinance`, `pandas`, `numpy`, `twstock` (optional), `requests`, `anthropic` (for signal filtering). API key stored in `.env` (`ANTHROPIC_API_KEY`). Data is cached as MD5-keyed pickles in `.cache/`.

## Architecture

### Core Modules

- **`strategy.py`** — Strategy logic engine. Defines `check_strategy_signal()` (buy B1-B7, sell S1-S4), `calculate_net_pnl()`, `calculate_fee()`, `calculate_tax()`, and `DEFAULT_CONFIG` with 40+ tunable parameters. This is the canonical source — `group_backtest.py` imports and re-exports these functions.
- **`group_backtest.py`** (~8K+ lines) — Monolithic backtest/simulation engine. `run_group_backtest()` is the main entry point. Also contains `INDUSTRY_CONFIGS` / `INDUSTRY_CONFIGS_V5` with per-industry parameter overrides, `PRESET_PERIODS` (7 market regimes), `reconstruct_market_history()` for market state pre-computation, and multiple operational modes (scan, verify, daily-vs-backtest comparison).
- **`run_daily_trading.py`** — Live daily execution via `DailyEngine` class. `step_day()` mirrors backtest logic exactly. State persisted in `output/daily_engine_state.json`. Imports strategy functions from `group_backtest.py` (which re-exports from `strategy.py`).
- **`run_full_backtest.py`** — Full backtest runner with equity curve plotting and 0050 DCA/signal-mirror benchmarks. Key settings: `INITIAL_CAPITAL=900_000`, `BUDGET_PER_TRADE=25_000`, `INDUSTRY='半導體業'`, `EXEC_MODE='next_open'`, `SLIPPAGE_PCT=0.003`.
- **`stock_utils.py`** — Data fetching (`get_stock_data()`, `batch_download_stocks()`), market analysis (`get_market_status()`), `build_info_dict()` for per-stock indicator computation, and MD5-keyed pickle cache in `.cache/`.
- **`theme_config.py`** — Sub-theme momentum boost system. Maps semiconductor stocks to sub-themes (AI_HPC設計, CoWoS封裝, etc.) via `THEME_MAP`. Calculates theme heat scores; controls `theme_max_hold=3` (R11: max 3 stocks per sub-theme).
- **`strategy_pullback.py`** — Alternative "Strategy B" focused on buying pullbacks (vs momentum-chasing Strategy A). Complete signal function with its own `PULLBACK_CONFIG`. Tested via `run_pullback_ablation.py`. Lower Sharpe than A — not deployed.
- **`strategy_midterm.py`** — "Strategy C" midterm trend-following. Buys pullbacks to MA20 support in uptrends, holds longer (90-day zombie vs 15-day), concentrated 4-position portfolio at 15% NAV each. Own `MIDTERM_CONFIG`. Tested via `run_midterm_backtest.py` / `r2.py` / `r3.py`.
- **`industry_manager.py` / `industry_lookup.py`** — Stock list management for 上市/上櫃 markets. `industry_lookup.py` is an interactive dev utility with `--check` mode for quarterly stock list maintenance.

### Strategy Variants

| Strategy | File | Focus | Status |
|----------|------|-------|--------|
| **A (A+)** | `strategy.py` / `group_backtest.py` | Momentum, 12 positions | ✅ **Active — daily trading** |
| **B** | `strategy_pullback.py` | Pullback to MA20 | ⚠️ Reference — lower Sharpe, not deployed |
| **C** | `strategy_midterm.py` | Midterm trend, 4 positions, 90-day hold | ⚠️ Optional alternative |
| **D** | `run_trend_persistence_*.py` | Trend persistence, 5 positions, 20-day asymmetric | 🔍 R2b exploration |

### Research Ablation Series (_r*.py)

These files are structured ablation experiments testing specific features (V30-V43), not production code. Each tests 8-20 parameter combinations across Train/Validation splits.

| File | Version | Tests | Integrated? |
|------|---------|-------|-------------|
| `_r11_theme_limit.py` | R11 | `theme_max_hold` 2-5 | ✅ theme_max_hold=3 |
| `_r12_factor.py` | V30 | Revenue growth + RS factor screening | 🔍 Exploration |
| `_r13_sizing.py` | — | Budget %, profit trailing, score sizing | 🔍 Exploration |
| `_r14_reduce.py` | R14 | Partial sell tiers, portfolio panic thresholds | ✅ Portfolio panic integrated |
| `_r15_analysis.py` | R15 | MTF trade-by-trade analysis (base vs MTF) | ✅ Analysis tool |
| `_r15_mtf_corr.py` | V31/V32 | Multi-timeframe + correlation filter | ✅ V31 mtf_block adopted |
| `_r16_new_directions.py` | V33-V35 | EWT filter, gradual re-entry, idle 0050 | 🔍 Exploration |
| `_r17_breadth_accel_fi.py` | V33-V35 | Breadth indicator, momentum accel, foreign investor | 🔍 Exploration |
| `_r18_midlong.py` | V36-V38 | Trend protect v2, ladder zombie, winner upgrade | 🔍 Exploration |
| `_r20_vix_fx.py` | V39/V40 | VIX panic filter (25/30), USD/TWD FX filter | ✅ VIX filter integrated |
| `_r21_crossmarket.py` | R21 | SOX, US10Y, TSM ADR, Korea semi overnight signals | 🔍 Exploration |
| `_r22_execution.py` | V41-V43 | Limit orders, entry/volume patterns | 🔍 Exploration |

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

**Buy conditions (B1-B7):** Trend confirmation (MA20>MA60), volume surge, price above prev close, bias limit (market-context-aware), fish tail momentum (5-day consecutive signals), breakout (10-day high), shooting star filter (disabled in A+ — ablation shows +0.10 Sharpe when off).

**Sell conditions (S1-S4):** Tiered stop-profit (tier_a 80%/tier_b 15% net in A+), break below MA60, hard stop-loss (-15% net default, dynamic -12% weak/-10% bear in A+), zombie cleanup (asymmetric: 15-day loss / 45-day profit in A+).

### Market State Classification

Evaluated on TWII/OTC indices: Bull → Neutral → Weak → Bear, with crash (1-day drop >-2.5%) and panic (1-day drop >-3% or 3-day cumulative >-3.5%) overlays.

### Active Risk Modules (A+ Configuration)

The A+ configuration (in `INDUSTRY_CONFIGS['半導體業']`) enables these modules vs `DEFAULT_CONFIG`:

| Module | Parameter | Effect |
|--------|-----------|--------|
| Portfolio Panic | `enable_portfolio_panic: True`, day -4%, 3d -6% | Sell losing positions, 3-day cooldown (R14: Shrp+0.04) |
| Dynamic Stop-Loss | `enable_dyn_stop: True`, weak -12%, bear -10% | Tighter stops in downtrends |
| S2 Buffer | `s2_buffer_enabled: True`, 10% profit → 5-day wait | Reduce MA60 false breakout exits (Ret+25%) |
| Asymmetric Zombie | `zombie_asymmetric: True`, loss 15d / profit 45d | Fast exit losers, let winners run (Ret+21%) |
| MTF Filter | `enable_mtf: True`, weekly MA20<MA60 → no new buys | R15: MDD 28.4%→24.1%, Calmar+0.09 |
| VIX Filter | `enable_vix_filter: True`, VIX>25 limit 2, VIX>30 halt | R20: Shrp+0.03, MDD-1.4%, Calmar+0.10 |
| Peer RS | `enable_val_peer_hold: True`, Z-score theme ranking | V19b: Shrp+0.08, Ret+18.4% |
| Theme Limit | `theme_max_hold: 3` | Max 3 stocks per sub-theme (R11: Shrp+0.02) |
| Sector Momentum | `enable_sector_momentum: True`, 20d avg | V8: Shrp+0.04 |
| Vol Sizing | `enable_vol_sizing: True`, target 2.5%, floor 70% | High-volatility position reduction |

**Disabled modules (ablation confirmed ineffective):** `enable_profit_trailing`, `enable_theme_boost`, `enable_theme_rotation`, `mtf_weekly_tighten`, B7 shooting star, B8/B9 pullback/dip entries, `enable_regime_adaptive`, `enable_conviction_hold`.

### Position Sizing (V7 NAV-based)

Each trade = NAV × 2.8%. Cash reserve: 10% of initial capital. Key constants in `group_backtest.py`:

- `DAILY_MAX_POSITIONS = 12` — max concurrent holdings
- `DAILY_MAX_NEW_BUY = 4` — max new buys per day
- `DAILY_MAX_SWAP = 1` — max sell-then-buy swaps per day
- `DAILY_BUDGET = 25,000` NTD per trade
- `INITIAL_CAPITAL = 900,000` NTD
- `DAILY_DEFAULT_INDUSTRIES = ['半導體業']`

### Three-tier Config System

1. `DEFAULT_CONFIG` in `strategy.py` — base parameters
2. `INDUSTRY_CONFIGS` in `group_backtest.py` — industry-specific overrides (currently only 半導體業 active for daily trading)
3. Runtime overrides — command-line or script-level modifications

**Critical:** Always load `INDUSTRY_CONFIGS.get(INDUSTRY, {}).get('config', {})` when running 半導體業 backtests. Using bare `DEFAULT_CONFIG` yields Sharpe ~0.43 instead of ~0.92.

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
- `output/daily_engine_state.json` — live engine state (positions, cash, NAV)
- `data/revenue_db.csv`, `data/revenue_monthly.csv` — revenue data for factor screening (R12)

## Critical Design Constraint

**Daily engine must produce 100% identical results to backtest engine.** Any strategy logic change must be reflected in both `group_backtest.py` and `run_daily_trading.py`. Use `python3 run_daily_trading.py verify` to confirm alignment after changes.

### Execution Modes

The backtest supports multiple execution timing modes via `EXEC_MODE`:
- `'next_open'` — execute at next day's open (default, most realistic)
- `'same_close'` — execute at same-day close
- `'close_open'` — execute at next day's open with previous close price

### Stock Pre-screening Filters

Before strategy evaluation, stocks must pass: `MIN_VOLUME_SHARES = 500,000`, `MIN_TURNOVER = 50,000,000` NTD, `MIN_PRICE = 10` NTD, `MIN_DATA_DAYS = 60`. Tickers ending in `.TW`/`.TWO` are normalized; tickers starting with `00` are treated as ETFs (different tax rate).

## Train / Validation Split

| Period | Dates | Description |
|--------|-------|-------------|
| **T (Training)** | 2021-01-01 ~ 2025-06-30 | Parameter tuning only (4.5 years) |
| T1 | 2022-01-01 ~ 2022-10-31 | Pure bear: TWII 18000→12600 |
| T2 | 2023-01-01 ~ 2024-07-31 | Pure bull: AI rally 14000→24000 |
| T3 | 2024-07-01 ~ 2025-01-31 | High correction: AI overheating |
| T4 | 2025-02-01 ~ 2025-06-30 | Trump tariff crash |
| **V (Validation)** | 2025-07-01 ~ now | Final validation only — no parameter tuning from V results |

**Anti-overfitting rule:** All ablation tuning uses only T periods. Once V is run, parameters are frozen.

## Ablation Testing

The ablation framework tests parameter changes across 7 market regimes defined in `PRESET_PERIODS`: full cycle, pure bear, pure bull, correction, crash (Trump tariff), V-recovery, and recent 6 months. A change is only accepted if it improves or maintains performance across all periods.

## Fee Calculations (Taiwan Market)

- Broker commission: 0.1425% × 0.6 discount (min 1-20 NTD) — applied on both buy and sell
- Tax: ETF 0.1%, stocks 0.3% — sell only
- Slippage: 0.3% (applied in backtest)
- All P&L figures are net of fees/tax via `calculate_net_pnl()` in `strategy.py`

## Troubleshooting: Sharpe ~0.43 instead of ~0.92

Most common causes:

1. **Missing INDUSTRY_CONFIGS** — must pass `INDUSTRY_CONFIGS.get('半導體業', {}).get('config', {})` as `config_override`
2. **Wrong time period** — use Training T (2021-01 ~ 2025-06), not validation-only or bear-only segments
3. **Wrong EXEC_MODE** — must be `'next_open'`
4. **Wrong capital** — `INITIAL_CAPITAL=900_000`, `BUDGET_PER_TRADE=25_000`
5. **Stale cache** — delete `.cache/` and re-run

## Industry Configs Reference

| Industry | Version | Sharpe | Status |
|----------|---------|--------|--------|
| 半導體業 | V19b A+ | 0.92 | ✅ Active daily trading |
| 通信網路業 | L3 | 1.23 | Validated, not in daily |
| 其他電子業 | L2 | 1.13 | Validated, not in daily |
| 電腦及週邊設備 | L2 | 1.06 | Validated, not in daily |
| 電子零組件 | L2 | 0.98 | Validated, not in daily |
| 電機機械 | L2 | 0.79 | Validated, not in daily |
| 光電業 | L2 | 0.38 | High MDD |
| 電子通路業 | — | -0.02 | Deprecated |

## Reference Documents

- **`STRATEGY_SUMMARY.md`** — Complete A+ strategy documentation: parameter diff table from DEFAULT_CONFIG→A+, performance data, daily usage guide, verify workflow, Sharpe 0.43 troubleshooting. **Read this first when onboarding.**
- **`README.md`** — Version marker (current: 0408 version)
