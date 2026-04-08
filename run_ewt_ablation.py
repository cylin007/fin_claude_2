#!/usr/bin/env python3
"""
EWT Ablation: Test EWT filter/accelerator/tighten combinations
on Training and Validation periods separately.

Usage:
    python3 run_ewt_ablation.py
"""

import sys
import os
import time
import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from group_backtest import (
    run_group_backtest, reconstruct_market_history,
    INDUSTRY_CONFIGS, MIN_DATA_DAYS
)
from industry_manager import get_stocks_by_industry
from stock_utils import batch_download_stocks

INDUSTRY = '半導體業'
INITIAL_CAPITAL = 900_000
BUDGET_PER_TRADE = 25_000
EXEC_MODE = 'next_open'

# Base config from INDUSTRY_CONFIGS
BASE_CONFIG = INDUSTRY_CONFIGS[INDUSTRY]['config'].copy()

# EWT ablation configs: each is a dict of overrides on top of BASE_CONFIG
EWT_CONFIGS = {
    'baseline': {
        'desc': 'EWT 全關 (現狀)',
        'overrides': {},
    },
    # === V6.6 Score Boost (連續分數, 主力測試) ===
    'boost_default': {
        'desc': 'Score Boost 預設門檻 (±0.5%/±2%)',
        'overrides': {
            'enable_ewt_score_boost': True,
        },
    },
    'boost_wide': {
        'desc': 'Score Boost 寬門檻 (±1%/±3%)',
        'overrides': {
            'enable_ewt_score_boost': True,
            'ewt_boost_strong_up': 0.03,
            'ewt_boost_up': 0.01,
            'ewt_boost_down': -0.01,
            'ewt_boost_strong_down': -0.03,
        },
    },
    'boost_tight': {
        'desc': 'Score Boost 窄門檻 (±0.3%/±1.5%)',
        'overrides': {
            'enable_ewt_score_boost': True,
            'ewt_boost_strong_up': 0.015,
            'ewt_boost_up': 0.003,
            'ewt_boost_down': -0.005,
            'ewt_boost_strong_down': -0.015,
        },
    },
    'boost_asym': {
        'desc': 'Score Boost 非對稱 (跌重罰, 漲輕獎)',
        'overrides': {
            'enable_ewt_score_boost': True,
            'ewt_boost_score_strong_up': 0.15,  # 漲: 小獎
            'ewt_boost_score_up': 0.05,
            'ewt_boost_score_down': -0.2,        # 跌: 重罰
            'ewt_boost_score_strong_down': -0.4,
        },
    },
    'boost_canbuy': {
        'desc': 'Score Boost + can_buy ±2 (大幅調量)',
        'overrides': {
            'enable_ewt_score_boost': True,
            'ewt_boost_can_buy_bonus': 2,
            'ewt_boost_can_buy_penalty': -2,
        },
    },
    'boost_score_only': {
        'desc': '純分數加減 (不調 can_buy)',
        'overrides': {
            'enable_ewt_score_boost': True,
            'ewt_boost_can_buy_bonus': 0,
            'ewt_boost_can_buy_penalty': 0,
        },
    },
    'boost_canbuy_only': {
        'desc': '純 can_buy 調整 (不動分數)',
        'overrides': {
            'enable_ewt_score_boost': True,
            'ewt_boost_score_strong_up': 0,
            'ewt_boost_score_up': 0,
            'ewt_boost_score_down': 0,
            'ewt_boost_score_strong_down': 0,
        },
    },
    # === V6.6b Market Adaptive (空頭自動關閉) ===
    'adapt_default': {
        'desc': 'Adaptive 預設 (空頭關/弱減半/多全開)',
        'overrides': {
            'enable_ewt_score_boost': True,
            'ewt_boost_market_adaptive': True,
        },
    },
    'adapt_asym': {
        'desc': 'Adaptive + 非對稱 (跌重罰/漲輕獎)',
        'overrides': {
            'enable_ewt_score_boost': True,
            'ewt_boost_market_adaptive': True,
            'ewt_boost_score_strong_up': 0.15,
            'ewt_boost_score_up': 0.05,
            'ewt_boost_score_down': -0.2,
            'ewt_boost_score_strong_down': -0.4,
        },
    },
    'adapt_canbuy_only': {
        'desc': 'Adaptive + 純 can_buy (不動分數)',
        'overrides': {
            'enable_ewt_score_boost': True,
            'ewt_boost_market_adaptive': True,
            'ewt_boost_score_strong_up': 0,
            'ewt_boost_score_up': 0,
            'ewt_boost_score_down': 0,
            'ewt_boost_score_strong_down': 0,
        },
    },
    'adapt_score_only': {
        'desc': 'Adaptive + 純分數 (不調 can_buy)',
        'overrides': {
            'enable_ewt_score_boost': True,
            'ewt_boost_market_adaptive': True,
            'ewt_boost_can_buy_bonus': 0,
            'ewt_boost_can_buy_penalty': 0,
        },
    },
    'adapt_wide': {
        'desc': 'Adaptive + 寬門檻 (±1%/±3%)',
        'overrides': {
            'enable_ewt_score_boost': True,
            'ewt_boost_market_adaptive': True,
            'ewt_boost_strong_up': 0.03,
            'ewt_boost_up': 0.01,
            'ewt_boost_down': -0.01,
            'ewt_boost_strong_down': -0.03,
        },
    },
    'adapt+F4': {
        'desc': 'Adaptive + F4 濾網',
        'overrides': {
            'enable_ewt_score_boost': True,
            'ewt_boost_market_adaptive': True,
            'enable_ewt_filter': True,
            'ewt_drop_threshold': -0.02,
            'ewt_3d_threshold': -0.035,
        },
    },
    'boost+F4': {
        'desc': 'Score Boost + F4 濾網 (無 adaptive)',
        'overrides': {
            'enable_ewt_score_boost': True,
            'enable_ewt_filter': True,
            'ewt_drop_threshold': -0.02,
            'ewt_3d_threshold': -0.035,
        },
    },
}

PERIODS = {
    'Training': ('2021-01-01', '2025-06-30'),
    'Validation': ('2025-07-01', '2026-03-27'),
}


def run_ablation():
    stocks = get_stocks_by_industry(INDUSTRY)
    print(f"{INDUSTRY}: {len(stocks)} stocks\n")

    all_period_results = {}

    for period_name, (start_date, end_date) in PERIODS.items():
        print(f"\n{'='*80}")
        print(f"  📅 {period_name}: {start_date} ~ {end_date}")
        print(f"{'='*80}")

        # Market map
        market_map = reconstruct_market_history(start_date, end_date)

        # Pre-download data (shared across all configs)
        dl_start = (pd.Timestamp(start_date) - pd.Timedelta(days=250)).strftime('%Y-%m-%d')
        dl_end = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        preloaded_data, skipped = batch_download_stocks(
            stocks, dl_start, dl_end, min_data_days=MIN_DATA_DAYS)
        print(f"  Valid stocks: {len(preloaded_data)}\n")

        period_results = {}
        for config_name, ewt_cfg in EWT_CONFIGS.items():
            # Merge base + overrides
            run_config = BASE_CONFIG.copy()
            run_config.update(ewt_cfg['overrides'])

            t0 = time.time()
            result = run_group_backtest(
                stock_list=stocks,
                start_date=start_date,
                end_date=end_date,
                budget_per_trade=BUDGET_PER_TRADE,
                market_map=market_map,
                exec_mode=EXEC_MODE,
                config_override=run_config,
                initial_capital=INITIAL_CAPITAL,
                preloaded_data=preloaded_data,
            )
            elapsed = time.time() - t0

            if result:
                period_results[config_name] = result
                ret = result.get('total_return_pct', 0)
                sharpe = result.get('sharpe_ratio', 0)
                mdd = result.get('mdd_pct', 0)
                calmar = result.get('calmar_ratio', 0)
                trades = result.get('trades', 0)
                wr = result.get('win_rate', 0)
                pf = result.get('profit_factor', 0)
                # Calculate profit factor from trade log
                tl = result.get('trade_log', [])
                sells = [t for t in tl if t['type'] == 'SELL' and t.get('profit') is not None]
                gw = sum(t['profit'] for t in sells if t['profit'] > 0)
                gl = abs(sum(t['profit'] for t in sells if t['profit'] <= 0))
                pf = gw / gl if gl > 0 else 999

                print(f"  {config_name:>15s}: Ret {ret:+7.1f}% | Sharpe {sharpe:5.2f} | "
                      f"MDD {mdd:5.1f}% | Calmar {calmar:5.2f} | "
                      f"PF {pf:4.2f} | WR {wr:4.1f}% | {trades}T | "
                      f"⏱{elapsed:.0f}s  — {ewt_cfg['desc']}")
            else:
                print(f"  {config_name:>15s}: FAILED  ⏱{elapsed:.0f}s")

        all_period_results[period_name] = period_results

    # === Summary comparison ===
    print(f"\n\n{'='*100}")
    print(f"  📊 EWT ABLATION SUMMARY")
    print(f"{'='*100}")

    header = f"{'Config':>15s} | {'Training Ret':>12s} {'Sharpe':>7s} {'MDD':>6s} {'Calmar':>7s} {'PF':>5s} | {'Valid Ret':>10s} {'Sharpe':>7s} {'MDD':>6s} {'Calmar':>7s} {'PF':>5s} | {'Δ Sharpe':>8s}"
    print(header)
    print("-" * len(header))

    baseline_t = all_period_results.get('Training', {}).get('baseline')
    baseline_v = all_period_results.get('Validation', {}).get('baseline')
    bl_sharpe_t = baseline_t.get('sharpe_ratio', 0) if baseline_t else 0
    bl_sharpe_v = baseline_v.get('sharpe_ratio', 0) if baseline_v else 0

    for config_name in EWT_CONFIGS:
        t_res = all_period_results.get('Training', {}).get(config_name)
        v_res = all_period_results.get('Validation', {}).get(config_name)

        if t_res and v_res:
            t_ret = t_res.get('total_return_pct', 0)
            t_sharpe = t_res.get('sharpe_ratio', 0)
            t_mdd = t_res.get('mdd_pct', 0)
            t_calmar = t_res.get('calmar_ratio', 0)
            tl_t = t_res.get('trade_log', [])
            sells_t = [t for t in tl_t if t['type'] == 'SELL' and t.get('profit') is not None]
            gw_t = sum(t['profit'] for t in sells_t if t['profit'] > 0)
            gl_t = abs(sum(t['profit'] for t in sells_t if t['profit'] <= 0))
            pf_t = gw_t / gl_t if gl_t > 0 else 999

            v_ret = v_res.get('total_return_pct', 0)
            v_sharpe = v_res.get('sharpe_ratio', 0)
            v_mdd = v_res.get('mdd_pct', 0)
            v_calmar = v_res.get('calmar_ratio', 0)
            tl_v = v_res.get('trade_log', [])
            sells_v = [t for t in tl_v if t['type'] == 'SELL' and t.get('profit') is not None]
            gw_v = sum(t['profit'] for t in sells_v if t['profit'] > 0)
            gl_v = abs(sum(t['profit'] for t in sells_v if t['profit'] <= 0))
            pf_v = gw_v / gl_v if gl_v > 0 else 999

            d_sharpe = v_sharpe - bl_sharpe_v  # delta vs baseline on validation

            marker = ""
            if config_name != 'baseline':
                if t_sharpe > bl_sharpe_t and v_sharpe > bl_sharpe_v:
                    marker = " ✅ BOTH UP"
                elif t_sharpe > bl_sharpe_t and v_sharpe <= bl_sharpe_v:
                    marker = " ⚠️ OVERFIT"
                elif v_sharpe > bl_sharpe_v:
                    marker = " 🔍 VAL-ONLY"

            print(f"{config_name:>15s} | {t_ret:+11.1f}% {t_sharpe:7.2f} {t_mdd:5.1f}% {t_calmar:7.2f} {pf_t:5.2f} | "
                  f"{v_ret:+9.1f}% {v_sharpe:7.2f} {v_mdd:5.1f}% {v_calmar:7.2f} {pf_v:5.2f} | "
                  f"{d_sharpe:+7.2f}{marker}")

    print(f"\n{'='*100}")
    print("  ✅ BOTH UP = Training 和 Validation 都優於 baseline → 可以採用")
    print("  ⚠️ OVERFIT = Training 變好但 Validation 變差 → 過擬合")
    print("  🔍 VAL-ONLY = 只有 Validation 變好 → 需進一步驗證")


if __name__ == '__main__':
    run_ablation()
