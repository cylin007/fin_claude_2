#!/usr/bin/env python3
"""
MDD Reduction Ablation: Test dynamic buy limit (A), tighter stops (B), and A+B combos.
Training + Validation split.
"""

import sys, os, time, warnings
import pandas as pd
import numpy as np

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
BASE_CONFIG = INDUSTRY_CONFIGS[INDUSTRY]['config'].copy()

CONFIGS = {
    'baseline': {
        'desc': '現狀 (全關)',
        'overrides': {},
    },
    # ==========================================
    # A: 動態限買 (dyn_buy_limit)
    # ==========================================
    'A1_dyn_buy': {
        'desc': 'A1: 動態限買 (weak=2/bear=1/panic=0)',
        'overrides': {
            'enable_dyn_buy_limit': True,
            'dyn_buy_bull': 4, 'dyn_buy_neutral': 4,
            'dyn_buy_weak': 2, 'dyn_buy_bear': 1, 'dyn_buy_panic': 0,
        },
    },
    'A1_tight': {
        'desc': 'A1 嚴格: 動態限買 (weak=1/bear=0/panic=0)',
        'overrides': {
            'enable_dyn_buy_limit': True,
            'dyn_buy_bull': 4, 'dyn_buy_neutral': 3,
            'dyn_buy_weak': 1, 'dyn_buy_bear': 0, 'dyn_buy_panic': 0,
        },
    },
    'A2_dyn_expo': {
        'desc': 'A2: 動態曝險 (weak=10/bear=6/panic=4)',
        'overrides': {
            'enable_dynamic_exposure': True,
            'dyn_max_bull': 12, 'dyn_max_neutral': 12,
            'dyn_max_weak': 10, 'dyn_max_bear': 6, 'dyn_max_panic': 4,
        },
    },
    'A2_tight': {
        'desc': 'A2 嚴格: 動態曝險 (weak=8/bear=4/panic=2)',
        'overrides': {
            'enable_dynamic_exposure': True,
            'dyn_max_bull': 12, 'dyn_max_neutral': 12,
            'dyn_max_weak': 8, 'dyn_max_bear': 4, 'dyn_max_panic': 2,
        },
    },
    'A_combo': {
        'desc': 'A1+A2: 限買+曝險 (溫和)',
        'overrides': {
            'enable_dyn_buy_limit': True,
            'dyn_buy_bull': 4, 'dyn_buy_neutral': 4,
            'dyn_buy_weak': 2, 'dyn_buy_bear': 1, 'dyn_buy_panic': 0,
            'enable_dynamic_exposure': True,
            'dyn_max_bull': 12, 'dyn_max_neutral': 12,
            'dyn_max_weak': 10, 'dyn_max_bear': 6, 'dyn_max_panic': 4,
        },
    },
    # ==========================================
    # B: 收緊停損
    # ==========================================
    'B1_tighter': {
        'desc': 'B1: 收緊停損 (weak=-10/bear=-8)',
        'overrides': {
            'enable_dyn_stop': True,
            'hard_stop_weak': -10,
            'hard_stop_bear': -8,
        },
    },
    'B1_very_tight': {
        'desc': 'B1 極緊: 停損 (weak=-8/bear=-6)',
        'overrides': {
            'enable_dyn_stop': True,
            'hard_stop_weak': -8,
            'hard_stop_bear': -6,
        },
    },
    'B2_zombie': {
        'desc': 'B2: 空頭加速殭屍 (z7 + ±3%)',
        'overrides': {
            'zombie_hold_days': 7,
            'zombie_net_range': 3.0,
        },
    },
    # ==========================================
    # A+B 組合
    # ==========================================
    'AB_moderate': {
        'desc': 'A+B 溫和: 限買+曝險+停損收緊',
        'overrides': {
            'enable_dyn_buy_limit': True,
            'dyn_buy_bull': 4, 'dyn_buy_neutral': 4,
            'dyn_buy_weak': 2, 'dyn_buy_bear': 1, 'dyn_buy_panic': 0,
            'enable_dynamic_exposure': True,
            'dyn_max_bull': 12, 'dyn_max_neutral': 12,
            'dyn_max_weak': 10, 'dyn_max_bear': 6, 'dyn_max_panic': 4,
            'enable_dyn_stop': True,
            'hard_stop_weak': -10,
            'hard_stop_bear': -8,
        },
    },
    'AB_tight': {
        'desc': 'A+B 嚴格: 限買嚴+曝險嚴+停損極緊',
        'overrides': {
            'enable_dyn_buy_limit': True,
            'dyn_buy_bull': 4, 'dyn_buy_neutral': 3,
            'dyn_buy_weak': 1, 'dyn_buy_bear': 0, 'dyn_buy_panic': 0,
            'enable_dynamic_exposure': True,
            'dyn_max_bull': 12, 'dyn_max_neutral': 12,
            'dyn_max_weak': 8, 'dyn_max_bear': 4, 'dyn_max_panic': 2,
            'enable_dyn_stop': True,
            'hard_stop_weak': -8,
            'hard_stop_bear': -6,
        },
    },
    'AB_balanced': {
        'desc': 'A+B 均衡: 限買溫和+停損溫和+殭屍加速',
        'overrides': {
            'enable_dyn_buy_limit': True,
            'dyn_buy_bull': 4, 'dyn_buy_neutral': 4,
            'dyn_buy_weak': 2, 'dyn_buy_bear': 1, 'dyn_buy_panic': 0,
            'enable_dyn_stop': True,
            'hard_stop_weak': -10,
            'hard_stop_bear': -8,
            'zombie_hold_days': 7,
            'zombie_net_range': 3.0,
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

        market_map = reconstruct_market_history(start_date, end_date)
        dl_start = (pd.Timestamp(start_date) - pd.Timedelta(days=250)).strftime('%Y-%m-%d')
        dl_end = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        preloaded_data, _ = batch_download_stocks(
            stocks, dl_start, dl_end, min_data_days=MIN_DATA_DAYS)
        print(f"  Valid stocks: {len(preloaded_data)}\n")

        period_results = {}
        for config_name, cfg_entry in CONFIGS.items():
            run_config = BASE_CONFIG.copy()
            run_config.update(cfg_entry['overrides'])

            t0 = time.time()
            result = run_group_backtest(
                stock_list=stocks, start_date=start_date, end_date=end_date,
                budget_per_trade=BUDGET_PER_TRADE, market_map=market_map,
                exec_mode=EXEC_MODE, config_override=run_config,
                initial_capital=INITIAL_CAPITAL, preloaded_data=preloaded_data,
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
                tl = result.get('trade_log', [])
                sells = [t for t in tl if t['type'] == 'SELL' and t.get('profit') is not None]
                gw = sum(t['profit'] for t in sells if t['profit'] > 0)
                gl = abs(sum(t['profit'] for t in sells if t['profit'] <= 0))
                pf = gw / gl if gl > 0 else 999

                print(f"  {config_name:>14s}: Ret {ret:+7.1f}% | Sharpe {sharpe:5.2f} | "
                      f"MDD {mdd:5.1f}% | Calmar {calmar:5.2f} | "
                      f"PF {pf:4.2f} | WR {wr:4.1f}% | {trades}T | "
                      f"⏱{elapsed:.0f}s  — {cfg_entry['desc']}")

        all_period_results[period_name] = period_results

    # === Summary ===
    print(f"\n\n{'='*120}")
    print(f"  📊 MDD REDUCTION ABLATION SUMMARY")
    print(f"{'='*120}")

    header = (f"{'Config':>14s} | {'Train Ret':>9s} {'Shrp':>5s} {'MDD':>6s} {'Clmr':>5s} {'PF':>5s} | "
              f"{'Val Ret':>8s} {'Shrp':>5s} {'MDD':>6s} {'Clmr':>5s} {'PF':>5s} | "
              f"{'ΔMDD-T':>7s} {'ΔMDD-V':>7s} {'判定':>8s}")
    print(header)
    print("-" * len(header))

    bl_t = all_period_results.get('Training', {}).get('baseline')
    bl_v = all_period_results.get('Validation', {}).get('baseline')
    bl_mdd_t = bl_t.get('mdd_pct', 0) if bl_t else 0
    bl_mdd_v = bl_v.get('mdd_pct', 0) if bl_v else 0
    bl_sharpe_t = bl_t.get('sharpe_ratio', 0) if bl_t else 0
    bl_sharpe_v = bl_v.get('sharpe_ratio', 0) if bl_v else 0

    for config_name in CONFIGS:
        t_res = all_period_results.get('Training', {}).get(config_name)
        v_res = all_period_results.get('Validation', {}).get(config_name)
        if not t_res or not v_res:
            continue

        def _metrics(res):
            ret = res.get('total_return_pct', 0)
            sharpe = res.get('sharpe_ratio', 0)
            mdd = res.get('mdd_pct', 0)
            calmar = res.get('calmar_ratio', 0)
            tl = res.get('trade_log', [])
            sells = [t for t in tl if t['type'] == 'SELL' and t.get('profit') is not None]
            gw = sum(t['profit'] for t in sells if t['profit'] > 0)
            gl = abs(sum(t['profit'] for t in sells if t['profit'] <= 0))
            pf = gw / gl if gl > 0 else 999
            return ret, sharpe, mdd, calmar, pf

        t_ret, t_sharpe, t_mdd, t_calmar, t_pf = _metrics(t_res)
        v_ret, v_sharpe, v_mdd, v_calmar, v_pf = _metrics(v_res)

        d_mdd_t = t_mdd - bl_mdd_t  # negative = MDD reduced (good)
        d_mdd_v = v_mdd - bl_mdd_v

        # Verdict
        marker = ""
        if config_name != 'baseline':
            t_better = (t_mdd < bl_mdd_t - 1) and (t_sharpe >= bl_sharpe_t * 0.9)
            v_better = (v_mdd <= bl_mdd_v + 1) and (v_sharpe >= bl_sharpe_v * 0.9)
            if t_better and v_better:
                marker = "✅ 雙贏"
            elif t_better and not v_better:
                marker = "⚠️ 驗證差"
            elif t_mdd < bl_mdd_t - 1:
                marker = "🔍 MDD降"
            else:
                marker = "❌ 無效"

        print(f"{config_name:>14s} | {t_ret:+8.1f}% {t_sharpe:5.2f} {t_mdd:5.1f}% {t_calmar:5.2f} {t_pf:5.2f} | "
              f"{v_ret:+7.1f}% {v_sharpe:5.2f} {v_mdd:5.1f}% {v_calmar:5.2f} {v_pf:5.2f} | "
              f"{d_mdd_t:+6.1f}% {d_mdd_v:+6.1f}% {marker}")

    print(f"\n{'='*120}")
    print("  ✅ 雙贏 = Training MDD 降且 Sharpe 維持 + Validation 不劣化")
    print("  ⚠️ 驗證差 = Training 改善但 Validation 劣化")
    print("  🔍 MDD降 = Training MDD 降但 Sharpe 掉太多")
    print("  ❌ 無效 = Training MDD 沒降")


if __name__ == '__main__':
    run_ablation()
