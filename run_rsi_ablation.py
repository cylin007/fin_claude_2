#!/usr/bin/env python3
"""
RSI Ablation: Test RSI minimum thresholds on 630K capital (70/30 mode).
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
INITIAL_CAPITAL = 630_000  # 70% of 900K
BUDGET_PER_TRADE = 17_500  # scaled from 25K
EXEC_MODE = 'next_open'
BASE_CONFIG = INDUSTRY_CONFIGS[INDUSTRY]['config'].copy()

CONFIGS = {
    'baseline': {
        'desc': 'RSI 無門檻 (現狀)',
        'overrides': {},
    },
    # === RSI 下限 (太弱不買) ===
    'rsi_30': {
        'desc': 'RSI > 30 (排除超賣)',
        'overrides': {'min_rsi': 30},
    },
    'rsi_40': {
        'desc': 'RSI > 40 (排除弱勢)',
        'overrides': {'min_rsi': 40},
    },
    'rsi_50': {
        'desc': 'RSI > 50 (只買中性以上)',
        'overrides': {'min_rsi': 50},
    },
    'rsi_55': {
        'desc': 'RSI > 55 (只買偏強)',
        'overrides': {'min_rsi': 55},
    },
    'rsi_60': {
        'desc': 'RSI > 60 (只買強勢)',
        'overrides': {'min_rsi': 60},
    },
    # === RSI + 其他動能確認 ===
    'rsi50_bb50': {
        'desc': 'RSI>50 + BB%B>0.5 (雙重動能)',
        'overrides': {'min_rsi': 50, 'min_bb_pct_b': 0.5},
    },
    'rsi50_10d3': {
        'desc': 'RSI>50 + 10日漲幅>3%',
        'overrides': {'min_rsi': 50, 'min_10d_return': 3},
    },
    'rsi40_vol12': {
        'desc': 'RSI>40 + 量比>1.2x',
        'overrides': {'min_rsi': 40, 'min_vol_ratio': 1.2},
    },
}

PERIODS = {
    'Train_2022': ('2022-01-01', '2025-06-30'),
    'Validation': ('2025-07-01', '2026-03-27'),
}


def run_ablation():
    stocks = get_stocks_by_industry(INDUSTRY)
    print(f"{INDUSTRY}: {len(stocks)} stocks")
    print(f"Capital: ${INITIAL_CAPITAL:,} (70% of 900K)\n")

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

                print(f"  {config_name:>14s}: Ret {ret:+7.1f}% | Shrp {sharpe:5.2f} | "
                      f"MDD {mdd:5.1f}% | Clmr {calmar:5.2f} | PF {pf:4.2f} | "
                      f"WR {wr:4.1f}% | {trades}T | "
                      f"⏱{elapsed:.0f}s  — {cfg_entry['desc']}")

        all_period_results[period_name] = period_results

    # === Summary ===
    print(f"\n\n{'='*110}")
    print(f"  📊 RSI ABLATION SUMMARY (630K capital)")
    print(f"{'='*110}")

    bl_t = all_period_results.get('Train_2022', {}).get('baseline')
    bl_v = all_period_results.get('Validation', {}).get('baseline')
    bl_sharpe_t = bl_t.get('sharpe_ratio', 0) if bl_t else 0
    bl_sharpe_v = bl_v.get('sharpe_ratio', 0) if bl_v else 0
    bl_mdd_t = bl_t.get('mdd_pct', 0) if bl_t else 0

    header = f"{'Config':>14s} | {'Train Ret':>9s} {'Shrp':>5s} {'MDD':>6s} {'Clmr':>5s} {'PF':>5s} {'WR':>5s} | {'Val Ret':>8s} {'Shrp':>5s} {'MDD':>6s} {'Clmr':>5s} | {'判定':>8s}"
    print(header)
    print("-" * len(header))

    for config_name in CONFIGS:
        t = all_period_results.get('Train_2022', {}).get(config_name)
        v = all_period_results.get('Validation', {}).get(config_name)
        if not t or not v:
            continue

        def _m(r):
            tl = r.get('trade_log', [])
            sells = [x for x in tl if x['type'] == 'SELL' and x.get('profit') is not None]
            gw = sum(x['profit'] for x in sells if x['profit'] > 0)
            gl = abs(sum(x['profit'] for x in sells if x['profit'] <= 0))
            return (r.get('total_return_pct', 0), r.get('sharpe_ratio', 0),
                    r.get('mdd_pct', 0), r.get('calmar_ratio', 0),
                    gw / gl if gl > 0 else 999, r.get('win_rate', 0))

        t_ret, t_shrp, t_mdd, t_clmr, t_pf, t_wr = _m(t)
        v_ret, v_shrp, v_mdd, v_clmr, v_pf, v_wr = _m(v)

        marker = ""
        if config_name != 'baseline':
            t_ok = (t_shrp >= bl_sharpe_t * 0.95) and (t_mdd <= bl_mdd_t + 2)
            v_ok = (v_shrp >= bl_sharpe_v * 0.90)
            if t_shrp > bl_sharpe_t and v_ok:
                marker = "✅ 雙贏"
            elif t_ok and v_ok:
                marker = "🟡 持平"
            elif t_shrp > bl_sharpe_t:
                marker = "⚠️ Val差"
            else:
                marker = "❌"

        print(f"{config_name:>14s} | {t_ret:+8.1f}% {t_shrp:5.2f} {t_mdd:5.1f}% {t_clmr:5.2f} {t_pf:5.2f} {t_wr:4.1f}% | "
              f"{v_ret:+7.1f}% {v_shrp:5.2f} {v_mdd:5.1f}% {v_clmr:5.2f} | {marker}")


if __name__ == '__main__':
    run_ablation()
