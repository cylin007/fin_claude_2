#!/usr/bin/env python3
"""R13: Budget sizing + Profit trailing + Score sizing"""
import sys, os, time, warnings
import pandas as pd, numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from group_backtest import (run_group_backtest, reconstruct_market_history,
                            INDUSTRY_CONFIGS, MIN_DATA_DAYS)
from industry_manager import get_stocks_by_industry
from stock_utils import batch_download_stocks

INDUSTRY = '半導體業'
IC = 900_000
BASE_A = INDUSTRY_CONFIGS[INDUSTRY]['config'].copy()
stocks = get_stocks_by_industry(INDUSTRY)

COMBOS = {
    'base':         {},

    # 1. Budget per trade
    'bud_2.0':      {'budget_pct': 2.0},
    'bud_2.5':      {'budget_pct': 2.5},
    'bud_3.5':      {'budget_pct': 3.5},
    'bud_4.0':      {'budget_pct': 4.0},
    'bud_5.0':      {'budget_pct': 5.0},

    # 2. Profit Trailing Stop
    'pt_15_8':      {'enable_profit_trailing': True, 'profit_trail_activation': 15,
                     'profit_trail_pct': -8, 'profit_trail_min_lock': 5},
    'pt_20_10':     {'enable_profit_trailing': True, 'profit_trail_activation': 20,
                     'profit_trail_pct': -10, 'profit_trail_min_lock': 8},
    'pt_10_5':      {'enable_profit_trailing': True, 'profit_trail_activation': 10,
                     'profit_trail_pct': -5, 'profit_trail_min_lock': 3},
    'pt_30_12':     {'enable_profit_trailing': True, 'profit_trail_activation': 30,
                     'profit_trail_pct': -12, 'profit_trail_min_lock': 10},

    # 3. Score Sizing (high score → bigger position)
    'ss_basic':     {'enable_score_sizing': True,
                     'score_high_threshold': 1.5, 'score_high_ratio': 1.5,
                     'score_low_threshold': 0.5, 'score_low_ratio': 0.7,
                     'score_mid_ratio': 1.0},
    'ss_aggr':      {'enable_score_sizing': True,
                     'score_high_threshold': 1.5, 'score_high_ratio': 2.0,
                     'score_low_threshold': 0.5, 'score_low_ratio': 0.5,
                     'score_mid_ratio': 1.0},
    'ss_mild':      {'enable_score_sizing': True,
                     'score_high_threshold': 1.2, 'score_high_ratio': 1.3,
                     'score_low_threshold': 0.8, 'score_low_ratio': 0.8,
                     'score_mid_ratio': 1.0},

    # Combos
    'bud35_pt20':   {'budget_pct': 3.5, 'enable_profit_trailing': True,
                     'profit_trail_activation': 20, 'profit_trail_pct': -10,
                     'profit_trail_min_lock': 8},
    'bud35_ss':     {'budget_pct': 3.5, 'enable_score_sizing': True,
                     'score_high_threshold': 1.5, 'score_high_ratio': 1.5,
                     'score_low_threshold': 0.5, 'score_low_ratio': 0.7,
                     'score_mid_ratio': 1.0},
    'pt20_ss':      {'enable_profit_trailing': True, 'profit_trail_activation': 20,
                     'profit_trail_pct': -10, 'profit_trail_min_lock': 8,
                     'enable_score_sizing': True,
                     'score_high_threshold': 1.5, 'score_high_ratio': 1.5,
                     'score_low_threshold': 0.5, 'score_low_ratio': 0.7,
                     'score_mid_ratio': 1.0},
}


def run1(cn, ov, stocks, start, end, mm, data):
    cfg = {**BASE_A, **ov}
    budget = int(IC * cfg.get('budget_pct', 2.8) / 100)
    t0 = time.time()
    r = run_group_backtest(stocks, start, end, budget, mm, exec_mode='next_open',
                           config_override=cfg, initial_capital=IC, preloaded_data=data)
    el = time.time() - t0
    if not r:
        return None
    tl = r.get('trade_log', [])
    sells = [x for x in tl if x['type'] == 'SELL' and x.get('profit') is not None]
    buys = [x for x in tl if x['type'] == 'BUY']
    gw = sum(x['profit'] for x in sells if x['profit'] > 0)
    gl = abs(sum(x['profit'] for x in sells if x['profit'] <= 0))
    wr = sum(1 for s in sells if s['profit'] > 0) / len(sells) * 100 if sells else 0
    pf = gw / gl if gl > 0 else 999
    return {'ret': r['total_return_pct'], 'shrp': r['sharpe_ratio'],
            'mdd': r['mdd_pct'], 'calmar': r['calmar_ratio'],
            'pf': pf, 'wr': wr, 'trades': len(buys) + len(sells), 'el': el}


print('=' * 130)
print('  R13: Budget + Profit Trailing + Score Sizing | Base=V19b+Z1.2+TM3')
print(f'  {len(COMBOS)} configs x 2 periods')
print('=' * 130)

for pn, (start, end) in [('Train', ('2021-01-01', '2025-06-30')),
                          ('Val', ('2025-07-01', '2026-03-28'))]:
    mm = reconstruct_market_history(start, end)
    dl_s = (pd.Timestamp(start) - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
    dl_e = (pd.Timestamp(end) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    data, _ = batch_download_stocks(stocks, dl_s, dl_e, min_data_days=60)
    print(f'\n=== {pn}: {start} ~ {end} ===')
    for cn, ov in COMBOS.items():
        m = run1(cn, ov, stocks, start, end, mm, data)
        if m:
            print(f'  {cn:>14s}: Ret{m["ret"]:+7.1f}% Shrp{m["shrp"]:5.2f} MDD{m["mdd"]:5.1f}% '
                  f'Clm{m["calmar"]:5.2f} PF{m["pf"]:5.2f} WR{m["wr"]:4.1f}% '
                  f'Tr{m["trades"]:>4d} T{m["el"]:.0f}s')
print('\nDone!')
