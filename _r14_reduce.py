#!/usr/bin/env python3
"""R14: Reduce (partial sell) + Swap margin + Portfolio panic tuning"""
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

    # 1. Reduce (partial sell) — currently disabled
    'red_20_40':    {'enable_reduce': True,
                     'reduce_tier1_net': 20, 'reduce_tier1_ratio': 0.5,
                     'reduce_tier2_net': 40, 'reduce_tier2_ratio': 0.5,
                     'reduce_cooldown_days': 3},
    'red_15_30':    {'enable_reduce': True,
                     'reduce_tier1_net': 15, 'reduce_tier1_ratio': 0.5,
                     'reduce_tier2_net': 30, 'reduce_tier2_ratio': 0.5,
                     'reduce_cooldown_days': 3},
    'red_25_50':    {'enable_reduce': True,
                     'reduce_tier1_net': 25, 'reduce_tier1_ratio': 0.5,
                     'reduce_tier2_net': 50, 'reduce_tier2_ratio': 0.5,
                     'reduce_cooldown_days': 5},
    'red_30_60':    {'enable_reduce': True,
                     'reduce_tier1_net': 30, 'reduce_tier1_ratio': 0.4,
                     'reduce_tier2_net': 60, 'reduce_tier2_ratio': 0.5,
                     'reduce_cooldown_days': 5},
    # Reduce with smaller first cut
    'red_20_40_30': {'enable_reduce': True,
                     'reduce_tier1_net': 20, 'reduce_tier1_ratio': 0.3,
                     'reduce_tier2_net': 40, 'reduce_tier2_ratio': 0.5,
                     'reduce_cooldown_days': 3},

    # 2. Swap margin
    'swap_1.5':     {'swap_score_margin': 1.5},
    'swap_2.0':     {'swap_score_margin': 2.0},
    'swap_0.5':     {'swap_score_margin': 0.5},

    # 3. Portfolio panic tuning
    'pp_25_45':     {'portfolio_panic_day_pct': -2.5, 'portfolio_panic_3d_pct': -4.5},
    'pp_35_55':     {'portfolio_panic_day_pct': -3.5, 'portfolio_panic_3d_pct': -5.5},
    'pp_40_60':     {'portfolio_panic_day_pct': -4.0, 'portfolio_panic_3d_pct': -6.0},

    # Combos
    'red20_sw15':   {'enable_reduce': True,
                     'reduce_tier1_net': 20, 'reduce_tier1_ratio': 0.5,
                     'reduce_tier2_net': 40, 'reduce_tier2_ratio': 0.5,
                     'swap_score_margin': 1.5},
    'red25_pp35':   {'enable_reduce': True,
                     'reduce_tier1_net': 25, 'reduce_tier1_ratio': 0.5,
                     'reduce_tier2_net': 50, 'reduce_tier2_ratio': 0.5,
                     'portfolio_panic_day_pct': -3.5, 'portfolio_panic_3d_pct': -5.5},
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
    reduces = [x for x in tl if x['type'] == 'REDUCE']
    buys = [x for x in tl if x['type'] == 'BUY']
    gw = sum(x['profit'] for x in sells if x['profit'] > 0)
    gl = abs(sum(x['profit'] for x in sells if x['profit'] <= 0))
    wr = sum(1 for s in sells if s['profit'] > 0) / len(sells) * 100 if sells else 0
    pf = gw / gl if gl > 0 else 999
    return {'ret': r['total_return_pct'], 'shrp': r['sharpe_ratio'],
            'mdd': r['mdd_pct'], 'calmar': r['calmar_ratio'],
            'pf': pf, 'wr': wr, 'trades': len(buys) + len(sells),
            'reduces': len(reduces), 'el': el}


print('=' * 130)
print('  R14: Reduce + Swap margin + PP tuning | Base=V19b+Z1.2+TM3')
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
            red_str = f' R{m["reduces"]}' if m['reduces'] > 0 else ''
            print(f'  {cn:>14s}: Ret{m["ret"]:+7.1f}% Shrp{m["shrp"]:5.2f} MDD{m["mdd"]:5.1f}% '
                  f'Clm{m["calmar"]:5.2f} PF{m["pf"]:5.2f} WR{m["wr"]:4.1f}% '
                  f'Tr{m["trades"]:>4d}{red_str} T{m["el"]:.0f}s')
print('\nDone!')
