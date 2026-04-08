#!/usr/bin/env python3
"""R15: V31 Multi-timeframe + V32 Correlation filter"""
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

    # V31: Multi-timeframe — weekly filter
    # A: block buy only (don't tighten exits)
    'mtf_block':    {'enable_mtf': True, 'mtf_weekly_block_buy': True, 'mtf_weekly_tighten': False},
    # B: tighten exits only (don't block buy)
    'mtf_tight':    {'enable_mtf': True, 'mtf_weekly_block_buy': False, 'mtf_weekly_tighten': True,
                     'mtf_tighten_tier_a': 50, 'mtf_tighten_zombie': 7},
    # C: both
    'mtf_both':     {'enable_mtf': True, 'mtf_weekly_block_buy': True, 'mtf_weekly_tighten': True,
                     'mtf_tighten_tier_a': 50, 'mtf_tighten_zombie': 7},
    # D: mild tighten
    'mtf_mild':     {'enable_mtf': True, 'mtf_weekly_block_buy': True, 'mtf_weekly_tighten': True,
                     'mtf_tighten_tier_a': 65, 'mtf_tighten_zombie': 8},

    # V32: Correlation filter
    'corr_07_2':    {'enable_corr_filter': True, 'corr_filter_threshold': 0.7, 'corr_filter_max_similar': 2},
    'corr_07_1':    {'enable_corr_filter': True, 'corr_filter_threshold': 0.7, 'corr_filter_max_similar': 1},
    'corr_06_2':    {'enable_corr_filter': True, 'corr_filter_threshold': 0.6, 'corr_filter_max_similar': 2},
    'corr_08_3':    {'enable_corr_filter': True, 'corr_filter_threshold': 0.8, 'corr_filter_max_similar': 3},

    # Combos
    'mtf_corr':     {'enable_mtf': True, 'mtf_weekly_block_buy': True, 'mtf_weekly_tighten': True,
                     'mtf_tighten_tier_a': 65, 'mtf_tighten_zombie': 8,
                     'enable_corr_filter': True, 'corr_filter_threshold': 0.7, 'corr_filter_max_similar': 2},
    'mtf_blk_corr': {'enable_mtf': True, 'mtf_weekly_block_buy': True, 'mtf_weekly_tighten': False,
                     'enable_corr_filter': True, 'corr_filter_threshold': 0.7, 'corr_filter_max_similar': 2},
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
print('  R15: V31 Multi-timeframe + V32 Correlation | Base=V19b+Z1.2+TM3+PP40')
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
