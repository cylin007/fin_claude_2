#!/usr/bin/env python3
"""R17: V33 Breadth + V34 Accel + V35 Foreign Investor"""
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
    'base':           {},

    # V33: Breadth filter
    'br_30_50':       {'enable_breadth_filter': True, 'breadth_block_pct': 30, 'breadth_caution_pct': 50,
                       'breadth_caution_max_buy': 2},
    'br_25_45':       {'enable_breadth_filter': True, 'breadth_block_pct': 25, 'breadth_caution_pct': 45,
                       'breadth_caution_max_buy': 2},
    'br_35_55':       {'enable_breadth_filter': True, 'breadth_block_pct': 35, 'breadth_caution_pct': 55,
                       'breadth_caution_max_buy': 2},

    # V34: Momentum acceleration
    'accel':          {'enable_momentum_accel': True, 'accel_bonus': 0.2, 'accel_penalty': -0.15},
    'accel_strong':   {'enable_momentum_accel': True, 'accel_bonus': 0.4, 'accel_penalty': -0.3},

    # V35: Foreign investor
    'fi_3d':          {'enable_foreign_investor': True, 'fi_consec_buy_days': 3, 'fi_bonus': 0.2,
                       'fi_consec_sell_days': 3, 'fi_tighten_dd': 0.50},
    'fi_5d':          {'enable_foreign_investor': True, 'fi_consec_buy_days': 5, 'fi_bonus': 0.3,
                       'fi_consec_sell_days': 5, 'fi_tighten_dd': 0.45},

    # Combos
    'br_accel':       {'enable_breadth_filter': True, 'breadth_block_pct': 30, 'breadth_caution_pct': 50,
                       'breadth_caution_max_buy': 2,
                       'enable_momentum_accel': True, 'accel_bonus': 0.2, 'accel_penalty': -0.15},
    'br_fi':          {'enable_breadth_filter': True, 'breadth_block_pct': 30, 'breadth_caution_pct': 50,
                       'breadth_caution_max_buy': 2,
                       'enable_foreign_investor': True, 'fi_consec_buy_days': 3, 'fi_bonus': 0.2},
    'accel_fi':       {'enable_momentum_accel': True, 'accel_bonus': 0.2, 'accel_penalty': -0.15,
                       'enable_foreign_investor': True, 'fi_consec_buy_days': 3, 'fi_bonus': 0.2},
    'all3':           {'enable_breadth_filter': True, 'breadth_block_pct': 30, 'breadth_caution_pct': 50,
                       'breadth_caution_max_buy': 2,
                       'enable_momentum_accel': True, 'accel_bonus': 0.2, 'accel_penalty': -0.15,
                       'enable_foreign_investor': True, 'fi_consec_buy_days': 3, 'fi_bonus': 0.2},
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
print('  R17: V33 Breadth + V34 Accel + V35 Foreign Investor')
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
