#!/usr/bin/env python3
"""R18: V36 Trend Protect v2 + V37 Ladder Zombie + V38 Winner Upgrade"""
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

    # V36: Trend protect v2 (improved — only protect profitable + added + trending)
    'tp2_b2_p0':      {'zombie_trend_protect_v2': True, 'ztp2_min_buys': 2, 'ztp2_min_profit': 0.0},
    'tp2_b2_p3':      {'zombie_trend_protect_v2': True, 'ztp2_min_buys': 2, 'ztp2_min_profit': 3.0},
    'tp2_b1_p0':      {'zombie_trend_protect_v2': True, 'ztp2_min_buys': 1, 'ztp2_min_profit': 0.0},
    'tp2_b3_p0':      {'zombie_trend_protect_v2': True, 'ztp2_min_buys': 3, 'ztp2_min_profit': 0.0},

    # V37: Ladder zombie (more adds = more patience)
    'lad_3':          {'zombie_ladder': True, 'zombie_ladder_extra_days': 3, 'zombie_ladder_cap': 25},
    'lad_5':          {'zombie_ladder': True, 'zombie_ladder_extra_days': 5, 'zombie_ladder_cap': 30},
    'lad_2':          {'zombie_ladder': True, 'zombie_ladder_extra_days': 2, 'zombie_ladder_cap': 20},

    # V38: Winner upgrade (profit + high adds → switch to long-term params)
    'wu_10_3':        {'winner_upgrade': True, 'winner_min_profit_pct': 10.0, 'winner_min_buys': 3,
                       'winner_tier_a': 120, 'winner_zombie_days': 45, 'winner_zombie_range': 0.0},
    'wu_15_3':        {'winner_upgrade': True, 'winner_min_profit_pct': 15.0, 'winner_min_buys': 3,
                       'winner_tier_a': 120, 'winner_zombie_days': 45, 'winner_zombie_range': 0.0},
    'wu_10_4':        {'winner_upgrade': True, 'winner_min_profit_pct': 10.0, 'winner_min_buys': 4,
                       'winner_tier_a': 100, 'winner_zombie_days': 30, 'winner_zombie_range': 0.0},
    'wu_5_2':         {'winner_upgrade': True, 'winner_min_profit_pct': 5.0, 'winner_min_buys': 2,
                       'winner_tier_a': 100, 'winner_zombie_days': 30, 'winner_zombie_range': 3.0},

    # Best combos
    'tp2_lad3':       {'zombie_trend_protect_v2': True, 'ztp2_min_buys': 2, 'ztp2_min_profit': 0.0,
                       'zombie_ladder': True, 'zombie_ladder_extra_days': 3, 'zombie_ladder_cap': 25},
    'tp2_wu10':       {'zombie_trend_protect_v2': True, 'ztp2_min_buys': 2, 'ztp2_min_profit': 0.0,
                       'winner_upgrade': True, 'winner_min_profit_pct': 10.0, 'winner_min_buys': 3,
                       'winner_tier_a': 120, 'winner_zombie_days': 45},
    'lad3_wu10':      {'zombie_ladder': True, 'zombie_ladder_extra_days': 3, 'zombie_ladder_cap': 25,
                       'winner_upgrade': True, 'winner_min_profit_pct': 10.0, 'winner_min_buys': 3,
                       'winner_tier_a': 120, 'winner_zombie_days': 45},
    'all3':           {'zombie_trend_protect_v2': True, 'ztp2_min_buys': 2, 'ztp2_min_profit': 0.0,
                       'zombie_ladder': True, 'zombie_ladder_extra_days': 3, 'zombie_ladder_cap': 25,
                       'winner_upgrade': True, 'winner_min_profit_pct': 10.0, 'winner_min_buys': 3,
                       'winner_tier_a': 120, 'winner_zombie_days': 45},
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
print('  R18: V36 TrendProtect v2 + V37 Ladder + V38 WinnerUpgrade')
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
