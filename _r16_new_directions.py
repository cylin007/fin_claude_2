#!/usr/bin/env python3
"""
R16: Three new directions
  1. SOX/EWT overnight signal — block buy if SOX dropped >2% overnight
  2. Gradual re-entry — limit new buys in first 2 weeks after weekly turns bull
  3. Idle cash → 0050 during weekly bear periods

These are tested as post-processing on the existing strategy results,
or as additional config parameters.
"""
import sys, os, time, warnings
import pandas as pd, numpy as np
import yfinance as yf
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

# ==========================================
# Test 1 & 2: SOX filter + Gradual re-entry
# These use existing engine config parameters
# ==========================================
# SOX filter: use EWT (already in market_map) as proxy
# EWT daily_chg < -2% → block buy next day
# This is similar to existing ewt_filter but we test it explicitly

COMBOS = {
    'base':           {},

    # 1. EWT/SOX overnight filter — block buy if EWT dropped significantly
    'ewt_filt':       {'enable_ewt_filter': True},  # existing feature, just enable

    # 2. Gradual re-entry — use weekly_max_buy to slow first weeks
    # Can't directly do "first 2 weeks after turn" with current engine,
    # but we can test max_new_buy reductions which approximate it
    'grad_buy2':      {'max_new_buy_per_day': 2},
    'grad_buy3':      {'max_new_buy_per_day': 3},
    'grad_w6':        {'weekly_max_buy': 6},

    # Combo: EWT filter + gradual
    'ewt_buy3':       {'enable_ewt_filter': True, 'max_new_buy_per_day': 3},
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
            'pf': pf, 'wr': wr, 'trades': len(buys) + len(sells),
            'snaps': r.get('daily_snapshots', []), 'el': el}


# ==========================================
# Test 3: Idle cash → 0050
# Simulated separately: overlay 0050 returns during idle periods
# ==========================================
def simulate_idle_0050(snaps, mm, initial_capital):
    """
    Simulate: when positions=0 AND weekly=bear, invest 80% cash in 0050.
    Returns adjusted NAV series.
    """
    # Download 0050
    old = sys.stderr; sys.stderr = open(os.devnull, 'w')
    etf = yf.download('0050.TW', start='2020-12-01', end='2026-04-05', progress=False)
    sys.stderr = old
    if isinstance(etf.columns, pd.MultiIndex):
        etf.columns = etf.columns.get_level_values(0)

    adjusted_navs = []
    etf_shares = 0
    etf_cost = 0
    in_0050 = False

    for s in snaps:
        d = s['date']
        nav = s['nav']
        pos = s['positions_count']
        weekly = mm.get(d, {}).get('weekly_bullish', True)

        # Get 0050 price
        ts = pd.Timestamp(d)
        valid = etf.index[etf.index <= ts]
        etf_close = float(etf.loc[valid[-1], 'Close']) if len(valid) > 0 else None

        if pos == 0 and not weekly and not in_0050 and etf_close:
            # Enter 0050 with 80% of cash
            invest = nav * 0.8
            etf_shares = int(invest / etf_close)
            etf_cost = etf_shares * etf_close
            in_0050 = True

        if in_0050 and (weekly or pos > 0):
            # Exit 0050
            if etf_close:
                etf_value = etf_shares * etf_close
                nav = nav + (etf_value - etf_cost)  # Add 0050 PnL
            etf_shares = 0
            etf_cost = 0
            in_0050 = False

        if in_0050 and etf_close:
            etf_value = etf_shares * etf_close
            nav = s['nav'] + (etf_value - etf_cost)

        adjusted_navs.append({'date': d, 'nav': nav})

    return adjusted_navs


print('=' * 130)
print('  R16: SOX/EWT filter + Gradual re-entry + Idle 0050')
print(f'  {len(COMBOS)} engine configs + idle 0050 simulation')
print('=' * 130)

all_snaps = {}

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
            if cn == 'base':
                all_snaps[pn] = m['snaps']
            print(f'  {cn:>14s}: Ret{m["ret"]:+7.1f}% Shrp{m["shrp"]:5.2f} MDD{m["mdd"]:5.1f}% '
                  f'Clm{m["calmar"]:5.2f} PF{m["pf"]:5.2f} WR{m["wr"]:4.1f}% '
                  f'Tr{m["trades"]:>4d} T{m["el"]:.0f}s')

    # Test 3: Idle 0050 overlay
    if pn in all_snaps:
        adj = simulate_idle_0050(all_snaps[pn], mm, IC)
        if adj:
            navs = [a['nav'] for a in adj]
            final = navs[-1]
            ret = (final / IC - 1) * 100
            # Calc Sharpe/MDD
            d_rets = []
            for i in range(1, len(navs)):
                if navs[i-1] > 0:
                    d_rets.append(navs[i] / navs[i-1] - 1)
            shrp = 0
            if d_rets and np.std(d_rets) > 0:
                rf_d = 0.015 / 245
                exc = [r - rf_d for r in d_rets]
                shrp = (np.mean(exc) * 245) / (np.std(d_rets) * np.sqrt(245))
            pk = navs[0]; mdd = 0
            for v in navs:
                if v > pk: pk = v
                dd = (pk - v) / pk * 100 if pk > 0 else 0
                if dd > mdd: mdd = dd
            days = (pd.Timestamp(adj[-1]['date']) - pd.Timestamp(adj[0]['date'])).days
            yrs = max(days / 365.25, 0.1)
            cagr = ((final / IC) ** (1/yrs) - 1) * 100 if final > 0 else 0
            calmar = cagr / mdd if mdd > 0 else 0
            print(f'  {"idle_0050":>14s}: Ret{ret:+7.1f}% Shrp{shrp:5.2f} MDD{mdd:5.1f}% '
                  f'Clm{calmar:5.2f} (overlay on base)')

print('\nDone!')
