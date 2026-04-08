#!/usr/bin/env python3
"""R15 analysis: Compare base vs mtf_block trade-by-trade"""
import sys, os, warnings
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

START, END = '2021-01-01', '2025-06-30'
mm = reconstruct_market_history(START, END)
dl_s = (pd.Timestamp(START) - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
dl_e = (pd.Timestamp(END) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
data, _ = batch_download_stocks(stocks, dl_s, dl_e, min_data_days=60)
budget = int(IC * BASE_A.get('budget_pct', 2.8) / 100)

# Run both configs
# base = current config WITH mtf enabled (what we just wrote in)
# no_mtf = same config but mtf disabled (what it was before R15)
cfg_mtf = dict(BASE_A)  # already has enable_mtf=True
cfg_no_mtf = dict(BASE_A)
cfg_no_mtf['enable_mtf'] = False

print('Running base (no MTF)...')
r_no = run_group_backtest(stocks, START, END, budget, mm, exec_mode='next_open',
                          config_override=cfg_no_mtf, initial_capital=IC, preloaded_data=data)
print('Running MTF...')
r_mtf = run_group_backtest(stocks, START, END, budget, mm, exec_mode='next_open',
                           config_override=cfg_mtf, initial_capital=IC, preloaded_data=data)

# 1. Monthly NAV comparison
snaps_no = r_no['daily_snapshots']
snaps_mtf = r_mtf['daily_snapshots']

print('\n' + '=' * 100)
print('  NAV Monthly Comparison: No-MTF vs MTF')
print('=' * 100)
print(f'  {"Month":>8s} | {"NoMTF NAV":>12s} {"NoMTF DD":>8s} {"NoMTF Pos":>4s} | '
      f'{"MTF NAV":>12s} {"MTF DD":>8s} {"MTF Pos":>4s} | {"Delta NAV":>10s} {"Weekly":>6s}')
print('-' * 100)

peak_no = IC
peak_mtf = IC
for s_no, s_mtf in zip(snaps_no, snaps_mtf):
    d = s_no['date']
    if not (d.endswith('-01') or d.endswith('-02') or d.endswith('-03')):
        continue
    if not (d >= '2021-10' and d <= '2023-07'):
        continue
    # DD calc
    if s_no['nav'] > peak_no: peak_no = s_no['nav']
    if s_mtf['nav'] > peak_mtf: peak_mtf = s_mtf['nav']
    dd_no = (peak_no - s_no['nav']) / peak_no * 100 if peak_no > 0 else 0
    dd_mtf = (peak_mtf - s_mtf['nav']) / peak_mtf * 100 if peak_mtf > 0 else 0
    delta = s_mtf['nav'] - s_no['nav']
    weekly = 'BULL' if mm.get(d, {}).get('weekly_bullish', True) else 'BEAR'
    print(f'  {d:>8s} | {s_no["nav"]:>12,} {dd_no:>7.1f}% {s_no["positions_count"]:>4d} | '
          f'{s_mtf["nav"]:>12,} {dd_mtf:>7.1f}% {s_mtf["positions_count"]:>4d} | '
          f'{delta:>+10,} {weekly:>6s}')

# 2. Weekly trend periods analysis
print('\n' + '=' * 100)
print('  Weekly Trend Periods')
print('=' * 100)

in_bear = False
bear_start = ''
for s in snaps_no:
    d = s['date']
    weekly = mm.get(d, {}).get('weekly_bullish', True)
    if not weekly and not in_bear:
        bear_start = d
        in_bear = True
    elif weekly and in_bear:
        print(f'  WEEKLY BEAR: {bear_start} ~ {d}')
        in_bear = False
if in_bear:
    print(f'  WEEKLY BEAR: {bear_start} ~ {snaps_no[-1]["date"]}')

# 3. Trades that MTF blocked (exist in no_mtf but not in mtf)
tl_no = r_no['trade_log']
tl_mtf = r_mtf['trade_log']

buys_no = [(t['date'], t['ticker']) for t in tl_no if t['type'] == 'BUY']
buys_mtf = [(t['date'], t['ticker']) for t in tl_mtf if t['type'] == 'BUY']
buys_mtf_set = set(buys_mtf)

blocked_buys = [(d, t) for d, t in buys_no if (d, t) not in buys_mtf_set]
print(f'\n  MTF blocked {len(blocked_buys)} buy trades')

# Find the sells for those blocked buys → how much did they lose?
blocked_tickers_dates = set(blocked_buys)
blocked_sells = []
for t in tl_no:
    if t['type'] == 'SELL' and t.get('profit') is not None:
        # Check if this sell corresponds to a blocked buy
        # Approximate: same ticker, buy date in blocked period
        for bd, bt in blocked_buys:
            if t['ticker'] == bt and t.get('buy_date', '') == bd:
                blocked_sells.append(t)
                break

# Since buy_date may not be in trade_log, use ticker matching in weekly bear periods
# Simpler: compare PnL by month
print('\n' + '=' * 100)
print('  Monthly PnL Comparison (NoMTF vs MTF)')
print('=' * 100)

sells_no = [t for t in tl_no if t['type'] == 'SELL' and t.get('profit') is not None]
sells_mtf = [t for t in tl_mtf if t['type'] == 'SELL' and t.get('profit') is not None]

for year in [2021, 2022, 2023]:
    print(f'\n  --- {year} ---')
    for month in range(1, 13):
        prefix = f'{year}-{month:02d}'
        ms_no = [s for s in sells_no if s['date'].startswith(prefix)]
        ms_mtf = [s for s in sells_mtf if s['date'].startswith(prefix)]
        pnl_no = sum(s['profit'] for s in ms_no)
        pnl_mtf = sum(s['profit'] for s in ms_mtf)
        wr_no = sum(1 for s in ms_no if s['profit'] > 0) / len(ms_no) * 100 if ms_no else 0
        wr_mtf = sum(1 for s in ms_mtf if s['profit'] > 0) / len(ms_mtf) * 100 if ms_mtf else 0
        weekly_bears = sum(1 for s in snaps_no if s['date'].startswith(prefix)
                           and not mm.get(s['date'], {}).get('weekly_bullish', True))
        weekly_total = sum(1 for s in snaps_no if s['date'].startswith(prefix))
        wk_str = f'BEAR{weekly_bears}d' if weekly_bears > 0 else 'BULL'

        if ms_no or ms_mtf:
            delta = pnl_mtf - pnl_no
            print(f'    {prefix}: NoMTF {len(ms_no):>2d}tx PnL{pnl_no:>+8,.0f} WR{wr_no:>3.0f}% | '
                  f'MTF {len(ms_mtf):>2d}tx PnL{pnl_mtf:>+8,.0f} WR{wr_mtf:>3.0f}% | '
                  f'Delta{delta:>+8,.0f} | {wk_str}')

# 4. Summary
print('\n' + '=' * 100)
print('  Summary')
print('=' * 100)

total_blocked_pnl = sum(s['profit'] for s in sells_no) - sum(s['profit'] for s in sells_mtf)
print(f'  NoMTF total sells PnL: {sum(s["profit"] for s in sells_no):+,.0f}')
print(f'  MTF   total sells PnL: {sum(s["profit"] for s in sells_mtf):+,.0f}')
print(f'  Blocked buys: {len(blocked_buys)}')
print(f'  NoMTF: {len(sells_no)} sells, WR={sum(1 for s in sells_no if s["profit"]>0)/len(sells_no)*100:.0f}%')
print(f'  MTF:   {len(sells_mtf)} sells, WR={sum(1 for s in sells_mtf if s["profit"]>0)/len(sells_mtf)*100:.0f}%')

print('\nDone!')
