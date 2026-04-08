#!/usr/bin/env python3
"""R21: Cross-market signals (SOX, US10Y, TSM ADR, Korea semi)
All use overnight data available before Taiwan market opens.
Implementation: Download data, compute signals, inject into market_map as custom fields,
then use in buy filtering."""
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


def download_cross_market(start, end):
    """Download all cross-market data and compute daily signals."""
    dl_start = (pd.Timestamp(start) - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
    dl_end = (pd.Timestamp(end) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')

    signals = {}  # {date_str: {sox_5d_ret, us10y_chg, tsm_chg, korea_chg}}

    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        sox = yf.download('^SOX', start=dl_start, end=dl_end, progress=False)
        us10y = yf.download('^TNX', start=dl_start, end=dl_end, progress=False)
        tsm = yf.download('TSM', start=dl_start, end=dl_end, progress=False)
        samsung = yf.download('005930.KS', start=dl_start, end=dl_end, progress=False)
    finally:
        sys.stderr = old_stderr

    for df in [sox, us10y, tsm, samsung]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    # Compute signals per trading day
    # Use T-1 US data for T Taiwan (overnight known)
    tw_dates = pd.bdate_range(start, end)

    for tw_date in tw_dates:
        d = tw_date.strftime('%Y-%m-%d')
        sig = {}

        # SOX: 5-day return (momentum)
        sox_valid = sox.index[sox.index < tw_date]
        if len(sox_valid) >= 6:
            sox_now = float(sox.loc[sox_valid[-1], 'Close'])
            sox_5d = float(sox.loc[sox_valid[-6], 'Close'])
            sig['sox_5d_ret'] = (sox_now / sox_5d - 1) * 100 if sox_5d > 0 else 0
            sig['sox_1d_chg'] = (sox_now / float(sox.loc[sox_valid[-2], 'Close']) - 1) * 100 if len(sox_valid) >= 2 else 0

        # US10Y: 20-day change (rising = bad for tech)
        us_valid = us10y.index[us10y.index < tw_date]
        if len(us_valid) >= 21:
            us_now = float(us10y.loc[us_valid[-1], 'Close'])
            us_20d = float(us10y.loc[us_valid[-21], 'Close'])
            sig['us10y_20d_chg'] = us_now - us_20d  # basis points change

        # TSM ADR: overnight change
        tsm_valid = tsm.index[tsm.index < tw_date]
        if len(tsm_valid) >= 2:
            tsm_now = float(tsm.loc[tsm_valid[-1], 'Close'])
            tsm_prev = float(tsm.loc[tsm_valid[-2], 'Close'])
            sig['tsm_1d_chg'] = (tsm_now / tsm_prev - 1) * 100 if tsm_prev > 0 else 0

        # Korea semi: Samsung daily change (Korea opens 1hr before Taiwan)
        kr_valid = samsung.index[samsung.index <= tw_date]
        if len(kr_valid) >= 2:
            kr_now = float(samsung.loc[kr_valid[-1], 'Close'])
            kr_prev = float(samsung.loc[kr_valid[-2], 'Close'])
            sig['korea_1d_chg'] = (kr_now / kr_prev - 1) * 100 if kr_prev > 0 else 0

        if sig:
            signals[d] = sig

    return signals


def run_with_signal(cn, cfg_overrides, stocks, start, end, mm, data, cross_signals):
    """Run backtest with cross-market signal filtering applied via config."""
    cfg = {**BASE_A, **cfg_overrides}
    budget = int(IC * cfg.get('budget_pct', 2.8) / 100)

    # Inject cross-market signals into market_map
    for d, sig in cross_signals.items():
        if d in mm:
            mm[d]['cross'] = sig

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


# Since cross-market signals aren't directly in the engine yet,
# we test via a simpler approach: pre-compute "bad days" and use
# the existing enable_ewt_filter-like mechanism
# Actually, the simplest test: just analyze correlation first

print('=' * 100)
print('  R21: Cross-market signal analysis + simple tests')
print('=' * 100)

for pn, (start, end) in [('Train', ('2021-01-01', '2025-06-30')),
                          ('Val', ('2025-07-01', '2026-03-28'))]:
    print(f'\n=== {pn}: Downloading cross-market data ===')
    signals = download_cross_market(start, end)
    print(f'  Got {len(signals)} days of cross-market signals')

    # Analyze: on days when SOX dropped > 2%, what happened to Taiwan semi next day?
    mm = reconstruct_market_history(start, end)

    # Correlation analysis
    twii_rets = []
    sox_rets = []
    us10y_chgs = []
    tsm_chgs = []
    korea_chgs = []

    for d, m in sorted(mm.items()):
        twii_chg = m.get('twii', {}).get('change_pct', 0)
        cross = signals.get(d, {})
        if cross.get('sox_1d_chg') is not None:
            twii_rets.append(twii_chg)
            sox_rets.append(cross.get('sox_1d_chg', 0))
            us10y_chgs.append(cross.get('us10y_20d_chg', 0))
            tsm_chgs.append(cross.get('tsm_1d_chg', 0))
            korea_chgs.append(cross.get('korea_1d_chg', 0))

    if twii_rets:
        from numpy import corrcoef
        print(f'\n  Correlation with TWII daily return (n={len(twii_rets)}):')
        for name, vals in [('SOX_1d', sox_rets), ('US10Y_20d', us10y_chgs),
                           ('TSM_ADR', tsm_chgs), ('Korea_1d', korea_chgs)]:
            corr = corrcoef(twii_rets, vals)[0, 1]
            print(f'    {name:>12s}: corr = {corr:+.3f}')

    # Simple test: SOX 5d momentum as buy/sell signal
    # Count: when SOX 5d ret > 5%, TWII next day up/down
    sox_bull_twii = []
    sox_bear_twii = []
    for d, m in sorted(mm.items()):
        twii_chg = m.get('twii', {}).get('change_pct', 0)
        cross = signals.get(d, {})
        sox_5d = cross.get('sox_5d_ret')
        if sox_5d is not None:
            if sox_5d > 3:
                sox_bull_twii.append(twii_chg)
            elif sox_5d < -3:
                sox_bear_twii.append(twii_chg)

    if sox_bull_twii:
        print(f'\n  SOX 5d > +3% ({len(sox_bull_twii)} days): TWII avg={np.mean(sox_bull_twii)*100:+.2f}%')
    if sox_bear_twii:
        print(f'  SOX 5d < -3% ({len(sox_bear_twii)} days): TWII avg={np.mean(sox_bear_twii)*100:+.2f}%')

print('\nDone!')
