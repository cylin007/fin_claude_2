#!/usr/bin/env python3
"""
Full Backtest: L3+panic strategy on semiconductors (2022-01-01 ~ 2026-03-13)
- Strategy equity curve
- Benchmark 1: 0050 DCA (Dollar-Cost Averaging)
- Benchmark 2: 0050 signal mirror (buy 0050 on strategy signal dates)
- Output: full_backtest_equity.png, full_backtest_trade_log.csv, performance table
"""

import sys
import os
import warnings
import time
import numpy as np
import pandas as pd
import yfinance as yf

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from group_backtest import run_group_backtest, reconstruct_market_history, INDUSTRY_CONFIGS
from industry_manager import get_stocks_by_industry
from stock_utils import batch_download_stocks

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# Parameters
# ==========================================
START_DATE = '2022-01-01'
END_DATE = '2026-03-13'
INITIAL_CAPITAL = 900_000
BUDGET_PER_TRADE = 25_000
INDUSTRY = '半導體業'
EXEC_MODE = 'next_open'  # 'next_open' / 'same_close' / 'close_open'

SLIPPAGE_PCT = 0.003
COMMISSION_RATE = 0.001425
COMMISSION_DISCOUNT = 0.6
ETF_TAX_RATE = 0.001

# L3+panic+SM config (全部已整合至 INDUSTRY_CONFIGS)
STRATEGY_CONFIG = INDUSTRY_CONFIGS.get(INDUSTRY, {}).get('config', {})


# ==========================================
# 0050 helpers
# ==========================================
def _download_0050(start_date, end_date):
    dl_start = (pd.Timestamp(start_date) - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
    dl_end = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    try:
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        try:
            df = yf.download('0050.TW', start=dl_start, end=dl_end, progress=False)
        finally:
            sys.stderr = old_stderr
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def _get_etf_prices(etf_df, ts):
    valid = etf_df.index[etf_df.index <= ts]
    if len(valid) == 0:
        return None, None
    row = etf_df.loc[valid[-1]]
    return float(row['Open']), float(row['Close'])


def _calc_0050_dca(equity_curve, etf_df, initial_capital):
    dates_all = [ec['date'] for ec in equity_curve]
    first_ts = pd.Timestamp(dates_all[0])
    last_ts = pd.Timestamp(dates_all[-1])
    n_months = max(1, (last_ts.year - first_ts.year) * 12 + (last_ts.month - first_ts.month) + 1)
    monthly_amount = initial_capital / n_months
    values = []
    cash = float(initial_capital)
    shares = 0
    last_buy_month = None
    for ec in equity_curve:
        ts = pd.Timestamp(ec['date'])
        open_p, close_p = _get_etf_prices(etf_df, ts)
        if open_p is None:
            values.append(cash + shares * 0)
            continue
        ym = (ts.year, ts.month)
        if ym != last_buy_month and cash >= monthly_amount:
            exec_price = open_p * (1 + SLIPPAGE_PCT)
            buy_shares = int(monthly_amount / exec_price)
            if buy_shares > 0:
                cost = buy_shares * exec_price
                fee = max(1, int(cost * COMMISSION_RATE * COMMISSION_DISCOUNT))
                if cost + fee <= cash:
                    cash -= (cost + fee)
                    shares += buy_shares
                    last_buy_month = ym
        values.append(cash + shares * close_p)
    return values, n_months, monthly_amount


def _calc_0050_mirror(equity_curve, etf_df, initial_capital):
    values = []
    cash = float(initial_capital)
    shares = 0
    prev_positions = 0
    for ec in equity_curve:
        n_pos = ec['positions']
        ts = pd.Timestamp(ec['date'])
        open_p, close_p = _get_etf_prices(etf_df, ts)
        if open_p is None:
            values.append(cash)
            prev_positions = n_pos
            continue
        new_buys = max(0, n_pos - prev_positions)
        if new_buys > 0:
            _mirror_pct = STRATEGY_CONFIG.get('budget_pct', 0)
            if _mirror_pct > 0:
                _mirror_nav = cash + shares * close_p
                buy_amount = int(_mirror_nav * _mirror_pct / 100) * new_buys
            else:
                buy_amount = BUDGET_PER_TRADE * new_buys
            exec_price = open_p * (1 + SLIPPAGE_PCT)
            buy_shares = int(buy_amount / exec_price)
            if buy_shares > 0:
                cost = buy_shares * exec_price
                fee = max(1, int(cost * COMMISSION_RATE * COMMISSION_DISCOUNT))
                if cost + fee <= cash:
                    cash -= (cost + fee)
                    shares += buy_shares
        if n_pos < prev_positions and prev_positions > 0 and shares > 0:
            sell_ratio = (prev_positions - n_pos) / prev_positions
            sell_shares = int(shares * sell_ratio)
            if sell_shares > 0:
                exec_price = open_p * (1 - SLIPPAGE_PCT)
                proceeds = sell_shares * exec_price
                fee = max(1, int(proceeds * COMMISSION_RATE * COMMISSION_DISCOUNT))
                tax = int(proceeds * ETF_TAX_RATE)
                cash += (proceeds - fee - tax)
                shares -= sell_shares
        values.append(cash + shares * close_p)
        prev_positions = n_pos
    return values


def _calc_metrics(values, initial_capital, dates):
    final = values[-1]
    ret = (final - initial_capital) / initial_capital * 100
    days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
    yrs = max(days / 365.25, 0.1)
    cagr = ((final / initial_capital) ** (1 / yrs) - 1) * 100 if final > 0 else 0
    d_rets = []
    for i in range(1, len(values)):
        if values[i-1] > 0:
            d_rets.append(values[i] / values[i-1] - 1)
    if d_rets and np.std(d_rets) > 0:
        rf_d = 0.015 / 245
        exc = [r - rf_d for r in d_rets]
        sharpe = (np.mean(exc) * 245) / (np.std(d_rets) * np.sqrt(245))
    else:
        sharpe = 0
    pk = values[0]
    max_dd = 0
    for v in values:
        if v > pk:
            pk = v
        dd = (pk - v) / pk * 100 if pk > 0 else 0
        if dd > max_dd:
            max_dd = dd
    calmar = cagr / max_dd if max_dd > 0 else 0
    return {'return': ret, 'cagr': cagr, 'sharpe': sharpe, 'mdd': max_dd, 'calmar': calmar, 'final': final}


# ==========================================
# Plotting
# ==========================================
def plot_equity(dates, strategy_vals, dca_vals, mirror_vals,
                initial_capital, strat_m, dca_m, mirror_m, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang TC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    pd_dates = pd.to_datetime(dates)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                    gridspec_kw={'height_ratios': [4, 1.5]},
                                    sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # --- Panel 1: Equity curves ---
    ax1.plot(pd_dates, strategy_vals, color='#1f77b4', linewidth=2.0,
             label=f"L3+panic (Return {strat_m['return']:+.1f}%, Sharpe {strat_m['sharpe']:.2f}, "
                   f"MDD {strat_m['mdd']:.1f}%)")
    ax1.plot(pd_dates, dca_vals, color='#ff7f0e', linewidth=1.5, linestyle='--', alpha=0.8,
             label=f"0050 DCA (Return {dca_m['return']:+.1f}%, Sharpe {dca_m['sharpe']:.2f}, "
                   f"MDD {dca_m['mdd']:.1f}%)")
    ax1.plot(pd_dates, mirror_vals, color='#2ca02c', linewidth=1.5, linestyle='--', alpha=0.8,
             label=f"0050 Signal Mirror (Return {mirror_m['return']:+.1f}%, Sharpe {mirror_m['sharpe']:.2f}, "
                   f"MDD {mirror_m['mdd']:.1f}%)")

    ax1.axhline(y=initial_capital, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
    ax1.fill_between(pd_dates, initial_capital, strategy_vals,
                     where=[v >= initial_capital for v in strategy_vals],
                     color='green', alpha=0.05)
    ax1.fill_between(pd_dates, initial_capital, strategy_vals,
                     where=[v < initial_capital for v in strategy_vals],
                     color='red', alpha=0.05)

    ax1.set_ylabel('Portfolio Value (NTD)')
    ax1.set_title(
        f'L3+panic Strategy vs 0050 Benchmarks  [{START_DATE} ~ {END_DATE}]\n'
        f'Strategy: Sharpe={strat_m["sharpe"]:.2f} | CAGR={strat_m["cagr"]:.1f}% | '
        f'MDD={strat_m["mdd"]:.1f}% | Calmar={strat_m["calmar"]:.2f}',
        fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # --- Panel 2: Drawdown ---
    pk = strategy_vals[0]
    drawdowns = []
    for v in strategy_vals:
        if v > pk:
            pk = v
        drawdowns.append(-((pk - v) / pk * 100) if pk > 0 else 0)

    ax2.fill_between(pd_dates, 0, drawdowns, color='#ff6b6b', alpha=0.4)
    ax2.plot(pd_dates, drawdowns, color='#cc0000', linewidth=0.8)
    ax2.set_ylabel('Drawdown %')
    ax2.set_ylim(min(drawdowns) * 1.2 if drawdowns else -10, 2)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    # annotate MDD
    min_dd_idx = np.argmin(drawdowns)
    ax2.annotate(f'MDD: {drawdowns[min_dd_idx]:.1f}%',
                 xy=(pd_dates[min_dd_idx], drawdowns[min_dd_idx]),
                 xytext=(20, 10), textcoords='offset points',
                 fontsize=8, color='darkred',
                 arrowprops=dict(arrowstyle='->', color='darkred', lw=0.8))

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nEquity chart saved: {output_path}")


# ==========================================
# Main
# ==========================================
def main():
    t0 = time.time()

    _mode_map = {'next_open': 'T+1 開盤', 'same_close': '當日收盤', 'close_open': '收盤+開盤雙買'}
    print("=" * 80)
    print(f"  Full Backtest: L3+panic Strategy")
    print(f"  Industry: {INDUSTRY}")
    print(f"  Period: {START_DATE} ~ {END_DATE}")
    print(f"  Capital: ${INITIAL_CAPITAL:,} | Budget: ${BUDGET_PER_TRADE:,}")
    print(f"  Exec Mode: {EXEC_MODE} ({_mode_map.get(EXEC_MODE, EXEC_MODE)})")
    print("=" * 80)

    # 1. Market history
    print("\nReconstructing market history...")
    market_map = reconstruct_market_history(START_DATE, END_DATE)
    if not market_map:
        print("ERROR: Failed to build market history")
        return

    # 2. Stock list
    stocks = get_stocks_by_industry(INDUSTRY)
    print(f"{INDUSTRY}: {len(stocks)} stocks")

    # 3. Pre-download stock data
    print("Pre-downloading stock data...")
    dl_start = (pd.Timestamp(START_DATE) - pd.Timedelta(days=250)).strftime('%Y-%m-%d')
    _today = pd.Timestamp.today().normalize()
    dl_end = (max(_today, pd.Timestamp(END_DATE)) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    _force = ('--fresh' in sys.argv or '--refresh' in sys.argv or '-f' in sys.argv)
    preloaded_data, skipped = batch_download_stocks(
        stocks, dl_start, dl_end, min_data_days=20, force_refresh=_force)
    print(f"  Valid stocks: {len(preloaded_data)}")

    # 4. Run backtest
    print(f"\nRunning backtest...")
    result = run_group_backtest(
        stock_list=stocks,
        start_date=START_DATE,
        end_date=END_DATE,
        budget_per_trade=BUDGET_PER_TRADE,
        market_map=market_map,
        exec_mode=EXEC_MODE,
        config_override=STRATEGY_CONFIG,
        initial_capital=INITIAL_CAPITAL,
        preloaded_data=preloaded_data,
    )

    if result is None:
        print("ERROR: Backtest failed")
        return

    # 5. Extract equity curve
    ec = result['equity_curve']
    dates = [e['date'] for e in ec]
    strategy_vals = [INITIAL_CAPITAL + e['equity'] for e in ec]
    positions_count = [e['positions'] for e in ec]

    # 6. Download 0050
    print("\nDownloading 0050.TW benchmark data...")
    etf_df = _download_0050(START_DATE, END_DATE)
    if etf_df.empty:
        print("WARNING: 0050 download failed, skipping benchmarks")
        return
    print(f"  0050.TW: {len(etf_df)} trading days")

    # 7. Compute benchmarks
    dca_vals, n_months, monthly_amount = _calc_0050_dca(ec, etf_df, INITIAL_CAPITAL)
    mirror_vals = _calc_0050_mirror(ec, etf_df, INITIAL_CAPITAL)
    print(f"  DCA: {n_months} months, ${monthly_amount:,.0f}/month")

    # 8. Metrics
    strat_m = _calc_metrics(strategy_vals, INITIAL_CAPITAL, dates)
    dca_m = _calc_metrics(dca_vals, INITIAL_CAPITAL, dates)
    mirror_m = _calc_metrics(mirror_vals, INITIAL_CAPITAL, dates)

    # Engine metrics
    eng_sharpe = result.get('sharpe_ratio', 0)
    eng_cagr = result.get('cagr', 0)
    eng_mdd = result.get('mdd_pct', 0)
    eng_calmar = result.get('calmar_ratio', 0)
    eng_wr = result.get('win_rate', 0)
    eng_trades = result.get('trades', 0)
    eng_wins = result.get('wins', 0)
    eng_losses = result.get('losses', 0)
    eng_ret = result.get('total_return_pct', 0)
    eng_final = result.get('final_total_value', 0)
    eng_cash = result.get('final_cash', 0)
    eng_stock = result.get('final_stock_value', 0)
    eng_realized = result.get('realized', 0)
    eng_unrealized = result.get('unrealized', 0)
    eng_fees = result.get('fees', 0)
    eng_bt_days = result.get('backtest_days', 0)
    eng_bt_years = result.get('backtest_years', 0)

    # Profit factor
    tl = result.get('trade_log', [])
    sells = [t for t in tl if t['type'] == 'SELL' and t.get('profit') is not None]
    gross_win = sum(t['profit'] for t in sells if t['profit'] > 0)
    gross_loss = abs(sum(t['profit'] for t in sells if t['profit'] <= 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    # ==========================================
    # Print performance table
    # ==========================================
    print(f"\n{'='*80}")
    print(f"  PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"  Period:          {START_DATE} ~ {END_DATE} ({eng_bt_days} days, {eng_bt_years:.2f} years)")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"  Final Value:     ${eng_final:,.0f}")
    print(f"  Final Cash:      ${eng_cash:,.0f}")
    print(f"  Final Stock Val: ${eng_stock:,.0f}")
    print(f"{'─'*80}")
    print(f"  Total Return:    {eng_ret:+.2f}%")
    print(f"  CAGR:            {eng_cagr:+.2f}%")
    print(f"  Sharpe Ratio:    {eng_sharpe:.2f}")
    print(f"  MDD:             {eng_mdd:.2f}%")
    print(f"  Calmar Ratio:    {eng_calmar:.2f}")
    print(f"{'─'*80}")
    print(f"  Trades:          {eng_trades}")
    print(f"  Wins / Losses:   {eng_wins} / {eng_losses}")
    print(f"  Win Rate:        {eng_wr:.1f}%")
    print(f"  Profit Factor:   {pf:.2f}")
    print(f"{'─'*80}")
    print(f"  Realized P&L:    ${eng_realized:+,.0f}")
    print(f"  Unrealized P&L:  ${eng_unrealized:+,.0f}")
    print(f"  Total Fees:      ${eng_fees:,.0f}")
    print(f"{'='*80}")

    # ==========================================
    # Benchmark comparison table
    # ==========================================
    print(f"\n{'='*80}")
    print(f"  BENCHMARK COMPARISON")
    print(f"{'='*80}")
    print(f"  {'':20s} {'Return':>10s} {'CAGR':>9s} {'Sharpe':>8s} {'MDD':>8s} {'Calmar':>9s} {'Final Value':>14s}")
    print(f"  {'─'*78}")
    for label, m in [('L3+panic Strategy', strat_m),
                     ('0050 DCA', dca_m),
                     ('0050 Signal Mirror', mirror_m)]:
        print(f"  {label:20s} {m['return']:>+9.1f}% {m['cagr']:>+8.1f}% "
              f"{m['sharpe']:>7.2f} {m['mdd']:>7.1f}% {m['calmar']:>8.2f} "
              f"${m['final']:>12,.0f}")
    print(f"  {'─'*78}")

    # vs 0050 DCA
    delta_ret = strat_m['return'] - dca_m['return']
    delta_sharpe = strat_m['sharpe'] - dca_m['sharpe']
    delta_mdd = strat_m['mdd'] - dca_m['mdd']
    print(f"\n  Strategy vs DCA 0050:")
    print(f"    Return delta:  {delta_ret:+.1f}%")
    print(f"    Sharpe delta:  {delta_sharpe:+.2f}")
    print(f"    MDD delta:     {delta_mdd:+.1f}%")
    print(f"{'='*80}")

    # ==========================================
    # Save trade log CSV
    # ==========================================
    if tl:
        tl_rows = []
        for t in tl:
            tl_rows.append({
                'Date': t['date'],
                'Ticker': t['ticker'],
                'Name': t.get('name', ''),
                'Type': t['type'],
                'Price': round(t['price'], 2),
                'Shares': t['shares'],
                'Fee': int(t.get('fee', 0)),
                'Profit': int(t['profit']) if t['profit'] is not None else '',
                'ROI%': round(t['roi'], 2) if t.get('roi') is not None else '',
                'Note': t.get('note', ''),
            })
        tl_df = pd.DataFrame(tl_rows)
        _suffix = f'_{EXEC_MODE}' if EXEC_MODE != 'next_open' else ''
        _date_tag = f'_{START_DATE}_{END_DATE}'
        _output_dir = os.path.join(_BASE_DIR, 'output')
        os.makedirs(_output_dir, exist_ok=True)
        tl_path = os.path.join(_output_dir, f'full_backtest_trade_log{_date_tag}{_suffix}.csv')
        tl_df.to_csv(tl_path, index=False, encoding='utf-8-sig')
        print(f"\nTrade log saved: {tl_path}  ({len(tl_df)} records)")

    # ==========================================
    # Plot equity curve
    # ==========================================
    _suffix2 = f'_{EXEC_MODE}' if EXEC_MODE != 'next_open' else ''
    _date_tag2 = f'_{START_DATE}_{END_DATE}'
    _output_dir2 = os.path.join(_BASE_DIR, 'output')
    os.makedirs(_output_dir2, exist_ok=True)
    png_path = os.path.join(_output_dir2, f'full_backtest_equity{_date_tag2}{_suffix2}.png')
    plot_equity(dates, strategy_vals, dca_vals, mirror_vals,
                INITIAL_CAPITAL, strat_m, dca_m, mirror_m, png_path)

    elapsed = time.time() - t0
    print(f"\nDone! Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    # CLI 參數解析: python3 run_full_backtest.py [start_date] [end_date] [exec_mode]
    # 範例:
    #   python3 run_full_backtest.py                          # 預設全部
    #   python3 run_full_backtest.py 2024-01-01 2026-03-13    # 指定日期
    #   python3 run_full_backtest.py 2024-01-01 2026-03-13 close_open  # 日期+模式
    #   python3 run_full_backtest.py close_open               # 只指定模式
    _dates = []
    for arg in sys.argv[1:]:
        if arg in ('next_open', 'same_close', 'close_open'):
            EXEC_MODE = arg
        elif len(arg) == 10 and arg[4] == '-' and arg[7] == '-':
            _dates.append(arg)
    if len(_dates) >= 2:
        START_DATE, END_DATE = _dates[0], _dates[1]
    elif len(_dates) == 1:
        START_DATE = _dates[0]
    if _dates:
        print(f"[CLI] period = {START_DATE} ~ {END_DATE}")
    if EXEC_MODE != 'next_open':
        print(f"[CLI] exec_mode = {EXEC_MODE}")
    main()
