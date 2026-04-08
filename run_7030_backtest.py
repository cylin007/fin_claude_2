#!/usr/bin/env python3
"""
70/30 Allocation Backtest:
  70% → 半導體動能策略 (initial_capital=630K)
  30% → 0050 定期定額 (270K, 每月第一交易日買入)

Output:
  - Equity curve PNG (baseline 900K vs 70/30 combo vs 0050 DCA 900K)
  - Trade log CSV
  - Performance summary
"""

import sys, os, warnings, time
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
TOTAL_CAPITAL = 900_000
STRATEGY_RATIO = 0.70
DCA_RATIO = 0.30

STRATEGY_CAPITAL = int(TOTAL_CAPITAL * STRATEGY_RATIO)  # 630,000
DCA_CAPITAL = int(TOTAL_CAPITAL * DCA_RATIO)              # 270,000

START_DATE = '2022-01-01'
END_DATE = '2026-03-27'
INDUSTRY = '半導體業'
EXEC_MODE = 'next_open'
BUDGET_PER_TRADE = int(25_000 * STRATEGY_RATIO / 1.0)  # scale down proportionally

STRATEGY_CONFIG = INDUSTRY_CONFIGS.get(INDUSTRY, {}).get('config', {})


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


def simulate_0050_dca(dates, etf_df, total_dca_capital):
    """
    模擬 0050 定期定額：
    - 每月第一個交易日買入
    - 月投入金額 = total_dca_capital / 回測總月數
    - 回傳每日 NAV 序列
    """
    first_ts = pd.Timestamp(dates[0])
    last_ts = pd.Timestamp(dates[-1])
    n_months = max(1, (last_ts.year - first_ts.year) * 12 + (last_ts.month - first_ts.month) + 1)
    monthly_amount = total_dca_capital / n_months

    shares = 0
    cash_remaining = 0  # 未投入的部分
    invested = 0
    last_buy_month = None
    navs = []
    buy_log = []

    for date_str in dates:
        ts = pd.Timestamp(date_str)
        month_key = (ts.year, ts.month)

        # 每月第一個交易日買入
        if month_key != last_buy_month:
            cash_remaining += monthly_amount
            invested += monthly_amount
            valid = etf_df.index[etf_df.index <= ts]
            if len(valid) > 0:
                price = float(etf_df.loc[valid[-1], 'Close'])
                if price > 0:
                    # 用整股計算 (0050 目前約 150-200)
                    can_buy_shares = int(cash_remaining / price)
                    if can_buy_shares > 0:
                        cost = can_buy_shares * price
                        shares += can_buy_shares
                        cash_remaining -= cost
                        buy_log.append({
                            'date': date_str, 'price': price,
                            'shares': can_buy_shares, 'total_shares': shares,
                            'monthly_amount': monthly_amount,
                        })
            last_buy_month = month_key

        # 每日估值
        valid = etf_df.index[etf_df.index <= ts]
        if len(valid) > 0 and shares > 0:
            cur_price = float(etf_df.loc[valid[-1], 'Close'])
            nav = shares * cur_price + cash_remaining
        else:
            nav = cash_remaining

        navs.append(nav)

    return navs, n_months, monthly_amount, buy_log


def calc_metrics(navs, initial_capital, dates):
    arr = np.array(navs, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak
    mdd = abs(dd.min()) * 100

    total_ret = (arr[-1] / initial_capital - 1) * 100

    daily_rets = np.diff(arr) / arr[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    sharpe = (np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)) if len(daily_rets) > 0 and np.std(daily_rets) > 0 else 0

    n_days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
    years = n_days / 365.25
    cagr = ((arr[-1] / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    calmar = cagr / mdd if mdd > 0 else 0

    return {
        'total_return_pct': total_ret,
        'cagr': cagr,
        'sharpe': sharpe,
        'mdd': mdd,
        'calmar': calmar,
        'final_value': arr[-1],
    }


def plot_equity(dates, baseline_navs, combo_navs, dca_navs,
                baseline_m, combo_m, dca_m,
                start_date, end_date, output_path):
    """Plot equity curves with performance table"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("WARNING: matplotlib not available, skipping plot")
        return

    date_objs = [pd.Timestamp(d) for d in dates]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                     gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'70/30 Allocation: Strategy vs 0050 DCA  [{start_date} ~ {end_date}]',
                 fontsize=14, fontweight='bold')

    # Equity curves
    ax1.plot(date_objs, baseline_navs, label=f'100% Strategy (Sharpe {baseline_m["sharpe"]:.2f})',
             color='#2196F3', linewidth=1.5, alpha=0.8)
    ax1.plot(date_objs, combo_navs, label=f'70/30 Combo (Sharpe {combo_m["sharpe"]:.2f})',
             color='#FF5722', linewidth=2.0)
    ax1.plot(date_objs, dca_navs, label=f'100% 0050 DCA (Sharpe {dca_m["sharpe"]:.2f})',
             color='#4CAF50', linewidth=1.5, alpha=0.8, linestyle='--')

    ax1.axhline(y=TOTAL_CAPITAL, color='gray', linestyle=':', alpha=0.5, label=f'Initial ${TOTAL_CAPITAL:,}')
    ax1.set_ylabel('Portfolio Value (NTD)')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Drawdown
    for navs, color, label in [
        (baseline_navs, '#2196F3', '100% Strategy'),
        (combo_navs, '#FF5722', '70/30 Combo'),
        (dca_navs, '#4CAF50', '0050 DCA'),
    ]:
        arr = np.array(navs)
        peak = np.maximum.accumulate(arr)
        dd = (arr - peak) / peak * 100
        ax2.fill_between(date_objs, dd, 0, alpha=0.3, color=color, label=label)
        ax2.plot(date_objs, dd, color=color, linewidth=0.8, alpha=0.6)

    ax2.set_ylabel('Drawdown %')
    ax2.set_xlabel('Date')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # Performance table as text
    table_text = (
        f"{'':>20s} {'Return':>8s} {'CAGR':>7s} {'Sharpe':>7s} {'MDD':>7s} {'Calmar':>7s} {'Final Value':>14s}\n"
        f"{'100% Strategy':>20s} {baseline_m['total_return_pct']:+7.1f}% {baseline_m['cagr']:+6.1f}% {baseline_m['sharpe']:6.2f} {baseline_m['mdd']:6.1f}% {baseline_m['calmar']:6.2f}  ${baseline_m['final_value']:>12,.0f}\n"
        f"{'70/30 Combo':>20s} {combo_m['total_return_pct']:+7.1f}% {combo_m['cagr']:+6.1f}% {combo_m['sharpe']:6.2f} {combo_m['mdd']:6.1f}% {combo_m['calmar']:6.2f}  ${combo_m['final_value']:>12,.0f}\n"
        f"{'100% 0050 DCA':>20s} {dca_m['total_return_pct']:+7.1f}% {dca_m['cagr']:+6.1f}% {dca_m['sharpe']:6.2f} {dca_m['mdd']:6.1f}% {dca_m['calmar']:6.2f}  ${dca_m['final_value']:>12,.0f}"
    )
    fig.text(0.12, -0.02, table_text, fontsize=9, fontfamily='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nEquity chart saved: {output_path}")


def main():
    # Parse CLI dates
    _dates = [a for a in sys.argv[1:] if not a.startswith('-') and '-' in a and len(a) == 10]
    global START_DATE, END_DATE
    if len(_dates) >= 2:
        START_DATE, END_DATE = _dates[0], _dates[1]
    elif len(_dates) == 1:
        START_DATE = _dates[0]
    print(f"{'='*80}")
    print(f"  70/30 Allocation Backtest")
    print(f"  {STRATEGY_RATIO*100:.0f}% Strategy (${STRATEGY_CAPITAL:,}) + {DCA_RATIO*100:.0f}% 0050 DCA (${DCA_CAPITAL:,})")
    print(f"  Period: {START_DATE} ~ {END_DATE}")
    print(f"  Total Capital: ${TOTAL_CAPITAL:,}")
    print(f"{'='*80}\n")

    # 1. Market map
    print("Reconstructing market history...")
    market_map = reconstruct_market_history(START_DATE, END_DATE)

    # 2. Stock data
    stocks = get_stocks_by_industry(INDUSTRY)
    print(f"\n{INDUSTRY}: {len(stocks)} stocks")
    dl_start = (pd.Timestamp(START_DATE) - pd.Timedelta(days=250)).strftime('%Y-%m-%d')
    dl_end = (pd.Timestamp(END_DATE) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    _force = ('--fresh' in sys.argv or '-f' in sys.argv)
    preloaded_data, _ = batch_download_stocks(stocks, dl_start, dl_end, min_data_days=20, force_refresh=_force)
    print(f"  Valid stocks: {len(preloaded_data)}")

    # 3. Download 0050
    print("\nDownloading 0050.TW...")
    etf_df = _download_0050(START_DATE, END_DATE)
    print(f"  0050.TW: {len(etf_df)} trading days")

    # ==========================================
    # Run 1: Baseline (100% strategy, 900K)
    # ==========================================
    print(f"\n{'─'*60}")
    print(f"  Run 1: Baseline (100% Strategy, ${TOTAL_CAPITAL:,})")
    print(f"{'─'*60}")
    result_baseline = run_group_backtest(
        stock_list=stocks, start_date=START_DATE, end_date=END_DATE,
        budget_per_trade=25_000, market_map=market_map,
        exec_mode=EXEC_MODE, config_override=STRATEGY_CONFIG,
        initial_capital=TOTAL_CAPITAL, preloaded_data=preloaded_data,
    )

    # ==========================================
    # Run 2: 70% Strategy (630K)
    # ==========================================
    print(f"\n{'─'*60}")
    print(f"  Run 2: 70% Strategy (${STRATEGY_CAPITAL:,})")
    print(f"{'─'*60}")
    result_70 = run_group_backtest(
        stock_list=stocks, start_date=START_DATE, end_date=END_DATE,
        budget_per_trade=BUDGET_PER_TRADE, market_map=market_map,
        exec_mode=EXEC_MODE, config_override=STRATEGY_CONFIG,
        initial_capital=STRATEGY_CAPITAL, preloaded_data=preloaded_data,
    )

    if not result_baseline or not result_70:
        print("ERROR: Backtest failed")
        return

    # ==========================================
    # Build equity curves
    # ==========================================
    ec_baseline = result_baseline['equity_curve']
    ec_70 = result_70['equity_curve']
    dates = [e['date'] for e in ec_baseline]

    # Baseline NAVs
    baseline_navs = [TOTAL_CAPITAL + e['equity'] for e in ec_baseline]

    # 70% strategy NAVs
    strat70_navs = [STRATEGY_CAPITAL + e['equity'] for e in ec_70]

    # 30% 0050 DCA
    dca30_navs, n_months, monthly_amt, dca_log = simulate_0050_dca(
        dates, etf_df, DCA_CAPITAL)

    # 100% 0050 DCA (benchmark)
    dca100_navs, _, monthly_100, _ = simulate_0050_dca(
        dates, etf_df, TOTAL_CAPITAL)

    # 70/30 combo
    combo_navs = [s + d for s, d in zip(strat70_navs, dca30_navs)]

    # ==========================================
    # Metrics
    # ==========================================
    baseline_m = calc_metrics(baseline_navs, TOTAL_CAPITAL, dates)
    combo_m = calc_metrics(combo_navs, TOTAL_CAPITAL, dates)
    dca_m = calc_metrics(dca100_navs, TOTAL_CAPITAL, dates)

    print(f"\n{'='*80}")
    print(f"  PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"  Period: {START_DATE} ~ {END_DATE} ({len(dates)} trading days)")
    print(f"  0050 DCA: {n_months} months, ${monthly_amt:,.0f}/month (30% pool)")
    print(f"{'─'*80}")
    fmt = "  {:<20s} {:>8s} {:>7s} {:>7s} {:>7s} {:>7s} {:>14s}"
    print(fmt.format('', 'Return', 'CAGR', 'Sharpe', 'MDD', 'Calmar', 'Final Value'))
    print(f"  {'─'*75}")

    for name, m in [('100% Strategy', baseline_m), ('70/30 Combo', combo_m), ('100% 0050 DCA', dca_m)]:
        print(f"  {name:<20s} {m['total_return_pct']:+7.1f}% {m['cagr']:+6.1f}% {m['sharpe']:6.2f} "
              f"{m['mdd']:6.1f}% {m['calmar']:6.2f}  ${m['final_value']:>12,.0f}")

    print(f"  {'─'*75}")
    print(f"  70/30 vs Baseline:  Return {combo_m['total_return_pct']-baseline_m['total_return_pct']:+.1f}%  "
          f"Sharpe {combo_m['sharpe']-baseline_m['sharpe']:+.2f}  "
          f"MDD {combo_m['mdd']-baseline_m['mdd']:+.1f}%  "
          f"Calmar {combo_m['calmar']-baseline_m['calmar']:+.2f}")
    print(f"  70/30 vs 0050 DCA:  Return {combo_m['total_return_pct']-dca_m['total_return_pct']:+.1f}%  "
          f"Sharpe {combo_m['sharpe']-dca_m['sharpe']:+.2f}  "
          f"MDD {combo_m['mdd']-dca_m['mdd']:+.1f}%")
    print(f"{'='*80}")

    # ==========================================
    # Output
    # ==========================================
    output_dir = os.path.join(_BASE_DIR, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Trade log (70% strategy)
    tl_path = os.path.join(output_dir, f'7030_trade_log_{START_DATE}_{END_DATE}.csv')
    tl = result_70.get('trade_log', [])
    if tl:
        pd.DataFrame(tl).to_csv(tl_path, index=False, encoding='utf-8-sig')
        print(f"\nTrade log saved: {tl_path} ({len(tl)} records)")

    # 0050 DCA log
    dca_path = os.path.join(output_dir, f'7030_0050_dca_log_{START_DATE}_{END_DATE}.csv')
    if dca_log:
        pd.DataFrame(dca_log).to_csv(dca_path, index=False, encoding='utf-8-sig')
        print(f"0050 DCA log saved: {dca_path} ({len(dca_log)} buys)")

    # Equity PNG
    png_path = os.path.join(output_dir, f'7030_equity_{START_DATE}_{END_DATE}.png')
    plot_equity(dates, baseline_navs, combo_navs, dca100_navs,
                baseline_m, combo_m, dca_m,
                START_DATE, END_DATE, png_path)


if __name__ == '__main__':
    main()
