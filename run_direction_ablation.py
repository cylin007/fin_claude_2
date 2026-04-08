#!/usr/bin/env python3
"""
Direction 2+3+1 Ablation:
  2: Weak時停買 (strict filter mode)
  3: 週線趨勢過濾 (weekly MA20>MA60)
  1: 資金配置 60/40 (post-hoc simulation)
"""

import sys, os, time, warnings
import pandas as pd
import numpy as np
import yfinance as yf

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from group_backtest import (
    run_group_backtest, reconstruct_market_history,
    INDUSTRY_CONFIGS, MIN_DATA_DAYS
)
from industry_manager import get_stocks_by_industry
from stock_utils import batch_download_stocks

INDUSTRY = '半導體業'
INITIAL_CAPITAL = 900_000
BUDGET_PER_TRADE = 25_000
EXEC_MODE = 'next_open'
BASE_CONFIG = INDUSTRY_CONFIGS[INDUSTRY]['config'].copy()

CONFIGS = {
    'baseline': {
        'desc': '現狀 (relaxed, 只擋bear)',
        'overrides': {},
    },
    # === 方向 2: Weak 時停買 ===
    'D2_strict': {
        'desc': '方向2: strict (weak+bear+crash全擋)',
        'overrides': {
            'market_filter_mode': 'strict',
        },
    },
    'D2_moderate': {
        'desc': '方向2: moderate (bear+crash擋, weak放行)',
        'overrides': {
            'market_filter_mode': 'moderate',
        },
    },
    # === 方向 3: 週線過濾 ===
    'D3_weekly': {
        'desc': '方向3: 週線過濾 (週MA20<MA60停買)',
        'overrides': {
            'enable_weekly_filter': True,
        },
    },
    'D3_weekly_relaxed': {
        'desc': '方向3+relaxed: 週線+日線relaxed',
        'overrides': {
            'enable_weekly_filter': True,
            # market_filter_mode stays 'relaxed' (default from BASE_CONFIG)
        },
    },
    # === 方向 2+3 組合 ===
    'D23_strict_weekly': {
        'desc': '方向2+3: strict + 週線',
        'overrides': {
            'market_filter_mode': 'strict',
            'enable_weekly_filter': True,
        },
    },
    'D23_moderate_weekly': {
        'desc': '方向2+3: moderate + 週線',
        'overrides': {
            'market_filter_mode': 'moderate',
            'enable_weekly_filter': True,
        },
    },
}

PERIODS = {
    'Train_2022': ('2022-01-01', '2025-06-30'),  # 公平比較 (無2021假高峰)
    'Training':   ('2021-01-01', '2025-06-30'),   # 完整 training
    'Validation': ('2025-07-01', '2026-03-27'),
}


def _download_0050(start, end):
    try:
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        try:
            df = yf.download('0050.TW', start=start, end=end, progress=False)
        finally:
            sys.stderr = old_stderr
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def simulate_direction1(strategy_ec, etf_df, initial_capital, ratio_strategy=0.6):
    """方向1: 資金配置 — ratio_strategy% 策略 + (1-ratio_strategy)% 0050 DCA"""
    cap_strat = initial_capital * ratio_strategy
    cap_0050 = initial_capital * (1 - ratio_strategy)

    dates = [e['date'] for e in strategy_ec]
    strat_navs = [cap_strat + e['equity'] * ratio_strategy for e in strategy_ec]

    # 0050 DCA: 每月投入固定金額
    first_ts = pd.Timestamp(dates[0])
    last_ts = pd.Timestamp(dates[-1])
    n_months = max(1, (last_ts.year - first_ts.year) * 12 + (last_ts.month - first_ts.month) + 1)
    monthly = cap_0050 / n_months

    etf_shares = 0
    etf_cash = 0
    invested = 0
    last_buy_month = None
    combo_navs = []

    for i, date_str in enumerate(dates):
        ts = pd.Timestamp(date_str)
        month_key = (ts.year, ts.month)

        # Monthly DCA buy
        if month_key != last_buy_month:
            etf_cash += monthly
            invested += monthly
            valid = etf_df.index[etf_df.index <= ts]
            if len(valid) > 0:
                price = float(etf_df.loc[valid[-1], 'Close'])
                if price > 0:
                    new_shares = etf_cash / price
                    etf_shares += new_shares
                    etf_cash = 0
            last_buy_month = month_key

        # Current 0050 value
        valid = etf_df.index[etf_df.index <= ts]
        if len(valid) > 0 and etf_shares > 0:
            cur_price = float(etf_df.loc[valid[-1], 'Close'])
            etf_val = etf_shares * cur_price + etf_cash
        else:
            etf_val = etf_cash + invested  # fallback

        combo_nav = strat_navs[i] + etf_val
        combo_navs.append(combo_nav)

    # Metrics
    combo_arr = np.array(combo_navs)
    peak = np.maximum.accumulate(combo_arr)
    dd = (combo_arr - peak) / peak
    mdd = abs(dd.min()) * 100

    total_ret = (combo_arr[-1] / initial_capital - 1) * 100

    daily_rets = np.diff(combo_arr) / combo_arr[:-1]
    sharpe = (np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)) if np.std(daily_rets) > 0 else 0

    n_days = (last_ts - first_ts).days
    years = n_days / 365.25
    cagr = ((combo_arr[-1] / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    calmar = cagr / mdd if mdd > 0 else 0

    return {
        'total_return_pct': total_ret,
        'sharpe_ratio': sharpe,
        'mdd_pct': mdd,
        'calmar_ratio': calmar,
        'final_nav': combo_arr[-1],
    }


def run_ablation():
    stocks = get_stocks_by_industry(INDUSTRY)
    print(f"{INDUSTRY}: {len(stocks)} stocks\n")

    # Pre-download 0050 for Direction 1
    print("Downloading 0050.TW for Direction 1...")
    etf_df = _download_0050('2020-06-01', '2026-04-01')
    print(f"  0050.TW: {len(etf_df)} trading days\n")

    all_period_results = {}

    for period_name, (start_date, end_date) in PERIODS.items():
        print(f"\n{'='*80}")
        print(f"  📅 {period_name}: {start_date} ~ {end_date}")
        print(f"{'='*80}")

        market_map = reconstruct_market_history(start_date, end_date)
        dl_start = (pd.Timestamp(start_date) - pd.Timedelta(days=250)).strftime('%Y-%m-%d')
        dl_end = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        preloaded_data, _ = batch_download_stocks(
            stocks, dl_start, dl_end, min_data_days=MIN_DATA_DAYS)
        print(f"  Valid stocks: {len(preloaded_data)}\n")

        period_results = {}
        for config_name, cfg_entry in CONFIGS.items():
            run_config = BASE_CONFIG.copy()
            run_config.update(cfg_entry['overrides'])

            t0 = time.time()
            result = run_group_backtest(
                stock_list=stocks, start_date=start_date, end_date=end_date,
                budget_per_trade=BUDGET_PER_TRADE, market_map=market_map,
                exec_mode=EXEC_MODE, config_override=run_config,
                initial_capital=INITIAL_CAPITAL, preloaded_data=preloaded_data,
            )
            elapsed = time.time() - t0

            if result:
                period_results[config_name] = result
                ret = result.get('total_return_pct', 0)
                sharpe = result.get('sharpe_ratio', 0)
                mdd = result.get('mdd_pct', 0)
                calmar = result.get('calmar_ratio', 0)
                tl = result.get('trade_log', [])
                sells = [t for t in tl if t['type'] == 'SELL' and t.get('profit') is not None]
                gw = sum(t['profit'] for t in sells if t['profit'] > 0)
                gl = abs(sum(t['profit'] for t in sells if t['profit'] <= 0))
                pf = gw / gl if gl > 0 else 999

                print(f"  {config_name:>20s}: Ret {ret:+7.1f}% | Shrp {sharpe:5.2f} | "
                      f"MDD {mdd:5.1f}% | Clmr {calmar:5.2f} | PF {pf:4.2f} | "
                      f"⏱{elapsed:.0f}s  — {cfg_entry['desc']}")

                # === 方向 1: 60/40 資金配置 ===
                if result.get('equity_curve') and not etf_df.empty:
                    for ratio in [0.6, 0.7]:
                        d1 = simulate_direction1(result['equity_curve'], etf_df,
                                                 INITIAL_CAPITAL, ratio)
                        tag = f"D1_{int(ratio*100)}_{config_name}"
                        period_results[tag] = d1
                        print(f"  {tag:>20s}: Ret {d1['total_return_pct']:+7.1f}% | Shrp {d1['sharpe_ratio']:5.2f} | "
                              f"MDD {d1['mdd_pct']:5.1f}% | Clmr {d1['calmar_ratio']:5.2f} |       "
                              f"— 方向1: {int(ratio*100)}%策略+{int((1-ratio)*100)}%0050")

        all_period_results[period_name] = period_results

    # === Summary (Train_2022 vs Validation) ===
    for compare_train in ['Train_2022', 'Training']:
        print(f"\n\n{'='*110}")
        print(f"  📊 SUMMARY ({compare_train} vs Validation)")
        print(f"{'='*110}")
        header = f"{'Config':>24s} | {compare_train+' Ret':>12s} {'Shrp':>5s} {'MDD':>6s} {'Clmr':>5s} | {'Val Ret':>8s} {'Shrp':>5s} {'MDD':>6s} {'Clmr':>5s}"
        print(header)
        print("-" * len(header))

        t_results = all_period_results.get(compare_train, {})
        v_results = all_period_results.get('Validation', {})

        # Only show configs that exist in both
        for name in list(CONFIGS.keys()):
            for key in [name] + [f'D1_60_{name}', f'D1_70_{name}']:
                t = t_results.get(key)
                v = v_results.get(key)
                if t and v:
                    t_ret = t.get('total_return_pct', t.get('roi', 0))
                    t_shrp = t.get('sharpe_ratio', 0)
                    t_mdd = t.get('mdd_pct', 0)
                    t_clmr = t.get('calmar_ratio', 0)
                    v_ret = v.get('total_return_pct', v.get('roi', 0))
                    v_shrp = v.get('sharpe_ratio', 0)
                    v_mdd = v.get('mdd_pct', 0)
                    v_clmr = v.get('calmar_ratio', 0)
                    print(f"{key:>24s} | {t_ret:+11.1f}% {t_shrp:5.2f} {t_mdd:5.1f}% {t_clmr:5.2f} | "
                          f"{v_ret:+7.1f}% {v_shrp:5.2f} {v_mdd:5.1f}% {v_clmr:5.2f}")


if __name__ == '__main__':
    run_ablation()
