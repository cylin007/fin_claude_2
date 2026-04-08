#!/usr/bin/env python3
"""
策略 C (中長期趨勢回檔) vs 策略 A (短線動能) 對比測試

Train: 2021-01-01 ~ 2025-06-30
Val:   2025-07-01 ~ 2026-03-28

同時跑多組參數變體，找最佳中長期策略設定
"""

import sys, os, time, warnings
import pandas as pd, numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from group_backtest import (run_group_backtest, reconstruct_market_history,
                            INDUSTRY_CONFIGS, MIN_DATA_DAYS)
from industry_manager import get_stocks_by_industry
from stock_utils import batch_download_stocks
from strategy_midterm import check_midterm_signal, MIDTERM_CONFIG

INDUSTRY = '半導體業'
INITIAL_CAPITAL = 900_000   # 與 run_full_backtest.py 一致
EXEC_MODE = 'next_open'
BASE_A = INDUSTRY_CONFIGS[INDUSTRY]['config'].copy()  # 策略 A (現有短線)

# ==========================================
# 策略 C 基礎設定 (覆蓋 backtest engine 的 position management)
# ==========================================
BASE_C = {
    # -- backtest engine 層 --
    'max_positions':         4,       # 4 檔集中持股
    'max_new_buy_per_day':   2,       # 每日最多買 2 檔
    'zombie_hold_days':      90,      # 90 天殭屍
    'zombie_net_range':      5.0,     # 殭屍淨利範圍
    'hard_stop_net':        -12,      # 硬停損 (engine 層備用，strategy 也有)
    'enable_zombie_cleanup': True,
    'enable_position_swap':  False,   # 中長期不做換股
    'max_add_per_stock':     1,       # 不加碼 (每檔只買一次)
    'budget_pct':            15.0,    # 每筆 15% NAV
    'cash_reserve_pct':      40.0,    # 40% 現金儲備

    # -- 關閉短線專用模組 --
    'enable_fish_tail':      False,
    'enable_breakout':       False,
    'enable_rs_filter':      False,
    'enable_sector_momentum': False,
    'enable_ewt_boost':      False,
    'enable_ewt_filter':     False,
    'enable_conviction_hold': False,
    'enable_regime_adaptive': False,
    'enable_peer_zscore':    False,
    'enable_weekly_filter':  False,   # 由 strategy_midterm 內部處理
    'enable_theme_boost':    False,
    'enable_quality_filter': False,
    'enable_dynamic_exposure': False,
    'enable_dyn_buy_limit':  False,
    'enable_dyn_stop':       False,
    'enable_vol_sizing':     False,
    'enable_profit_trailing': False,  # 由 strategy_midterm 內部處理
    'enable_trailing_stop':  False,
    'enable_pullback_buy':   False,
    'enable_dip_buy':        False,
    'market_filter_mode':    'off',   # 大盤過濾由 strategy_midterm 處理
    'min_rsi':               0,       # RSI 由 strategy_midterm 處理

    # -- 中長期策略專用參數 (傳給 check_midterm_signal) --
    **MIDTERM_CONFIG,
}

# ==========================================
# 測試矩陣
# ==========================================
CONFIGS = {
    # --- Baseline: 策略 A ---
    'A_baseline': {
        'desc': '策略A: 短線動能 (現狀)',
        'signal_func': None,
        'config': BASE_A,
        'budget': 25_000,
        'capital': INITIAL_CAPITAL,
    },

    # --- 策略 C 主力 ---
    'C_default': {
        'desc': '策略C: 中長期回檔 (預設)',
        'signal_func': check_midterm_signal,
        'config': BASE_C,
        'budget': int(INITIAL_CAPITAL * 0.15),  # 135K
        'capital': INITIAL_CAPITAL,
    },

    # --- 策略 C 變體 ---
    'C_no_weekly': {
        'desc': '策略C: 不看週線',
        'signal_func': check_midterm_signal,
        'config': {**BASE_C, 'mt_require_weekly': False},
        'budget': int(INITIAL_CAPITAL * 0.15),
        'capital': INITIAL_CAPITAL,
    },
    'C_wider': {
        'desc': '策略C: 回檔放寬 (-8%~+5%)',
        'signal_func': check_midterm_signal,
        'config': {**BASE_C, 'mt_pullback_lo': -8.0, 'mt_pullback_hi': 5.0},
        'budget': int(INITIAL_CAPITAL * 0.15),
        'capital': INITIAL_CAPITAL,
    },
    'C_5pos': {
        'desc': '策略C: 5檔 (每檔12%)',
        'signal_func': check_midterm_signal,
        'config': {**BASE_C, 'mt_max_positions': 5, 'max_positions': 5,
                   'mt_budget_pct': 12.0, 'budget_pct': 12.0, 'cash_reserve_pct': 40.0},
        'budget': int(INITIAL_CAPITAL * 0.12),
        'capital': INITIAL_CAPITAL,
    },
    'C_trail_30': {
        'desc': '策略C: 追蹤停利30%觸發/10%回撤',
        'signal_func': check_midterm_signal,
        'config': {**BASE_C, 'mt_trail_trigger': 30.0, 'mt_trail_drop': 10.0},
        'budget': int(INITIAL_CAPITAL * 0.15),
        'capital': INITIAL_CAPITAL,
    },
    'C_trail_50': {
        'desc': '策略C: 追蹤停利50%觸發/20%回撤',
        'signal_func': check_midterm_signal,
        'config': {**BASE_C, 'mt_trail_trigger': 50.0, 'mt_trail_drop': 20.0},
        'budget': int(INITIAL_CAPITAL * 0.15),
        'capital': INITIAL_CAPITAL,
    },
    'C_no_vol': {
        'desc': '策略C: 不看量縮',
        'signal_func': check_midterm_signal,
        'config': {**BASE_C, 'mt_vol_shrink_ratio': 999.0},
        'budget': int(INITIAL_CAPITAL * 0.15),
        'capital': INITIAL_CAPITAL,
    },
    'C_relax_bear': {
        'desc': '策略C: 弱勢也能買',
        'signal_func': check_midterm_signal,
        'config': {**BASE_C, 'mt_block_bear': False},
        'budget': int(INITIAL_CAPITAL * 0.15),
        'capital': INITIAL_CAPITAL,
    },
    'C_ma60_1day': {
        'desc': '策略C: 跌破MA60隔天就賣',
        'signal_func': check_midterm_signal,
        'config': {**BASE_C, 'mt_ma60_break_days': 1},
        'budget': int(INITIAL_CAPITAL * 0.15),
        'capital': INITIAL_CAPITAL,
    },
    'C_zombie_60': {
        'desc': '策略C: zombie 60天+3%',
        'signal_func': check_midterm_signal,
        'config': {**BASE_C, 'mt_zombie_days': 60, 'mt_zombie_min_profit': 3.0,
                   'zombie_hold_days': 60, 'zombie_net_range': 3.0},
        'budget': int(INITIAL_CAPITAL * 0.15),
        'capital': INITIAL_CAPITAL,
    },
}

PERIODS = {
    'Train': ('2021-01-01', '2025-06-30'),
    'Val':   ('2025-07-01', '2026-03-28'),
}


def _extract_metrics(r):
    """從 backtest result 提取關鍵指標"""
    if not r:
        return {'ret': 0, 'shrp': 0, 'mdd': 0, 'pf': 0, 'wr': 0,
                'buys': 0, 'sells': 0, 'avg_hold': 0, 'calmar': 0,
                'cagr': 0, 'final': 0}
    tl = r.get('trade_log', [])
    buys = [x for x in tl if x['type'] == 'BUY']
    sells = [x for x in tl if x['type'] == 'SELL' and x.get('profit') is not None]
    gw = sum(x['profit'] for x in sells if x['profit'] > 0)
    gl = abs(sum(x['profit'] for x in sells if x['profit'] <= 0))
    wr = sum(1 for s in sells if s['profit'] > 0) / len(sells) * 100 if sells else 0
    hold_days = []
    for s in sells:
        bd, sd = s.get('buy_date', ''), s.get('date', '')
        if bd and sd:
            try:
                hold_days.append((pd.Timestamp(sd) - pd.Timestamp(bd)).days)
            except Exception:
                pass
    return {
        'ret': r.get('total_return_pct', 0),
        'shrp': r.get('sharpe_ratio', 0),
        'mdd': r.get('mdd_pct', 0),
        'pf': gw / gl if gl > 0 else 999,
        'wr': wr,
        'buys': len(buys),
        'sells': len(sells),
        'avg_hold': np.mean(hold_days) if hold_days else 0,
        'calmar': r.get('calmar_ratio', 0),
        'cagr': r.get('cagr', 0),
        'final': r.get('final_total_value', 0),
    }


def run_backtest():
    stocks = get_stocks_by_industry(INDUSTRY)
    print(f"{'='*110}")
    print(f"  策略 C (中長期趨勢回檔) vs 策略 A (短線動能) 對比測試")
    print(f"  {INDUSTRY}: {len(stocks)} stocks | Capital: ${INITIAL_CAPITAL:,}")
    print(f"{'='*110}\n")

    all_results = {}
    for period_name, (start, end) in PERIODS.items():
        print(f"\n{'='*90}")
        print(f"  📅 {period_name}: {start} ~ {end}")
        print(f"{'='*90}")

        mm = reconstruct_market_history(start, end)
        dl_s = (pd.Timestamp(start) - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
        dl_e = (pd.Timestamp(end) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        data, _ = batch_download_stocks(stocks, dl_s, dl_e, min_data_days=MIN_DATA_DAYS)
        print(f"  Valid: {len(data)}\n")

        pr = {}
        for cn, ce in CONFIGS.items():
            t0 = time.time()
            r = run_group_backtest(
                stocks, start, end, ce['budget'], mm,
                exec_mode=EXEC_MODE,
                config_override=ce['config'],
                initial_capital=ce['capital'],
                preloaded_data=data,
                signal_func=ce['signal_func'],
            )
            el = time.time() - t0
            if r:
                pr[cn] = r
                m = _extract_metrics(r)
                print(f"  {cn:>14s}: Ret{m['ret']:+7.1f}% CAGR{m['cagr']:+6.1f}% "
                      f"Shrp{m['shrp']:5.2f} MDD{m['mdd']:5.1f}% Clm{m['calmar']:5.2f} "
                      f"PF{m['pf']:5.2f} WR{m['wr']:4.1f}% "
                      f"買{m['buys']:>3d}次 賣{m['sells']:>3d}次 "
                      f"均持{m['avg_hold']:.0f}天 ⏱{el:.0f}s — {ce['desc']}")
            else:
                print(f"  {cn:>14s}: ❌ No result — {ce['desc']}")
                pr[cn] = None
        all_results[period_name] = pr

    # ==========================================
    # Summary Table
    # ==========================================
    print(f"\n\n{'='*140}")
    print(f"  📊 策略 C (中長期) vs A (短線) 完整對比")
    print(f"{'='*140}")

    header = (f"{'Config':>14s} | {'Train':>62s} | {'Val':>62s}")
    sub    = (f"{'':>14s} | {'Ret':>7s} {'CAGR':>6s} {'Shrp':>5s} {'MDD':>5s} {'Clm':>5s} "
              f"{'PF':>5s} {'WR':>5s} {'買':>3s} {'均持':>4s} | "
              f"{'Ret':>7s} {'CAGR':>6s} {'Shrp':>5s} {'MDD':>5s} {'Clm':>5s} "
              f"{'PF':>5s} {'WR':>5s} {'買':>3s} {'均持':>4s}")
    print(header)
    print(sub)
    print("-" * 140)

    for cn in CONFIGS:
        t_r = all_results.get('Train', {}).get(cn)
        v_r = all_results.get('Val', {}).get(cn)
        t = _extract_metrics(t_r)
        v = _extract_metrics(v_r)

        # 標記最佳
        marker = ''
        if cn.startswith('C_') and t['shrp'] > 0 and v['shrp'] > 0:
            marker = ' ⭐'

        print(f"{cn:>14s} | "
              f"{t['ret']:+6.1f}% {t['cagr']:+5.1f}% {t['shrp']:5.2f} "
              f"{t['mdd']:4.1f}% {t['calmar']:5.2f} "
              f"{t['pf']:5.2f} {t['wr']:4.1f}% {t['buys']:>3d} {t['avg_hold']:4.0f}d | "
              f"{v['ret']:+6.1f}% {v['cagr']:+5.1f}% {v['shrp']:5.2f} "
              f"{v['mdd']:4.1f}% {v['calmar']:5.2f} "
              f"{v['pf']:5.2f} {v['wr']:4.1f}% {v['buys']:>3d} {v['avg_hold']:4.0f}d{marker}")

    print(f"{'='*140}")

    # ==========================================
    # Trade log 輸出 (最佳 C 變體)
    # ==========================================
    _output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(_output_dir, exist_ok=True)

    for period_name in PERIODS:
        for cn in CONFIGS:
            if not cn.startswith('C_'):
                continue
            r = all_results.get(period_name, {}).get(cn)
            if r and r.get('trade_log'):
                tl = r['trade_log']
                rows = []
                for t in tl:
                    rows.append({
                        'Date': t['date'],
                        'Ticker': t['ticker'],
                        'Name': t.get('name', ''),
                        'Type': t['type'],
                        'Price': round(t['price'], 2),
                        'Shares': t['shares'],
                        'Profit': int(t['profit']) if t['profit'] is not None else '',
                        'ROI%': round(t['roi'], 2) if t.get('roi') is not None else '',
                        'Note': t.get('note', ''),
                    })
                df = pd.DataFrame(rows)
                path = os.path.join(_output_dir, f'midterm_{cn}_{period_name}.csv')
                df.to_csv(path, index=False, encoding='utf-8-sig')

    print(f"\nTrade logs saved to output/midterm_*.csv")


if __name__ == '__main__':
    run_backtest()
