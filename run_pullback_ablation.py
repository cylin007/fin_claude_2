#!/usr/bin/env python3
"""策略 B (回檔買進) vs 策略 A (動能追漲) 對比測試"""

import sys, os, time, warnings
import pandas as pd, numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from group_backtest import (run_group_backtest, reconstruct_market_history,
                            INDUSTRY_CONFIGS, MIN_DATA_DAYS)
from industry_manager import get_stocks_by_industry
from stock_utils import batch_download_stocks
from strategy_pullback import check_pullback_signal, PULLBACK_CONFIG

INDUSTRY = '半導體業'
INITIAL_CAPITAL = 630_000
EXEC_MODE = 'next_open'
BASE_A = INDUSTRY_CONFIGS[INDUSTRY]['config'].copy()  # 策略 A (現有)

# 策略 B 需要不同的 position management 參數
BASE_B = {
    'zombie_hold_days': 45,       # 更長的持有期
    'zombie_net_range': 5.0,
    'hard_stop_net': -8,          # PB 自帶停損，這裡是備用
    'enable_zombie_cleanup': True,
    'enable_position_swap': False, # PB 不做換股
    'max_add_per_stock': 1,       # 不加碼 (PB 每檔只買一次)
    'budget_pct': 15.0,           # 大倉位
    'min_rsi': 0,                 # RSI 由 PB 自己管
    'enable_rs_filter': False,
    'enable_sector_momentum': False,
    'enable_ewt_boost': False,
    'enable_conviction_hold': False,
    'enable_regime_adaptive': False,
    'enable_peer_zscore': False,
    'enable_weekly_filter': False,
    'market_filter_mode': 'relaxed',
    # 加入 PB 特定參數
    **PULLBACK_CONFIG,
}

CONFIGS = {
    'A_baseline': {
        'desc': '策略A: 動能追漲 (現狀)',
        'signal_func': None,  # default
        'config': BASE_A,
        'budget': 17_500,
        'capital': INITIAL_CAPITAL,
    },

    # === 策略 B variants ===
    'B_default': {
        'desc': '策略B: 回檔買 (預設)',
        'signal_func': check_pullback_signal,
        'config': BASE_B,
        'budget': 94_500,  # 630K * 15%
        'capital': INITIAL_CAPITAL,
    },
    'B_wider': {
        'desc': '策略B: 回檔區間放寬 (-8%~+5%)',
        'signal_func': check_pullback_signal,
        'config': {**BASE_B, 'pb_pullback_lo': -8.0, 'pb_pullback_hi': 5.0},
        'budget': 94_500,
        'capital': INITIAL_CAPITAL,
    },
    'B_no_vol': {
        'desc': '策略B: 不看量縮',
        'signal_func': check_pullback_signal,
        'config': {**BASE_B, 'pb_vol_shrink': False},
        'budget': 94_500,
        'capital': INITIAL_CAPITAL,
    },
    'B_wider_no_vol': {
        'desc': '策略B: 放寬+不看量',
        'signal_func': check_pullback_signal,
        'config': {**BASE_B, 'pb_pullback_lo': -8.0, 'pb_pullback_hi': 5.0, 'pb_vol_shrink': False},
        'budget': 94_500,
        'capital': INITIAL_CAPITAL,
    },
    'B_rsi_wide': {
        'desc': '策略B: RSI放寬 30-70',
        'signal_func': check_pullback_signal,
        'config': {**BASE_B, 'pb_rsi_lo': 30, 'pb_rsi_hi': 70, 'pb_vol_shrink': False},
        'budget': 94_500,
        'capital': INITIAL_CAPITAL,
    },
    'B_trail_15': {
        'desc': '策略B: 追蹤停利15%觸發/8%回撤',
        'signal_func': check_pullback_signal,
        'config': {**BASE_B, 'pb_trail_trigger': 15.0, 'pb_trail_drop': 8.0, 'pb_vol_shrink': False},
        'budget': 94_500,
        'capital': INITIAL_CAPITAL,
    },
    'B_small_pos': {
        'desc': '策略B: 小倉位5% (10檔)',
        'signal_func': check_pullback_signal,
        'config': {**BASE_B, 'pb_budget_pct': 5.0, 'pb_max_positions': 10, 'pb_vol_shrink': False},
        'budget': 31_500,  # 630K * 5%
        'capital': INITIAL_CAPITAL,
    },
}

PERIODS = {
    'Train': ('2022-01-01', '2025-06-30'),
    'Val':   ('2025-07-01', '2026-03-27'),
}

def run_ablation():
    stocks = get_stocks_by_industry(INDUSTRY)
    print(f"{'='*100}")
    print(f"  策略 A (動能追漲) vs 策略 B (回檔買進) 對比測試")
    print(f"  {INDUSTRY}: {len(stocks)} stocks")
    print(f"{'='*100}\n")

    all_results = {}
    for period_name, (start, end) in PERIODS.items():
        print(f"\n{'='*80}\n  📅 {period_name}: {start} ~ {end}\n{'='*80}")
        mm = reconstruct_market_history(start, end)
        dl_s = (pd.Timestamp(start) - pd.Timedelta(days=250)).strftime('%Y-%m-%d')
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
                ret = r.get('total_return_pct', 0)
                sh = r.get('sharpe_ratio', 0)
                mdd = r.get('mdd_pct', 0)
                clm = r.get('calmar_ratio', 0)
                tl = r.get('trade_log', [])
                buys = [t for t in tl if t['type'] == 'BUY']
                sells = [t for t in tl if t['type'] == 'SELL' and t.get('profit') is not None]
                gw = sum(t['profit'] for t in sells if t['profit'] > 0)
                gl = abs(sum(t['profit'] for t in sells if t['profit'] <= 0))
                pf = gw / gl if gl > 0 else 999
                wr = sum(1 for s in sells if s['profit'] > 0) / len(sells) * 100 if sells else 0
                avg_hold = 0
                if sells:
                    hold_days = []
                    for s in sells:
                        bd = s.get('buy_date', '')
                        sd = s.get('date', '')
                        if bd and sd:
                            try:
                                hd = (pd.Timestamp(sd) - pd.Timestamp(bd)).days
                                hold_days.append(hd)
                            except:
                                pass
                    avg_hold = np.mean(hold_days) if hold_days else 0

                print(f"  {cn:>14s}: Ret{ret:+7.1f}% Shrp{sh:5.2f} MDD{mdd:5.1f}% PF{pf:4.2f} "
                      f"WR{wr:4.1f}% 買{len(buys):>3d}次 賣{len(sells):>3d}次 "
                      f"均持{avg_hold:.0f}天 ⏱{el:.0f}s — {ce['desc']}")
            else:
                print(f"  {cn:>14s}: ❌ No result — {ce['desc']}")
                pr[cn] = None
        all_results[period_name] = pr

    # Summary
    print(f"\n\n{'='*120}")
    print(f"  📊 策略 A vs B 完整對比")
    print(f"{'='*120}")

    header = (f"{'Config':>14s} | {'Train':>46s} | {'Val':>46s}")
    sub    = (f"{'':>14s} | {'Ret':>7s} {'Shrp':>5s} {'MDD':>5s} {'PF':>4s} {'WR':>5s} {'買':>3s} {'均持':>4s} | "
              f"{'Ret':>7s} {'Shrp':>5s} {'MDD':>5s} {'PF':>4s} {'WR':>5s} {'買':>3s} {'均持':>4s}")
    print(header)
    print(sub)
    print("-" * 120)

    for cn in CONFIGS:
        t = all_results.get('Train', {}).get(cn)
        v = all_results.get('Val', {}).get(cn)

        def _extract(r):
            if not r:
                return (0, 0, 0, 0, 0, 0, 0)
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
                    try: hold_days.append((pd.Timestamp(sd) - pd.Timestamp(bd)).days)
                    except: pass
            return (r.get('total_return_pct', 0), r.get('sharpe_ratio', 0), r.get('mdd_pct', 0),
                    gw / gl if gl > 0 else 999, wr, len(buys),
                    np.mean(hold_days) if hold_days else 0)

        tr, ts, tm, tp, tw, tb, th = _extract(t)
        vr, vs, vm, vp, vw, vb, vh = _extract(v)
        print(f"{cn:>14s} | {tr:+6.1f}% {ts:5.2f} {tm:4.1f}% {tp:4.2f} {tw:4.1f}% {tb:>3d} {th:4.0f}d | "
              f"{vr:+6.1f}% {vs:5.2f} {vm:4.1f}% {vp:4.2f} {vw:4.1f}% {vb:>3d} {vh:4.0f}d")

    print(f"{'='*120}")

if __name__ == '__main__':
    run_ablation()
