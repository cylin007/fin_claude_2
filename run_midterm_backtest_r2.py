#!/usr/bin/env python3
"""
策略 C 第二輪優化: 基於 Round 1 發現

Round 1 關鍵發現:
  - 量縮 0.7x 是最大殺手 → 本輪全部關掉量縮或大幅放寬
  - C_no_vol 是唯一正報酬變體 (+4.8% Train)
  - C_wider 表現接近 default (回檔區間可放寬)
  - C_5pos 分散有助降低波動

Round 2 設計:
  1. 以 C_no_vol 為基底 (不看量縮)
  2. 測試多個子維度: 回檔區間 / RSI / 停利 / 大盤過濾 / 持股數 / MA60出場
  3. 最後組合最佳子項成 combo

Train: 2021-01-01 ~ 2025-06-30
Val:   2025-07-01 ~ 2026-03-28
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
INITIAL_CAPITAL = 900_000
EXEC_MODE = 'next_open'
BASE_A = INDUSTRY_CONFIGS[INDUSTRY]['config'].copy()

# ==========================================
# Round 2 基底: C_no_vol (量縮關閉)
# ==========================================
R2_BASE = {
    # -- backtest engine 層 --
    'max_positions':         4,
    'max_new_buy_per_day':   2,
    'zombie_hold_days':      90,
    'zombie_net_range':      5.0,
    'hard_stop_net':        -12,
    'enable_zombie_cleanup': True,
    'enable_position_swap':  False,
    'max_add_per_stock':     1,
    'budget_pct':            15.0,
    'cash_reserve_pct':      40.0,

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
    'enable_weekly_filter':  False,
    'enable_theme_boost':    False,
    'enable_quality_filter': False,
    'enable_dynamic_exposure': False,
    'enable_dyn_buy_limit':  False,
    'enable_dyn_stop':       False,
    'enable_vol_sizing':     False,
    'enable_profit_trailing': False,
    'enable_trailing_stop':  False,
    'enable_pullback_buy':   False,
    'enable_dip_buy':        False,
    'market_filter_mode':    'off',
    'min_rsi':               0,

    # -- 中長期策略參數 (基於 C_no_vol) --
    **MIDTERM_CONFIG,
    'mt_vol_shrink_ratio':   999.0,   # ← 關閉量縮 (Round 1 最大改進)
}


def _mk(desc, overrides, max_pos=4, budget_pct=15.0):
    """建立 config dict 的快捷函式"""
    cfg = {**R2_BASE, **overrides}
    # 同步 engine 層與 strategy 層
    cfg['max_positions'] = max_pos
    cfg['mt_max_positions'] = max_pos
    cfg['budget_pct'] = budget_pct
    cfg['mt_budget_pct'] = budget_pct
    return {
        'desc': desc,
        'signal_func': check_midterm_signal,
        'config': cfg,
        'budget': int(INITIAL_CAPITAL * budget_pct / 100),
        'capital': INITIAL_CAPITAL,
    }


# ==========================================
# 測試矩陣 — Round 2
# ==========================================
CONFIGS = {}

# === Baseline ===
CONFIGS['A_baseline'] = {
    'desc': '策略A: 短線動能 (現狀)',
    'signal_func': None,
    'config': BASE_A,
    'budget': 25_000,
    'capital': INITIAL_CAPITAL,
}
CONFIGS['R1_no_vol'] = _mk('R1最佳: C_no_vol (基準線)', {})

# ==========================================
# Dimension 1: 回檔區間
# ==========================================
CONFIGS['D1_tight'] = _mk(
    '回檔: 緊 (-3%~+2%)',
    {'mt_pullback_lo': -3.0, 'mt_pullback_hi': 2.0})

CONFIGS['D1_wide'] = _mk(
    '回檔: 寬 (-8%~+5%)',
    {'mt_pullback_lo': -8.0, 'mt_pullback_hi': 5.0})

CONFIGS['D1_vwide'] = _mk(
    '回檔: 超寬 (-10%~+8%)',
    {'mt_pullback_lo': -10.0, 'mt_pullback_hi': 8.0})

# ==========================================
# Dimension 2: RSI
# ==========================================
CONFIGS['D2_rsi60'] = _mk(
    'RSI≤60 (偏保守)',
    {'mt_rsi_max': 60})

CONFIGS['D2_rsi80'] = _mk(
    'RSI≤80 (放寬)',
    {'mt_rsi_max': 80})

# ==========================================
# Dimension 3: 停利機制
# ==========================================
CONFIGS['D3_trail20_8'] = _mk(
    '停利: 20%觸發/8%回撤 (緊)',
    {'mt_trail_trigger': 20.0, 'mt_trail_drop': 8.0})

CONFIGS['D3_trail25_10'] = _mk(
    '停利: 25%觸發/10%回撤',
    {'mt_trail_trigger': 25.0, 'mt_trail_drop': 10.0})

CONFIGS['D3_trail50_15'] = _mk(
    '停利: 50%觸發/15%回撤 (寬鬆)',
    {'mt_trail_trigger': 50.0, 'mt_trail_drop': 15.0})

CONFIGS['D3_trail60_20'] = _mk(
    '停利: 60%觸發/20%回撤 (超寬鬆)',
    {'mt_trail_trigger': 60.0, 'mt_trail_drop': 20.0})

# ==========================================
# Dimension 4: 大盤過濾
# ==========================================
CONFIGS['D4_no_filter'] = _mk(
    '大盤: 完全不過濾',
    {'mt_block_bear': False, 'mt_block_panic': False})

CONFIGS['D4_bear_only'] = _mk(
    '大盤: 只擋 bear',
    {'mt_block_bear': True, 'mt_block_panic': False})

CONFIGS['D4_panic_only'] = _mk(
    '大盤: 只擋 panic',
    {'mt_block_bear': False, 'mt_block_panic': True})

# ==========================================
# Dimension 5: 持股數 & 資金配置
# ==========================================
CONFIGS['D5_3pos'] = _mk(
    '3檔 (每檔18%)',
    {}, max_pos=3, budget_pct=18.0)

CONFIGS['D5_5pos'] = _mk(
    '5檔 (每檔12%)',
    {'cash_reserve_pct': 40.0}, max_pos=5, budget_pct=12.0)

CONFIGS['D5_6pos'] = _mk(
    '6檔 (每檔10%)',
    {'cash_reserve_pct': 40.0}, max_pos=6, budget_pct=10.0)

# ==========================================
# Dimension 6: MA60 出場速度
# ==========================================
CONFIGS['D6_ma60_1d'] = _mk(
    'MA60跌破: 隔天出場',
    {'mt_ma60_break_days': 1})

CONFIGS['D6_ma60_2d'] = _mk(
    'MA60跌破: 連2天出場',
    {'mt_ma60_break_days': 2})

CONFIGS['D6_ma60_5d'] = _mk(
    'MA60跌破: 連5天出場 (寬鬆)',
    {'mt_ma60_break_days': 5})

# ==========================================
# Dimension 7: Zombie 清除
# ==========================================
CONFIGS['D7_zombie45'] = _mk(
    'Zombie: 45天+3%',
    {'mt_zombie_days': 45, 'mt_zombie_min_profit': 3.0,
     'zombie_hold_days': 45, 'zombie_net_range': 3.0})

CONFIGS['D7_zombie60'] = _mk(
    'Zombie: 60天+5%',
    {'mt_zombie_days': 60, 'mt_zombie_min_profit': 5.0,
     'zombie_hold_days': 60, 'zombie_net_range': 5.0})

CONFIGS['D7_zombie120'] = _mk(
    'Zombie: 120天+8%',
    {'mt_zombie_days': 120, 'mt_zombie_min_profit': 8.0,
     'zombie_hold_days': 120, 'zombie_net_range': 8.0})

# ==========================================
# Dimension 8: 硬停損
# ==========================================
CONFIGS['D8_stop8'] = _mk(
    '硬停損: -8%',
    {'mt_hard_stop_pct': -8.0, 'hard_stop_net': -8.0})

CONFIGS['D8_stop15'] = _mk(
    '硬停損: -15%',
    {'mt_hard_stop_pct': -15.0, 'hard_stop_net': -15.0})

CONFIGS['D8_stop20'] = _mk(
    '硬停損: -20%',
    {'mt_hard_stop_pct': -20.0, 'hard_stop_net': -20.0})

# ==========================================
# Dimension 9: 週線過濾
# ==========================================
CONFIGS['D9_no_weekly'] = _mk(
    '不看週線',
    {'mt_require_weekly': False})

# ==========================================
# Dimension 10: 量縮 (微放寬 vs 完全關閉確認)
# ==========================================
CONFIGS['D10_vol1.5'] = _mk(
    '量縮: vol < 1.5x MA5 (微放寬)',
    {'mt_vol_shrink_ratio': 1.5})

CONFIGS['D10_vol2.0'] = _mk(
    '量縮: vol < 2.0x MA5 (幾乎不濾)',
    {'mt_vol_shrink_ratio': 2.0})

# ==========================================
# Combo: 組合最佳子項 (根據 Round 1 推斷)
# ==========================================
CONFIGS['COMBO_A'] = _mk(
    'Combo: 寬回檔+5檔+不看週線+25/10停利',
    {'mt_pullback_lo': -8.0, 'mt_pullback_hi': 5.0,
     'mt_require_weekly': False,
     'mt_trail_trigger': 25.0, 'mt_trail_drop': 10.0,
     'cash_reserve_pct': 40.0},
    max_pos=5, budget_pct=12.0)

CONFIGS['COMBO_B'] = _mk(
    'Combo: 超寬回檔+6檔+RSI80+MA60_2天+停損15%',
    {'mt_pullback_lo': -10.0, 'mt_pullback_hi': 8.0,
     'mt_rsi_max': 80,
     'mt_ma60_break_days': 2,
     'mt_hard_stop_pct': -15.0, 'hard_stop_net': -15.0,
     'mt_require_weekly': False,
     'cash_reserve_pct': 40.0},
    max_pos=6, budget_pct=10.0)

CONFIGS['COMBO_C'] = _mk(
    'Combo: 寬回檔+4檔+不擋bear+50/15停利+zombie120',
    {'mt_pullback_lo': -8.0, 'mt_pullback_hi': 5.0,
     'mt_block_bear': False, 'mt_block_panic': True,
     'mt_trail_trigger': 50.0, 'mt_trail_drop': 15.0,
     'mt_zombie_days': 120, 'mt_zombie_min_profit': 8.0,
     'zombie_hold_days': 120, 'zombie_net_range': 8.0,
     'mt_require_weekly': False})

CONFIGS['COMBO_D'] = _mk(
    'Combo: 寬回檔+5檔+bear不擋+RSI80+25/10停利+MA60_2天',
    {'mt_pullback_lo': -8.0, 'mt_pullback_hi': 5.0,
     'mt_rsi_max': 80,
     'mt_block_bear': False, 'mt_block_panic': True,
     'mt_ma60_break_days': 2,
     'mt_trail_trigger': 25.0, 'mt_trail_drop': 10.0,
     'mt_require_weekly': False,
     'cash_reserve_pct': 40.0},
    max_pos=5, budget_pct=12.0)


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
    print(f"{'='*120}")
    print(f"  策略 C — Round 2 優化 (基於 R1_no_vol)")
    print(f"  {INDUSTRY}: {len(stocks)} stocks | Capital: ${INITIAL_CAPITAL:,}")
    print(f"  共 {len(CONFIGS)} 組測試 (2 baseline + {len(CONFIGS)-2} 變體)")
    print(f"{'='*120}\n")

    all_results = {}
    for period_name, (start, end) in PERIODS.items():
        print(f"\n{'='*100}")
        print(f"  📅 {period_name}: {start} ~ {end}")
        print(f"{'='*100}")

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
    # Summary Table — 按 dimension 分組
    # ==========================================
    print(f"\n\n{'='*150}")
    print(f"  📊 Round 2 完整結果 — 以 R1_no_vol 為基準，各維度逐一優化")
    print(f"{'='*150}")

    header = (f"{'Config':>14s} | {'--- Train ---':>65s} | {'--- Val ---':>65s}")
    sub    = (f"{'':>14s} | {'Ret':>7s} {'CAGR':>6s} {'Shrp':>5s} {'MDD':>5s} {'Clm':>5s} "
              f"{'PF':>5s} {'WR':>5s} {'買':>3s} {'賣':>3s} {'均持':>4s} | "
              f"{'Ret':>7s} {'CAGR':>6s} {'Shrp':>5s} {'MDD':>5s} {'Clm':>5s} "
              f"{'PF':>5s} {'WR':>5s} {'買':>3s} {'賣':>3s} {'均持':>4s}")
    print(header)
    print(sub)
    print("-" * 150)

    # 按維度分組
    dimensions = {
        'Baseline':  ['A_baseline', 'R1_no_vol'],
        'D1-回檔':   [k for k in CONFIGS if k.startswith('D1_')],
        'D2-RSI':    [k for k in CONFIGS if k.startswith('D2_')],
        'D3-停利':   [k for k in CONFIGS if k.startswith('D3_')],
        'D4-大盤':   [k for k in CONFIGS if k.startswith('D4_')],
        'D5-持股':   [k for k in CONFIGS if k.startswith('D5_')],
        'D6-MA60':   [k for k in CONFIGS if k.startswith('D6_')],
        'D7-Zombie': [k for k in CONFIGS if k.startswith('D7_')],
        'D8-停損':   [k for k in CONFIGS if k.startswith('D8_')],
        'D9-週線':   [k for k in CONFIGS if k.startswith('D9_')],
        'D10-量縮':  [k for k in CONFIGS if k.startswith('D10_')],
        'Combo':     [k for k in CONFIGS if k.startswith('COMBO_')],
    }

    # R1 基準 metrics
    r1_train = _extract_metrics(all_results.get('Train', {}).get('R1_no_vol'))

    for dim_name, keys in dimensions.items():
        print(f"\n  【{dim_name}】")
        for cn in keys:
            t_r = all_results.get('Train', {}).get(cn)
            v_r = all_results.get('Val', {}).get(cn)
            t = _extract_metrics(t_r)
            v = _extract_metrics(v_r)

            # 標記相對 R1 改善的
            markers = []
            if cn not in ('A_baseline', 'R1_no_vol'):
                if t['ret'] > r1_train['ret'] * 1.05:
                    markers.append('↑ret')
                if t['shrp'] > r1_train['shrp'] + 0.1:
                    markers.append('↑shrp')
                if t['mdd'] < r1_train['mdd'] * 0.9:
                    markers.append('↑mdd')
            marker = ' ⭐' + ','.join(markers) if markers else ''

            print(f"  {cn:>14s} | "
                  f"{t['ret']:+6.1f}% {t['cagr']:+5.1f}% {t['shrp']:5.2f} "
                  f"{t['mdd']:4.1f}% {t['calmar']:5.2f} "
                  f"{t['pf']:5.2f} {t['wr']:4.1f}% {t['buys']:>3d} {t['sells']:>3d} {t['avg_hold']:4.0f}d | "
                  f"{v['ret']:+6.1f}% {v['cagr']:+5.1f}% {v['shrp']:5.2f} "
                  f"{v['mdd']:4.1f}% {v['calmar']:5.2f} "
                  f"{v['pf']:5.2f} {v['wr']:4.1f}% {v['buys']:>3d} {v['sells']:>3d} {v['avg_hold']:4.0f}d{marker}")

    print(f"\n{'='*150}")

    # ==========================================
    # 各維度最佳選擇
    # ==========================================
    print(f"\n\n📋 各維度最佳選擇 (Train Ret 排序):\n")
    for dim_name, keys in dimensions.items():
        if dim_name in ('Baseline', 'Combo'):
            continue
        best = None
        best_ret = -999
        for cn in keys:
            t = _extract_metrics(all_results.get('Train', {}).get(cn))
            if t['ret'] > best_ret:
                best_ret = t['ret']
                best = cn
        if best:
            t = _extract_metrics(all_results.get('Train', {}).get(best))
            v = _extract_metrics(all_results.get('Val', {}).get(best))
            diff = best_ret - r1_train['ret']
            print(f"  {dim_name:>10s}: {best:>14s} "
                  f"Train={t['ret']:+.1f}% (Δ{diff:+.1f}%) Shrp={t['shrp']:.2f} MDD={t['mdd']:.1f}% | "
                  f"Val={v['ret']:+.1f}% Shrp={v['shrp']:.2f}")

    # ==========================================
    # Combo 排名
    # ==========================================
    print(f"\n\n🏆 Combo 排名:\n")
    combo_keys = [k for k in CONFIGS if k.startswith('COMBO_')]
    combo_scores = []
    for cn in combo_keys:
        t = _extract_metrics(all_results.get('Train', {}).get(cn))
        v = _extract_metrics(all_results.get('Val', {}).get(cn))
        # 綜合分 = Train Sharpe × 40% + Val Sharpe × 30% + (1 - MDD/50) × 30%
        score = t['shrp'] * 0.4 + v['shrp'] * 0.3 + (1 - abs(t['mdd'])/50) * 0.3
        combo_scores.append((cn, score, t, v))
    combo_scores.sort(key=lambda x: -x[1])

    for cn, score, t, v in combo_scores:
        print(f"  {cn:>10s}: Score={score:.3f} | "
              f"Train: Ret={t['ret']:+.1f}% Shrp={t['shrp']:.2f} MDD={t['mdd']:.1f}% PF={t['pf']:.2f} WR={t['wr']:.0f}% | "
              f"Val: Ret={v['ret']:+.1f}% Shrp={v['shrp']:.2f} MDD={v['mdd']:.1f}%"
              f" — {CONFIGS[cn]['desc']}")

    # ==========================================
    # Trade log 輸出
    # ==========================================
    _output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(_output_dir, exist_ok=True)

    for period_name in PERIODS:
        for cn in CONFIGS:
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
                path = os.path.join(_output_dir, f'midterm_r2_{cn}_{period_name}.csv')
                df.to_csv(path, index=False, encoding='utf-8-sig')

    print(f"\nTrade logs saved to output/midterm_r2_*.csv")


if __name__ == '__main__':
    run_backtest()
