#!/usr/bin/env python3
"""
策略 D: 趨勢存續型持股 (Trend-Persistence Strategy) — Ablation Backtest

核心改變 vs C:
  1. 非對稱出場: 虧損快砍 / 小獲利中等 / 大獲利寬鬆 (趨勢存續判斷)
  2. 進場模式: momentum / pullback / hybrid
  3. 降頻: 5檔集中, 每週限買, 高現金儲備

Train: 2021-01-01 ~ 2025-06-30
Val:   2025-07-01 ~ 2026-03-28

測試矩陣:
  L0: 三種進場模式 (baseline)
  L1: 非對稱出場參數掃描
  L2: 部位管理 (檔數 / budget / 週限)
  L3: 最佳組合
"""

import sys, os, time, warnings
import pandas as pd, numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from group_backtest import (run_group_backtest, reconstruct_market_history,
                            INDUSTRY_CONFIGS, MIN_DATA_DAYS)
from industry_manager import get_stocks_by_industry
from stock_utils import batch_download_stocks
from strategy_midterm import (check_midterm_signal, check_trend_persistence_signal,
                              MIDTERM_CONFIG, TREND_PERSISTENCE_CONFIG)

INDUSTRY = '半導體業'
INITIAL_CAPITAL = 900_000
EXEC_MODE = 'next_open'
BASE_A = INDUSTRY_CONFIGS[INDUSTRY]['config'].copy()

# ==========================================
# 策略D 基底參數
# ==========================================
TP_BASE = {
    # -- backtest engine 層 --
    'max_positions':         5,
    'max_new_buy_per_day':   2,
    'zombie_hold_days':      20,         # 虧損zombie由策略D的非對稱邏輯接管
    'zombie_net_range':      0.0,        # 只砍虧損的 (net < 0 = ±0% range)
    'hard_stop_net':        -10,
    'enable_zombie_cleanup': True,
    'enable_position_swap':  False,
    'max_add_per_stock':     1,
    'budget_pct':            12.0,
    'cash_reserve_pct':      15.0,
    'weekly_max_buy':        2,          # V14: 每週最多新買2檔

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

    # -- 策略D 專用參數 --
    **TREND_PERSISTENCE_CONFIG,
}


def _mk(desc, overrides, max_pos=5, budget_pct=12.0, weekly_max=2):
    """建構測試配置"""
    cfg = {**TP_BASE, **overrides}
    cfg['max_positions'] = max_pos
    cfg['tp_max_positions'] = max_pos
    cfg['budget_pct'] = budget_pct
    cfg['tp_budget_pct'] = budget_pct
    cfg['weekly_max_buy'] = weekly_max
    return {
        'desc': desc,
        'signal_func': check_trend_persistence_signal,
        'config': cfg,
        'budget': int(INITIAL_CAPITAL * budget_pct / 100),
        'capital': INITIAL_CAPITAL,
    }


# ==========================================
# 測試配置矩陣
# ==========================================
CONFIGS = {}

# === Baselines ===
CONFIGS['A_baseline'] = {
    'desc': '策略A: 短線動能 (對照)',
    'signal_func': None,
    'config': BASE_A,
    'budget': 25_000,
    'capital': INITIAL_CAPITAL,
}
# 策略C R3 最佳 (對照)
_c_r3_cfg = {
    **TP_BASE,
    **MIDTERM_CONFIG,
    'mt_pullback_lo': -10.0, 'mt_pullback_hi': 8.0,
    'mt_ma60_break_days': 5, 'mt_vol_shrink_ratio': 1.5,
    'mt_rsi_max': 60,
    'max_positions': 4, 'budget_pct': 15.0,
    'zombie_hold_days': 90, 'zombie_net_range': 5.0,
    'weekly_max_buy': 0,  # C 不限週買
}
CONFIGS['C_r3_best'] = {
    'desc': '策略C R3底座+RSI60 (對照)',
    'signal_func': check_midterm_signal,
    'config': _c_r3_cfg,
    'budget': int(INITIAL_CAPITAL * 15 / 100),
    'capital': INITIAL_CAPITAL,
}

# ==========================================
# Layer 0: 三種進場模式
# ==========================================
CONFIGS['D_pullback'] = _mk(
    'D: pullback進場 (default)',
    {'tp_entry_mode': 'pullback'})

CONFIGS['D_momentum'] = _mk(
    'D: momentum進場',
    {'tp_entry_mode': 'momentum'})

CONFIGS['D_hybrid'] = _mk(
    'D: hybrid進場 (momentum優先)',
    {'tp_entry_mode': 'hybrid'})

# ==========================================
# Layer 1: 非對稱出場參數掃描
# ==========================================

# --- 虧損層 ---
CONFIGS['L1_stop8'] = _mk(
    '+虧損硬停-8%',
    {'tp_loss_hard_stop': -8.0})

CONFIGS['L1_stop12'] = _mk(
    '+虧損硬停-12%',
    {'tp_loss_hard_stop': -12.0})

CONFIGS['L1_loss_ma5d'] = _mk(
    '+虧損MA60破5天',
    {'tp_loss_ma60_break_days': 5})

CONFIGS['L1_loss_z30'] = _mk(
    '+虧損zombie 30天',
    {'tp_loss_zombie_days': 30, 'zombie_hold_days': 30, 'zombie_net_range': 0.0})

CONFIGS['L1_loss_z15'] = _mk(
    '+虧損zombie 15天',
    {'tp_loss_zombie_days': 15, 'zombie_hold_days': 15, 'zombie_net_range': 0.0})

# --- 小獲利層 ---
CONFIGS['L1_sm_ma3d'] = _mk(
    '+小獲利MA60破3天',
    {'tp_small_ma60_break_days': 3})

CONFIGS['L1_sm_ma7d'] = _mk(
    '+小獲利MA60破7天',
    {'tp_small_ma60_break_days': 7})

CONFIGS['L1_sm_nohealth'] = _mk(
    '+小獲利不看趨勢健康',
    {'tp_small_trend_health': False})

CONFIGS['L1_sm_z45'] = _mk(
    '+小獲利zombie 45天/5%',
    {'tp_small_zombie_days': 45, 'tp_small_zombie_min_pct': 5.0})

CONFIGS['L1_sm_z90'] = _mk(
    '+小獲利zombie 90天/10%',
    {'tp_small_zombie_days': 90, 'tp_small_zombie_min_pct': 10.0})

# --- 大獲利層 ---
CONFIGS['L1_big10'] = _mk(
    '+大獲利門檻10%',
    {'tp_big_profit_threshold': 10.0})

CONFIGS['L1_big20'] = _mk(
    '+大獲利門檻20%',
    {'tp_big_profit_threshold': 20.0})

CONFIGS['L1_big_dd20'] = _mk(
    '+大獲利保底回撤20%',
    {'tp_big_max_drawdown': 20.0})

CONFIGS['L1_big_dd30'] = _mk(
    '+大獲利保底回撤30%',
    {'tp_big_max_drawdown': 30.0})

CONFIGS['L1_big_noweekly'] = _mk(
    '+大獲利不需週線翻空',
    {'tp_big_require_weekly_bear': False})

CONFIGS['L1_big_slope3'] = _mk(
    '+大獲利斜率負3天',
    {'tp_big_ma60_slope_days': 3})

CONFIGS['L1_big_slope7'] = _mk(
    '+大獲利斜率負7天',
    {'tp_big_ma60_slope_days': 7})

CONFIGS['L1_big_notrendx'] = _mk(
    '+大獲利不用趨勢出場(純MA60)',
    {'tp_big_use_trend_exit': False})

# ==========================================
# Layer 2: 部位管理
# ==========================================
CONFIGS['L2_4pos'] = _mk(
    '4檔 (15%)',
    {}, max_pos=4, budget_pct=15.0)

CONFIGS['L2_6pos'] = _mk(
    '6檔 (10%)',
    {}, max_pos=6, budget_pct=10.0)

CONFIGS['L2_5pos_15pct'] = _mk(
    '5檔 (15%)',
    {}, max_pos=5, budget_pct=15.0)

CONFIGS['L2_weekly1'] = _mk(
    '每週限買1檔',
    {}, weekly_max=1)

CONFIGS['L2_weekly3'] = _mk(
    '每週限買3檔',
    {}, weekly_max=3)

CONFIGS['L2_weekly0'] = _mk(
    '每週不限買',
    {}, weekly_max=0)

CONFIGS['L2_cash10'] = _mk(
    '現金儲備10%',
    {'tp_cash_reserve_pct': 10.0, 'cash_reserve_pct': 10.0})

CONFIGS['L2_cash25'] = _mk(
    '現金儲備25%',
    {'tp_cash_reserve_pct': 25.0, 'cash_reserve_pct': 25.0})

# ==========================================
# Layer 2B: 進場參數 (pullback模式)
# ==========================================
CONFIGS['L2B_wide'] = _mk(
    'pullback寬區間 (-10~+8)',
    {'tp_pullback_lo': -10.0, 'tp_pullback_hi': 8.0})

CONFIGS['L2B_narrow'] = _mk(
    'pullback窄區間 (-5~+3)',
    {'tp_pullback_lo': -5.0, 'tp_pullback_hi': 3.0})

CONFIGS['L2B_rsi70'] = _mk(
    'RSI上限70',
    {'tp_rsi_max': 70})

CONFIGS['L2B_rsi60'] = _mk(
    'RSI上限60',
    {'tp_rsi_max': 60})

CONFIGS['L2B_vol08'] = _mk(
    '量比上限0.8 (嚴格量縮)',
    {'tp_vol_ratio_max': 0.8})

CONFIGS['L2B_vol20'] = _mk(
    '量比上限2.0 (幾乎不濾)',
    {'tp_vol_ratio_max': 2.0})

CONFIGS['L2B_noweekly'] = _mk(
    '不要求週線多頭',
    {'tp_require_weekly_bull': False})


# ==========================================
# 測試週期
# ==========================================
PERIODS = {
    'Train': ('2021-01-01', '2025-06-30'),
    'Val':   ('2025-07-01', '2026-03-28'),
}


def _extract_metrics(r):
    """從回測結果中提取關鍵指標"""
    if not r:
        return {'ret': 0, 'shrp': 0, 'mdd': 0, 'pf': 0, 'wr': 0,
                'buys': 0, 'sells': 0, 'avg_hold': 0, 'calmar': 0,
                'cagr': 0, 'final': 0, 'trades': 0}
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
        'trades': len(buys) + len(sells),
    }


def run_backtest():
    stocks = get_stocks_by_industry(INDUSTRY)
    print(f"{'='*130}")
    print(f"  策略 D: 趨勢存續型持股 — Ablation Backtest")
    print(f"  核心: 非對稱出場 (虧損快砍 / 小獲利中等 / 大獲利寬鬆)")
    print(f"  {INDUSTRY}: {len(stocks)} stocks | Capital: ${INITIAL_CAPITAL:,}")
    print(f"  共 {len(CONFIGS)} 組測試")
    print(f"{'='*130}\n")

    all_results = {}
    for period_name, (start, end) in PERIODS.items():
        print(f"\n{'='*110}")
        print(f"  {period_name}: {start} ~ {end}")
        print(f"{'='*110}")

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
                print(f"  {cn:>20s}: Ret{m['ret']:+7.1f}% CAGR{m['cagr']:+6.1f}% "
                      f"Shrp{m['shrp']:5.2f} MDD{m['mdd']:5.1f}% Clm{m['calmar']:5.2f} "
                      f"PF{m['pf']:5.2f} WR{m['wr']:4.1f}% "
                      f"買{m['buys']:>3d} 賣{m['sells']:>3d} "
                      f"均持{m['avg_hold']:.0f}天 交易{m['trades']:>3d}次 "
                      f"⏱{el:.0f}s — {ce['desc']}")
            else:
                print(f"  {cn:>20s}: No result — {ce['desc']}")
                pr[cn] = None
        all_results[period_name] = pr

    # ==========================================
    # Summary Table
    # ==========================================
    print(f"\n\n{'='*170}")
    print(f"  策略D 完整結果 — 非對稱出場 + 趨勢存續")
    print(f"{'='*170}")

    header = (f"{'Config':>20s} | {'--- Train ---':>76s} | {'--- Val ---':>76s}")
    sub    = (f"{'':>20s} | {'Ret':>7s} {'CAGR':>6s} {'Shrp':>5s} {'MDD':>5s} {'Clm':>5s} "
              f"{'PF':>5s} {'WR':>5s} {'買':>3s} {'賣':>3s} {'均持':>4s} {'交易':>4s} | "
              f"{'Ret':>7s} {'CAGR':>6s} {'Shrp':>5s} {'MDD':>5s} {'Clm':>5s} "
              f"{'PF':>5s} {'WR':>5s} {'買':>3s} {'賣':>3s} {'均持':>4s} {'交易':>4s}")
    print(header)
    print(sub)
    print("-" * 170)

    # 依 Layer 分組顯示
    layers = {
        'Baseline':      ['A_baseline', 'C_r3_best'],
        'L0 進場模式':    [k for k in CONFIGS if k.startswith('D_')],
        'L1 出場掃描':    [k for k in CONFIGS if k.startswith('L1_')],
        'L2 部位管理':    [k for k in CONFIGS if k.startswith('L2_') and not k.startswith('L2B_')],
        'L2B 進場參數':   [k for k in CONFIGS if k.startswith('L2B_')],
    }

    # 找基準
    base_key = 'D_pullback'
    base_train = _extract_metrics(all_results.get('Train', {}).get(base_key))

    for layer_name, keys in layers.items():
        print(f"\n  [{layer_name}]")
        for cn in keys:
            t_r = all_results.get('Train', {}).get(cn)
            v_r = all_results.get('Val', {}).get(cn)
            t = _extract_metrics(t_r)
            v = _extract_metrics(v_r)

            markers = []
            if cn not in ('A_baseline', 'C_r3_best', base_key):
                if t['mdd'] < base_train['mdd'] * 0.85:
                    markers.append('MDD')
                if t['wr'] > base_train['wr'] + 3:
                    markers.append('WR')
                if t['shrp'] > base_train['shrp'] + 0.05:
                    markers.append('Shrp')
                if t['avg_hold'] > base_train['avg_hold'] * 1.3:
                    markers.append('Hold')
            marker = ' ' + ','.join(markers) if markers else ''

            print(f"  {cn:>20s} | "
                  f"{t['ret']:+6.1f}% {t['cagr']:+5.1f}% {t['shrp']:5.2f} "
                  f"{t['mdd']:4.1f}% {t['calmar']:5.2f} "
                  f"{t['pf']:5.2f} {t['wr']:4.1f}% {t['buys']:>3d} {t['sells']:>3d} "
                  f"{t['avg_hold']:4.0f}d {t['trades']:>4d}  | "
                  f"{v['ret']:+6.1f}% {v['cagr']:+5.1f}% {v['shrp']:5.2f} "
                  f"{v['mdd']:4.1f}% {v['calmar']:5.2f} "
                  f"{v['pf']:5.2f} {v['wr']:4.1f}% {v['buys']:>3d} {v['sells']:>3d} "
                  f"{v['avg_hold']:4.0f}d {v['trades']:>4d} {marker}")

    # ==========================================
    # MDD < 25% 篩選
    # ==========================================
    print(f"\n\n{'='*120}")
    print(f"  MDD < 25% 篩選 (目標: MDD降低, 勝率提升, 持有天數增加)")
    print(f"{'='*120}\n")

    candidates = []
    for cn in CONFIGS:
        if cn in ('A_baseline', 'C_r3_best'):
            continue
        t = _extract_metrics(all_results.get('Train', {}).get(cn))
        v = _extract_metrics(all_results.get('Val', {}).get(cn))
        if t['mdd'] > 0 and abs(t['mdd']) < 25:
            score = (t['shrp'] * 0.30 +
                     v['shrp'] * 0.20 +
                     (1 - abs(t['mdd'])/40) * 0.25 +
                     min(t['wr'], 50) / 100 * 0.15 +
                     min(t['avg_hold'], 60) / 100 * 0.10)
            candidates.append((cn, score, t, v))

    candidates.sort(key=lambda x: -x[1])
    if candidates:
        for i, (cn, score, t, v) in enumerate(candidates[:10]):
            medal = ['1', '2', '3'][i] if i < 3 else ' '
            print(f"  {medal} {cn:>20s}: Score={score:.3f} | "
                  f"T: Ret={t['ret']:+.1f}% Shrp={t['shrp']:.2f} MDD={abs(t['mdd']):.1f}% "
                  f"WR={t['wr']:.0f}% 均持{t['avg_hold']:.0f}d 交易{t['trades']}次 | "
                  f"V: Ret={v['ret']:+.1f}% MDD={abs(v['mdd']):.1f}% "
                  f"— {CONFIGS[cn]['desc']}")
    else:
        print("  沒有找到 MDD < 25% 的配置")

    # ==========================================
    # 邊際效果分析
    # ==========================================
    print(f"\n\n L1 邊際效果 (vs {base_key} Train={base_train['ret']:+.1f}% "
          f"MDD={base_train['mdd']:.1f}% WR={base_train['wr']:.0f}%):\n")
    l1_keys = [k for k in CONFIGS if k.startswith('L1_')]
    l1_effects = []
    for cn in l1_keys:
        t = _extract_metrics(all_results.get('Train', {}).get(cn))
        delta_mdd = t['mdd'] - base_train['mdd']  # negative = better
        delta_wr = t['wr'] - base_train['wr']
        delta_shrp = t['shrp'] - base_train['shrp']
        l1_effects.append((cn, delta_mdd, delta_wr, delta_shrp, t))

    # 按 MDD 改善排序 (越負越好)
    l1_effects.sort(key=lambda x: x[1])
    for cn, d_mdd, d_wr, d_shrp, t in l1_effects:
        print(f"  {cn:>20s}: DMDD={d_mdd:+5.1f}% DWR={d_wr:+4.1f}% DShrp={d_shrp:+.2f} | "
              f"MDD={t['mdd']:.1f}% WR={t['wr']:.0f}% 均持{t['avg_hold']:.0f}d "
              f"— {CONFIGS[cn]['desc']}")

    # ==========================================
    # 最終排名 (綜合 Sharpe + MDD + WR + 持有天數)
    # ==========================================
    print(f"\n\n {'='*80}")
    print(f"  綜合排名 (Score = Sharpe×0.3 + (1-MDD/40)×0.25 + ValShrp×0.2 + WR/100×0.15 + Hold/100×0.1)")
    print(f" {'='*80}\n")

    all_scores = []
    for cn in CONFIGS:
        if cn in ('A_baseline', 'C_r3_best'):
            continue
        t = _extract_metrics(all_results.get('Train', {}).get(cn))
        v = _extract_metrics(all_results.get('Val', {}).get(cn))
        if t['shrp'] == 0 and t['ret'] == 0:
            continue
        score = (t['shrp'] * 0.30 +
                 (1 - abs(t['mdd'])/40) * 0.25 +
                 v['shrp'] * 0.20 +
                 min(t['wr'], 50) / 100 * 0.15 +
                 min(t['avg_hold'], 60) / 100 * 0.10)
        all_scores.append((cn, score, t, v))

    all_scores.sort(key=lambda x: -x[1])
    for i, (cn, score, t, v) in enumerate(all_scores[:15]):
        medal = ['1', '2', '3'][i] if i < 3 else ' '
        mdd_ok = 'MDD<25' if abs(t['mdd']) < 25 else ''
        print(f"  {medal} {cn:>20s}: Score={score:.3f} | "
              f"T: Ret={t['ret']:+.1f}% Shrp={t['shrp']:.2f} MDD={abs(t['mdd']):.1f}% "
              f"WR={t['wr']:.0f}% 均持{t['avg_hold']:.0f}d | "
              f"V: Ret={v['ret']:+.1f}% Shrp={v['shrp']:.2f} MDD={abs(v['mdd']):.1f}% "
              f"{mdd_ok} — {CONFIGS[cn]['desc']}")

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
                path = os.path.join(_output_dir, f'tp_{cn}_{period_name}.csv')
                df.to_csv(path, index=False, encoding='utf-8-sig')

    print(f"\nTrade logs saved to output/tp_*.csv")


if __name__ == '__main__':
    run_backtest()
