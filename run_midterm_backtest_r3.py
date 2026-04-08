#!/usr/bin/env python3
"""
策略 C — Round 3 精細組合優化

Round 2 各維度最佳:
  D1: 超寬回檔 (-10%~+8%)    Train +39.7%  ← 核心
  D6: MA60 連5天出場          Train +26.1%  ← 第二大
  D10: 量縮 vol<1.5x          Train +21.0%  ← 第三大
  D2: RSI≤60                  Train +14.0%
  D3: 停利 25%/10%            Train +12.5%
  D7: Zombie 45天/3%          Train +9.1%
  D5: 3檔集中                 Train +6.2%
  D4: 只擋 bear               Train +4.8% (預設)
  D8: 停損 -15%               Train +3.1%

Round 3 策略:
  1. 以 Top-3 因子 (超寬回檔 + MA60_5d + vol1.5x) 為核心底座
  2. 逐步疊加其他因子，觀察邊際效果
  3. 最終精選 3-4 個最佳全組合
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
# Round 3 基底: Top-3 因子組合
# ==========================================
R3_BASE = {
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

    # -- 中長期策略參數 --
    **MIDTERM_CONFIG,

    # === Round 3 核心底座: Top-3 因子 ===
    'mt_pullback_lo':       -10.0,    # D1_vwide: 超寬回檔
    'mt_pullback_hi':        8.0,
    'mt_ma60_break_days':    5,       # D6_ma60_5d: 連5天才出場
    'mt_vol_shrink_ratio':   1.5,     # D10_vol1.5: 微放寬量縮
}


def _mk(desc, overrides, max_pos=4, budget_pct=15.0):
    cfg = {**R3_BASE, **overrides}
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


CONFIGS = {}

# === Baselines ===
CONFIGS['A_baseline'] = {
    'desc': '策略A: 短線動能',
    'signal_func': None,
    'config': BASE_A,
    'budget': 25_000,
    'capital': INITIAL_CAPITAL,
}
CONFIGS['R3_core'] = _mk('R3底座: vwide+MA60_5d+vol1.5', {})

# ==========================================
# Layer 1: 在底座上 +1 因子 (邊際效果)
# ==========================================
CONFIGS['L1_rsi60'] = _mk(
    '+RSI≤60',
    {'mt_rsi_max': 60})

CONFIGS['L1_trail25'] = _mk(
    '+停利25%/10%',
    {'mt_trail_trigger': 25.0, 'mt_trail_drop': 10.0})

CONFIGS['L1_zombie45'] = _mk(
    '+Zombie 45天/3%',
    {'mt_zombie_days': 45, 'mt_zombie_min_profit': 3.0,
     'zombie_hold_days': 45, 'zombie_net_range': 3.0})

CONFIGS['L1_3pos'] = _mk(
    '+3檔集中 (18%)',
    {}, max_pos=3, budget_pct=18.0)

CONFIGS['L1_stop15'] = _mk(
    '+停損-15%',
    {'mt_hard_stop_pct': -15.0, 'hard_stop_net': -15.0})

CONFIGS['L1_no_weekly'] = _mk(
    '+不看週線',
    {'mt_require_weekly': False})

CONFIGS['L1_trail60_20'] = _mk(
    '+停利60%/20% (寬鬆)',
    {'mt_trail_trigger': 60.0, 'mt_trail_drop': 20.0})

# ==========================================
# Layer 2: 底座 + 2 因子 (最佳 pair)
# ==========================================
CONFIGS['L2_rsi60_trail25'] = _mk(
    '+RSI60+停利25/10',
    {'mt_rsi_max': 60,
     'mt_trail_trigger': 25.0, 'mt_trail_drop': 10.0})

CONFIGS['L2_rsi60_zombie45'] = _mk(
    '+RSI60+Zombie45',
    {'mt_rsi_max': 60,
     'mt_zombie_days': 45, 'mt_zombie_min_profit': 3.0,
     'zombie_hold_days': 45, 'zombie_net_range': 3.0})

CONFIGS['L2_rsi60_3pos'] = _mk(
    '+RSI60+3檔',
    {'mt_rsi_max': 60}, max_pos=3, budget_pct=18.0)

CONFIGS['L2_trail25_zombie45'] = _mk(
    '+停利25/10+Zombie45',
    {'mt_trail_trigger': 25.0, 'mt_trail_drop': 10.0,
     'mt_zombie_days': 45, 'mt_zombie_min_profit': 3.0,
     'zombie_hold_days': 45, 'zombie_net_range': 3.0})

CONFIGS['L2_trail25_3pos'] = _mk(
    '+停利25/10+3檔',
    {'mt_trail_trigger': 25.0, 'mt_trail_drop': 10.0},
    max_pos=3, budget_pct=18.0)

CONFIGS['L2_zombie45_3pos'] = _mk(
    '+Zombie45+3檔',
    {'mt_zombie_days': 45, 'mt_zombie_min_profit': 3.0,
     'zombie_hold_days': 45, 'zombie_net_range': 3.0},
    max_pos=3, budget_pct=18.0)

CONFIGS['L2_rsi60_noweekly'] = _mk(
    '+RSI60+不看週線',
    {'mt_rsi_max': 60, 'mt_require_weekly': False})

# ==========================================
# Layer 3: 底座 + 3~4 因子 (精選全組合)
# ==========================================
CONFIGS['L3_best3'] = _mk(
    '+RSI60+停利25/10+Zombie45',
    {'mt_rsi_max': 60,
     'mt_trail_trigger': 25.0, 'mt_trail_drop': 10.0,
     'mt_zombie_days': 45, 'mt_zombie_min_profit': 3.0,
     'zombie_hold_days': 45, 'zombie_net_range': 3.0})

CONFIGS['L3_best3_3pos'] = _mk(
    '+RSI60+停利25/10+Zombie45+3檔',
    {'mt_rsi_max': 60,
     'mt_trail_trigger': 25.0, 'mt_trail_drop': 10.0,
     'mt_zombie_days': 45, 'mt_zombie_min_profit': 3.0,
     'zombie_hold_days': 45, 'zombie_net_range': 3.0},
    max_pos=3, budget_pct=18.0)

CONFIGS['L3_best4'] = _mk(
    '+RSI60+停利25/10+Zombie45+停損15%',
    {'mt_rsi_max': 60,
     'mt_trail_trigger': 25.0, 'mt_trail_drop': 10.0,
     'mt_zombie_days': 45, 'mt_zombie_min_profit': 3.0,
     'zombie_hold_days': 45, 'zombie_net_range': 3.0,
     'mt_hard_stop_pct': -15.0, 'hard_stop_net': -15.0})

CONFIGS['L3_best4_3pos'] = _mk(
    '+RSI60+停利25/10+Zombie45+停損15%+3檔',
    {'mt_rsi_max': 60,
     'mt_trail_trigger': 25.0, 'mt_trail_drop': 10.0,
     'mt_zombie_days': 45, 'mt_zombie_min_profit': 3.0,
     'zombie_hold_days': 45, 'zombie_net_range': 3.0,
     'mt_hard_stop_pct': -15.0, 'hard_stop_net': -15.0},
    max_pos=3, budget_pct=18.0)

# ==========================================
# Layer 4: 最終候選 (微調停利)
# ==========================================
CONFIGS['FINAL_A'] = _mk(
    'Final: RSI60+trail25/10+Z45+3pos',
    {'mt_rsi_max': 60,
     'mt_trail_trigger': 25.0, 'mt_trail_drop': 10.0,
     'mt_zombie_days': 45, 'mt_zombie_min_profit': 3.0,
     'zombie_hold_days': 45, 'zombie_net_range': 3.0},
    max_pos=3, budget_pct=18.0)

CONFIGS['FINAL_B'] = _mk(
    'Final: RSI60+trail60/20+Z45+3pos',
    {'mt_rsi_max': 60,
     'mt_trail_trigger': 60.0, 'mt_trail_drop': 20.0,
     'mt_zombie_days': 45, 'mt_zombie_min_profit': 3.0,
     'zombie_hold_days': 45, 'zombie_net_range': 3.0},
    max_pos=3, budget_pct=18.0)

CONFIGS['FINAL_C'] = _mk(
    'Final: RSI60+trail40/15+Z45+4pos (原始停利)',
    {'mt_rsi_max': 60,
     'mt_zombie_days': 45, 'mt_zombie_min_profit': 3.0,
     'zombie_hold_days': 45, 'zombie_net_range': 3.0})

CONFIGS['FINAL_D'] = _mk(
    'Final: RSI60+trail25/10+Z45+4pos+noWeekly',
    {'mt_rsi_max': 60,
     'mt_trail_trigger': 25.0, 'mt_trail_drop': 10.0,
     'mt_zombie_days': 45, 'mt_zombie_min_profit': 3.0,
     'zombie_hold_days': 45, 'zombie_net_range': 3.0,
     'mt_require_weekly': False})

CONFIGS['FINAL_E'] = _mk(
    'Final: RSI60+trail25/10+Z45+3pos+noWeekly',
    {'mt_rsi_max': 60,
     'mt_trail_trigger': 25.0, 'mt_trail_drop': 10.0,
     'mt_zombie_days': 45, 'mt_zombie_min_profit': 3.0,
     'zombie_hold_days': 45, 'zombie_net_range': 3.0,
     'mt_require_weekly': False},
    max_pos=3, budget_pct=18.0)

CONFIGS['FINAL_F'] = _mk(
    'Final: RSI60+trail40/15+Z45+3pos+noWeekly',
    {'mt_rsi_max': 60,
     'mt_zombie_days': 45, 'mt_zombie_min_profit': 3.0,
     'zombie_hold_days': 45, 'zombie_net_range': 3.0,
     'mt_require_weekly': False},
    max_pos=3, budget_pct=18.0)


PERIODS = {
    'Train': ('2021-01-01', '2025-06-30'),
    'Val':   ('2025-07-01', '2026-03-28'),
}


def _extract_metrics(r):
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
    print(f"  策略 C — Round 3: 精細組合優化")
    print(f"  底座: 超寬回檔(-10%~+8%) + MA60連5天 + 量縮1.5x")
    print(f"  {INDUSTRY}: {len(stocks)} stocks | Capital: ${INITIAL_CAPITAL:,}")
    print(f"  共 {len(CONFIGS)} 組測試")
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
                print(f"  {cn:>20s}: Ret{m['ret']:+7.1f}% CAGR{m['cagr']:+6.1f}% "
                      f"Shrp{m['shrp']:5.2f} MDD{m['mdd']:5.1f}% Clm{m['calmar']:5.2f} "
                      f"PF{m['pf']:5.2f} WR{m['wr']:4.1f}% "
                      f"買{m['buys']:>3d}次 賣{m['sells']:>3d}次 "
                      f"均持{m['avg_hold']:.0f}天 ⏱{el:.0f}s — {ce['desc']}")
            else:
                print(f"  {cn:>20s}: ❌ No result — {ce['desc']}")
                pr[cn] = None
        all_results[period_name] = pr

    # ==========================================
    # Summary Table
    # ==========================================
    print(f"\n\n{'='*160}")
    print(f"  📊 Round 3 完整結果 — 底座: vwide+MA60_5d+vol1.5x，逐層疊加")
    print(f"{'='*160}")

    header = (f"{'Config':>20s} | {'--- Train ---':>68s} | {'--- Val ---':>68s}")
    sub    = (f"{'':>20s} | {'Ret':>7s} {'CAGR':>6s} {'Shrp':>5s} {'MDD':>5s} {'Clm':>5s} "
              f"{'PF':>5s} {'WR':>5s} {'買':>3s} {'賣':>3s} {'均持':>4s} | "
              f"{'Ret':>7s} {'CAGR':>6s} {'Shrp':>5s} {'MDD':>5s} {'Clm':>5s} "
              f"{'PF':>5s} {'WR':>5s} {'買':>3s} {'賣':>3s} {'均持':>4s}")
    print(header)
    print(sub)
    print("-" * 160)

    layers = {
        'Baseline':  ['A_baseline', 'R3_core'],
        'L1 (+1因子)': [k for k in CONFIGS if k.startswith('L1_')],
        'L2 (+2因子)': [k for k in CONFIGS if k.startswith('L2_')],
        'L3 (+3~4因子)': [k for k in CONFIGS if k.startswith('L3_')],
        'FINAL 候選': [k for k in CONFIGS if k.startswith('FINAL_')],
    }

    core_train = _extract_metrics(all_results.get('Train', {}).get('R3_core'))

    for layer_name, keys in layers.items():
        print(f"\n  【{layer_name}】")
        for cn in keys:
            t_r = all_results.get('Train', {}).get(cn)
            v_r = all_results.get('Val', {}).get(cn)
            t = _extract_metrics(t_r)
            v = _extract_metrics(v_r)

            markers = []
            if cn not in ('A_baseline', 'R3_core'):
                if t['ret'] > core_train['ret'] * 1.05:
                    markers.append('↑ret')
                if t['shrp'] > core_train['shrp'] + 0.05:
                    markers.append('↑shrp')
                if t['mdd'] < core_train['mdd'] * 0.9:
                    markers.append('↓mdd')
            marker = ' ⭐' + ','.join(markers) if markers else ''

            print(f"  {cn:>20s} | "
                  f"{t['ret']:+6.1f}% {t['cagr']:+5.1f}% {t['shrp']:5.2f} "
                  f"{t['mdd']:4.1f}% {t['calmar']:5.2f} "
                  f"{t['pf']:5.2f} {t['wr']:4.1f}% {t['buys']:>3d} {t['sells']:>3d} {t['avg_hold']:4.0f}d | "
                  f"{v['ret']:+6.1f}% {v['cagr']:+5.1f}% {v['shrp']:5.2f} "
                  f"{v['mdd']:4.1f}% {v['calmar']:5.2f} "
                  f"{v['pf']:5.2f} {v['wr']:4.1f}% {v['buys']:>3d} {v['sells']:>3d} {v['avg_hold']:4.0f}d{marker}")

    # ==========================================
    # 邊際效果分析
    # ==========================================
    print(f"\n\n📋 L1 邊際效果 (vs R3_core Train={core_train['ret']:+.1f}%):\n")
    l1_keys = [k for k in CONFIGS if k.startswith('L1_')]
    l1_effects = []
    for cn in l1_keys:
        t = _extract_metrics(all_results.get('Train', {}).get(cn))
        v = _extract_metrics(all_results.get('Val', {}).get(cn))
        delta_ret = t['ret'] - core_train['ret']
        delta_shrp = t['shrp'] - core_train['shrp']
        l1_effects.append((cn, delta_ret, delta_shrp, t, v))
    l1_effects.sort(key=lambda x: -x[1])

    for cn, dr, ds, t, v in l1_effects:
        print(f"  {cn:>16s}: ΔRet={dr:+5.1f}% ΔShrp={ds:+.2f} | "
              f"Train={t['ret']:+.1f}% Shrp={t['shrp']:.2f} MDD={t['mdd']:.1f}% | "
              f"Val={v['ret']:+.1f}% Shrp={v['shrp']:.2f} — {CONFIGS[cn]['desc']}")

    # ==========================================
    # FINAL 排名
    # ==========================================
    print(f"\n\n🏆 FINAL 候選排名 (綜合分 = Train_Shrp×0.35 + Val_Shrp×0.25 + Train_Ret/100×0.2 + (1-MDD/40)×0.2):\n")
    final_keys = [k for k in CONFIGS if k.startswith('FINAL_')]
    final_scores = []
    for cn in final_keys:
        t = _extract_metrics(all_results.get('Train', {}).get(cn))
        v = _extract_metrics(all_results.get('Val', {}).get(cn))
        score = (t['shrp'] * 0.35 +
                 v['shrp'] * 0.25 +
                 t['ret'] / 100 * 0.2 +
                 (1 - abs(t['mdd'])/40) * 0.2)
        final_scores.append((cn, score, t, v))
    final_scores.sort(key=lambda x: -x[1])

    for i, (cn, score, t, v) in enumerate(final_scores):
        medal = ['🥇','🥈','🥉'][i] if i < 3 else '  '
        print(f"  {medal} {cn:>12s}: Score={score:.3f} | "
              f"Train: Ret={t['ret']:+.1f}% CAGR={t['cagr']:+.1f}% Shrp={t['shrp']:.2f} "
              f"MDD={t['mdd']:.1f}% PF={t['pf']:.2f} WR={t['wr']:.0f}% | "
              f"Val: Ret={v['ret']:+.1f}% Shrp={v['shrp']:.2f} MDD={v['mdd']:.1f}%"
              f" — {CONFIGS[cn]['desc']}")

    # ==========================================
    # 最終推薦
    # ==========================================
    if final_scores:
        best_cn, best_score, best_t, best_v = final_scores[0]
        print(f"\n\n{'='*80}")
        print(f"  🎯 最終推薦: {best_cn}")
        print(f"  {CONFIGS[best_cn]['desc']}")
        print(f"  Train: Ret={best_t['ret']:+.1f}% CAGR={best_t['cagr']:+.1f}% "
              f"Sharpe={best_t['shrp']:.2f} MDD={best_t['mdd']:.1f}% "
              f"PF={best_t['pf']:.2f} WR={best_t['wr']:.0f}%")
        print(f"  Val:   Ret={best_v['ret']:+.1f}% CAGR={best_v['cagr']:+.1f}% "
              f"Sharpe={best_v['shrp']:.2f} MDD={best_v['mdd']:.1f}%")
        print(f"{'='*80}")

        # 輸出最終策略參數
        print(f"\n  📝 策略參數:")
        best_cfg = CONFIGS[best_cn]['config']
        params = [
            ('回檔區間', f"{best_cfg['mt_pullback_lo']}% ~ +{best_cfg['mt_pullback_hi']}%"),
            ('MA60出場', f"連{best_cfg['mt_ma60_break_days']}天跌破"),
            ('量縮門檻', f"vol < {best_cfg['mt_vol_shrink_ratio']}x MA5"),
            ('RSI上限', f"≤{best_cfg['mt_rsi_max']}"),
            ('停利', f"{best_cfg['mt_trail_trigger']}%觸發/{best_cfg['mt_trail_drop']}%回撤"),
            ('硬停損', f"{best_cfg['mt_hard_stop_pct']}%"),
            ('Zombie', f"{best_cfg.get('mt_zombie_days',90)}天/{best_cfg.get('mt_zombie_min_profit',5)}%"),
            ('持股數', f"{best_cfg['max_positions']}檔 (每檔{best_cfg['budget_pct']}%)"),
            ('大盤過濾', f"bear={'擋' if best_cfg['mt_block_bear'] else '不擋'} panic={'擋' if best_cfg['mt_block_panic'] else '不擋'}"),
            ('週線過濾', '開' if best_cfg['mt_require_weekly'] else '關'),
        ]
        for name, val in params:
            print(f"    {name:>8s}: {val}")

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
                path = os.path.join(_output_dir, f'midterm_r3_{cn}_{period_name}.csv')
                df.to_csv(path, index=False, encoding='utf-8-sig')

    print(f"\nTrade logs saved to output/midterm_r3_*.csv")


if __name__ == '__main__':
    run_backtest()
