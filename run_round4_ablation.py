#!/usr/bin/env python3
"""
Round 4 Ablation: 出場/風控優化 (Phase A) + 選股精煉 (Phase B)
目標: MDD 33.2% → <28%, WR 27.6% → >30%
Training + Validation split.
"""

import sys, os, time, warnings
import pandas as pd
import numpy as np

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

# ============================================================
#  Phase A: 出場/風控優化 (目標: 降 MDD, 維持 Sharpe)
# ============================================================
PHASE_A_CONFIGS = {
    'A_baseline': {
        'desc': '策略A 現狀基準',
        'overrides': {},
    },
    # --- 單因子測試 ---
    'A1_trail': {
        'desc': '獲利追蹤停利 (act15%/dd-8%/lock5%)',
        'overrides': {
            'enable_profit_trailing': True,
            'profit_trail_activation': 15,
            'profit_trail_pct': -8,
            'profit_trail_min_lock': 5,
        },
    },
    'A2_grad_pos': {
        'desc': '漸進式降倉 (neutral=10/weak=8/bear=5/panic=3)',
        'overrides': {
            'enable_dyn_buy_limit': True,
            'dyn_buy_bull': 4, 'dyn_buy_neutral': 3,
            'dyn_buy_weak': 2, 'dyn_buy_bear': 1, 'dyn_buy_panic': 0,
            'enable_dynamic_exposure': True,
            'dyn_max_bull': 12, 'dyn_max_neutral': 10,
            'dyn_max_weak': 8, 'dyn_max_bear': 5, 'dyn_max_panic': 3,
        },
    },
    'A3_tierB_dd60': {
        'desc': 'Tier B drawdown 0.6x (允許60%回撤)',
        'overrides': {
            'tier_b_drawdown': 0.6,
        },
    },
    'A4_tierB_dd50': {
        'desc': 'Tier B drawdown 0.5x (允許50%回撤)',
        'overrides': {
            'tier_b_drawdown': 0.5,
        },
    },
    'A5_s2_no_buf': {
        'desc': 'S2 無緩衝 (跌破季線立即賣)',
        'overrides': {
            's2_buffer_enabled': False,
        },
    },
    'A6_regime': {
        'desc': 'Regime-Adaptive (unsafe: zombie7d/tierB10%/maxBuy2)',
        'overrides': {
            'enable_regime_adaptive': True,
            'regime_unsafe_zombie_days': 7,
            'regime_unsafe_tier_b_net': 10,
            'regime_unsafe_max_new_buy': 2,
        },
    },
    # --- 組合測試 ---
    'A7_combo_mod': {
        'desc': '溫和組合: 漸進降倉+tierB0.6+regime',
        'overrides': {
            'enable_dyn_buy_limit': True,
            'dyn_buy_bull': 4, 'dyn_buy_neutral': 3,
            'dyn_buy_weak': 2, 'dyn_buy_bear': 1, 'dyn_buy_panic': 0,
            'enable_dynamic_exposure': True,
            'dyn_max_bull': 12, 'dyn_max_neutral': 10,
            'dyn_max_weak': 8, 'dyn_max_bear': 5, 'dyn_max_panic': 3,
            'tier_b_drawdown': 0.6,
            'enable_regime_adaptive': True,
            'regime_unsafe_zombie_days': 7,
            'regime_unsafe_tier_b_net': 10,
            'regime_unsafe_max_new_buy': 2,
        },
    },
    'A8_combo_agg': {
        'desc': '激進組合: 漸進降倉+tierB0.5+regime+stop-12+zombie7d',
        'overrides': {
            'enable_dyn_buy_limit': True,
            'dyn_buy_bull': 4, 'dyn_buy_neutral': 3,
            'dyn_buy_weak': 2, 'dyn_buy_bear': 1, 'dyn_buy_panic': 0,
            'enable_dynamic_exposure': True,
            'dyn_max_bull': 12, 'dyn_max_neutral': 10,
            'dyn_max_weak': 8, 'dyn_max_bear': 5, 'dyn_max_panic': 3,
            'tier_b_drawdown': 0.5,
            'enable_regime_adaptive': True,
            'regime_unsafe_zombie_days': 7,
            'regime_unsafe_tier_b_net': 10,
            'regime_unsafe_max_new_buy': 2,
            'hard_stop_net': -12,
            'zombie_hold_days': 7,
            'zombie_net_range': 3.0,
        },
    },
}

# ============================================================
#  Phase B: 選股精煉 (目標: 提升 WR/PF, 維持 Return)
# ============================================================
PHASE_B_CONFIGS = {
    'B1_peer_def': {
        'desc': 'Peer Z-score default (z>1.5扣/z<-1加/z>2.5擋)',
        'overrides': {
            'enable_peer_zscore': True,
        },
    },
    'B2_peer_blk': {
        'desc': 'Peer Z: 純擋買 (z>2.0擋, 不加減分)',
        'overrides': {
            'enable_peer_zscore': True,
            'peer_zscore_penalty': 0,
            'peer_zscore_bonus': 0,
            'peer_zscore_block': 2.0,
        },
    },
    'B3_rs_30': {
        'desc': 'RS filter: 底部30%不買',
        'overrides': {
            'enable_rs_filter': True,
            'rs_lookback': 60,
            'rs_cutoff_bottom_pct': 30,
        },
    },
    'B4_rs_20': {
        'desc': 'RS filter: 底部20%不買 (溫和)',
        'overrides': {
            'enable_rs_filter': True,
            'rs_lookback': 60,
            'rs_cutoff_bottom_pct': 20,
        },
    },
    'B5_bias_tight': {
        'desc': 'Bias收緊: bull=20/neutral=18/bear=15',
        'overrides': {
            'bias_limit_bull': 20,
            'bias_limit_neutral': 18,
            'bias_limit_bear': 15,
        },
    },
    'B6_rs_peer': {
        'desc': 'RS30% + Peer Z-score (雙重篩選)',
        'overrides': {
            'enable_rs_filter': True,
            'rs_lookback': 60,
            'rs_cutoff_bottom_pct': 30,
            'enable_peer_zscore': True,
        },
    },
    'B7_rsi50_rs': {
        'desc': 'RSI>50 + RS30% (雙動能門檻)',
        'overrides': {
            'min_rsi': 50,
            'enable_rs_filter': True,
            'rs_lookback': 60,
            'rs_cutoff_bottom_pct': 30,
        },
    },
}

# Merge all configs
ALL_CONFIGS = {}
ALL_CONFIGS.update(PHASE_A_CONFIGS)
ALL_CONFIGS.update(PHASE_B_CONFIGS)

PERIODS = {
    'Training': ('2021-01-01', '2025-06-30'),
    'Validation': ('2025-07-01', '2026-03-27'),
}


def _metrics(res):
    ret = res.get('total_return_pct', 0)
    sharpe = res.get('sharpe_ratio', 0)
    mdd = res.get('mdd_pct', 0)
    calmar = res.get('calmar_ratio', 0)
    wr = res.get('win_rate', 0)
    tl = res.get('trade_log', [])
    buys = sum(1 for t in tl if t['type'] == 'BUY')
    sells = [t for t in tl if t['type'] == 'SELL' and t.get('profit') is not None]
    gw = sum(t['profit'] for t in sells if t['profit'] > 0)
    gl = abs(sum(t['profit'] for t in sells if t['profit'] <= 0))
    pf = gw / gl if gl > 0 else 999
    return ret, sharpe, mdd, calmar, pf, wr, buys, len(sells)


def run_ablation():
    stocks = get_stocks_by_industry(INDUSTRY)
    print(f"{'='*120}")
    print(f"  Round 4: 出場/風控優化 (Phase A) + 選股精煉 (Phase B)")
    print(f"  {INDUSTRY}: {len(stocks)} stocks | Capital: ${INITIAL_CAPITAL:,}")
    print(f"  共 {len(ALL_CONFIGS)} 組測試")
    print(f"{'='*120}\n")

    all_period_results = {}

    for period_name, (start_date, end_date) in PERIODS.items():
        print(f"\n{'='*100}")
        print(f"  📅 {period_name}: {start_date} ~ {end_date}")
        print(f"{'='*100}")

        market_map = reconstruct_market_history(start_date, end_date)
        dl_start = (pd.Timestamp(start_date) - pd.Timedelta(days=250)).strftime('%Y-%m-%d')
        dl_end = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        preloaded_data, _ = batch_download_stocks(
            stocks, dl_start, dl_end, min_data_days=MIN_DATA_DAYS)
        print(f"  Valid: {len(preloaded_data)}\n")

        period_results = {}
        for config_name, cfg_entry in ALL_CONFIGS.items():
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
                ret, sharpe, mdd, calmar, pf, wr, buys, sells = _metrics(result)
                print(f"  {config_name:>14s}: Ret {ret:+7.1f}% CAGR {result.get('cagr_pct',0):+5.1f}% "
                      f"Shrp {sharpe:5.2f} MDD {mdd:5.1f}% Clm {calmar:5.2f} "
                      f"PF {pf:4.2f} WR{wr:4.1f}% 買{buys}次 賣{sells}次 "
                      f"⏱{elapsed:.0f}s — {cfg_entry['desc']}")

        all_period_results[period_name] = period_results

    # === Phase A Summary ===
    print(f"\n\n{'='*130}")
    print(f"  📊 PHASE A: 出場/風控優化 SUMMARY")
    print(f"{'='*130}")

    header_a = (f"{'Config':>14s} | {'T-Ret':>7s} {'T-Shrp':>6s} {'T-MDD':>6s} {'T-Clm':>6s} {'T-PF':>5s} {'T-WR':>5s} | "
                f"{'V-Ret':>7s} {'V-Shrp':>6s} {'V-MDD':>6s} {'V-Clm':>6s} {'V-PF':>5s} {'V-WR':>5s} | "
                f"{'ΔMDD-T':>7s} {'ΔMDD-V':>7s} {'判定':>8s}")
    print(header_a)
    print("-" * len(header_a))

    bl_t = all_period_results.get('Training', {}).get('A_baseline')
    bl_v = all_period_results.get('Validation', {}).get('A_baseline')
    bl_mdd_t = bl_t.get('mdd_pct', 0) if bl_t else 0
    bl_mdd_v = bl_v.get('mdd_pct', 0) if bl_v else 0
    bl_sharpe_t = bl_t.get('sharpe_ratio', 0) if bl_t else 0
    bl_sharpe_v = bl_v.get('sharpe_ratio', 0) if bl_v else 0

    for config_name in PHASE_A_CONFIGS:
        t_res = all_period_results.get('Training', {}).get(config_name)
        v_res = all_period_results.get('Validation', {}).get(config_name)
        if not t_res or not v_res:
            continue

        t_ret, t_sharpe, t_mdd, t_calmar, t_pf, t_wr, _, _ = _metrics(t_res)
        v_ret, v_sharpe, v_mdd, v_calmar, v_pf, v_wr, _, _ = _metrics(v_res)

        d_mdd_t = t_mdd - bl_mdd_t
        d_mdd_v = v_mdd - bl_mdd_v

        marker = ""
        if config_name != 'A_baseline':
            t_better = (t_mdd < bl_mdd_t - 1) and (t_sharpe >= bl_sharpe_t * 0.9)
            v_ok = (v_mdd <= bl_mdd_v + 1) and (v_sharpe >= bl_sharpe_v * 0.9)
            if t_better and v_ok:
                marker = "✅ 雙贏"
            elif t_better and not v_ok:
                marker = "⚠️ 驗證差"
            elif t_mdd < bl_mdd_t - 1:
                marker = "🔍 MDD降"
            else:
                marker = "❌ 無效"

        print(f"{config_name:>14s} | {t_ret:+6.1f}% {t_sharpe:6.2f} {t_mdd:5.1f}% {t_calmar:6.2f} {t_pf:5.2f} {t_wr:4.1f}% | "
              f"{v_ret:+6.1f}% {v_sharpe:6.2f} {v_mdd:5.1f}% {v_calmar:6.2f} {v_pf:5.2f} {v_wr:4.1f}% | "
              f"{d_mdd_t:+6.1f}% {d_mdd_v:+6.1f}% {marker}")

    # === Phase B Summary ===
    print(f"\n\n{'='*130}")
    print(f"  📊 PHASE B: 選股精煉 SUMMARY")
    print(f"{'='*130}")

    # Reuse baseline from Phase A
    bl_wr_t = _metrics(bl_t)[5] if bl_t else 0
    bl_wr_v = _metrics(bl_v)[5] if bl_v else 0
    bl_pf_t = _metrics(bl_t)[4] if bl_t else 0
    bl_pf_v = _metrics(bl_v)[4] if bl_v else 0
    bl_ret_t = _metrics(bl_t)[0] if bl_t else 0

    header_b = (f"{'Config':>14s} | {'T-Ret':>7s} {'T-Shrp':>6s} {'T-MDD':>6s} {'T-PF':>5s} {'T-WR':>5s} | "
                f"{'V-Ret':>7s} {'V-Shrp':>6s} {'V-MDD':>6s} {'V-PF':>5s} {'V-WR':>5s} | "
                f"{'ΔWR-T':>6s} {'ΔPF-T':>6s} {'ΔWR-V':>6s} {'判定':>8s}")
    print(header_b)
    print("-" * len(header_b))

    # Print baseline row first
    if bl_t and bl_v:
        t_ret, t_sharpe, t_mdd, t_calmar, t_pf, t_wr, _, _ = _metrics(bl_t)
        v_ret, v_sharpe, v_mdd, v_calmar, v_pf, v_wr, _, _ = _metrics(bl_v)
        print(f"{'A_baseline':>14s} | {t_ret:+6.1f}% {t_sharpe:6.2f} {t_mdd:5.1f}% {t_pf:5.2f} {t_wr:4.1f}% | "
              f"{v_ret:+6.1f}% {v_sharpe:6.2f} {v_mdd:5.1f}% {v_pf:5.2f} {v_wr:4.1f}% | "
              f"{'---':>6s} {'---':>6s} {'---':>6s} {'基準':>8s}")

    for config_name in PHASE_B_CONFIGS:
        t_res = all_period_results.get('Training', {}).get(config_name)
        v_res = all_period_results.get('Validation', {}).get(config_name)
        if not t_res or not v_res:
            continue

        t_ret, t_sharpe, t_mdd, t_calmar, t_pf, t_wr, _, _ = _metrics(t_res)
        v_ret, v_sharpe, v_mdd, v_calmar, v_pf, v_wr, _, _ = _metrics(v_res)

        d_wr_t = t_wr - bl_wr_t
        d_pf_t = t_pf - bl_pf_t
        d_wr_v = v_wr - bl_wr_v

        marker = ""
        t_improved = (d_wr_t > 2 or d_pf_t > 0.1) and (t_ret >= bl_ret_t * 0.95)
        v_ok = (v_sharpe >= bl_sharpe_v * 0.9) and (v_wr >= bl_wr_v - 2)
        if t_improved and v_ok:
            marker = "✅ 雙贏"
        elif t_improved and not v_ok:
            marker = "⚠️ 驗證差"
        elif d_wr_t > 1 or d_pf_t > 0.05:
            marker = "🔍 微升"
        else:
            marker = "❌ 無效"

        print(f"{config_name:>14s} | {t_ret:+6.1f}% {t_sharpe:6.2f} {t_mdd:5.1f}% {t_pf:5.2f} {t_wr:4.1f}% | "
              f"{v_ret:+6.1f}% {v_sharpe:6.2f} {v_mdd:5.1f}% {v_pf:5.2f} {v_wr:4.1f}% | "
              f"{d_wr_t:+5.1f}% {d_pf_t:+5.2f} {d_wr_v:+5.1f}% {marker}")

    # === Legend ===
    print(f"\n{'='*130}")
    print("  Phase A 判定: ✅雙贏=Train MDD降>1pp+Sharpe≥90%+Val不劣化 | ⚠️驗證差=Train好Val差 | 🔍MDD降=MDD降但Sharpe掉 | ❌無效")
    print("  Phase B 判定: ✅雙贏=Train WR升>2pp或PF升>0.1+Ret≥95%+Val持平 | ⚠️驗證差=Train好Val差 | 🔍微升=小幅改善 | ❌無效")
    print(f"{'='*130}")


if __name__ == '__main__':
    run_ablation()
