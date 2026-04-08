#!/usr/bin/env python3
"""Conviction Hold + Regime-Adaptive Ablation (630K, rsi40 baseline)"""

import sys, os, time, warnings
import pandas as pd, numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from group_backtest import (run_group_backtest, reconstruct_market_history,
                            INDUSTRY_CONFIGS, MIN_DATA_DAYS)
from industry_manager import get_stocks_by_industry
from stock_utils import batch_download_stocks

INDUSTRY = '半導體業'
INITIAL_CAPITAL = 630_000
BUDGET_PER_TRADE = 17_500
EXEC_MODE = 'next_open'
BASE_CONFIG = INDUSTRY_CONFIGS[INDUSTRY]['config'].copy()

CONFIGS = {
    'baseline': {
        'desc': '現狀 (rsi40)',
        'overrides': {},
    },

    # === A: Conviction Hold variants ===
    'conv_default': {
        'desc': 'Conviction: ≥3次, zombie+5天/±3%, stop-3%, tierA+15%',
        'overrides': {'enable_conviction_hold': True},
    },
    'conv_2buys': {
        'desc': 'Conviction: ≥2次觸發 (寬鬆)',
        'overrides': {'enable_conviction_hold': True, 'conviction_min_buys': 2},
    },
    'conv_zombie_only': {
        'desc': 'Conviction: 只放寬zombie (不改stop/tier)',
        'overrides': {
            'enable_conviction_hold': True,
            'conviction_stop_extra': 0,
            'conviction_tier_a_extra': 0,
        },
    },
    'conv_tier_only': {
        'desc': 'Conviction: 只放寬tierA (+20%)',
        'overrides': {
            'enable_conviction_hold': True,
            'conviction_zombie_extra': 0,
            'conviction_zombie_range': 0,
            'conviction_stop_extra': 0,
            'conviction_tier_a_extra': 20.0,
        },
    },
    'conv_stop_only': {
        'desc': 'Conviction: 只放寬stop (-5%)',
        'overrides': {
            'enable_conviction_hold': True,
            'conviction_zombie_extra': 0,
            'conviction_zombie_range': 0,
            'conviction_stop_extra': 5.0,
            'conviction_tier_a_extra': 0,
        },
    },

    # === B: Regime-Adaptive variants ===
    'regime_default': {
        'desc': 'Regime: unsafe→zombie7天/tierB10%/買1檔',
        'overrides': {'enable_regime_adaptive': True},
    },
    'regime_zombie_only': {
        'desc': 'Regime: unsafe→zombie7天 (其他不變)',
        'overrides': {
            'enable_regime_adaptive': True,
            'regime_unsafe_tier_b_net': 15,  # same as normal
            'regime_unsafe_max_new_buy': 99,  # no limit
        },
    },
    'regime_buy_limit': {
        'desc': 'Regime: unsafe→每日最多買1檔 (其他不變)',
        'overrides': {
            'enable_regime_adaptive': True,
            'regime_unsafe_zombie_days': 99,  # no change
            'regime_unsafe_tier_b_net': 15,   # same as normal
            'regime_unsafe_max_new_buy': 1,
        },
    },
    'regime_tierb_only': {
        'desc': 'Regime: unsafe→tierB 10% 提早停利 (其他不變)',
        'overrides': {
            'enable_regime_adaptive': True,
            'regime_unsafe_zombie_days': 99,
            'regime_unsafe_tier_b_net': 10,
            'regime_unsafe_max_new_buy': 99,
        },
    },
    'regime_no_buy': {
        'desc': 'Regime: unsafe→完全不買',
        'overrides': {
            'enable_regime_adaptive': True,
            'regime_unsafe_zombie_days': 99,
            'regime_unsafe_tier_b_net': 15,
            'regime_unsafe_max_new_buy': 0,
        },
    },

    # === C: Combinations ===
    'conv_regime': {
        'desc': 'Conviction + Regime 全開',
        'overrides': {
            'enable_conviction_hold': True,
            'enable_regime_adaptive': True,
        },
    },
}

PERIODS = {
    'Train': ('2022-01-01', '2025-06-30'),
    'Val':   ('2025-07-01', '2026-03-27'),
}

def run_ablation():
    stocks = get_stocks_by_industry(INDUSTRY)
    print(f"{INDUSTRY}: {len(stocks)} stocks, Capital: ${INITIAL_CAPITAL:,}\n")

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
            rc = BASE_CONFIG.copy()
            rc.update(ce['overrides'])
            t0 = time.time()
            r = run_group_backtest(stocks, start, end, BUDGET_PER_TRADE, mm,
                                   exec_mode=EXEC_MODE, config_override=rc,
                                   initial_capital=INITIAL_CAPITAL, preloaded_data=data)
            el = time.time() - t0
            if r:
                pr[cn] = r
                ret=r.get('total_return_pct',0); sh=r.get('sharpe_ratio',0)
                mdd=r.get('mdd_pct',0); clm=r.get('calmar_ratio',0)
                tl=r.get('trade_log',[]); sells=[t for t in tl if t['type']=='SELL' and t.get('profit') is not None]
                gw=sum(t['profit'] for t in sells if t['profit']>0)
                gl=abs(sum(t['profit'] for t in sells if t['profit']<=0))
                pf=gw/gl if gl>0 else 999
                wr=sum(1 for s in sells if s['profit']>0)/len(sells)*100 if sells else 0
                print(f"  {cn:>16s}: Ret{ret:+7.1f}% Shrp{sh:5.2f} MDD{mdd:5.1f}% Clmr{clm:5.2f} PF{pf:4.2f} WR{wr:4.1f}% ⏱{el:.0f}s — {ce['desc']}")
        all_results[period_name] = pr

    # Summary
    print(f"\n\n{'='*110}")
    print(f"  📊 CONVICTION HOLD + REGIME-ADAPTIVE ABLATION SUMMARY")
    print(f"{'='*110}")

    bl_t = all_results.get('Train',{}).get('baseline')
    bl_v = all_results.get('Val',{}).get('baseline')
    bl_st = bl_t.get('sharpe_ratio',0) if bl_t else 0
    bl_sv = bl_v.get('sharpe_ratio',0) if bl_v else 0

    header = f"{'Config':>16s} | {'Train Ret':>9s} {'Shrp':>5s} {'MDD':>6s} {'Clmr':>5s} {'PF':>5s} {'WR':>5s} | {'Val Ret':>8s} {'Shrp':>5s} {'MDD':>6s} {'Clmr':>5s} {'PF':>5s} | {'判定':>8s}"
    print(header)
    print("-"*len(header))

    for cn in CONFIGS:
        t = all_results.get('Train',{}).get(cn)
        v = all_results.get('Val',{}).get(cn)
        if not t or not v: continue
        def _m(r):
            tl=r.get('trade_log',[]); sells=[x for x in tl if x['type']=='SELL' and x.get('profit') is not None]
            gw=sum(x['profit'] for x in sells if x['profit']>0); gl=abs(sum(x['profit'] for x in sells if x['profit']<=0))
            wr=sum(1 for s in sells if s['profit']>0)/len(sells)*100 if sells else 0
            return (r.get('total_return_pct',0),r.get('sharpe_ratio',0),r.get('mdd_pct',0),r.get('calmar_ratio',0),gw/gl if gl>0 else 999,wr)
        tr,ts,tm,tc,tp,tw=_m(t); vr,vs,vm,vc,vp,vw=_m(v)
        mk=""
        if cn != 'baseline':
            t_better = ts > bl_st * 1.02  # Train Sharpe 提升 >2%
            v_ok = vs >= bl_sv * 0.90     # Val Sharpe 不掉 >10%
            if t_better and v_ok: mk="✅ 採用"
            elif ts >= bl_st * 0.98 and v_ok: mk="🟡 持平"
            elif t_better: mk="⚠️ Val差"
            else: mk="❌"
        print(f"{cn:>16s} | {tr:+8.1f}% {ts:5.2f} {tm:5.1f}% {tc:5.2f} {tp:5.2f} {tw:4.1f}% | {vr:+7.1f}% {vs:5.2f} {vm:5.1f}% {vc:5.2f} {vp:5.2f} | {mk}")

    print(f"\n{'='*110}")

if __name__ == '__main__':
    run_ablation()
