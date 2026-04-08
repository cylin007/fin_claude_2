#!/usr/bin/env python3
"""Profit Trailing Stop Ablation (630K, rsi40 baseline)"""

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
        'desc': '現狀 (rsi40, trailing關)',
        'overrides': {},
    },
    # === Activation threshold variants ===
    'trail_act10_dd6': {
        'desc': '啟動10%, 回撤6%',
        'overrides': {
            'enable_profit_trailing': True,
            'profit_trail_activation': 10,
            'profit_trail_pct': -6,
            'profit_trail_min_lock': 3,
        },
    },
    'trail_act15_dd8': {
        'desc': '啟動15%, 回撤8% (預設)',
        'overrides': {
            'enable_profit_trailing': True,
            'profit_trail_activation': 15,
            'profit_trail_pct': -8,
            'profit_trail_min_lock': 5,
        },
    },
    'trail_act20_dd8': {
        'desc': '啟動20%, 回撤8%',
        'overrides': {
            'enable_profit_trailing': True,
            'profit_trail_activation': 20,
            'profit_trail_pct': -8,
            'profit_trail_min_lock': 5,
        },
    },
    'trail_act20_dd10': {
        'desc': '啟動20%, 回撤10%',
        'overrides': {
            'enable_profit_trailing': True,
            'profit_trail_activation': 20,
            'profit_trail_pct': -10,
            'profit_trail_min_lock': 5,
        },
    },
    'trail_act25_dd10': {
        'desc': '啟動25%, 回撤10%',
        'overrides': {
            'enable_profit_trailing': True,
            'profit_trail_activation': 25,
            'profit_trail_pct': -10,
            'profit_trail_min_lock': 8,
        },
    },
    'trail_act15_dd5': {
        'desc': '啟動15%, 回撤5% (緊)',
        'overrides': {
            'enable_profit_trailing': True,
            'profit_trail_activation': 15,
            'profit_trail_pct': -5,
            'profit_trail_min_lock': 5,
        },
    },
    'trail_act30_dd12': {
        'desc': '啟動30%, 回撤12% (鬆)',
        'overrides': {
            'enable_profit_trailing': True,
            'profit_trail_activation': 30,
            'profit_trail_pct': -12,
            'profit_trail_min_lock': 10,
        },
    },
}

PERIODS = {
    'Training': ('2022-01-01', '2025-06-30'),
    'Validation': ('2025-07-01', '2026-03-27'),
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
                wr=sum(1 for t in sells if t['profit']>0)/len(sells)*100 if sells else 0
                print(f"  {cn:>18s}: Ret {ret:+7.1f}% | Shrp {sh:5.2f} | MDD {mdd:5.1f}% | Clmr {clm:5.2f} | PF {pf:4.2f} | WR {wr:4.1f}% | ⏱{el:.0f}s — {ce['desc']}")
        all_results[period_name] = pr

    # Summary
    print(f"\n\n{'='*100}")
    print(f"  📊 PROFIT TRAILING STOP ABLATION SUMMARY")
    print(f"{'='*100}")
    bl_t = all_results.get('Training',{}).get('baseline')
    bl_v = all_results.get('Validation',{}).get('baseline')

    header = f"{'Config':>18s} | {'Train':>55s} | {'Validation':>45s} | {'判定':>6s}"
    sub    = f"{'':>18s} | {'Ret':>8s} {'Shrp':>5s} {'MDD':>6s} {'Clmr':>5s} {'PF':>5s} {'WR':>5s} | {'Ret':>8s} {'Shrp':>5s} {'MDD':>6s} {'Clmr':>5s} {'PF':>5s} | {'':>6s}"
    print(header); print(sub); print("-"*130)

    for cn in CONFIGS:
        t=all_results.get('Training',{}).get(cn); v=all_results.get('Validation',{}).get(cn)
        if not t or not v: continue
        def _m(r):
            tl=r.get('trade_log',[]); sells=[x for x in tl if x['type']=='SELL' and x.get('profit') is not None]
            gw=sum(x['profit'] for x in sells if x['profit']>0); gl=abs(sum(x['profit'] for x in sells if x['profit']<=0))
            wr=sum(1 for x in sells if x['profit']>0)/len(sells)*100 if sells else 0
            return (r.get('total_return_pct',0),r.get('sharpe_ratio',0),r.get('mdd_pct',0),r.get('calmar_ratio',0),gw/gl if gl>0 else 999,wr)
        tr,ts,tm,tc,tp,tw=_m(t); vr,vs,vm,vc,vp,vw=_m(v)
        bl_ts=bl_t.get('sharpe_ratio',0) if bl_t else 0; bl_vs=bl_v.get('sharpe_ratio',0) if bl_v else 0
        bl_tm=bl_t.get('mdd_pct',0) if bl_t else 0; bl_vm=bl_v.get('mdd_pct',0) if bl_v else 0
        mk=""
        if cn!='baseline':
            t_better = ts>bl_ts or (ts>=bl_ts*0.95 and tm<bl_tm)
            v_ok = vs>=bl_vs*0.85
            if t_better and v_ok: mk="✅"
            elif ts>=bl_ts*0.95 and v_ok: mk="🟡"
            else: mk="❌"
        print(f"{cn:>18s} | {tr:+8.1f}% {ts:5.2f} {tm:5.1f}% {tc:5.2f} {tp:5.2f} {tw:4.1f}% | {vr:+7.1f}% {vs:5.2f} {vm:5.1f}% {vc:5.2f} {vp:5.2f} | {mk:>6s}")

if __name__ == '__main__':
    run_ablation()
