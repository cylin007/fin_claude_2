#!/usr/bin/env python3
"""Peer Group Z-score Ablation (630K capital, with min_rsi=40 as new baseline)"""

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
BASE_CONFIG = INDUSTRY_CONFIGS[INDUSTRY]['config'].copy()  # already includes min_rsi=40

CONFIGS = {
    'baseline': {
        'desc': '現狀 (rsi40, peer關)',
        'overrides': {},
    },
    # === 族群 Z-score 不同參數 ===
    'peer_default': {
        'desc': 'Peer Z: 預設 (z>1.5扣/z<-1加/z>2.5擋)',
        'overrides': {'enable_peer_zscore': True},
    },
    'peer_loose': {
        'desc': 'Peer Z 寬鬆 (z>2.0扣/z>3.0擋)',
        'overrides': {
            'enable_peer_zscore': True,
            'peer_zscore_expensive': 2.0,
            'peer_zscore_block': 3.0,
        },
    },
    'peer_tight': {
        'desc': 'Peer Z 嚴格 (z>1.0扣/z>2.0擋)',
        'overrides': {
            'enable_peer_zscore': True,
            'peer_zscore_expensive': 1.0,
            'peer_zscore_block': 2.0,
        },
    },
    'peer_30d': {
        'desc': 'Peer Z 短期 (30日報酬)',
        'overrides': {
            'enable_peer_zscore': True,
            'peer_zscore_lookback': 30,
        },
    },
    'peer_90d': {
        'desc': 'Peer Z 長期 (90日報酬)',
        'overrides': {
            'enable_peer_zscore': True,
            'peer_zscore_lookback': 90,
        },
    },
    'peer_score_only': {
        'desc': 'Peer Z 純加減分 (不擋買)',
        'overrides': {
            'enable_peer_zscore': True,
            'peer_zscore_block': 999,  # never block
        },
    },
    'peer_block_only': {
        'desc': 'Peer Z 純擋買 (z>2.0擋, 不加減分)',
        'overrides': {
            'enable_peer_zscore': True,
            'peer_zscore_penalty': 0,
            'peer_zscore_bonus': 0,
            'peer_zscore_block': 2.0,
        },
    },
}

PERIODS = {
    'Train_2022': ('2022-01-01', '2025-06-30'),
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
                print(f"  {cn:>14s}: Ret {ret:+7.1f}% | Shrp {sh:5.2f} | MDD {mdd:5.1f}% | Clmr {clm:5.2f} | PF {pf:4.2f} | ⏱{el:.0f}s — {ce['desc']}")
        all_results[period_name] = pr

    # Summary
    print(f"\n\n{'='*100}\n  📊 PEER Z-SCORE ABLATION SUMMARY\n{'='*100}")
    bl_t = all_results.get('Train_2022',{}).get('baseline')
    bl_v = all_results.get('Validation',{}).get('baseline')
    bl_st = bl_t.get('sharpe_ratio',0) if bl_t else 0
    bl_sv = bl_v.get('sharpe_ratio',0) if bl_v else 0

    header = f"{'Config':>14s} | {'Train Ret':>9s} {'Shrp':>5s} {'MDD':>6s} {'Clmr':>5s} {'PF':>5s} | {'Val Ret':>8s} {'Shrp':>5s} {'MDD':>6s} {'Clmr':>5s} | {'判定':>8s}"
    print(header); print("-"*len(header))

    for cn in CONFIGS:
        t=all_results.get('Train_2022',{}).get(cn); v=all_results.get('Validation',{}).get(cn)
        if not t or not v: continue
        def _m(r):
            tl=r.get('trade_log',[]); sells=[x for x in tl if x['type']=='SELL' and x.get('profit') is not None]
            gw=sum(x['profit'] for x in sells if x['profit']>0); gl=abs(sum(x['profit'] for x in sells if x['profit']<=0))
            return (r.get('total_return_pct',0),r.get('sharpe_ratio',0),r.get('mdd_pct',0),r.get('calmar_ratio',0),gw/gl if gl>0 else 999)
        tr,ts,tm,tc,tp=_m(t); vr,vs,vm,vc,vp=_m(v)
        mk=""
        if cn!='baseline':
            if ts>bl_st and vs>=bl_sv*0.9: mk="✅ 雙贏"
            elif ts>=bl_st*0.95 and vs>=bl_sv*0.9: mk="🟡 持平"
            elif ts>bl_st: mk="⚠️ Val差"
            else: mk="❌"
        print(f"{cn:>14s} | {tr:+8.1f}% {ts:5.2f} {tm:5.1f}% {tc:5.2f} {tp:5.2f} | {vr:+7.1f}% {vs:5.2f} {vm:5.1f}% {vc:5.2f} | {mk}")

if __name__ == '__main__':
    run_ablation()
