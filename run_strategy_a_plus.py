#!/usr/bin/env python3
"""
策略 A+: 短線動能 + 中長期出場增強

策略A已經 Train +112%, Shrp 0.78, Calmar 0.60 (全面贏 0050 DCA)
目標: 保持報酬率, MDD 30.5% → <25%, WR 25.9% → >30%

手段:
  1. 非對稱殭屍: 虧損快砍(15天) / 獲利慢砍(45天+須>8%)
  2. 降頻: max_positions 12→8~10, max_new_buy 4→2~3, 加入每週上限
  3. 出場放寬: tier_a 放更寬讓大贏家跑, tier_b_drawdown 微調
  4. 不改進場 (策略A的B1-B7不動)

0050 DCA 基準 (from R2):
  Train: Ret+49.1% Shrp 0.59 MDD 26.6% Clm 0.35
  Val:   Ret+26.9% Shrp 2.16 MDD 8.5% Clm 4.48
"""

import sys, os, time, warnings
import pandas as pd, numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from group_backtest import (run_group_backtest, reconstruct_market_history,
                            INDUSTRY_CONFIGS, MIN_DATA_DAYS)
from industry_manager import get_stocks_by_industry
from stock_utils import batch_download_stocks

INDUSTRY = '半導體業'
INITIAL_CAPITAL = 900_000
EXEC_MODE = 'next_open'

# 現有策略A半導體config
BASE_A = INDUSTRY_CONFIGS[INDUSTRY]['config'].copy()

# 0050 DCA 基準
DCA = {
    'Train': {'ret': 49.1, 'shrp': 0.59, 'mdd': 26.6, 'calmar': 0.35},
    'Val':   {'ret': 26.9, 'shrp': 2.16, 'mdd': 8.5, 'calmar': 4.48},
}

def _mk(desc, overrides):
    """用策略A base + overrides"""
    cfg = {**BASE_A, **overrides}
    budget = int(INITIAL_CAPITAL * cfg.get('budget_pct', 2.8) / 100)
    return {'desc': desc, 'signal_func': None, 'config': cfg,
            'budget': budget, 'capital': INITIAL_CAPITAL}


CONFIGS = {}

# ==========================================
# Baseline
# ==========================================
CONFIGS['A_base'] = _mk('策略A 原始', {})

# ==========================================
# D1: 非對稱殭屍 (虧損快砍 / 獲利慢砍)
# ==========================================
# 原始: zombie_hold_days=10, zombie_net_range=5.0 (10天±5%內就砍)
# 非對稱: 虧損15天砍, 獲利要45天且<8%才砍

CONFIGS['D1_asym_15_45'] = _mk(
    '非對稱zombie 虧15/利45/8%',
    {'zombie_asymmetric': True,
     'zombie_loss_days': 15, 'zombie_profit_days': 45,
     'zombie_profit_min_pct': 8.0})

CONFIGS['D1_asym_10_30'] = _mk(
    '非對稱zombie 虧10/利30/5%',
    {'zombie_asymmetric': True,
     'zombie_loss_days': 10, 'zombie_profit_days': 30,
     'zombie_profit_min_pct': 5.0})

CONFIGS['D1_asym_10_45'] = _mk(
    '非對稱zombie 虧10/利45/8%',
    {'zombie_asymmetric': True,
     'zombie_loss_days': 10, 'zombie_profit_days': 45,
     'zombie_profit_min_pct': 8.0})

CONFIGS['D1_asym_15_60'] = _mk(
    '非對稱zombie 虧15/利60/10%',
    {'zombie_asymmetric': True,
     'zombie_loss_days': 15, 'zombie_profit_days': 60,
     'zombie_profit_min_pct': 10.0})

CONFIGS['D1_asym_20_60'] = _mk(
    '非對稱zombie 虧20/利60/8%',
    {'zombie_asymmetric': True,
     'zombie_loss_days': 20, 'zombie_profit_days': 60,
     'zombie_profit_min_pct': 8.0})

# 關閉原始zombie, 只用非對稱
CONFIGS['D1_asym_only'] = _mk(
    '純非對稱zombie (關原始)',
    {'zombie_asymmetric': True, 'enable_zombie_cleanup': False,
     'zombie_loss_days': 15, 'zombie_profit_days': 45,
     'zombie_profit_min_pct': 8.0})

# ==========================================
# D2: 降頻 — 持倉數 + 每日買入上限
# ==========================================
# 原始: max_positions=12, max_new_buy=4
CONFIGS['D2_p10_b3'] = _mk('10檔/日買3', {'max_positions': 10, 'max_new_buy_per_day': 3})
CONFIGS['D2_p10_b2'] = _mk('10檔/日買2', {'max_positions': 10, 'max_new_buy_per_day': 2})
CONFIGS['D2_p8_b3'] = _mk('8檔/日買3', {'max_positions': 8, 'max_new_buy_per_day': 3})
CONFIGS['D2_p8_b2'] = _mk('8檔/日買2', {'max_positions': 8, 'max_new_buy_per_day': 2})

# 每週上限
CONFIGS['D2_w4'] = _mk('原始+週限4', {'weekly_max_buy': 4})
CONFIGS['D2_w6'] = _mk('原始+週限6', {'weekly_max_buy': 6})
CONFIGS['D2_w8'] = _mk('原始+週限8', {'weekly_max_buy': 8})

# ==========================================
# D3: 出場放寬 — 讓大贏家跑更久
# ==========================================
# 原始: tier_a_net=65, tier_b_net=15, tier_b_drawdown=0.6
CONFIGS['D3_ta80'] = _mk('tier_a 80%', {'tier_a_net': 80})
CONFIGS['D3_ta100'] = _mk('tier_a 100%', {'tier_a_net': 100})
CONFIGS['D3_tb_dd07'] = _mk('tier_b回撤0.7', {'tier_b_drawdown': 0.7})
CONFIGS['D3_tb20'] = _mk('tier_b門檻20%', {'tier_b_net': 20})
CONFIGS['D3_tb20_dd07'] = _mk('tier_b 20%+回撤0.7', {'tier_b_net': 20, 'tier_b_drawdown': 0.7})
CONFIGS['D3_s2buf5'] = _mk('S2緩衝5天', {'s2_buffer_days': 5})

# ==========================================
# D4: 組合 — 非對稱zombie + 降頻
# ==========================================
_ASYM = {'zombie_asymmetric': True,
         'zombie_loss_days': 15, 'zombie_profit_days': 45,
         'zombie_profit_min_pct': 8.0}

CONFIGS['D4_asym_p10b3'] = _mk(
    '非對稱+10檔/買3',
    {**_ASYM, 'max_positions': 10, 'max_new_buy_per_day': 3})

CONFIGS['D4_asym_p10b2'] = _mk(
    '非對稱+10檔/買2',
    {**_ASYM, 'max_positions': 10, 'max_new_buy_per_day': 2})

CONFIGS['D4_asym_p8b2'] = _mk(
    '非對稱+8檔/買2',
    {**_ASYM, 'max_positions': 8, 'max_new_buy_per_day': 2})

CONFIGS['D4_asym_p8b3'] = _mk(
    '非對稱+8檔/買3',
    {**_ASYM, 'max_positions': 8, 'max_new_buy_per_day': 3})

# ==========================================
# D5: 全組合 — 非對稱 + 降頻 + 出場放寬
# ==========================================
CONFIGS['D5_full_a'] = _mk(
    '全組合A: asym+10/3+ta80',
    {**_ASYM, 'max_positions': 10, 'max_new_buy_per_day': 3,
     'tier_a_net': 80})

CONFIGS['D5_full_b'] = _mk(
    '全組合B: asym+10/3+ta80+tb20/0.7',
    {**_ASYM, 'max_positions': 10, 'max_new_buy_per_day': 3,
     'tier_a_net': 80, 'tier_b_net': 20, 'tier_b_drawdown': 0.7})

CONFIGS['D5_full_c'] = _mk(
    '全組合C: asym+8/2+ta80+tb20/0.7',
    {**_ASYM, 'max_positions': 8, 'max_new_buy_per_day': 2,
     'tier_a_net': 80, 'tier_b_net': 20, 'tier_b_drawdown': 0.7})

CONFIGS['D5_full_d'] = _mk(
    '全組合D: asym+10/3+ta100+tb20/0.7',
    {**_ASYM, 'max_positions': 10, 'max_new_buy_per_day': 3,
     'tier_a_net': 100, 'tier_b_net': 20, 'tier_b_drawdown': 0.7})

CONFIGS['D5_full_e'] = _mk(
    '全組合E: asym+10/3+ta80+w6',
    {**_ASYM, 'max_positions': 10, 'max_new_buy_per_day': 3,
     'tier_a_net': 80, 'weekly_max_buy': 6})

CONFIGS['D5_full_f'] = _mk(
    '全組合F: asym+8/3+ta80+w6',
    {**_ASYM, 'max_positions': 8, 'max_new_buy_per_day': 3,
     'tier_a_net': 80, 'weekly_max_buy': 6})

# ==========================================
# 測試
# ==========================================
PERIODS = {
    'Train': ('2021-01-01', '2025-06-30'),
    'Val':   ('2025-07-01', '2026-03-28'),
}


def _extract_metrics(r):
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
        'buys': len(buys), 'sells': len(sells),
        'avg_hold': np.mean(hold_days) if hold_days else 0,
        'calmar': r.get('calmar_ratio', 0),
        'cagr': r.get('cagr', 0),
        'final': r.get('final_total_value', 0),
        'trades': len(buys) + len(sells),
    }


def run_backtest():
    stocks = get_stocks_by_industry(INDUSTRY)
    print(f"{'='*130}")
    print(f"  策略 A+: 短線動能 + 中長期出場增強")
    print(f"  0050 DCA: Train Shrp=0.59 MDD=26.6% Clm=0.35 | Val Shrp=2.16")
    print(f"  目標: 保持Shrp>0.59, MDD<25%, Calmar>0.35")
    print(f"  {INDUSTRY}: {len(stocks)} stocks | {len(CONFIGS)} configs")
    print(f"{'='*130}\n")

    all_results = {}
    for pn, (start, end) in PERIODS.items():
        print(f"\n{'='*100}")
        print(f"  {pn}: {start} ~ {end}")
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
                exec_mode=EXEC_MODE, config_override=ce['config'],
                initial_capital=ce['capital'], preloaded_data=data,
                signal_func=ce['signal_func'],
            )
            el = time.time() - t0
            if r:
                pr[cn] = r
                m = _extract_metrics(r)
                dca = DCA[pn]
                beat = 'v0050' if (m['shrp'] > dca['shrp'] and m['calmar'] > dca['calmar']) else ''
                print(f"  {cn:>18s}: Ret{m['ret']:+7.1f}% Shrp{m['shrp']:5.2f} "
                      f"MDD{m['mdd']:5.1f}% Clm{m['calmar']:5.2f} "
                      f"PF{m['pf']:5.2f} WR{m['wr']:4.1f}% "
                      f"交{m['trades']:>4d} 均持{m['avg_hold']:.0f}d "
                      f"⏱{el:.0f}s {beat} — {ce['desc']}")
            else:
                print(f"  {cn:>18s}: No result")
                pr[cn] = None
        all_results[pn] = pr

    # ==========================================
    # 綜合比較
    # ==========================================
    print(f"\n\n{'='*160}")
    print(f"  策略A+ 完整結果 (0050 DCA: T:Shrp=0.59 Clm=0.35 MDD=26.6% | V:Shrp=2.16 Clm=4.48)")
    print(f"{'='*160}")

    layers = {
        'Baseline':     ['A_base'],
        'D1 非對稱zombie': [k for k in CONFIGS if k.startswith('D1_')],
        'D2 降頻':       [k for k in CONFIGS if k.startswith('D2_')],
        'D3 出場放寬':    [k for k in CONFIGS if k.startswith('D3_')],
        'D4 asym+降頻':  [k for k in CONFIGS if k.startswith('D4_')],
        'D5 全組合':      [k for k in CONFIGS if k.startswith('D5_')],
    }

    for layer_name, keys in layers.items():
        print(f"\n  [{layer_name}]")
        for cn in keys:
            t = _extract_metrics(all_results.get('Train', {}).get(cn))
            v = _extract_metrics(all_results.get('Val', {}).get(cn))
            dca_t, dca_v = DCA['Train'], DCA['Val']

            tags = []
            if t['shrp'] > dca_t['shrp'] and t['calmar'] > dca_t['calmar']:
                tags.append('T>0050')
            if v['shrp'] > dca_v['shrp']:
                tags.append('V>0050')
            if abs(t['mdd']) < 25:
                tags.append('MDD<25')
            if t['wr'] > 30:
                tags.append('WR>30')
            tag = ' ' + ','.join(tags) if tags else ''

            print(f"  {cn:>18s} | T:{t['ret']:+6.1f}% Shrp{t['shrp']:5.2f} "
                  f"MDD{t['mdd']:5.1f}% Clm{t['calmar']:5.2f} "
                  f"PF{t['pf']:5.2f} WR{t['wr']:4.1f}% 交{t['trades']:>4d} | "
                  f"V:{v['ret']:+6.1f}% Shrp{v['shrp']:5.2f} "
                  f"MDD{v['mdd']:5.1f}%{tag}")

    # ==========================================
    # 排名: 贏 0050 的
    # ==========================================
    print(f"\n\n{'='*120}")
    print(f"  排名 — Train Sharpe>{DCA['Train']['shrp']} 且 Calmar>{DCA['Train']['calmar']} (贏0050)")
    print(f"{'='*120}\n")

    all_scores = []
    for cn in CONFIGS:
        t = _extract_metrics(all_results.get('Train', {}).get(cn))
        v = _extract_metrics(all_results.get('Val', {}).get(cn))
        if t['ret'] == 0 and t['shrp'] == 0:
            continue
        score = (t['shrp'] * 0.25 + (1 - abs(t['mdd'])/40) * 0.25 +
                 t['calmar'] * 0.20 + v['shrp'] * 0.15 +
                 min(t['wr'], 40) / 100 * 0.15)
        all_scores.append((cn, score, t, v))

    # 贏 0050 的
    winners = [(cn, s, t, v) for cn, s, t, v in all_scores
               if t['shrp'] > DCA['Train']['shrp'] and t['calmar'] > DCA['Train']['calmar']]
    winners.sort(key=lambda x: -x[1])

    if winners:
        for i, (cn, score, t, v) in enumerate(winners[:15]):
            medal = ['1', '2', '3'][i] if i < 3 else ' '
            mdd_f = ' MDD<25' if abs(t['mdd']) < 25 else ''
            print(f"  {medal} {cn:>18s}: Score={score:.3f} "
                  f"T:Ret={t['ret']:+.1f}% Shrp={t['shrp']:.2f} "
                  f"MDD={abs(t['mdd']):.1f}% Clm={t['calmar']:.2f} "
                  f"PF={t['pf']:.2f} WR={t['wr']:.0f}% 交{t['trades']} | "
                  f"V:Ret={v['ret']:+.1f}% Shrp={v['shrp']:.2f}"
                  f"{mdd_f} — {CONFIGS[cn]['desc']}")
    else:
        print("  (無配置同時贏過 0050 的 Sharpe 和 Calmar)")

    # 全部排名
    print(f"\n  全部排名 Top 15:")
    all_scores.sort(key=lambda x: -x[1])
    for i, (cn, score, t, v) in enumerate(all_scores[:15]):
        dca_t = DCA['Train']
        beat = 'v0050' if (t['shrp'] > dca_t['shrp'] and t['calmar'] > dca_t['calmar']) else '     '
        print(f"  {cn:>18s}: Score={score:.3f} {beat} "
              f"T:Shrp={t['shrp']:.2f} MDD={abs(t['mdd']):.1f}% "
              f"Clm={t['calmar']:.2f} PF={t['pf']:.2f} WR={t['wr']:.0f}% | "
              f"V:Shrp={v['shrp']:.2f}")

    # ==========================================
    # vs A_base delta
    # ==========================================
    base_t = _extract_metrics(all_results.get('Train', {}).get('A_base'))
    print(f"\n\n  Delta vs A_base (T:Ret={base_t['ret']:+.1f}% Shrp={base_t['shrp']:.2f} "
          f"MDD={base_t['mdd']:.1f}% WR={base_t['wr']:.0f}%):\n")

    deltas = []
    for cn, s, t, v in all_scores:
        if cn == 'A_base':
            continue
        d_mdd = t['mdd'] - base_t['mdd']  # neg = better
        d_shrp = t['shrp'] - base_t['shrp']
        d_wr = t['wr'] - base_t['wr']
        d_ret = t['ret'] - base_t['ret']
        deltas.append((cn, d_mdd, d_shrp, d_wr, d_ret, t))
    deltas.sort(key=lambda x: x[1])  # sort by MDD improvement

    for cn, dm, ds, dw, dr, t in deltas[:15]:
        print(f"  {cn:>18s}: DMDD={dm:+5.1f}% DShrp={ds:+5.2f} "
              f"DWR={dw:+4.1f}% DRet={dr:+6.1f}% | "
              f"MDD={t['mdd']:.1f}% Shrp={t['shrp']:.2f} — {CONFIGS[cn]['desc']}")

    # Trade logs
    _out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(_out, exist_ok=True)
    for pn in PERIODS:
        for cn in CONFIGS:
            r = all_results.get(pn, {}).get(cn)
            if r and r.get('trade_log'):
                rows = [{'Date': t['date'], 'Ticker': t['ticker'],
                         'Name': t.get('name',''), 'Type': t['type'],
                         'Price': round(t['price'],2), 'Shares': t['shares'],
                         'Profit': int(t['profit']) if t['profit'] is not None else '',
                         'ROI%': round(t['roi'],2) if t.get('roi') is not None else '',
                         'Note': t.get('note',''),
                         } for t in r['trade_log']]
                pd.DataFrame(rows).to_csv(
                    os.path.join(_out, f'aplus_{cn}_{pn}.csv'),
                    index=False, encoding='utf-8-sig')
    print(f"\nTrade logs → output/aplus_*.csv")


if __name__ == '__main__':
    run_backtest()
