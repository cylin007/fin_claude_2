#!/usr/bin/env python3
"""
策略 D R2b: 只跑 R2 中未完成的 L4-L6 + 全部 Val
加入 0050 DCA 比較
"""
import sys, os, time, warnings
import pandas as pd, numpy as np
import yfinance as yf
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from group_backtest import (run_group_backtest, reconstruct_market_history,
                            INDUSTRY_CONFIGS, MIN_DATA_DAYS)
from industry_manager import get_stocks_by_industry
from stock_utils import batch_download_stocks
from strategy_midterm import (check_trend_persistence_signal,
                              TREND_PERSISTENCE_CONFIG)

INDUSTRY = '半導體業'
INITIAL_CAPITAL = 900_000
EXEC_MODE = 'next_open'

# ==========================================
# 0050 DCA (from R2 run)
# ==========================================
DCA_METRICS = {
    'Train': {'ret': 49.1, 'shrp': 0.59, 'mdd': 26.6, 'calmar': 0.35},
    'Val':   {'ret': 26.9, 'shrp': 2.16, 'mdd': 8.5, 'calmar': 4.48},
}

# ==========================================
# R2 已知結果 (from partial run, Train only)
# ==========================================
R2_KNOWN_TRAIN = {
    'L1_rsi60':       {'ret': 30.1, 'shrp': 0.32, 'mdd': 29.6, 'calmar': 0.20, 'pf': 1.38, 'wr': 25.7},
    'L2_rsi60_stop12':{'ret': 34.1, 'shrp': 0.35, 'mdd': 29.7, 'calmar': 0.23, 'pf': 1.45, 'wr': 24.8},
    'L2_rsi60_sm3d':  {'ret': 22.5, 'shrp': 0.26, 'mdd': 27.3, 'calmar': 0.17, 'pf': 1.25, 'wr': 28.7},
    'L3_rsi_sm_b10':  {'ret': 22.9, 'shrp': 0.26, 'mdd': 27.9, 'calmar': 0.17, 'pf': 1.27, 'wr': 29.1},
    'L3_rsi_sm_s12':  {'ret': 19.7, 'shrp': 0.23, 'mdd': 27.5, 'calmar': 0.15, 'pf': 1.23, 'wr': 28.4},
    'L1_sm3d':        {'ret': 24.2, 'shrp': 0.27, 'mdd': 29.7, 'calmar': 0.17, 'pf': 1.28, 'wr': 28.8},
}

# ==========================================
# 基底
# ==========================================
R2_BASE = {
    'max_positions': 5, 'max_new_buy_per_day': 2,
    'zombie_hold_days': 20, 'zombie_net_range': 0.0,
    'hard_stop_net': -10, 'enable_zombie_cleanup': True,
    'enable_position_swap': False, 'max_add_per_stock': 1,
    'budget_pct': 12.0, 'cash_reserve_pct': 15.0, 'weekly_max_buy': 2,
    'enable_fish_tail': False, 'enable_breakout': False,
    'enable_rs_filter': False, 'enable_sector_momentum': False,
    'enable_ewt_boost': False, 'enable_ewt_filter': False,
    'enable_conviction_hold': False, 'enable_regime_adaptive': False,
    'enable_peer_zscore': False, 'enable_weekly_filter': False,
    'enable_theme_boost': False, 'enable_quality_filter': False,
    'enable_dynamic_exposure': False, 'enable_dyn_buy_limit': False,
    'enable_dyn_stop': False, 'enable_vol_sizing': False,
    'enable_profit_trailing': False, 'enable_trailing_stop': False,
    'enable_pullback_buy': False, 'enable_dip_buy': False,
    'market_filter_mode': 'off', 'min_rsi': 0,
    **TREND_PERSISTENCE_CONFIG,
}


def _mk(desc, overrides, max_pos=5, budget_pct=12.0, weekly_max=2):
    cfg = {**R2_BASE, **overrides}
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


# R2 最佳因子
_BEST = {'tp_rsi_max': 60, 'tp_small_ma60_break_days': 3,
         'tp_loss_hard_stop': -12.0, 'tp_big_profit_threshold': 10.0}

CONFIGS = {}

# === R2 最佳單/兩/三因子 (需要 Val 數據) ===
CONFIGS['best_rsi60'] = _mk('RSI60', {'tp_rsi_max': 60})
CONFIGS['best_rsi_s12'] = _mk('RSI60+stop12',
    {'tp_rsi_max': 60, 'tp_loss_hard_stop': -12.0})
CONFIGS['best_rsi_sm3d'] = _mk('RSI60+sm3d',
    {'tp_rsi_max': 60, 'tp_small_ma60_break_days': 3})
CONFIGS['best3_sm_b10'] = _mk('RSI60+sm3d+big10',
    {'tp_rsi_max': 60, 'tp_small_ma60_break_days': 3,
     'tp_big_profit_threshold': 10.0})
CONFIGS['best3_sm_s12'] = _mk('RSI60+sm3d+stop12',
    {'tp_rsi_max': 60, 'tp_small_ma60_break_days': 3,
     'tp_loss_hard_stop': -12.0})

# === L4: 四因子組合 ===
CONFIGS['L4_s12_b10'] = _mk('RSI60+sm3d+stop12+big10', _BEST)
CONFIGS['L4_s12_b8'] = _mk('RSI60+sm3d+stop12+big8',
    {**_BEST, 'tp_big_profit_threshold': 8.0})
CONFIGS['L4_s8_b10'] = _mk('RSI60+sm3d+stop8+big10',
    {**_BEST, 'tp_loss_hard_stop': -8.0})

# === L5: 檔數 × budget (基於 L4 最佳) ===
CONFIGS['L5_4p15'] = _mk('4檔15%', _BEST, max_pos=4, budget_pct=15.0)
CONFIGS['L5_5p15'] = _mk('5檔15%', _BEST, max_pos=5, budget_pct=15.0)
CONFIGS['L5_6p10'] = _mk('6檔10%', _BEST, max_pos=6, budget_pct=10.0)
CONFIGS['L5_6p12'] = _mk('6檔12%', _BEST, max_pos=6, budget_pct=12.0)
CONFIGS['L5_4p12'] = _mk('4檔12%', _BEST, max_pos=4, budget_pct=12.0)

# === L6: 週限/週線/回撤 ===
CONFIGS['L6_w1'] = _mk('週限1', _BEST, weekly_max=1)
CONFIGS['L6_w3'] = _mk('週限3', _BEST, weekly_max=3)
CONFIGS['L6_w0'] = _mk('不限週', _BEST, weekly_max=0)
CONFIGS['L6_nowk'] = _mk('不看週線',
    {**_BEST, 'tp_require_weekly_bull': False})
CONFIGS['L6_dd20'] = _mk('保底回撤20%', {**_BEST, 'tp_big_max_drawdown': 20.0})
CONFIGS['L6_dd30'] = _mk('保底回撤30%', {**_BEST, 'tp_big_max_drawdown': 30.0})
CONFIGS['L6_big_ntrx'] = _mk('大獲利純MA60出場',
    {**_BEST, 'tp_big_use_trend_exit': False})

# 寬鬆 bear 過濾 (只擋 panic 不擋 weak)
CONFIGS['L6_nobear'] = _mk('不擋bear/weak',
    {**_BEST, 'tp_block_bear': False})

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
    print(f"  策略D R2b: L4-L6 + Val + 0050 DCA 比較")
    print(f"  0050 DCA Train: Ret+49.1% Shrp 0.59 MDD 26.6% Clm 0.35")
    print(f"  0050 DCA Val:   Ret+26.9% Shrp 2.16 MDD 8.5%  Clm 4.48")
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
                dca = DCA_METRICS[pn]
                beat = 'v0050' if (m['shrp'] > dca['shrp'] and m['calmar'] > dca['calmar']) else ''
                print(f"  {cn:>16s}: Ret{m['ret']:+7.1f}% Shrp{m['shrp']:5.2f} "
                      f"MDD{m['mdd']:5.1f}% Clm{m['calmar']:5.2f} "
                      f"PF{m['pf']:5.2f} WR{m['wr']:4.1f}% "
                      f"買{m['buys']:>3d} 賣{m['sells']:>3d} 均持{m['avg_hold']:.0f}d "
                      f"⏱{el:.0f}s {beat} — {ce['desc']}")
            else:
                print(f"  {cn:>16s}: No result — {ce['desc']}")
                pr[cn] = None
        all_results[pn] = pr

    # ==========================================
    # 綜合比較表
    # ==========================================
    print(f"\n\n{'='*160}")
    print(f"  策略D R2b 綜合比較 (0050 DCA: Train Shrp=0.59 Clm=0.35 | Val Shrp=2.16 Clm=4.48)")
    print(f"{'='*160}")
    print(f"  {'Config':>16s} | {'--- Train ---':>60s} | {'--- Val ---':>50s} | vs0050")
    print(f"-"*160)

    all_scores = []
    for cn in CONFIGS:
        t = _extract_metrics(all_results.get('Train', {}).get(cn))
        v = _extract_metrics(all_results.get('Val', {}).get(cn))
        if t['ret'] == 0 and t['shrp'] == 0:
            continue
        dca_t = DCA_METRICS['Train']
        dca_v = DCA_METRICS['Val']
        beat_t = t['shrp'] > dca_t['shrp'] and t['calmar'] > dca_t['calmar']
        beat_v = v['shrp'] > dca_v['shrp'] and v['calmar'] > dca_v['calmar']
        beat_str = ('TT' if (beat_t and beat_v) else
                    'T-' if beat_t else
                    '-V' if beat_v else '--')

        score = (t['shrp'] * 0.30 + (1 - abs(t['mdd'])/40) * 0.25 +
                 v['shrp'] * 0.20 + min(t['wr'], 50)/100 * 0.15 +
                 min(t['avg_hold'], 60)/100 * 0.10)
        all_scores.append((cn, score, t, v, beat_str))

        print(f"  {cn:>16s} | Ret{t['ret']:+6.1f}% Shrp{t['shrp']:5.2f} "
              f"MDD{t['mdd']:5.1f}% Clm{t['calmar']:5.2f} PF{t['pf']:5.2f} "
              f"WR{t['wr']:4.1f}% 交{t['trades']:>3d} | "
              f"Ret{v['ret']:+6.1f}% Shrp{v['shrp']:5.2f} "
              f"MDD{v['mdd']:5.1f}% Clm{v['calmar']:5.2f} | {beat_str}")

    all_scores.sort(key=lambda x: -x[1])
    print(f"\n\n Top 10 by Score:")
    for i, (cn, score, t, v, bs) in enumerate(all_scores[:10]):
        medal = ['1', '2', '3'][i] if i < 3 else ' '
        print(f"  {medal} {cn:>16s}: Score={score:.3f} "
              f"T:Shrp={t['shrp']:.2f} MDD={abs(t['mdd']):.1f}% Clm={t['calmar']:.2f} "
              f"PF={t['pf']:.2f} WR={t['wr']:.0f}% | "
              f"V:Shrp={v['shrp']:.2f} MDD={abs(v['mdd']):.1f}% | "
              f"vs0050={bs} — {CONFIGS[cn]['desc']}")

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
                    os.path.join(_out, f'tp_r2b_{cn}_{pn}.csv'),
                    index=False, encoding='utf-8-sig')
    print(f"\nTrade logs → output/tp_r2b_*.csv")


if __name__ == '__main__':
    run_backtest()
