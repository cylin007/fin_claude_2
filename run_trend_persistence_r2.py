#!/usr/bin/env python3
"""
策略 D — Round 2: 最佳因子組合

Round 1 發現:
  - 最佳因子: L2B_rsi60 (Shrp 0.32, MDD 29.6%), L1_sm_ma3d (WR 28.8%, MDD 29.7%)
  - momentum 進場已修復 (移除10日高突破條件)
  - 很多大獲利層參數沒效果 → 門檻太高, 需降低
  - zombie 參數無效 → 大部分出場由策略D的非對稱邏輯接管

Round 2 策略:
  L0: 修復後的三種進場模式
  L1: 組合最佳因子 (RSI60 + sm_ma3d + stop8/12)
  L2: 降低大獲利門檻 (15% → 8/10%)
  L3: 檔數 × budget 最佳組合
  L4: 全組合精調
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
from strategy_midterm import (check_midterm_signal, check_trend_persistence_signal,
                              MIDTERM_CONFIG, TREND_PERSISTENCE_CONFIG)

# ==========================================
# 0050 DCA Benchmark
# ==========================================
COMMISSION_RATE = 0.001425
COMMISSION_DISCOUNT = 0.6
ETF_TAX_RATE = 0.001
SLIPPAGE_PCT = 0.003


def _download_0050(start_date, end_date):
    dl_s = (pd.Timestamp(start_date) - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
    dl_e = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    try:
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        try:
            df = yf.download('0050.TW', start=dl_s, end=dl_e, progress=False)
        finally:
            sys.stderr = old_stderr
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def _calc_0050_dca_metrics(start, end, initial_capital):
    """計算 0050 定期定額的 Sharpe / MDD / Calmar / PF 等指標"""
    etf_df = _download_0050(start, end)
    if etf_df.empty:
        return None

    # 產生交易日列表
    mask = (etf_df.index >= pd.Timestamp(start)) & (etf_df.index <= pd.Timestamp(end))
    trading_days = etf_df.index[mask]
    if len(trading_days) < 10:
        return None

    first_ts = trading_days[0]
    last_ts = trading_days[-1]
    n_months = max(1, (last_ts.year - first_ts.year) * 12 + (last_ts.month - first_ts.month) + 1)
    monthly_amount = initial_capital / n_months

    cash = float(initial_capital)
    shares = 0
    last_buy_month = None
    values = []

    for ts in trading_days:
        row = etf_df.loc[ts]
        open_p = float(row['Open'])
        close_p = float(row['Close'])
        ym = (ts.year, ts.month)
        if ym != last_buy_month and cash >= monthly_amount:
            exec_price = open_p * (1 + SLIPPAGE_PCT)
            buy_shares = int(monthly_amount / exec_price)
            if buy_shares > 0:
                cost = buy_shares * exec_price
                fee = max(1, int(cost * COMMISSION_RATE * COMMISSION_DISCOUNT))
                if cost + fee <= cash:
                    cash -= (cost + fee)
                    shares += buy_shares
                    last_buy_month = ym
        values.append(cash + shares * close_p)

    if not values or values[0] <= 0:
        return None

    # 計算指標
    final = values[-1]
    days = (last_ts - first_ts).days
    yrs = max(days / 365.25, 0.1)
    ret = (final - initial_capital) / initial_capital * 100
    cagr = ((final / initial_capital) ** (1 / yrs) - 1) * 100 if final > 0 else 0

    d_rets = []
    for i in range(1, len(values)):
        if values[i-1] > 0:
            d_rets.append(values[i] / values[i-1] - 1)

    if d_rets and np.std(d_rets) > 0:
        rf_d = 0.015 / 245
        exc = [r - rf_d for r in d_rets]
        sharpe = (np.mean(exc) * 245) / (np.std(d_rets) * np.sqrt(245))
    else:
        sharpe = 0

    pk = values[0]
    max_dd = 0
    for v in values:
        if v > pk:
            pk = v
        dd = (pk - v) / pk * 100 if pk > 0 else 0
        if dd > max_dd:
            max_dd = dd

    calmar = cagr / max_dd if max_dd > 0 else 0
    return {'ret': ret, 'cagr': cagr, 'shrp': sharpe, 'mdd': max_dd,
            'calmar': calmar, 'final': final}

INDUSTRY = '半導體業'
INITIAL_CAPITAL = 900_000
EXEC_MODE = 'next_open'
BASE_A = INDUSTRY_CONFIGS[INDUSTRY]['config'].copy()

# ==========================================
# Round 2 基底: Round 1 最佳底座
# ==========================================
R2_BASE = {
    # -- backtest engine 層 --
    'max_positions':         5,
    'max_new_buy_per_day':   2,
    'zombie_hold_days':      20,
    'zombie_net_range':      0.0,
    'hard_stop_net':        -10,
    'enable_zombie_cleanup': True,
    'enable_position_swap':  False,
    'max_add_per_stock':     1,
    'budget_pct':            12.0,
    'cash_reserve_pct':      15.0,
    'weekly_max_buy':        2,

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

    # -- 策略D 參數 --
    **TREND_PERSISTENCE_CONFIG,
}


def _mk(desc, overrides, max_pos=5, budget_pct=12.0, weekly_max=2,
        entry_mode='pullback'):
    cfg = {**R2_BASE, **overrides}
    cfg['max_positions'] = max_pos
    cfg['tp_max_positions'] = max_pos
    cfg['budget_pct'] = budget_pct
    cfg['tp_budget_pct'] = budget_pct
    cfg['weekly_max_buy'] = weekly_max
    cfg['tp_entry_mode'] = entry_mode
    return {
        'desc': desc,
        'signal_func': check_trend_persistence_signal,
        'config': cfg,
        'budget': int(INITIAL_CAPITAL * budget_pct / 100),
        'capital': INITIAL_CAPITAL,
    }


CONFIGS = {}

# ==========================================
# Baselines
# ==========================================
CONFIGS['A_baseline'] = {
    'desc': '策略A (對照)',
    'signal_func': None,
    'config': BASE_A,
    'budget': 25_000,
    'capital': INITIAL_CAPITAL,
}

# ==========================================
# L0: 修復後三種進場模式
# ==========================================
CONFIGS['L0_pull'] = _mk('pullback進場', {})
CONFIGS['L0_mom'] = _mk('momentum進場 (修復)', {}, entry_mode='momentum')
CONFIGS['L0_hyb'] = _mk('hybrid進場', {}, entry_mode='hybrid')

# ==========================================
# L1: 單因子確認 (基於 pullback)
# ==========================================
CONFIGS['L1_rsi60'] = _mk('+RSI≤60', {'tp_rsi_max': 60})
CONFIGS['L1_sm3d'] = _mk('+小獲利MA60破3天', {'tp_small_ma60_break_days': 3})
CONFIGS['L1_stop8'] = _mk('+硬停-8%', {'tp_loss_hard_stop': -8.0})
CONFIGS['L1_stop12'] = _mk('+硬停-12%', {'tp_loss_hard_stop': -12.0})
CONFIGS['L1_big10'] = _mk('+大獲利門檻10%', {'tp_big_profit_threshold': 10.0})
CONFIGS['L1_big8'] = _mk('+大獲利門檻8%', {'tp_big_profit_threshold': 8.0})

# ==========================================
# L2: 兩因子組合
# ==========================================
CONFIGS['L2_rsi60_sm3d'] = _mk(
    '+RSI60+sm3d',
    {'tp_rsi_max': 60, 'tp_small_ma60_break_days': 3})

CONFIGS['L2_rsi60_stop8'] = _mk(
    '+RSI60+stop8',
    {'tp_rsi_max': 60, 'tp_loss_hard_stop': -8.0})

CONFIGS['L2_rsi60_stop12'] = _mk(
    '+RSI60+stop12',
    {'tp_rsi_max': 60, 'tp_loss_hard_stop': -12.0})

CONFIGS['L2_rsi60_big10'] = _mk(
    '+RSI60+big10',
    {'tp_rsi_max': 60, 'tp_big_profit_threshold': 10.0})

CONFIGS['L2_rsi60_big8'] = _mk(
    '+RSI60+big8',
    {'tp_rsi_max': 60, 'tp_big_profit_threshold': 8.0})

CONFIGS['L2_sm3d_stop8'] = _mk(
    '+sm3d+stop8',
    {'tp_small_ma60_break_days': 3, 'tp_loss_hard_stop': -8.0})

CONFIGS['L2_sm3d_big10'] = _mk(
    '+sm3d+big10',
    {'tp_small_ma60_break_days': 3, 'tp_big_profit_threshold': 10.0})

# ==========================================
# L3: 三因子組合
# ==========================================
CONFIGS['L3_rsi_sm_s8'] = _mk(
    '+RSI60+sm3d+stop8',
    {'tp_rsi_max': 60, 'tp_small_ma60_break_days': 3,
     'tp_loss_hard_stop': -8.0})

CONFIGS['L3_rsi_sm_s12'] = _mk(
    '+RSI60+sm3d+stop12',
    {'tp_rsi_max': 60, 'tp_small_ma60_break_days': 3,
     'tp_loss_hard_stop': -12.0})

CONFIGS['L3_rsi_sm_b10'] = _mk(
    '+RSI60+sm3d+big10',
    {'tp_rsi_max': 60, 'tp_small_ma60_break_days': 3,
     'tp_big_profit_threshold': 10.0})

CONFIGS['L3_rsi_sm_b8'] = _mk(
    '+RSI60+sm3d+big8',
    {'tp_rsi_max': 60, 'tp_small_ma60_break_days': 3,
     'tp_big_profit_threshold': 8.0})

CONFIGS['L3_rsi_s8_b10'] = _mk(
    '+RSI60+stop8+big10',
    {'tp_rsi_max': 60, 'tp_loss_hard_stop': -8.0,
     'tp_big_profit_threshold': 10.0})

# ==========================================
# L4: 四因子全組合
# ==========================================
CONFIGS['L4_all_s8_b10'] = _mk(
    '+RSI60+sm3d+stop8+big10',
    {'tp_rsi_max': 60, 'tp_small_ma60_break_days': 3,
     'tp_loss_hard_stop': -8.0, 'tp_big_profit_threshold': 10.0})

CONFIGS['L4_all_s8_b8'] = _mk(
    '+RSI60+sm3d+stop8+big8',
    {'tp_rsi_max': 60, 'tp_small_ma60_break_days': 3,
     'tp_loss_hard_stop': -8.0, 'tp_big_profit_threshold': 8.0})

CONFIGS['L4_all_s12_b10'] = _mk(
    '+RSI60+sm3d+stop12+big10',
    {'tp_rsi_max': 60, 'tp_small_ma60_break_days': 3,
     'tp_loss_hard_stop': -12.0, 'tp_big_profit_threshold': 10.0})

CONFIGS['L4_all_s12_b8'] = _mk(
    '+RSI60+sm3d+stop12+big8',
    {'tp_rsi_max': 60, 'tp_small_ma60_break_days': 3,
     'tp_loss_hard_stop': -12.0, 'tp_big_profit_threshold': 8.0})

# ==========================================
# L5: 最佳因子 × 進場模式
# ==========================================
# 用 L4 最佳因子組合, 搭配不同進場
_L5_FACTORS = {'tp_rsi_max': 60, 'tp_small_ma60_break_days': 3,
               'tp_loss_hard_stop': -8.0, 'tp_big_profit_threshold': 10.0}

CONFIGS['L5_mom'] = _mk(
    'L4best+momentum', _L5_FACTORS, entry_mode='momentum')

CONFIGS['L5_hyb'] = _mk(
    'L4best+hybrid', _L5_FACTORS, entry_mode='hybrid')

CONFIGS['L5_pull'] = _mk(
    'L4best+pullback', _L5_FACTORS, entry_mode='pullback')

# ==========================================
# L6: 檔數 × budget 精調 (基於 L4 最佳因子)
# ==========================================
CONFIGS['L6_4p15'] = _mk('L4best+4檔15%', _L5_FACTORS, max_pos=4, budget_pct=15.0)
CONFIGS['L6_5p15'] = _mk('L4best+5檔15%', _L5_FACTORS, max_pos=5, budget_pct=15.0)
CONFIGS['L6_6p10'] = _mk('L4best+6檔10%', _L5_FACTORS, max_pos=6, budget_pct=10.0)
CONFIGS['L6_6p12'] = _mk('L4best+6檔12%', _L5_FACTORS, max_pos=6, budget_pct=12.0)
CONFIGS['L6_4p12'] = _mk('L4best+4檔12%', _L5_FACTORS, max_pos=4, budget_pct=12.0)

# 週限
CONFIGS['L6_w1'] = _mk('L4best+週限1', _L5_FACTORS, weekly_max=1)
CONFIGS['L6_w3'] = _mk('L4best+週限3', _L5_FACTORS, weekly_max=3)
CONFIGS['L6_w0'] = _mk('L4best+不限週', _L5_FACTORS, weekly_max=0)

# 不看週線 (R1 中 L2B_noweekly 表現好)
CONFIGS['L6_noweekly'] = _mk(
    'L4best+不看週線',
    {**_L5_FACTORS, 'tp_require_weekly_bull': False})

# 大獲利回撤放寬/收緊
CONFIGS['L6_dd20'] = _mk(
    'L4best+保底回撤20%',
    {**_L5_FACTORS, 'tp_big_max_drawdown': 20.0})

CONFIGS['L6_dd30'] = _mk(
    'L4best+保底回撤30%',
    {**_L5_FACTORS, 'tp_big_max_drawdown': 30.0})


# ==========================================
# 測試週期
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
    print(f"  策略 D — Round 2: 最佳因子組合")
    print(f"  {INDUSTRY}: {len(stocks)} stocks | Capital: ${INITIAL_CAPITAL:,}")
    print(f"  共 {len(CONFIGS)} 組測試")
    print(f"{'='*130}\n")

    # ==========================================
    # 0050 DCA Benchmark
    # ==========================================
    print(f"\n  計算 0050 定期定額基準...")
    dca_metrics = {}
    for period_name, (start, end) in PERIODS.items():
        m = _calc_0050_dca_metrics(start, end, INITIAL_CAPITAL)
        if m:
            dca_metrics[period_name] = m
            print(f"    {period_name}: Ret{m['ret']:+7.1f}% CAGR{m['cagr']:+6.1f}% "
                  f"Shrp{m['shrp']:5.2f} MDD{m['mdd']:5.1f}% Clm{m['calmar']:5.2f}")
        else:
            dca_metrics[period_name] = {'ret': 0, 'shrp': 0, 'mdd': 99, 'calmar': 0, 'cagr': 0}
            print(f"    {period_name}: 0050 資料不足")

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
                      f"均持{m['avg_hold']:.0f}d 交易{m['trades']:>3d} "
                      f"⏱{el:.0f}s — {ce['desc']}")
            else:
                print(f"  {cn:>20s}: No result — {ce['desc']}")
                pr[cn] = None
        all_results[period_name] = pr

    # ==========================================
    # Summary Table (含 0050 DCA 比較)
    # ==========================================
    dca_t = dca_metrics.get('Train', {})
    dca_v = dca_metrics.get('Val', {})

    print(f"\n\n{'='*170}")
    print(f"  策略D R2 完整結果")
    print(f"  0050 DCA 基準 — Train: Ret{dca_t.get('ret',0):+.1f}% Shrp{dca_t.get('shrp',0):.2f} "
          f"MDD{dca_t.get('mdd',0):.1f}% Clm{dca_t.get('calmar',0):.2f} | "
          f"Val: Ret{dca_v.get('ret',0):+.1f}% Shrp{dca_v.get('shrp',0):.2f} "
          f"MDD{dca_v.get('mdd',0):.1f}% Clm{dca_v.get('calmar',0):.2f}")
    print(f"{'='*170}")

    layers = {
        'Baseline':       ['A_baseline'],
        'L0 進場模式':     [k for k in CONFIGS if k.startswith('L0_')],
        'L1 單因子':       [k for k in CONFIGS if k.startswith('L1_')],
        'L2 兩因子':       [k for k in CONFIGS if k.startswith('L2_')],
        'L3 三因子':       [k for k in CONFIGS if k.startswith('L3_')],
        'L4 四因子':       [k for k in CONFIGS if k.startswith('L4_')],
        'L5 進場×因子':    [k for k in CONFIGS if k.startswith('L5_')],
        'L6 部位精調':     [k for k in CONFIGS if k.startswith('L6_')],
    }

    base_key = 'L0_pull'
    base_train = _extract_metrics(all_results.get('Train', {}).get(base_key))

    for layer_name, keys in layers.items():
        print(f"\n  [{layer_name}]")
        for cn in keys:
            t_r = all_results.get('Train', {}).get(cn)
            v_r = all_results.get('Val', {}).get(cn)
            t = _extract_metrics(t_r)
            v = _extract_metrics(v_r)

            markers = []
            if t['mdd'] > 0 and abs(t['mdd']) < 25:
                markers.append('MDD<25')
            if t['wr'] > 30:
                markers.append('WR>30')
            # 是否贏過 0050 DCA (Train 期間的 Sharpe, Calmar, PF)
            beats_0050 = (t['shrp'] > dca_t.get('shrp', 0) and
                          t['calmar'] > dca_t.get('calmar', 0))
            if beats_0050:
                markers.append('>0050')

            marker = ' ' + ','.join(markers) if markers else ''

            print(f"  {cn:>20s} | "
                  f"T: {t['ret']:+6.1f}% Shrp{t['shrp']:5.2f} "
                  f"MDD{t['mdd']:5.1f}% Clm{t['calmar']:5.2f} "
                  f"WR{t['wr']:4.1f}% PF{t['pf']:5.2f} "
                  f"均持{t['avg_hold']:4.0f}d 交{t['trades']:>4d} | "
                  f"V: {v['ret']:+6.1f}% Shrp{v['shrp']:5.2f} "
                  f"MDD{v['mdd']:5.1f}% WR{v['wr']:4.1f}%{marker} "
                  f"— {CONFIGS[cn]['desc']}")

    # ==========================================
    # 綜合排名 (只顯示 Train 贏過 0050 DCA 的)
    # ==========================================
    _dca_shrp = dca_t.get('shrp', 0)
    _dca_calmar = dca_t.get('calmar', 0)
    _dca_pf = 999  # 0050 DCA 沒有 PF 概念

    print(f"\n\n{'='*130}")
    print(f"  綜合排名 — 只列 Train Sharpe>{_dca_shrp:.2f} 且 Calmar>{_dca_calmar:.2f} (贏0050DCA)")
    print(f"  Score = Sharpe×0.30 + (1-MDD/40)×0.25 + ValShrp×0.20 + WR/100×0.15 + Hold/100×0.10")
    print(f"{'='*130}\n")

    all_scores = []
    all_scores_raw = []  # 含不及格的
    for cn in CONFIGS:
        if cn == 'A_baseline':
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
        entry = (cn, score, t, v)
        all_scores_raw.append(entry)
        # 過濾: Train Sharpe 和 Calmar 都要贏 0050
        if t['shrp'] > _dca_shrp and t['calmar'] > _dca_calmar:
            all_scores.append(entry)

    all_scores.sort(key=lambda x: -x[1])
    if all_scores:
        for i, (cn, score, t, v) in enumerate(all_scores[:20]):
            medal = ['1', '2', '3'][i] if i < 3 else ' '
            mdd_flag = ' MDD<25' if abs(t['mdd']) < 25 else ''
            print(f"  {medal} {cn:>20s}: Score={score:.3f} | "
                  f"T: Ret={t['ret']:+.1f}% Shrp={t['shrp']:.2f} MDD={abs(t['mdd']):.1f}% "
                  f"Clm={t['calmar']:.2f} PF={t['pf']:.2f} WR={t['wr']:.0f}% "
                  f"均持{t['avg_hold']:.0f}d 交易{t['trades']}次 | "
                  f"V: Ret={v['ret']:+.1f}% Shrp={v['shrp']:.2f} MDD={abs(v['mdd']):.1f}%"
                  f"{mdd_flag} — {CONFIGS[cn]['desc']}")
    else:
        print("  (沒有配置在 Train 期間同時贏過 0050 DCA 的 Sharpe 和 Calmar)")
        print(f"\n  退而求其次, 列出 Train Sharpe 最高的前10:")
        all_scores_raw.sort(key=lambda x: -x[2]['shrp'])
        for i, (cn, score, t, v) in enumerate(all_scores_raw[:10]):
            print(f"    {cn:>20s}: Shrp={t['shrp']:.2f} Clm={t['calmar']:.2f} "
                  f"MDD={abs(t['mdd']):.1f}% — {CONFIGS[cn]['desc']}")

    # ==========================================
    # MDD < 25% 且贏 0050 篩選
    # ==========================================
    print(f"\n\n{'='*100}")
    print(f"  MDD < 25% 且 Train 贏 0050 DCA 候選")
    print(f"{'='*100}\n")
    mdd25 = [(cn, s, t, v) for cn, s, t, v in all_scores_raw
             if abs(t['mdd']) < 25 and t['shrp'] > _dca_shrp]
    if mdd25:
        for cn, s, t, v in mdd25[:10]:
            print(f"  {cn:>20s}: Score={s:.3f} T:Ret={t['ret']:+.1f}% "
                  f"MDD={abs(t['mdd']):.1f}% WR={t['wr']:.0f}% | "
                  f"V:Ret={v['ret']:+.1f}% MDD={abs(v['mdd']):.1f}%")
    else:
        print("  (無)")

    # ==========================================
    # 最終推薦 (分兩個維度)
    # ==========================================
    if all_scores:
        best = all_scores[0]
        print(f"\n\n{'='*80}")
        print(f"  1 最高綜合分: {best[0]}")
        print(f"     {CONFIGS[best[0]]['desc']}")
        print(f"     Train: Ret={best[2]['ret']:+.1f}% Sharpe={best[2]['shrp']:.2f} "
              f"MDD={abs(best[2]['mdd']):.1f}% WR={best[2]['wr']:.0f}% 均持{best[2]['avg_hold']:.0f}d")
        print(f"     Val:   Ret={best[3]['ret']:+.1f}% Sharpe={best[3]['shrp']:.2f} "
              f"MDD={abs(best[3]['mdd']):.1f}%")

        # 最低 MDD
        lowest_mdd = min(all_scores, key=lambda x: abs(x[2]['mdd']))
        if lowest_mdd[0] != best[0]:
            print(f"\n  2 最低MDD: {lowest_mdd[0]}")
            print(f"     {CONFIGS[lowest_mdd[0]]['desc']}")
            print(f"     Train: Ret={lowest_mdd[2]['ret']:+.1f}% MDD={abs(lowest_mdd[2]['mdd']):.1f}% "
                  f"WR={lowest_mdd[2]['wr']:.0f}%")
        print(f"{'='*80}")

    # Trade logs
    _output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(_output_dir, exist_ok=True)
    for period_name in PERIODS:
        for cn in CONFIGS:
            r = all_results.get(period_name, {}).get(cn)
            if r and r.get('trade_log'):
                rows = [{'Date': t['date'], 'Ticker': t['ticker'],
                         'Name': t.get('name',''), 'Type': t['type'],
                         'Price': round(t['price'],2), 'Shares': t['shares'],
                         'Profit': int(t['profit']) if t['profit'] is not None else '',
                         'ROI%': round(t['roi'],2) if t.get('roi') is not None else '',
                         'Note': t.get('note',''),
                         } for t in r['trade_log']]
                df = pd.DataFrame(rows)
                df.to_csv(os.path.join(_output_dir, f'tp_r2_{cn}_{period_name}.csv'),
                          index=False, encoding='utf-8-sig')
    print(f"\nTrade logs saved to output/tp_r2_*.csv")


if __name__ == '__main__':
    run_backtest()
