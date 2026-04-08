import pandas as pd
import numpy as np
import datetime
import sys
import os
import warnings
import time
import shutil
import argparse
import re
import csv
import json
import yfinance as yf

warnings.simplefilter(action='ignore', category=FutureWarning)

from strategy import check_strategy_signal, calculate_fee, calculate_tax, calculate_net_pnl, _evaluate_pullback_buy
from stock_utils import (get_stock_data, build_info_dict, batch_download_stocks,
                         check_cache_vs_latest, print_cache_check,
                         check_dividend_adjustment, print_dividend_check,
                         get_market_status, print_market_status)

# ==========================================
# 📋 預設測試區間 (涵蓋不同市況)
# ==========================================
PRESET_PERIODS = {
    '1': {
        'name': '🔄 完整週期 (2022~2026-02-11)',
        'start': '2022-01-01', 'end': '2026-02-11',
        'desc': '空頭→復甦→AI狂飆→關稅崩盤→V轉，4年完整考驗 (與V5 ablation一致)',
    },
    '2': {
        'name': '🐻 純空頭',
        'start': '2022-01-01', 'end': '2022-10-31',
        'desc': '加權 18000→12600，測試停損/濾網是否有效擋住虧損',
    },
    '3': {
        'name': '🐂 純多頭',
        'start': '2023-01-01', 'end': '2024-07-31',
        'desc': '加權 14000→24000，測試策略能不能吃到完整漲幅',
    },
    '4': {
        'name': '📉 高檔修正',
        'start': '2024-07-01', 'end': '2025-01-31',
        'desc': 'AI 過熱後回檔，測試停利/過熱濾網的效果',
    },
    '5': {
        'name': '🩸 急殺崩盤 (川普關稅)',
        'start': '2025-02-01', 'end': '2025-07-31',
        'desc': '川普關稅急跌，測試策略能否躲過系統性風險',
    },
    '6': {
        'name': '🚀 V轉狂牛',
        'start': '2025-08-01', 'end': datetime.datetime.now().strftime('%Y-%m-%d'),
        'desc': '急跌後反彈，測試策略能否迅速反手做多吃到漲幅',
    },
    '7': {
        'name': '📆 近半年',
        'start': (datetime.datetime.now() - datetime.timedelta(days=180)).strftime('%Y-%m-%d'),
        'end': datetime.datetime.now().strftime('%Y-%m-%d'),
        'desc': '最近半年的實際表現',
    },
    # ==========================================
    # 🧪 Train / Validation Split (防 overfit)
    # ==========================================
    # Training: 只在這些期間跑 ablation 調參數
    # Validation: 參數定案後只跑一次，結果不回頭改參數
    'T': {
        'name': '🏋️ Training 完整 (2021-01~2025-06)',
        'start': '2021-01-01', 'end': '2025-06-30',
        'desc': '調參專用 — 含空頭/多頭/V轉/恐慌，4.5年完整市況',
    },
    'T1': {
        'name': '🏋️ Train: 純空頭 (2022)',
        'start': '2022-01-01', 'end': '2022-10-31',
        'desc': 'Training — 加權 18000→12600',
    },
    'T2': {
        'name': '🏋️ Train: 純多頭 (2023~2024)',
        'start': '2023-01-01', 'end': '2024-07-31',
        'desc': 'Training — 加權 14000→24000 AI狂飆',
    },
    'T3': {
        'name': '🏋️ Train: 高檔修正 (2024H2)',
        'start': '2024-07-01', 'end': '2025-01-31',
        'desc': 'Training — AI 過熱回檔',
    },
    'T4': {
        'name': '🏋️ Train: 關稅崩盤 (2025H1)',
        'start': '2025-02-01', 'end': '2025-06-30',
        'desc': 'Training — 川普關稅急跌 (截止至 training 邊界)',
    },
    'V': {
        'name': '✅ Validation (2025-07~now)',
        'start': '2025-07-01', 'end': datetime.datetime.now().strftime('%Y-%m-%d'),
        'desc': '驗證專用 — 參數定案後只跑一次，禁止回頭改參數！',
    },
}


# ==========================================
# 📊 大盤歷史狀態重建 (匹配 stock_utils.py 邏輯)
# ==========================================
def _download_index(symbol, start, end):
    """下載指數資料 (靜音 yfinance 錯誤)"""
    try:
        with open(os.devnull, 'w') as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                df = yf.download(symbol, start=start, end=end, progress=False)
            finally:
                sys.stderr = old_stderr
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def _calc_index_indicators(df):
    """為指數 DataFrame 計算 MA20/MA60/Change/Bias/Cum3d"""
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    df['Change_pct'] = df['Close'].pct_change()
    df['Bias_pct'] = (df['Close'] - df['MA20']) / df['MA20']
    df['Cum_3d'] = df['Close'].pct_change(periods=3)
    return df


def _eval_index_at_row(row):
    """從單一 row 算出 trend / is_crash (匹配 evaluate_single_index)"""
    if pd.isna(row.get('MA20')) or pd.isna(row.get('MA60')):
        return None

    price = float(row['Close'])
    ma20 = float(row['MA20'])
    ma60 = float(row['MA60'])
    change_pct = float(row['Change_pct']) if not pd.isna(row['Change_pct']) else 0
    bias_pct = float(row['Bias_pct']) if not pd.isna(row['Bias_pct']) else 0
    cum_3d = float(row['Cum_3d']) if not pd.isna(row['Cum_3d']) else 0

    if price < ma20 and price < ma60:
        trend = 'bear'
    elif price < ma20:
        trend = 'weak'
    elif price > ma20 and price > ma60:
        trend = 'bull'
    else:
        trend = 'neutral'

    is_crash = (change_pct < -0.025) or (price < ma20 and bias_pct < -0.02)

    return {
        'price': price, 'ma20': ma20, 'bias_pct': bias_pct,
        'change_pct': change_pct, 'cum_3d': cum_3d,
        'trend': trend, 'is_crash': bool(is_crash),
    }


def reconstruct_market_history(start_date, end_date, otc_unsafe_mode='or'):
    """
    預先計算回測期間每個交易日的大盤狀態。
    V4.20: 加入櫃買指數 (^TWOII) 重建，與 stock_utils.get_market_status 一致。

    Args:
        otc_unsafe_mode: 'or'  = TWII 或 OTC 任一偏空即不安全 (最嚴格, 匹配 Mode 5)
                         'and' = TWII 且 OTC 都偏空才不安全 (共識法)
                         'off' = 只看 TWII (v2.11 相容, 無 OTC)
    """
    _mode_desc = {'or': 'TWII∪OTC(嚴格)', 'and': 'TWII∩OTC(共識)', 'off': 'TWII-only(v2.11)'}
    print(f"⏳ 正在重建大盤歷史狀態... [{_mode_desc.get(otc_unsafe_mode, otc_unsafe_mode)}]")

    download_start = (pd.Timestamp(start_date) - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
    download_end = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')

    # 下載加權指數
    twii_df = _download_index('^TWII', download_start, download_end)
    if twii_df.empty or len(twii_df) < 61:
        print("❌ 加權指數資料不足")
        return {}
    _calc_index_indicators(twii_df)

    # V4.20: 下載櫃買指數 (嘗試多個符號, ^TWOII 為 2026 實測可用)
    otc_df = pd.DataFrame()
    for otc_symbol in ['^TWOII', '^TWO', '^TWOTCI', 'OTCI.TW']:
        otc_df = _download_index(otc_symbol, download_start, download_end)
        if not otc_df.empty and len(otc_df) >= 61:
            break
    otc_available = not otc_df.empty and len(otc_df) >= 61
    if otc_available:
        _calc_index_indicators(otc_df)
        print(f"   ✅ 櫃買指數: {len(otc_df)} 個交易日 ({otc_symbol})")
    else:
        print(f"   ⚠️ 櫃買指數無資料，回測只使用加權指數")

    # V6.4: 下載 EWT (iShares MSCI Taiwan ETF) — 美股隔夜情緒
    ewt_df = _download_index('EWT', download_start, download_end)
    ewt_available = not ewt_df.empty and len(ewt_df) >= 21
    if ewt_available:
        _calc_index_indicators(ewt_df)
        print(f"   ✅ EWT: {len(ewt_df)} 個交易日")
    else:
        print(f"   ⚠️ EWT 無資料，EWT 濾網不可用")

    market_history = {}

    # V6.7: 週線趨勢過濾 — 用 TWII 週 K 計算 MA20/MA60
    #   weekly_ma20 > weekly_ma60 → 週線多頭 → 允許買進
    #   weekly_ma20 < weekly_ma60 → 週線空頭 → 停止新建倉
    twii_weekly = twii_df['Close'].resample('W-FRI').last().dropna()
    twii_weekly_ma20 = twii_weekly.rolling(20, min_periods=20).mean()
    twii_weekly_ma60 = twii_weekly.rolling(60, min_periods=60).mean()
    # 預建: 每個交易日 → 最近完整週的週線狀態
    weekly_trend_map = {}  # date_str -> bool (True=多頭)
    for idx in range(len(twii_df)):
        d = twii_df.index[idx]
        ds = d.strftime('%Y-%m-%d')
        # 找最近的已完成週五 (含當天如果是週五)
        recent_fri = twii_weekly.index[twii_weekly.index <= d]
        if len(recent_fri) > 0:
            fri = recent_fri[-1]
            w_ma20 = twii_weekly_ma20.get(fri)
            w_ma60 = twii_weekly_ma60.get(fri)
            if pd.notna(w_ma20) and pd.notna(w_ma60) and w_ma60 > 0:
                weekly_trend_map[ds] = bool(w_ma20 > w_ma60)
            else:
                weekly_trend_map[ds] = True  # 資料不足預設允許
        else:
            weekly_trend_map[ds] = True

    # 預建 OTC 日期索引 (避免迴圈中重複 strftime)
    otc_date_map = {}
    if otc_available:
        for oi in range(len(otc_df)):
            od = otc_df.index[oi].strftime('%Y-%m-%d')
            otc_date_map[od] = oi

    # V6.4: 預建 EWT 日期映射 (US T-1 → Taiwan T)
    # EWT 收盤 16:00 ET ≈ 台灣次日凌晨 04:00-05:00
    # 台灣 date T → 查找 EWT index <= (T - 1 day) 的最近交易日
    ewt_date_map = {}  # taiwan_date_str -> ewt_df iloc index
    if ewt_available:
        ewt_index = ewt_df.index
        for idx_tw in range(len(twii_df)):
            tw_date = twii_df.index[idx_tw]
            tw_date_str = tw_date.strftime('%Y-%m-%d')
            target_us = tw_date - pd.Timedelta(days=1)
            actual_us = ewt_index.asof(target_us)
            if not pd.isna(actual_us):
                ewt_date_map[tw_date_str] = ewt_index.get_loc(actual_us)

    for idx in range(len(twii_df)):
        date_str = twii_df.index[idx].strftime('%Y-%m-%d')

        twii_eval = _eval_index_at_row(twii_df.iloc[idx])
        if twii_eval is None:
            continue

        # --- is_unsafe: TWII + OTC (V4.20: 匹配 stock_utils.get_market_status) ---
        is_twii_bad = (twii_eval['trend'] in ['bear', 'weak']) or twii_eval['is_crash']

        is_otc_bad = False
        if otc_available and date_str in otc_date_map:
            otc_eval = _eval_index_at_row(otc_df.iloc[otc_date_map[date_str]])
            if otc_eval is not None:
                is_otc_bad = (otc_eval['trend'] in ['bear', 'weak']) or otc_eval['is_crash']

        # V4.20: OTC 合併模式 (由 otc_unsafe_mode 參數控制)
        if otc_unsafe_mode == 'or':
            is_unsafe = is_twii_bad or is_otc_bad
        elif otc_unsafe_mode == 'and':
            is_unsafe = is_twii_bad and is_otc_bad
        else:  # 'off'
            is_unsafe = is_twii_bad

        # --- is_overheated ---
        is_overheated = twii_eval['bias_pct'] > 0.08

        # --- is_panic ---
        is_panic = (twii_eval['change_pct'] < -0.03) or (twii_eval['cum_3d'] < -0.035)

        # --- V6.4: EWT 隔夜情緒 ---
        ewt_info = None
        if ewt_available and date_str in ewt_date_map:
            ewt_iloc = ewt_date_map[date_str]
            ewt_row = ewt_df.iloc[ewt_iloc]
            ewt_close = float(ewt_row['Close'])
            ewt_change = float(ewt_row['Change_pct']) if not pd.isna(ewt_row.get('Change_pct')) else 0
            ewt_cum3d = float(ewt_row['Cum_3d']) if not pd.isna(ewt_row.get('Cum_3d')) else 0
            ewt_ma20 = float(ewt_row['MA20']) if not pd.isna(ewt_row.get('MA20')) else ewt_close
            ewt_info = {
                'daily_chg': ewt_change,
                'cum_3d_chg': ewt_cum3d,
                'close': ewt_close,
                'above_ma20': ewt_close > ewt_ma20,
            }

        market_history[date_str] = {
            'date': date_str,
            'is_unsafe': bool(is_unsafe),
            'is_overheated': bool(is_overheated),
            'is_panic': bool(is_panic),
            'weekly_bullish': weekly_trend_map.get(date_str, True),
            'twii': twii_eval,
            'ewt': ewt_info,
        }

    print(f"   ✅ 大盤狀態: {len(market_history)} 個交易日 (含前置MA計算期)")

    # 統計各狀態天數 (只算回測區間內)
    in_range = {k: v for k, v in market_history.items() if start_date <= k <= end_date}
    n_total = len(in_range)
    n_unsafe = sum(1 for v in in_range.values() if v['is_unsafe'])
    n_hot = sum(1 for v in in_range.values() if v['is_overheated'])
    n_panic = sum(1 for v in in_range.values() if v['is_panic'])
    n_safe = n_total - n_unsafe
    if n_total > 0:
        print(f"   📊 回測區間 {n_total} 天: "
              f"可買入 {n_safe} 天 ({n_safe/n_total*100:.0f}%) | "
              f"偏空 {n_unsafe} 天 ({n_unsafe/n_total*100:.0f}%) | "
              f"過熱 {n_hot} 天 | 恐慌 {n_panic} 天")

        # V6.7: 週線統計
        n_weekly_bull = sum(1 for v in in_range.values() if v.get('weekly_bullish', True))
        n_weekly_bear = n_total - n_weekly_bull
        print(f"   📊 週線趨勢: 多頭 {n_weekly_bull} 天 ({n_weekly_bull/n_total*100:.0f}%) | "
              f"空頭 {n_weekly_bear} 天 ({n_weekly_bear/n_total*100:.0f}%)")

        # V6.4: EWT 統計
        if ewt_available:
            n_ewt_weak = sum(1 for v in in_range.values()
                             if v.get('ewt') and v['ewt']['daily_chg'] < -0.02)
            n_ewt_3d = sum(1 for v in in_range.values()
                           if v.get('ewt') and v['ewt']['cum_3d_chg'] < -0.035)
            print(f"   📊 EWT 統計: 日跌>2% {n_ewt_weak} 天 | 3日累跌>3.5% {n_ewt_3d} 天")

        # V4.20: OTC 影響分析 (debug 用)
        if otc_available:
            n_twii_only = 0
            n_otc_only = 0
            n_both = 0
            for k, v in in_range.items():
                _twii = v.get('twii', {})
                _twii_bad = (_twii.get('trend') in ['bear', 'weak']) or _twii.get('is_crash', False)
                _otc_bad_day = False
                if k in otc_date_map:
                    _oe = _eval_index_at_row(otc_df.iloc[otc_date_map[k]])
                    if _oe:
                        _otc_bad_day = (_oe['trend'] in ['bear', 'weak']) or _oe['is_crash']
                if _twii_bad and _otc_bad_day:
                    n_both += 1
                elif _twii_bad:
                    n_twii_only += 1
                elif _otc_bad_day:
                    n_otc_only += 1
            n_twii_total = n_twii_only + n_both
            n_otc_total = n_otc_only + n_both
            print(f"   📊 偏空拆解: "
                  f"上市(TWII)偏空 {n_twii_total} 天 | "
                  f"上櫃(OTC)偏空 {n_otc_total} 天 | "
                  f"兩者皆空 {n_both} 天 | "
                  f"OTC 額外增加 {n_otc_only} 天 "
                  f"({n_otc_only/max(1,n_total)*100:.0f}%)")
    else:
        print(f"   ⚠️ 回測區間 0 天 (指定日期可能非交易日)")

    return market_history


# ==========================================
# 📅 回測區間選擇
# ==========================================
def select_period():
    """選擇或自訂回測區間"""
    print("\n📅 選擇回測區間:")
    _printed_train_header = False
    for key, p in PRESET_PERIODS.items():
        # 在 Training 區塊前加分隔線
        if key == 'T' and not _printed_train_header:
            print(f"   {'─'*60}")
            print(f"   🧪 Train/Validation Split (防 overfit)")
            print(f"   {'─'*60}")
            _printed_train_header = True
        print(f"   {key:>2}. {p['name']}  ({p['start']} ~ {p['end']})")
        print(f"       {p['desc']}")
    print(f"   C. 自訂日期")

    choice = input("👉 選擇 (Enter=1): ").strip()

    if choice in PRESET_PERIODS:
        p = PRESET_PERIODS[choice]
        return p['start'], p['end']
    elif choice.upper() == 'C':
        s = input("   起始日 (YYYY-MM-DD): ").strip()
        e = input("   結束日 (YYYY-MM-DD): ").strip()
        return s, e
    else:
        p = PRESET_PERIODS['1']
        return p['start'], p['end']


# ==========================================
# ⏱️ 成交模式選擇
# ==========================================
def select_exec_mode():
    """選擇成交模式"""
    print("\n⏱️ 選擇成交模式:")
    print("   1. T+1 開盤成交 (昨日訊號→今日開盤下單，推薦)")
    print("   2. 當日收盤成交 (當日訊號→當日收盤下單)")
    print("   3. 收盤+開盤雙買 (T日收盤買+T+1開盤再買，積極進場)")
    choice = input("👉 選擇 (Enter=1): ").strip()
    if choice == '2':
        return 'same_close'
    if choice == '3':
        return 'close_open'
    return 'next_open'

# 嘗試匯入產業管理模組
try:
    from industry_manager import get_all_companies, list_industries, get_stocks_by_industry
except ImportError:
    print("⚠️ 缺少 industry_manager.py，無法使用產業功能")
    def get_all_companies(): return pd.DataFrame()
    def list_industries(df): return []
    def get_stocks_by_industry(name): return []


# ==========================================
# 📋 Group Scanner 篩選門檻 (與 group_scanner.py 一致)
# ==========================================
MIN_VOLUME_SHARES = 500_000
MIN_TURNOVER      = 50_000_000
MIN_PRICE         = 10
MIN_DATA_DAYS     = 60


# ==========================================
# 🏭 產業專屬參數 (ablation 實證最佳化)
#
#   選擇「標準回測」時, 若只選了一個產業且有對應的專屬參數,
#   會自動套用該產業的 config_override, 不再使用 DEFAULT_CONFIG。
#   若選了多個產業, 則使用 DEFAULT_CONFIG (因為各產業最佳參數不同)。
# ==========================================
INDUSTRY_CONFIGS = {
    # ==========================================
    #   v2.11 L2 交叉驗證更新 (2026-02)
    #   方法: L1 單因子掃描 → L2 多因子交叉 → 6 區間一致性驗證
    #   淘汰: 電子通路業 (baseline Sharpe -0.02, 不適合動能策略)
    #   共同發現: brk5 和 f1_strict 是 L1→L2 一致性陷阱
    # ==========================================

    # 半導體業: L3+panic+SM+vol_C 冠軍 (Sharpe 1.36)
    #   L2: b30/z10 (Sharpe 1.32) → V2 ablation 發現 tierB_10 唯一有效
    #   L3 (V3 ablation): tierB10+tierA35 穩定性驗證 6/6 段不差, 4/6 更好
    #   V4 (panic ablation): portfolio_panic + dyn_stop + s2_buffer
    #   V8 (SM ablation): sector_momentum Sharpe +0.04
    #   V6.5 (vol_C): vol_sizing 2.5%/floor70% → Sharpe 1.35→1.36, Return +226.7→+229.6%
    #     穩定性驗證: 6段 1勝0敗5平, 無任何區間劣化
    #   決策: tier_b_net 20→10, tier_a_net 40→35, SM開, panic開, vol_sizing開
    '半導體業': {
        'desc': '🏆 V19b+R6+R11+R14+R15: A+ (Shrp 0.92, Clm 0.89, Ret+138%, MDD 24.1%)',
        'config': {
            # --- V7: NAV-based position sizing ---
            'budget_pct': 2.8,               # 每筆 = NAV * 2.8% (25K/900K ≈ 2.78%)
            # --- L3 base ---
            'zombie_hold_days': 10,          # L2: z10 > z15(default), 半導體週轉快
            'tier_b_net': 15,                # L3: tierB 停利門檻 10→15%, 測試放寬停利
            'tier_a_net': 80,                # V19b A+: 65→80, 讓大贏家跑更久 (ablation: Ret+6.6%, PF+0.11)
            # --- V7.3: Tier B drawdown 收緊 ---
            'tier_b_drawdown': 0.6,          # V7.3: 0.7→0.6, R4 ablation: Train Shrp 0.73→0.77, Ret+6.5%, MDD-1.4%, Val全面更優
            # --- V6.8: RSI 動能門檻 ---
            'min_rsi': 40,                   # RSI<40不買 (Train Sharpe 0.82→0.87, Val持平, PF 1.76→1.81)
            # bias 保留 default (bull=30, neutral=25, bear=20) — L2 驗證 b30 最優
            # --- V9: 魚尾回看優化 ---
            'fish_tail_lookback': 3,         # V9: 5→3, ablation Sharpe 1.18→1.21, Return +204→+214%, MDD持平
            # --- V7.3: Portfolio Panic + 動態停損 ---
            'enable_portfolio_panic': True,          # V7.3: R4 ablation MDD 37.4→31.4%(-6pp), Ret+4.8%, Calmar 0.48→0.59
            'portfolio_panic_day_pct': -4.0,         # R14: -3.0→-4.0, 放寬恐慌閾值 (Shrp 0.90→0.94, Clm 0.75→0.80, Ret+10%)
            'portfolio_panic_3d_pct': -6.0,          # R14: -5.0→-6.0, 減少正常回檔時的誤觸發
            'portfolio_panic_action': 'sell_losers',  # 賣虧損股
            'portfolio_panic_loss_threshold': 0.0,   # 淨利 <0% 才賣
            'portfolio_panic_cooldown': 3,           # 冷卻3天
            'enable_dyn_stop': True,                 # 動態停損
            'hard_stop_weak': -12,                   # 偏弱時 -12%
            'hard_stop_bear': -10,                   # 空頭時 -10%
            's2_buffer_enabled': True,               # S2 跌破季線緩衝
            's2_buffer_net_pct': 10,                 # 淨利>10%才給緩衝
            's2_buffer_days': 5,                     # V19b A+: 3→5, 減少假跌破被洗出 (ablation: Ret+25%, Shrp+0.10)
            # --- V8: 產業動能 ---
            'enable_sector_momentum': True,  # 產業動能開關
            'sm_lookback': 20,               # 20日產業平均報酬
            'sm_positive_bonus': 0.15,       # 產業正向加分
            'sm_negative_penalty': -0.3,     # 產業負向扣分
            'sm_strong_threshold': 0.02,     # 強勁門檻
            'sm_strong_bonus': 0.2,          # 強勁加分
            'sm_weak_threshold': -0.03,      # 疲弱門檻
            'sm_weak_penalty': -0.5,         # 疲弱扣分
            # --- V6.5: 波動率倉位控制 ---
            'enable_vol_sizing': True,       # 高波動時自動縮小單筆金額
            'vol_target_pct': 2.5,           # 目標波動率 2.5%
            'vol_scale_floor_pct': 70,       # 最低縮至原金額 70%
            # --- V8+V8.1: 子題材動量 Boost + 市場自適應 ---
            'enable_theme_boost': False,
            # --- V11: 子題材輪動 ---
            'enable_theme_rotation': False,
            # --- V15: 非對稱殭屍 (A+ ablation: Ret+21%, PF 2.01, WR+2.1%) ---
            'zombie_asymmetric': True,
            'zombie_loss_days': 15,          # 虧損持倉>15天 → 快砍
            'zombie_profit_days': 45,        # 獲利持倉>45天 + 獲利<8% → 效率清除
            'zombie_profit_min_pct': 8.0,
            # --- V19b: Peer RS 子題材分群 (Shrp+0.08, Clm+0.10, Ret+18.4%) ---
            'enable_val_peer_hold': True,    # 族群內相對強度影響出場
            'val_peer_use_theme': True,      # 用 THEME_MAP 子題材分群 (vs 全產業)
            'val_peer_lookback': 60,         # 60日報酬率比較
            'val_peer_expensive_z': 1.2,     # R6: 1.5→1.2, 更積極收緊族群內偏貴股 (Shrp 0.86→0.88, Clm 0.70→0.73, Ret+6.4%)
            'val_peer_expensive_dd': 0.48,   # R6: 0.50→0.48, 配合 Z 收緊
            'val_peer_weak_z': -1.0,         # Z<-1 族群內最弱 → zombie 提前 3 天
            'val_peer_weak_zombie_cut': 3,
            # --- V29: 子題材持倉集中度限制 (R11: Shrp 0.88→0.90, Ret+4.3%, Val持平) ---
            'theme_max_hold': 3,             # 同子題材最多持3檔, 強迫分散
            # --- V31: 多時間框架 (R15: MDD 28.4→24.1%, Calmar 0.80→0.89, Val不變) ---
            'enable_mtf': True,
            'mtf_weekly_block_buy': True,    # 週線空頭(MA20<MA60)時不開新倉
            'mtf_weekly_tighten': False,     # 不額外收緊出場(已有其他機制)
            # --- F4: EWT 隔夜濾網 (R16b: 5時段交叉驗證零惡化, MDD -0.3~-0.9%) ---
            'enable_ewt_filter': True,
            'ewt_drop_threshold': -0.03,
            'ewt_3d_threshold': -0.05,
            # --- V39: VIX 恐慌指數 (R20: Shrp 0.93→0.96, MDD 24.4→23.0%, Clm 0.88→0.98, Val不變) ---
            'enable_vix_filter': True,
            'vix_caution_threshold': 25,     # VIX > 25 → 每日最多買 2 檔
            'vix_block_threshold': 30,       # VIX > 30 → 全停
            'vix_caution_max_buy': 2,
            # --- R23: market_filter_mode moderate (Clm 0.98→1.02, MDD 23.0→22.2%, Val Shrp+0.04) ---
            'market_filter_mode': 'moderate',
            # --- R24: 加碼上限 (Shrp 0.96→1.01, Clm 1.02→1.06, Ret+6.4%, Val+3.2%) ---
            'max_add_per_stock': 8,          # 每檔最多加碼8次, 防止單股集中度>30% NAV
        },
    },
    # 其他電子業: L2冠軍 b15/z30 (Sharpe 1.13)
    #   L1: bias10 > bias15, z30 冠軍, brk5 看似好
    #   L2: bias 從 10→15 提升穩定度; brk5 交叉後劣化 → REJECT
    #   決策: bias15 統一三態, z30 大幅延長清理
    '其他電子業': {
        'desc': '🏆 L2冠軍 b15/z30 (Sharpe 1.13, brk5 rejected)',
        'config': {
            'bias_limit_bull': 15,
            'bias_limit_neutral': 15,
            'bias_limit_bear': 15,
            'zombie_hold_days': 30,          # L2: z30 >> z25, 其他電子需要長清理
            'enable_breakout': False,        # L2: brk5 交叉劣化, 關閉突破
            # S1 保留 default loose — L1 未見 S1 顯著差異
            # F1 保留 default relaxed — L1/L2 未測試 F1 維度
        },
    },
    # 通信網路業: L3冠軍 bias_35+tierB_10 (Sharpe 1.23)
    #   L2 base: bias_30+z20 (Sharpe 1.13, CAGR 25.1%, MDD 17.8%)
    #   L3: bias_35+tierB_10 穩定性驗證通過
    #     全期: Sharpe 1.13→1.23(+0.099), MDD 17.8%→14.4%, CAGR 25.1%→28.0%
    #   決策: bias 30→35 (放寬乖離追動能), tier_b_net 20→10 (更敏感回撤停利)
    '通信網路業': {
        'desc': '🏆 L3冠軍 bias_35+tierB_10 (Sharpe 1.23, L3穩定驗證通過)',
        'config': {
            'bias_limit_bull': 35,
            'bias_limit_neutral': 35,
            'bias_limit_bear': 35,
            'zombie_hold_days': 20,          # L2: z20 > z15
            'tier_b_net': 10,                # L3: tierB 停利門檻 20→10%, 更敏感回撤停利
            # tier_a_net 保留 default 40 — L3 base 未含 tight, ablation 驗證 default 更優
            # breakout 保留 default brk10 — L3 base 未含 brk20
            # F1 保留 default relaxed
        },
    },
    # 電腦及週邊設備業: L2冠軍 b30/loose/z20 (Sharpe 1.06)
    #   L1: bias25/30 好, s1loose 冠軍, z20 冠軍, brk5 看似好
    #   L2: brk5 全期最高(1.09)但 MDD 72.6%且急殺/V轉劣化 → REJECT
    #   L2: b30/loose/z20 = 1.06, V轉期唯一正數(+0.33), 防禦最穩
    #   決策: b30 統一三態, loose 保留 default, z20 延長清理
    '電腦及週邊設備業': {
        'desc': '🏆 L2冠軍 b30/loose/z20 (Sharpe 1.06, V轉唯一正數)',
        'config': {
            'bias_limit_bull': 30,
            'bias_limit_neutral': 30,
            'bias_limit_bear': 30,
            'zombie_hold_days': 20,          # L2: z20 >> z15 (全期+0.21, 多頭+0.39)
            'enable_breakout': False,        # L2: brk5 V轉/急殺劣化, REJECT
            # S1 保留 default loose — L2 驗證 loose 勝出 (V轉差距 1.20!)
            # F1 保留 default relaxed — L1/L2 未測試 F1 維度
        },
    },
    # 電子零組件業: L2冠軍 b10/z25 (Sharpe 0.98)
    #   L1: bias10 壓 MDD, z20 冠軍
    #   L2: zombie 從 20→25 進一步改善; bias10 維持
    #   決策: 保守型配置, bias10 壓風險 + z25 延長清理
    '電子零組件業': {
        'desc': '🏆 L2冠軍 b10/z25 (Sharpe 0.98, 防禦導向)',
        'config': {
            'bias_limit_bull': 10,
            'bias_limit_neutral': 10,
            'bias_limit_bear': 10,
            'zombie_hold_days': 25,          # L2: z25 > z20(L1冠軍), 零組件需較長清理
            # S1 保留 default loose — L2 未測試 S1 維度
            # F1 保留 default relaxed — L1/L2 未測試 F1 維度
            # breakout 保留 default True — L2 未測試 brk 維度
        },
    },
    # 光電業: 本輪未重新 ablation, 保留上一版 L2 結果
    #   L2冠軍: b15/s1tight/z30 (Sharpe 0.38, MDD 28.6% vs baseline 72%)
    '光電業': {
        'desc': '🏭 L2冠軍 b15/s1tight/z30 (Sharpe 0.38, MDD 28.6% vs baseline 72%)',
        'config': {
            'bias_limit_bull': 15,
            'bias_limit_neutral': 15,
            'bias_limit_bear': 15,
            'tier_a_net': 20,
            'tier_a_ma_buf': 0.98,
            'tier_b_net': 10,
            'tier_b_drawdown': 0.5,
            'zombie_hold_days': 30,
        },
    },
    # 電機機械: L2冠軍 b20/z10/f1_mod (Sharpe 0.79)
    #   L1: bias10/15 好, z10 冠軍, brk5 看似好, f1_mod 有效
    #   L2: brk5 交叉後高檔修正災難級劣化 → REJECT
    #   L2: bias20/z10/f1_mod 六期最穩, 空頭唯一正數(+0.03)
    #   決策: bias20 統一三態, z10 快速清理, f1_moderate 收緊濾網
    '電機機械': {
        'desc': '🏆 L2冠軍 b20/z10/f1mod (Sharpe 0.79, brk5 rejected)',
        'config': {
            'bias_limit_bull': 20,
            'bias_limit_neutral': 20,
            'bias_limit_bear': 20,
            'zombie_hold_days': 10,          # L2: z10 快速清理, 傳產週轉快
            'market_filter_mode': 'moderate', # L2: f1_mod 提升防禦 (空頭+0.03)
            # S1 保留 default loose — L2 未測試 S1 維度
            # breakout 保留 default True — brk5 REJECT, 保留 default brk10
        },
    },
    # 電子通路業: v2.11 淘汰 (baseline Sharpe -0.02, L1 最佳僅 0.40)
    #   V4.15: 套用電子零組件業參數 (同為電子供應鏈, 防禦導向)
    #   用於 portfolio_other.csv 持有判定 (sell/hold only)
    '電子通路業': {
        'desc': '📎 套用電子零組件業參數 (V4.15, sell/hold 判定用)',
        'config': {
            'bias_limit_bull': 10,
            'bias_limit_neutral': 10,
            'bias_limit_bear': 10,
            'zombie_hold_days': 25,
        },
    },
    # 金融保險業: 無 ablation, V4.15 套用光電業參數
    #   理由: 金融股低波動, 需要保守停利 (TierA=20%, TierB=10%)
    #   半導體 TierA=40% 對金融股永遠不觸發, 光電業最適配
    '金融保險業': {
        'desc': '📎 套用光電業參數 (V4.15, 低波動適配, TierA=20%)',
        'config': {
            'bias_limit_bull': 15,
            'bias_limit_neutral': 15,
            'bias_limit_bear': 15,
            'tier_a_net': 20,
            'tier_a_ma_buf': 0.98,
            'tier_b_net': 10,
            'tier_b_drawdown': 0.5,
            'zombie_hold_days': 30,
        },
    },
    # 塑膠工業: 無 ablation, V4.15 套用電子零組件業參數
    #   理由: 傳產低波動, 需最保守乖離限制 (bias 10%)
    '塑膠工業': {
        'desc': '📎 套用電子零組件業參數 (V4.15, 低波動傳產)',
        'config': {
            'bias_limit_bull': 10,
            'bias_limit_neutral': 10,
            'bias_limit_bear': 10,
            'zombie_hold_days': 25,
        },
    },
    # 其他業: 無 ablation 報告, 使用半導體 default
    '其他業': {
        'desc': '⚠️ 無 ablation 資料, 暫用 DEFAULT_CONFIG (待優化)',
        'config': {},  # 空 dict = 使用 DEFAULT_CONFIG
    },
}

# ==========================================
# 🏆 V5冠軍升級配置 (實驗性, 尚未驗證與 mode 1 一致)
#   在 L3 基礎上加 6 因子, ablation 結果:
#     MDD 51%→34%, Sharpe 1.35→1.33, Calmar 0.64→0.71
#   使用方式: mode 1/7 選半導體時會出現 A/B/C 選項
# ==========================================
INDUSTRY_CONFIGS_V5 = {
    '半導體業': {
        'desc': '🧪 V5冠軍 6因子 (Calmar 0.71, MDD 34%, Sharpe 1.33) [實驗]',
        'config': {
            # --- 原始 L3 base ---
            'zombie_hold_days': 10,
            'tier_a_net': 35,
            # --- combo ablation 冠軍 3 因子 ---
            'tier_b_net': 5,                 # tb_5: 10→5
            'max_add_per_stock': 3,          # add_3: 限加碼3次
            'hard_stop_net': -12,            # hs_12: 停損 -15→-12%
            # --- panic ablation 冠軍 3 因子 ---
            'enable_portfolio_panic': True,  # panic_d3
            'portfolio_panic_day_pct': -3.0,
            'portfolio_panic_3d_pct': -6.0,
            'portfolio_panic_action': 'sell_losers',
            'portfolio_panic_loss_threshold': 0.0,
            'portfolio_panic_cooldown': 3,
            'enable_dyn_stop': True,         # dyn_stop_mod
            'hard_stop_weak': -12,
            'hard_stop_bear': -10,
            's2_buffer_enabled': True,       # s2_buf_3d
            's2_buffer_net_pct': 10,
            's2_buffer_days': 3,
        },
    },
}

# ==========================================
# 📊 產業配額制 (v2.11 倉位實驗)
#   按 L2 全期 Sharpe 加權分配 15 倉位
#   半導體 1.32 / 其他電子 1.13 / 電腦週邊 1.06 / 電子零組件 0.98 / 電機機械 0.79 / 通信網路 0.78
# ==========================================
INDUSTRY_QUOTA = {
    '半導體業':         4,
    '其他電子業':       3,
    '電腦及週邊設備業': 3,
    '電子零組件業':     3,
    '電機機械':         1,
    '通信網路業':       1,
}  # 合計 15


# ==========================================
# 📊 Mode 5: 每日實盤訊號常數
# ==========================================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_STRATEGY_FILE = os.path.join(_BASE_DIR, 'portfolio_strategy.csv')
PORTFOLIO_OTHER_FILE = os.path.join(_BASE_DIR, 'portfolio_other.csv')
INDUSTRY_PORTFOLIO_FILES = {
    '通信網路業': os.path.join(_BASE_DIR, 'portfolio_telecom.csv'),
    # 其他產業使用預設 PORTFOLIO_STRATEGY_FILE (portfolio_strategy.csv)
}
TRADE_LOG_FILE = os.path.join(_BASE_DIR, 'trade_log.csv')
PERFORMANCE_LOG_FILE = os.path.join(_BASE_DIR, 'performance_log.csv')
DAILY_DEFAULT_INDUSTRIES = ['半導體業']          # V4.15: 純半導體 (L3: Sharpe 1.35/MDD 30.6%/CAGR 32.5%)
DAILY_BUDGET = 25000  # V4.18 ablation: 25K > 15K (Sharpe 1.32 vs 1.11)
DAILY_MAX_POSITIONS = 12   # V4.22: 20→12, 與 DEFAULT_CONFIG 一致
DAILY_MAX_NEW_BUY = 4
DAILY_MAX_SWAP = 1         # V4.22: 3→1, 與 DEFAULT_CONFIG 一致
_PORTFOLIO_COLS = ['ticker', 'name', 'industry', 'shares', 'avg_cost', 'buy_price', 'peak_since_entry', 'note']


# ==========================================
# 📈 Group Scanner 驅動回測引擎
#
# 模擬你的實際操作流程:
#   1. 每天跑 Group Scanner → 發現買進訊號
#   2. T+1 開盤買入 (固定金額)
#   3. 已持有 → 每天跑 Daily Scanner 邏輯
#      - 加碼訊號 → T+1 加碼
#      - 賣出訊號 → T+1 賣出
#   4. 統計跨多檔的總績效
# ==========================================
def _get_ticker_config(ticker, industry_map, per_industry_config, config_override):
    """根據 ticker 取得對應策略參數

    V4.20 修正: per-industry 與 config_override 合併而非二擇一。
    合併順序: per_industry 為基底 → config_override 疊加覆蓋。
    這樣 ablation 的 config_override (如 limit_up_retry) 能和
    產業 L2 參數 (如 zombie_hold_days) 同時生效。

    回傳值直接傳給 check_strategy_signal 的 config 參數。
    """
    result = {}
    # 1. 先套用產業專屬參數 (L2 最優)
    if per_industry_config and industry_map:
        ind = industry_map.get(ticker)
        if ind and ind in per_industry_config:
            result.update(per_industry_config[ind])
    # 2. 再疊加 config_override (ablation 參數優先)
    if config_override:
        result.update(config_override)
    return result if result else config_override


# === CSV-driven helpers ===

def load_positions_csv(filepath):
    """Load positions CSV → dict compatible with initial_positions parameter."""
    positions = {}
    if not os.path.exists(filepath):
        return positions
    df = pd.read_csv(filepath, dtype={'ticker': str}, encoding='utf-8-sig')
    for _, row in df.iterrows():
        ticker = str(row['ticker']).strip()
        positions[ticker] = {
            'shares': int(row['shares']),
            'avg_cost': float(row['avg_cost']),
            'cost_total': float(row.get('cost_total', row['avg_cost'] * row['shares'])),
            'buy_price': float(row.get('buy_price', row['avg_cost'])),
            'name': str(row.get('name', '')),
            'buy_count': int(row.get('buy_count', 1)),
            'last_buy_date_idx': int(row.get('last_buy_date_idx', 0)),
            'reduce_stage': int(row.get('reduce_stage', 0)),
            'last_reduce_date_idx': int(row.get('last_reduce_date_idx', -99)),
            'peak_since_entry': float(row.get('peak_since_entry', row['avg_cost'])),
        }
    return positions


def load_state_json(filepath):
    """Load state JSON → dict compatible with initial_state parameter."""
    with open(filepath, 'r', encoding='utf-8') as f:
        state = json.load(f)
    return {
        'cash': state['cash'],
        'prev_nav': state['prev_nav'],
        'peak_nav': state['peak_nav'],
        'vol_rets': state.get('vol_rets', []),
        'pp_daily_returns': state.get('pp_daily_returns', []),
        'pp_last_trigger_idx': state.get('pp_last_trigger_idx', -99),
        'realized_profit': state.get('realized_profit', 0),
        'total_fees': state.get('total_fees', 0),
        'trade_count': state.get('trade_count', 0),
        'win_count': state.get('win_count', 0),
        'loss_count': state.get('loss_count', 0),
    }


def load_pending_from_state(filepath):
    """Load pending orders from state JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        state = json.load(f)
    return state.get('pending', {})


def find_latest_csv_date(csv_dir):
    """Find the most recent date with positions CSV in csv_dir."""
    import glob
    files = glob.glob(os.path.join(csv_dir, 'positions_*.csv'))
    if not files:
        return None
    dates = []
    for f in files:
        basename = os.path.basename(f)
        date_str = basename.replace('positions_', '').replace('.csv', '')
        dates.append(date_str)
    dates.sort()
    return dates[-1] if dates else None


def run_group_backtest(stock_list, start_date, end_date, budget_per_trade,
                       market_map, exec_mode='next_open', min_hold_days=1,
                       config_override=None, initial_capital=None,
                       preloaded_data=None, force_refresh=False,
                       industry_map=None, per_industry_config=None,
                       industry_quota=None,
                       initial_positions=None, initial_pending=None,
                       _capture_positions=False,
                       _cash_reserve_override=None,
                       _revenue_whitelist=None,
                       _revenue_blacklist=None,
                       initial_state=None,
                       initial_day_idx=0,
                       csv_output_dir=None,
                       signal_func=None):
    """
    Args:
        stock_list:        [(ticker, name), ...] 要掃描的股票清單
        start_date:        回測起始日
        end_date:          回測結束日
        budget_per_trade:  每次買入/加碼的金額
        market_map:        大盤歷史狀態 dict
        exec_mode:         'next_open' / 'same_close'
        min_hold_days:     買入後最少持有天數
        config_override:   dict, 覆蓋 DEFAULT_CONFIG (策略參數 + 上車限制)
        initial_capital:   初始總資金 (None=自動計算: budget × max_positions × 1.5)
        preloaded_data:    預載入的股票資料 {ticker: {'df': DataFrame, 'name': str}}
                           傳入時跳過下載，直接使用 (Ablation 共享資料用)
        force_refresh:     True → 強制重新下載 (忽略快取)
        industry_map:      {ticker: industry_name} 每檔股票所屬產業 (多產業混合模式用)
        per_industry_config: {industry_name: config_dict} 每產業策略參數覆蓋
                             (搭配 industry_map 使用, 讓不同產業用不同策略)
        industry_quota:    {industry_name: int} 每產業最大持倉數 (配額制模式)
                           None = 統一池模式 (不限產業分配)
        initial_positions: {ticker: {shares, avg_cost, name, buy_count,
                            last_buy_date_idx, reduce_stage,
                            peak_since_entry}} 初始持倉
                           None = 空倉開始 (標準回測模式)
                           用於 daily_signal 從現有庫存開始跑引擎
        initial_pending:   {ticker: {action, reason, ...}} 初始掛單
                           None = 無掛單 (標準回測模式)
                           用於驗證模式：帶入 split 點的 pending，第一天開盤執行
        _capture_positions: bool, True = 每日記錄 positions 快照 (驗證用, 會增加記憶體)

    Returns:
        dict with portfolio-level stats + per-stock details
    """
    from strategy import DEFAULT_CONFIG
    cfg = {**DEFAULT_CONFIG}
    if config_override:
        cfg.update(config_override)

    max_positions = cfg.get('max_positions', 12)
    max_new_buy_per_day = cfg.get('max_new_buy_per_day', 4)
    entry_sort_by = cfg.get('entry_sort_by', 'score')

    # V3.9 倉位品質管理
    enable_zombie = cfg.get('enable_zombie_cleanup', True)
    zombie_days = cfg.get('zombie_hold_days', 15)
    zombie_range = cfg.get('zombie_net_range', 5.0)
    enable_swap = cfg.get('enable_position_swap', True)
    swap_margin = cfg.get('swap_score_margin', 1.0)
    max_swap_per_day = cfg.get('max_swap_per_day', 1)

    # V15: 非對稱殭屍 — 虧損股快砍, 獲利股慢砍
    _z_asym_enabled = cfg.get('zombie_asymmetric', False)
    _z_asym_loss_days = cfg.get('zombie_loss_days', 15)        # 虧損持有>N天 → 砍
    _z_asym_profit_days = cfg.get('zombie_profit_days', 45)    # 獲利不足持有>N天 → 砍
    _z_asym_profit_min = cfg.get('zombie_profit_min_pct', 8.0) # 獲利至少>X%才留

    # V13: RS 自適應殭屍 — RS 強的慢爬股延長觀察天數
    _zombie_rs_adaptive = cfg.get('zombie_rs_adaptive', False)
    _zombie_rs_lookback = cfg.get('zombie_rs_lookback', 60)
    _zombie_rs_top_pct = cfg.get('zombie_rs_top_pct', 30)
    _zombie_rs_extra_days = cfg.get('zombie_rs_extra_days', 10)
    _zombie_rs_extra_range = cfg.get('zombie_rs_extra_range', 3.0)

    # V6.9: 信念持股 (Conviction Hold)
    _conviction_enabled = cfg.get('enable_conviction_hold', False)
    _conviction_min_buys = cfg.get('conviction_min_buys', 3)
    _conviction_zombie_extra = cfg.get('conviction_zombie_extra', 5)
    _conviction_zombie_range = cfg.get('conviction_zombie_range', 3.0)
    _conviction_stop_extra = cfg.get('conviction_stop_extra', 3.0)
    _conviction_tier_a_extra = cfg.get('conviction_tier_a_extra', 15.0)

    # V6.10: 市場狀態自適應 (Regime-Adaptive)
    _regime_adaptive = cfg.get('enable_regime_adaptive', False)
    _regime_unsafe_zombie = cfg.get('regime_unsafe_zombie_days', 7)
    _regime_unsafe_tier_b = cfg.get('regime_unsafe_tier_b_net', 10)
    _regime_unsafe_max_buy = cfg.get('regime_unsafe_max_new_buy', 1)

    # V4.11: 現金管理
    _cash_check = cfg.get('enable_cash_check', True)
    _cash_reserve_pct = cfg.get('cash_reserve_pct', 0.0)
    _enable_add = cfg.get('enable_add', True)
    _max_add_per_stock = cfg.get('max_add_per_stock', 99)
    _max_add_per_day = cfg.get('max_add_per_day', 99)  # V8: 每日加碼上限 (99=無限制)

    # V4.13: 滑價
    _slippage = cfg.get('slippage_pct', 0.0) / 100  # 轉為小數 (0.3% → 0.003)

    # V4.16: 漲停跳過
    _skip_limit_up = cfg.get('skip_limit_up', True)
    _limit_up_threshold = cfg.get('limit_up_threshold', 1.095)  # 開盤 >= 前收*1.095 視為漲停
    _limit_up_skipped = []  # 記錄被跳過的 tickers (供 Mode 5 顯示備選)

    # V4.18: 漲停遞補 (新買入漲停 → 備選遞補; 加碼漲停 → 暫緩)
    _enable_backup_fill = cfg.get('enable_backup_fill', True)
    _backup_queue = []  # T日排序後未被選中的新買入候選 (每日重算)

    # V4.20: 漲停遞延 (T+1漲停 → T+2再試)
    _limit_up_retry = cfg.get('limit_up_retry', False)
    _limit_up_max_retry = cfg.get('limit_up_max_retry', 1)

    # V12: 2年/1年新高過濾 — 只買突破 N 日新高的股票
    _require_period_high = cfg.get('require_2y_high', False)
    _period_high_lookback = cfg.get('2y_high_lookback', 490)  # 490≈2年, 245≈1年

    # V4.21: 動態曝險管理 (大盤弱勢時降低持倉上限, 強制賣出超額弱勢持倉)
    _dyn_enabled = cfg.get('enable_dynamic_exposure', False)
    _dyn_max_map = {
        'bull':    cfg.get('dyn_max_bull', max_positions),
        'neutral': cfg.get('dyn_max_neutral', max_positions),
        'weak':    cfg.get('dyn_max_weak', max_positions),
        'bear':    cfg.get('dyn_max_bear', max_positions),
    }
    _dyn_max_panic = cfg.get('dyn_max_panic', max_positions)
    _dyn_force_sell_count = 0  # 追蹤動態曝險強制賣出次數

    # V4.21: 動態限買 (大盤弱勢時降低每日新買上限)
    _dyn_buy_enabled = cfg.get('enable_dyn_buy_limit', False)
    _dyn_buy_map = {
        'bull':    cfg.get('dyn_buy_bull', max_new_buy_per_day),
        'neutral': cfg.get('dyn_buy_neutral', max_new_buy_per_day),
        'weak':    cfg.get('dyn_buy_weak', max_new_buy_per_day),
        'bear':    cfg.get('dyn_buy_bear', max_new_buy_per_day),
    }
    _dyn_buy_panic = cfg.get('dyn_buy_panic', 0)

    # V14: 每週買入上限 (降低交易頻率, 控制 MDD)
    _weekly_max_buy = cfg.get('weekly_max_buy', 0)  # 0=不限制
    _weekly_buy_count = 0
    _last_week_num = -1

    # V4.21: 動態停損 (大盤弱勢時收緊 hard_stop)
    _dyn_stop_enabled = cfg.get('enable_dyn_stop', False)
    _hard_stop_override = {
        'weak': cfg.get('hard_stop_weak', cfg.get('hard_stop_net', -15)),
        'bear': cfg.get('hard_stop_bear', cfg.get('hard_stop_net', -15)),
    }

    # V16: Portfolio DD 斷路器 — NAV 回撤超過門檻時暫停新買
    _dd_breaker_enabled = cfg.get('enable_dd_breaker', False)
    _dd_breaker_lv1_pct = cfg.get('dd_breaker_lv1_pct', 12.0)   # DD > 此% → 暫停 N 天
    _dd_breaker_lv1_days = cfg.get('dd_breaker_lv1_days', 3)
    _dd_breaker_lv2_pct = cfg.get('dd_breaker_lv2_pct', 18.0)   # DD > 此% → 暫停更多天
    _dd_breaker_lv2_days = cfg.get('dd_breaker_lv2_days', 7)
    _dd_breaker_cooldown_until = -1  # day_idx until which buying is frozen

    # V16b: 趨勢保護型 zombie — zombie 天數到了但趨勢仍在 → 不砍
    _zombie_trend_protect = cfg.get('zombie_trend_protect', False)

    # V18: 乖離率分層出場 — 股價離 MA60 越遠, tier_b 越緊
    _val_bias_enabled = cfg.get('enable_val_bias_exit', False)
    _val_bias_lv2_ratio = cfg.get('val_bias_lv2_ratio', 1.25)   # price/MA60 > 1.25 → 偏貴
    _val_bias_lv2_dd = cfg.get('val_bias_lv2_dd', 0.55)         # tier_b_drawdown 收緊
    _val_bias_lv3_ratio = cfg.get('val_bias_lv3_ratio', 1.40)   # price/MA60 > 1.40 → 極貴
    _val_bias_lv3_dd = cfg.get('val_bias_lv3_dd', 0.45)         # 更緊
    _val_bias_cheap_ratio = cfg.get('val_bias_cheap_ratio', 1.05) # price/MA60 < 1.05 → 便宜
    _val_bias_cheap_dd = cfg.get('val_bias_cheap_dd', 0.80)      # tier_b 放寬

    # V20: 營收動能出場 — 連續營收年減 → 收緊停利 / 加速出場
    _val_rev_enabled = cfg.get('enable_val_revenue', False)
    _val_rev_decline_months = cfg.get('val_rev_decline_months', 2)  # 連續N月YoY負成長
    _val_rev_action = cfg.get('val_rev_action', 'tighten')  # 'tighten' or 'force_sell'
    _val_rev_tighten_dd = cfg.get('val_rev_tighten_dd', 0.45)  # 營收衰退時 tier_b 收緊
    _val_rev_data = {}  # {ticker: {(year,month): yoy_pct}} — 在外部預載

    # V19: Peer RS 持倉調整 — 族群內相對強度影響出場
    _val_peer_hold_enabled = cfg.get('enable_val_peer_hold', False)
    _val_peer_lookback = cfg.get('val_peer_lookback', 60)
    _val_peer_expensive_z = cfg.get('val_peer_expensive_z', 1.5)  # z > 1.5 → 族群內最貴
    _val_peer_expensive_dd = cfg.get('val_peer_expensive_dd', 0.50) # 收緊 tier_b
    _val_peer_weak_z = cfg.get('val_peer_weak_z', -1.0)           # z < -1 → 族群內最弱
    _val_peer_weak_zombie_cut = cfg.get('val_peer_weak_zombie_cut', 3) # zombie 天數減少
    _val_peer_use_theme = cfg.get('val_peer_use_theme', False)    # True=用THEME_MAP分群, False=全產業
    _val_peer_theme_map = {}  # {ticker_id: [peer_ticker_ids]} — 同子題材
    if _val_peer_use_theme:
        try:
            from theme_config import THEME_MAP
            _ticker_to_theme = {}
            for theme_name, tickers in THEME_MAP.items():
                for t in tickers:
                    _ticker_to_theme[t] = theme_name
            for theme_name, tickers in THEME_MAP.items():
                for t in tickers:
                    # 同子題材的其他股票 (含 .TW/.TWO 變體)
                    peers = [p for p in tickers if p != t]
                    _val_peer_theme_map[t] = peers
                    _val_peer_theme_map[t + '.TW'] = [p + '.TW' for p in peers]
                    _val_peer_theme_map[t + '.TWO'] = [p + '.TWO' for p in peers]
        except ImportError:
            pass

    # V36: 趨勢保護 zombie（改良版）— 獲利+加碼+趨勢完好 → 不砍
    _zombie_trend_protect_v2 = cfg.get('zombie_trend_protect_v2', False)
    _ztp2_min_buys = cfg.get('ztp2_min_buys', 2)         # 至少加碼過 N-1 次
    _ztp2_min_profit = cfg.get('ztp2_min_profit', 0.0)    # 淨利 > 此% 才保護

    # V37: 階梯式 zombie — 加碼越多 zombie 天數越長
    _zombie_ladder = cfg.get('zombie_ladder', False)
    _zombie_ladder_extra = cfg.get('zombie_ladder_extra_days', 3)  # 每多加碼一次 +N天
    _zombie_ladder_cap = cfg.get('zombie_ladder_cap', 25)          # zombie 最多不超過此天數

    # V38: 贏家升級 — 獲利+高加碼次數 → 切換到長線參數
    _winner_upgrade = cfg.get('winner_upgrade', False)
    _winner_min_profit = cfg.get('winner_min_profit_pct', 10.0)  # 淨利 > 此%
    _winner_min_buys = cfg.get('winner_min_buys', 3)             # 至少加碼過 N-1 次
    _winner_tier_a = cfg.get('winner_tier_a', 120)               # 升級後 tier_a
    _winner_zombie_days = cfg.get('winner_zombie_days', 45)      # 升級後 zombie
    _winner_zombie_range = cfg.get('winner_zombie_range', 0.0)   # 升級後 zombie range (0=只砍虧)

    # V35: 外資買賣超 — 外資連續買超的股票加分, 連續賣超的收緊
    _fi_enabled = cfg.get('enable_foreign_investor', False)
    _fi_consec_buy_bonus = cfg.get('fi_consec_buy_days', 3)     # 連續N天買超 → 加分
    _fi_bonus = cfg.get('fi_bonus', 0.2)
    _fi_consec_sell_tighten = cfg.get('fi_consec_sell_days', 3)  # 連續N天賣超 → 收緊 tier_b
    _fi_tighten_dd = cfg.get('fi_tighten_dd', 0.50)
    _fi_data = {}  # {ticker_id: {date_str: net_buy_shares}}
    if _fi_enabled:
        _fi_cache = os.path.join('.cache', 'foreign_investor_data.json')
        if os.path.exists(_fi_cache):
            import json as _json_fi
            with open(_fi_cache) as _f:
                _fi_data = _json_fi.load(_f)
            print(f"   📊 V35 外資: 載入 {len(_fi_data)} 檔外資買賣超數據")

    # V41: 限價單模擬 — 開盤價追高太多時不買 (改善進場價格)
    _limit_order_enabled = cfg.get('enable_limit_order', False)
    _limit_order_premium = cfg.get('limit_order_max_premium', 0.02)  # 開盤價 > 昨收 × (1+此值) → 不買

    # V42: 信號前型態 — 前N天走勢影響信號品質
    _pattern_enabled = cfg.get('enable_entry_pattern', False)
    _pattern_lookback = cfg.get('pattern_lookback', 3)
    _pattern_pullback_bonus = cfg.get('pattern_pullback_bonus', 0.2)   # 前幾天下跌後突破 → 加分
    _pattern_chase_penalty = cfg.get('pattern_chase_penalty', -0.15)   # 前幾天連漲後突破 → 扣分
    _pattern_chase_days = cfg.get('pattern_chase_days', 4)             # 連漲N天才算追高

    # V43: 量能型態 — 成交量趨勢
    _vol_pattern_enabled = cfg.get('enable_vol_pattern', False)
    _vol_pattern_expanding_bonus = cfg.get('vol_expanding_bonus', 0.15)   # 量持續放大 → 加分
    _vol_pattern_spike_penalty = cfg.get('vol_spike_penalty', -0.1)       # 只有今天爆量 → 扣分

    # V39: VIX 風險指標 — VIX > 閾值時限制買入
    _vix_enabled = cfg.get('enable_vix_filter', False)
    _vix_block_threshold = cfg.get('vix_block_threshold', 30)   # VIX > 此值 → 停買
    _vix_caution_threshold = cfg.get('vix_caution_threshold', 25) # VIX > 此值 → 減買
    _vix_caution_max_buy = cfg.get('vix_caution_max_buy', 2)
    _vix_data = {}  # {date_str: vix_close}
    if _vix_enabled:
        try:
            import yfinance as _yf_vix
            _vix_start = (pd.Timestamp(start_date) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
            _vix_end = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
            _old_stderr = sys.stderr; sys.stderr = open(os.devnull, 'w')
            _vix_df = _yf_vix.download('^VIX', start=_vix_start, end=_vix_end, progress=False)
            sys.stderr = _old_stderr
            if isinstance(_vix_df.columns, pd.MultiIndex):
                _vix_df.columns = _vix_df.columns.get_level_values(0)
            for _vix_idx in _vix_df.index:
                _vix_data[_vix_idx.strftime('%Y-%m-%d')] = float(_vix_df.loc[_vix_idx, 'Close'])
            print(f"   📊 V39 VIX: 載入 {len(_vix_data)} 天 VIX 數據")
        except Exception as _e:
            print(f"   ⚠️ V39 VIX: 載入失敗 ({_e})")

    # V40: USD/TWD 匯率 — 台幣急貶時限制買入
    _fx_enabled = cfg.get('enable_fx_filter', False)
    _fx_depreciation_threshold = cfg.get('fx_depreciation_pct', 2.0)  # 20日台幣貶值 > 此% → 限買
    _fx_caution_max_buy = cfg.get('fx_caution_max_buy', 2)
    _fx_data = {}  # {date_str: usd_twd_close}
    if _fx_enabled:
        try:
            import yfinance as _yf_fx
            _fx_start = (pd.Timestamp(start_date) - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
            _fx_end = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
            _old_stderr = sys.stderr; sys.stderr = open(os.devnull, 'w')
            _fx_df = _yf_fx.download('TWD=X', start=_fx_start, end=_fx_end, progress=False)
            sys.stderr = _old_stderr
            if isinstance(_fx_df.columns, pd.MultiIndex):
                _fx_df.columns = _fx_df.columns.get_level_values(0)
            for _fx_idx in _fx_df.index:
                _fx_data[_fx_idx.strftime('%Y-%m-%d')] = float(_fx_df.loc[_fx_idx, 'Close'])
            print(f"   📊 V40 FX: 載入 {len(_fx_data)} 天 USD/TWD 數據")
        except Exception as _e:
            print(f"   ⚠️ V40 FX: 載入失敗 ({_e})")

    # V33: 半導體族群市場廣度 — 族群中 > MA20 的比例作為健康度指標
    _breadth_enabled = cfg.get('enable_breadth_filter', False)
    _breadth_block_threshold = cfg.get('breadth_block_pct', 30)   # 廣度 < 30% → 停買
    _breadth_caution_threshold = cfg.get('breadth_caution_pct', 50) # 廣度 < 50% → 減買
    _breadth_caution_max_buy = cfg.get('breadth_caution_max_buy', 2)

    # V34: 動能加速度 — 5日報酬 vs 20日報酬, 加速中的動能信號更可靠
    _accel_enabled = cfg.get('enable_momentum_accel', False)
    _accel_bonus = cfg.get('accel_bonus', 0.2)       # 加速動能加分
    _accel_penalty = cfg.get('accel_penalty', -0.15)  # 減速動能扣分

    # V31: 多時間框架 — 週線趨勢影響日線行為
    _mtf_enabled = cfg.get('enable_mtf', False)
    _mtf_weekly_block = cfg.get('mtf_weekly_block_buy', True)    # 週線空頭時不開新倉
    _mtf_weekly_tighten = cfg.get('mtf_weekly_tighten', True)    # 週線空頭時收緊出場
    _mtf_tighten_tier_a = cfg.get('mtf_tighten_tier_a', 50)     # 週線空頭時 tier_a 收到此值
    _mtf_tighten_zombie = cfg.get('mtf_tighten_zombie', 7)      # 週線空頭時 zombie 收到此天數

    # V32: 持倉相關性過濾 — 新候選跟現有持倉高度相關則不買
    _corr_filter_enabled = cfg.get('enable_corr_filter', False)
    _corr_filter_lookback = cfg.get('corr_filter_lookback', 20)  # 報酬率相關性計算天數
    _corr_filter_threshold = cfg.get('corr_filter_threshold', 0.7) # 相關性 > 此值 → 不買
    _corr_filter_max_similar = cfg.get('corr_filter_max_similar', 2) # 跟持倉中 N 檔以上高相關 → 不買

    # V30: 因子模型預篩 — 營收動能 + RS 雙因子篩選候選
    _factor_enabled = cfg.get('enable_factor_screen', False)
    _factor_rev_exclude = cfg.get('factor_rev_exclude_months', 2)    # 連續N月營收YoY負 → 排除
    _factor_rev_bonus_threshold = cfg.get('factor_rev_bonus_threshold', 20.0)  # YoY > N% → 加分
    _factor_rev_bonus = cfg.get('factor_rev_bonus', 0.3)
    _factor_rs_exclude_pct = cfg.get('factor_rs_exclude_bottom_pct', 0)  # RS 底部N% → 排除 (0=不排除)
    _factor_rs_bonus_pct = cfg.get('factor_rs_bonus_top_pct', 30)      # RS 頂部N% → 加分
    _factor_rs_bonus = cfg.get('factor_rs_bonus', 0.2)
    _factor_rs_lookback = cfg.get('factor_rs_lookback', 60)
    _factor_rev_data = {}  # {ticker_id: {'Y-M': {'revenue': int, 'yoy': float}}}
    if _factor_enabled:
        _rev_cache = os.path.join('.cache', 'revenue_data.json')
        if os.path.exists(_rev_cache):
            import json
            with open(_rev_cache) as _f:
                _factor_rev_data = json.load(_f)
            print(f"   📊 V30 因子模型: 載入 {len(_factor_rev_data)} 檔營收數據")

    # V29: 子題材持倉集中度限制 — 同一子題材最多持 N 檔
    _theme_max_hold = cfg.get('theme_max_hold', 0)  # 0=不限, N=同題材最多N檔

    # V28: 市場狀態自適應參數 — 不同 regime 用不同 tier/zombie/s2
    _regime_param_enabled = cfg.get('enable_regime_params', False)
    _regime_param_map = {}  # {trend: {config_overrides}}
    if _regime_param_enabled:
        for _rp_regime in ['bull', 'neutral', 'weak', 'bear']:
            _rp_prefix = f'regime_{_rp_regime}_'
            _rp_overrides = {}
            for _rp_key in ['tier_a_net', 'tier_b_net', 'tier_b_drawdown',
                            'zombie_hold_days', 'zombie_net_range',
                            's2_buffer_days', 'hard_stop_net',
                            'zombie_loss_days', 'zombie_profit_days']:
                _rp_val = cfg.get(_rp_prefix + _rp_key)
                if _rp_val is not None:
                    _rp_overrides[_rp_key] = _rp_val
            if _rp_overrides:
                _regime_param_map[_rp_regime] = _rp_overrides

    # V26: 加碼要求獲利 — 持倉必須已獲利才允許加碼 (趨勢確認後才加注)
    _add_require_profit = cfg.get('add_require_profit', False)
    _add_require_profit_pct = cfg.get('add_require_profit_pct', 0.0)  # 淨利>此%才允許加碼

    # V27: 無加碼快砍 — 持倉超過N天仍未被加碼 + 虧損 → 快砍 (方向判斷錯誤)
    _no_add_fast_cut = cfg.get('no_add_fast_cut', False)
    _no_add_fast_cut_days = cfg.get('no_add_fast_cut_days', 5)     # N天內沒加碼
    _no_add_fast_cut_loss = cfg.get('no_add_fast_cut_loss', -3.0)  # 且虧損>此% → 砍

    # V24: 進場確認延遲 — 信號出現後等1天確認 (next_open模式: T+2 執行)
    _entry_confirm_enabled = cfg.get('entry_confirm_delay', False)
    _entry_confirm_pending = {}  # {ticker: {'day_idx': int, 'close': float, 'reason': str}}

    # V25: 換股保護 — 獲利持倉不參與換股
    _swap_protect_profit = cfg.get('swap_protect_profit', False)
    _swap_protect_threshold = cfg.get('swap_protect_threshold', 0.0)  # 淨利 > 此% 的不被 swap

    # V23: 進場品質提升
    #   A1: 量增門檻 — vol > vol_ma5 * min_vol_multiplier 才買 (提高確信度)
    _entry_min_vol_mult = cfg.get('entry_min_vol_multiplier', 1.0)  # 1.0=現有, 1.3=嚴格
    #   A2: 大盤趨勢停手 — TWII MA20 < MA60 時不開新倉
    _entry_twii_trend_gate = cfg.get('entry_twii_trend_gate', False)
    #   C1: 波動率門檻 — TWII 20日實現波動率 > 閾值時不開新倉
    _entry_vol_gate = cfg.get('entry_vol_gate', False)
    _entry_vol_gate_threshold = cfg.get('entry_vol_gate_threshold', 2.0)  # 日波動率%
    _entry_vol_gate_lookback = cfg.get('entry_vol_gate_lookback', 20)

    # V21: 子題材進場加分 — 同題材有 2+ 檔在漲時, 新買入加分 (題材輪動跟風)
    _theme_entry_boost_enabled = cfg.get('enable_theme_entry_boost', False)
    _theme_entry_min_hot = cfg.get('theme_entry_min_hot', 2)
    _theme_entry_bonus = cfg.get('theme_entry_bonus', 0.3)
    _theme_entry_map = {}
    if _theme_entry_boost_enabled or (_val_peer_hold_enabled and _val_peer_use_theme):
        try:
            from theme_config import THEME_MAP as _TM
            for _tn, _tl in _TM.items():
                for _tt in _tl:
                    _theme_entry_map[_tt] = _tn
                    _theme_entry_map[_tt + '.TW'] = _tn
                    _theme_entry_map[_tt + '.TWO'] = _tn
        except ImportError:
            pass

    # V22: 未分類股票自動歸群 — 用報酬率相關性歸入最近的子題材
    _auto_group_enabled = cfg.get('enable_auto_group', False)
    _auto_group_lookback = cfg.get('auto_group_lookback', 60)
    _auto_group_min_corr = cfg.get('auto_group_min_corr', 0.5)

    # V17: 漸進式 DD 預算縮放 — DD 越深, 單筆越小 (漸進減碼, 不停手)
    _dd_budget_enabled = cfg.get('enable_dd_budget_scale', False)
    _dd_budget_lv1_pct = cfg.get('dd_budget_lv1_pct', 10.0)   # DD > 10% → 縮到 lv1_scale
    _dd_budget_lv1_scale = cfg.get('dd_budget_lv1_scale', 0.7)
    _dd_budget_lv2_pct = cfg.get('dd_budget_lv2_pct', 18.0)   # DD > 18% → 縮到 lv2_scale
    _dd_budget_lv2_scale = cfg.get('dd_budget_lv2_scale', 0.45)
    _dd_budget_lv3_pct = cfg.get('dd_budget_lv3_pct', 25.0)   # DD > 25% → 縮到 lv3_scale
    _dd_budget_lv3_scale = cfg.get('dd_budget_lv3_scale', 0.25)

    # V4.21: 波動率倉位控制 (高波動時縮小單筆金額)
    _vol_sizing_enabled = cfg.get('enable_vol_sizing', False)
    _vol_lookback = cfg.get('vol_lookback', 20)
    _vol_target = cfg.get('vol_target_pct', 1.5)
    _vol_floor = cfg.get('vol_scale_floor_pct', 50) / 100  # 轉小數
    _vol_rets = []  # 每日報酬率% (供波動率計算)

    # V6.5: 趨勢預算 (大盤趨勢調整單筆金額, 與 vol_sizing 可疊加)
    _trend_budget_enabled = cfg.get('enable_trend_budget', False)
    _trend_budget_map = {
        'bull':    cfg.get('trend_budget_bull', budget_per_trade),
        'neutral': cfg.get('trend_budget_neutral', budget_per_trade),
        'weak':    cfg.get('trend_budget_weak', budget_per_trade),
        'bear':    cfg.get('trend_budget_bear', budget_per_trade),
    }

    # V6.1: 品質預篩
    _qf_enabled = cfg.get('enable_quality_filter', False)
    _qf_min_trade_ratio = cfg.get('qf_min_trade_ratio', 0.80)
    _qf_vol_stab_min = cfg.get('qf_vol_stability_min', 0.3)
    _qf_vol_stab_max = cfg.get('qf_vol_stability_max', 2.5)
    _qf_min_turnover_20d = cfg.get('qf_min_avg_turnover_20d', 30_000_000)

    # V6.8: 族群相對估值 (Peer Z-score)
    _peer_z_enabled = cfg.get('enable_peer_zscore', False)
    _peer_z_lookback = cfg.get('peer_zscore_lookback', 60)
    _peer_z_expensive = cfg.get('peer_zscore_expensive', 1.5)
    _peer_z_cheap = cfg.get('peer_zscore_cheap', -1.0)
    _peer_z_penalty = cfg.get('peer_zscore_penalty', -0.3)
    _peer_z_bonus = cfg.get('peer_zscore_bonus', 0.15)
    _peer_z_block = cfg.get('peer_zscore_block', 2.5)
    _peer_z_min = cfg.get('peer_min_stocks', 3)

    # V6.8: 預建 ticker → theme 反查 + theme → tickers 正查
    #   stock_data 尚未就緒，延遲到第一個交易日再建立
    _peer_ticker_theme = {}  # ticker (with suffix) → theme_name
    _peer_theme_tickers = {}  # theme_name → [ticker_with_suffix, ...]
    _peer_maps_built = False
    if _peer_z_enabled:
        from theme_config import THEME_MAP as _PEER_THEME_MAP
    else:
        _PEER_THEME_MAP = {}

    # V6.2: 相對強度 (RS)
    _rs_enabled = cfg.get('enable_rs_filter', False)
    _rs_lookback = cfg.get('rs_lookback', 60)
    _rs_cutoff_pct = cfg.get('rs_cutoff_bottom_pct', 30)
    _rs_bonus_top_pct = cfg.get('rs_bonus_top_pct', 20)
    _rs_bonus_score = cfg.get('rs_bonus_score', 0.3)

    # V6.3: 產業動能
    _sm_enabled = cfg.get('enable_sector_momentum', False)
    _sm_lookback = cfg.get('sm_lookback', 20)
    _sm_pos_bonus = cfg.get('sm_positive_bonus', 0.15)
    _sm_neg_penalty = cfg.get('sm_negative_penalty', -0.3)
    _sm_strong_th = cfg.get('sm_strong_threshold', 0.02)
    _sm_strong_bonus = cfg.get('sm_strong_bonus', 0.2)
    _sm_weak_th = cfg.get('sm_weak_threshold', -0.03)
    _sm_weak_penalty = cfg.get('sm_weak_penalty', -0.5)

    # V6.6: EWT Score Boost (連續分數, 非二元)
    _ewt_boost_enabled = cfg.get('enable_ewt_score_boost', False)
    _ewt_boost_strong_up = cfg.get('ewt_boost_strong_up', 0.02)
    _ewt_boost_up = cfg.get('ewt_boost_up', 0.005)
    _ewt_boost_down = cfg.get('ewt_boost_down', -0.01)
    _ewt_boost_strong_down = cfg.get('ewt_boost_strong_down', -0.02)
    _ewt_boost_score_strong_up = cfg.get('ewt_boost_score_strong_up', 0.3)
    _ewt_boost_score_up = cfg.get('ewt_boost_score_up', 0.15)
    _ewt_boost_score_down = cfg.get('ewt_boost_score_down', -0.15)
    _ewt_boost_score_strong_down = cfg.get('ewt_boost_score_strong_down', -0.3)
    _ewt_boost_can_buy_bonus = cfg.get('ewt_boost_can_buy_bonus', 1)
    _ewt_boost_can_buy_penalty = cfg.get('ewt_boost_can_buy_penalty', -1)
    _ewt_boost_adaptive = cfg.get('ewt_boost_market_adaptive', False)

    # V7: NAV-based position sizing
    _budget_pct = cfg.get('budget_pct', 0)

    # V14: 金字塔加碼 — 前幾次小額插旗，確認趨勢後放大
    _pyramid_enabled = cfg.get('enable_pyramid', False)
    _pyramid_ratios = cfg.get('pyramid_ratios', [1.0, 1.0, 1.0, 1.0])  # 相對於 _today_budget 的倍率

    # V8: 子題材動量 Boost (Theme Momentum Boost)
    _theme_enabled = cfg.get('enable_theme_boost', False)
    _theme_boost_max = cfg.get('theme_boost_max', 0.3)
    _theme_lookback = cfg.get('theme_lookback', 20)
    _theme_min_stocks = cfg.get('theme_min_stocks', 3)
    _theme_scores = {}  # 每日更新: {theme_name: float(0~1)}
    # V8.1: 市場自適應 Boost
    _theme_market_adaptive = cfg.get('theme_market_adaptive', False)
    # V8.2: 題材動量方向過濾
    _theme_direction_filter = cfg.get('theme_direction_filter', False)
    _theme_returns = {}  # 每日更新: {theme_name: float (%)} — 方向過濾用
    # V11: 子題材輪動 (Theme Rotation)
    _theme_rotation_enabled = cfg.get('enable_theme_rotation', False) and _theme_enabled
    _theme_rotation_min_score = cfg.get('theme_rotation_min_score', 0.4)
    _theme_rotation_min_themes = cfg.get('theme_rotation_min_themes', 3)
    _theme_rotation_max_themes = cfg.get('theme_rotation_max_themes', 8)
    _theme_rotation_unmapped = cfg.get('theme_rotation_unmapped', 'allow')
    _theme_rotation_unmapped_penalty = cfg.get('theme_rotation_unmapped_penalty', -0.3)
    _allowed_themes = set()  # 每日重建

    # V10: 訊號強度加權倉位 (score-based sizing)
    _score_sizing_enabled = cfg.get('enable_score_sizing', False)
    _score_high_threshold = cfg.get('score_high_threshold', 1.2)   # score >= 此值 → 加大倉位
    _score_low_threshold = cfg.get('score_low_threshold', 0.8)     # score < 此值 → 縮小倉位
    _score_high_ratio = cfg.get('score_high_ratio', 1.3)           # 高分倉位比例 (1.3 = 130%)
    _score_mid_ratio = cfg.get('score_mid_ratio', 1.0)             # 中分倉位比例 (標準)
    _score_low_ratio = cfg.get('score_low_ratio', 0.7)             # 低分倉位比例 (0.7 = 70%)

    # V4.1: 初始資金 (現金帳戶)
    if initial_capital is None:
        if _budget_pct > 0:
            raise ValueError("budget_pct 模式必須指定 initial_capital")
        initial_capital = int(budget_per_trade * max_positions * 1.5)

    if _cash_reserve_override is not None:
        _cash_reserve = _cash_reserve_override  # 驗證模式: 與全程回測一致
    else:
        _cash_reserve = int(initial_capital * _cash_reserve_pct / 100)  # V4.11: 保留金額

    sim_start = pd.Timestamp(start_date)
    sim_end = pd.Timestamp(end_date)
    download_start = (sim_start - pd.Timedelta(days=180)).strftime('%Y-%m-%d')

    # V8: 擴大股票池 (跨產業題材股)
    if _theme_enabled:
        from theme_config import EXTRA_THEME_STOCKS, get_stock_theme, compute_all_theme_scores, compute_theme_returns
        _existing_tickers = {t for t, _ in stock_list}
        _extra_added = 0
        for _et, _en in EXTRA_THEME_STOCKS:
            if _et not in _existing_tickers:
                stock_list = list(stock_list) + [(_et, _en)]
                _existing_tickers.add(_et)
                _extra_added += 1
        if _extra_added > 0:
            print(f"   🏷️ Theme Boost: 加入 {_extra_added} 檔跨產業題材股")

    # =========================================
    # 1. 預先下載所有股票資料 (支援預載入 + 批次下載 + 快取)
    # =========================================
    if preloaded_data is not None:
        # Ablation 共享模式: 直接使用預載入資料，不重複下載
        stock_data = preloaded_data
        # V8: 補下載跨產業題材股 (preloaded_data 可能不含 EXTRA_THEME_STOCKS)
        if _theme_enabled:
            _theme_missing = [(t, n) for t, n in stock_list if t not in stock_data]
            if _theme_missing:
                download_end = (sim_end + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
                _theme_extra, _ = batch_download_stocks(
                    _theme_missing, download_start, download_end,
                    min_data_days=20)
                stock_data.update(_theme_extra)
    else:
        download_end = (sim_end + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        print(f"\n⏳ 下載 {len(stock_list)} 檔股票資料...")
        stock_data, skipped = batch_download_stocks(
            stock_list, download_start, download_end,
            min_data_days=MIN_DATA_DAYS, force_refresh=force_refresh,
        )
        print(f"   ✅ 有效標的: {len(stock_data)} 檔")
        if skipped['download_fail'] > 0:
            print(f"   ⚠️ 下載失敗: {skipped['download_fail']} 檔")
        if skipped['low_data'] > 0:
            print(f"   ⚠️ 資料不足: {skipped['low_data']} 檔")

        # V4.12: 除權息驗證 (資料品質檢查)
        from stock_utils import check_dividend_adjustment, print_dividend_check as _pdc
        _div_check = check_dividend_adjustment(stock_data, sample_n=10)
        _pdc(_div_check)

    if not stock_data:
        print("❌ 無有效標的")
        return None

    # =========================================
    # 2. 建立統一交易日曆 (用 market_map 的交易日為準)
    # =========================================
    # 用 market_map 的日期 (=大盤指數交易日), 避免個股錯誤資料引入假交易日
    market_dates = sorted(pd.Timestamp(d) for d in market_map.keys())
    trading_dates = [d for d in market_dates if sim_start <= d <= sim_end]

    if not trading_dates:
        print("❌ 回測區間無交易日")
        return None

    print(f"   📅 回測交易日: {len(trading_dates)} 天 ({trading_dates[0].strftime('%Y-%m-%d')} ~ {trading_dates[-1].strftime('%Y-%m-%d')})")

    # =========================================
    # 2.5 V4.9: 依 config 門檻重新推導大盤旗標
    #     (允許 ablation 覆蓋 crash/panic/overheat 門檻)
    # =========================================
    _crash_day_th  = cfg.get('crash_day_threshold', -0.025)
    _crash_bias_th = cfg.get('crash_bias_threshold', -0.02)
    _panic_day_th  = cfg.get('panic_day_threshold', -0.03)
    _panic_3d_th   = cfg.get('panic_3d_threshold', -0.035)
    _overheat_th   = cfg.get('overheat_bias_threshold', 0.08)

    # 檢查是否使用非預設門檻 → 需要重新推導
    _need_rederive = (
        _crash_day_th != -0.025 or _crash_bias_th != -0.02 or
        _panic_day_th != -0.03 or _panic_3d_th != -0.035 or
        _overheat_th != 0.08
    )
    if _need_rederive and market_map:
        for date_key, mdata in market_map.items():
            twii = mdata.get('twii', {})
            chg = twii.get('change_pct', 0)
            bias = twii.get('bias_pct', 0)
            cum3 = twii.get('cum_3d', 0)
            price = twii.get('price', 0)
            ma20 = twii.get('ma20', price)
            trend = twii.get('trend', 'neutral')

            is_crash = (chg < _crash_day_th) or (price < ma20 and bias < _crash_bias_th)
            is_unsafe = (trend in ['bear', 'weak']) or is_crash
            is_panic = (chg < _panic_day_th) or (cum3 < _panic_3d_th)
            is_overheated = bias > _overheat_th

            mdata['is_unsafe'] = bool(is_unsafe)
            mdata['is_panic'] = bool(is_panic)
            mdata['is_overheated'] = bool(is_overheated)
            twii['is_crash'] = bool(is_crash)

    # =========================================
    # 3. 持倉管理
    # =========================================
    # positions[ticker] = {
    #   'shares': int, 'cost_total': float, 'avg_cost': float,
    #   'buy_count': int, 'last_buy_date_idx': int, 'name': str,
    #   'reduce_stage': int (0/1/2), 'last_reduce_date_idx': int  # V4.8 減碼追蹤
    # }
    positions = {}

    # V3.0 daily: 注入初始持倉 (從 portfolio.csv 帶入)
    if initial_positions:
        for _ip_ticker, _ip_data in initial_positions.items():
            _ip_cost = _ip_data['avg_cost'] * _ip_data['shares']
            positions[_ip_ticker] = {
                'shares': _ip_data['shares'],
                'cost_total': _ip_data.get('cost_total', _ip_cost),
                'avg_cost': _ip_data['avg_cost'],
                'buy_price': _ip_data.get('buy_price', _ip_data['avg_cost']),  # V6: 最近買入價
                'buy_count': _ip_data.get('buy_count', 1),
                'last_buy_date_idx': _ip_data.get('last_buy_date_idx', -10),
                'name': _ip_data.get('name', _ip_ticker),
                'reduce_stage': _ip_data.get('reduce_stage', 0),
                'last_reduce_date_idx': _ip_data.get('last_reduce_date_idx', -99),
                'peak_since_entry': _ip_data.get('peak_since_entry', _ip_data['avg_cost']),  # V4.20
            }

    # 掛單管理 (T+1)
    # pending[ticker] = {'action': 'buy'/'sell'/'reduce', 'reason': str,
    #                     'reduce_ratio': float, 'reduce_stage': int}  # V4.8
    pending = {}
    # V3.0 daily: 注入初始掛單 (驗證模式: 帶入 split 點的 pending)
    if initial_pending:
        pending = {k: dict(v) for k, v in initial_pending.items()}

    # V4.1: 現金帳戶
    cash = initial_capital
    # V3.0 daily: 初始持倉佔用的資金從 cash 扣除 (用 cost_total 含手續費)
    if initial_positions:
        _init_cost = sum(
            p.get('cost_total', p['avg_cost'] * p['shares'])
            for p in initial_positions.values()
        )
        cash -= _init_cost
    max_overdraft = 0           # 最大透支金額 (cash < 0 時記錄)

    # V4.10: 持倉組合級恐慌追蹤
    _pp_enabled = cfg.get('enable_portfolio_panic', False)  # V4.22 fix: 與 DEFAULT_CONFIG 一致 (預設 False)
    _pp_day_th = cfg.get('portfolio_panic_day_pct', -4.0)
    _pp_3d_th = cfg.get('portfolio_panic_3d_pct', -7.0)
    _pp_action = cfg.get('portfolio_panic_action', 'sell_losers')
    _pp_loss_th = cfg.get('portfolio_panic_loss_threshold', 0.0)
    _pp_cooldown = cfg.get('portfolio_panic_cooldown', 3)
    _pp_last_trigger_idx = -99  # 上次觸發的 day_idx
    _pp_daily_returns = []      # 最近 N 日組合報酬率 (用於3日累計)

    # 績效追蹤
    realized_profit = 0
    total_fees = 0
    trade_count = 0  # 完整來回次數
    win_count = 0
    loss_count = 0
    trade_log = []  # 全部交易紀錄
    new_buy_candidates = []  # V4.16: 最後一天的候選 (供 Mode 5 備選)

    # 回撤追蹤 (V4.15: 改用 NAV-based drawdown, 分母=peak_nav 而非 initial_capital)
    peak_equity = 0
    peak_nav = initial_capital        # V4.15: 追蹤帳戶淨值高水位
    max_drawdown = 0
    max_drawdown_nav = 0              # V4.15: NAV-based 最大回撤金額
    max_dd_pct = 0                    # V4.21: 正確的 MDD% (當日 dd / 當時 peak)
    max_capital_invested = 0
    equity_curve = []
    daily_snapshots = []   # V4.7: 每日快照 (供 CSV 輸出)
    _positions_history = [] if _capture_positions else None  # V3.0 驗證用
    prev_nav = initial_capital  # 前一日帳戶淨值 (算每日報酬率用)

    # V9: 注入引擎狀態 (DailyEngine 共用回測邏輯時使用)
    if initial_state:
        cash = initial_state['cash']
        prev_nav = initial_state['prev_nav']
        peak_nav = initial_state['peak_nav']
        _vol_rets = list(initial_state.get('vol_rets', []))
        _pp_daily_returns = list(initial_state.get('pp_daily_returns', []))
        _pp_last_trigger_idx = initial_state.get('pp_last_trigger_idx', -99)
        realized_profit = initial_state.get('realized_profit', 0)
        total_fees = initial_state.get('total_fees', 0)
        trade_count = initial_state.get('trade_count', 0)
        win_count = initial_state.get('win_count', 0)
        loss_count = initial_state.get('loss_count', 0)

    # V4.30: Forward-fill 診斷計數器 (追蹤 data gap 影響)
    _ffill_total_lookups = 0   # 總查詢次數
    _ffill_gap_count = 0       # forward-fill 次數 (當日無報價, 用前一日)

    # =========================================
    # 4. 主迴圈: 逐日模擬
    # =========================================
    # V22: 自動歸群 — 用報酬率相關性把未分類股票歸入最近的子題材
    if _auto_group_enabled and stock_data and _theme_entry_map:
        from theme_config import THEME_MAP as _ag_TM
        # 建立每個子題材的代表報酬序列 (用主要成分股的平均)
        _ag_theme_rets = {}  # {theme: pd.Series of daily returns}
        for _ag_theme, _ag_tickers in _ag_TM.items():
            _ag_series = []
            for _ag_t in _ag_tickers:
                for _ag_sfx in ['.TW', '.TWO']:
                    _ag_key = _ag_t + _ag_sfx
                    if _ag_key in stock_data and len(stock_data[_ag_key]['df']) >= _auto_group_lookback:
                        _ag_s = stock_data[_ag_key]['df']['Close'].pct_change().dropna().tail(_auto_group_lookback)
                        if len(_ag_s) >= _auto_group_lookback * 0.8:
                            _ag_series.append(_ag_s)
                        break
            if len(_ag_series) >= 2:
                _ag_avg = pd.concat(_ag_series, axis=1).mean(axis=1)
                _ag_theme_rets[_ag_theme] = _ag_avg

        # 對每個未分類股票, 找相關性最高的子題材
        _ag_assigned = 0
        for _ag_ticker in stock_data:
            _ag_tid = _ag_ticker.replace('.TW', '').replace('.TWO', '')
            if _ag_tid in _theme_entry_map or _ag_ticker in _theme_entry_map:
                continue  # 已有分類
            if len(stock_data[_ag_ticker]['df']) < _auto_group_lookback:
                continue
            _ag_stock_ret = stock_data[_ag_ticker]['df']['Close'].pct_change().dropna().tail(_auto_group_lookback)
            if len(_ag_stock_ret) < _auto_group_lookback * 0.8:
                continue
            _ag_best_corr = 0
            _ag_best_theme = None
            for _ag_theme, _ag_tret in _ag_theme_rets.items():
                try:
                    _ag_common = pd.concat([_ag_stock_ret, _ag_tret], axis=1).dropna()
                    if len(_ag_common) >= 20:
                        _ag_corr = _ag_common.iloc[:, 0].corr(_ag_common.iloc[:, 1])
                        if _ag_corr > _ag_best_corr:
                            _ag_best_corr = _ag_corr
                            _ag_best_theme = _ag_theme
                except Exception:
                    pass
            if _ag_best_theme and _ag_best_corr >= _auto_group_min_corr:
                _theme_entry_map[_ag_ticker] = _ag_best_theme
                _theme_entry_map[_ag_tid] = _ag_best_theme
                # 也更新 peer theme map
                if _val_peer_use_theme:
                    _existing_peers = [t + '.TW' for t in _ag_TM.get(_ag_best_theme, [])] + \
                                      [t + '.TWO' for t in _ag_TM.get(_ag_best_theme, [])]
                    _val_peer_theme_map[_ag_ticker] = [p for p in _existing_peers if p != _ag_ticker]
                _ag_assigned += 1
        if _ag_assigned > 0:
            print(f"   📊 V22 自動歸群: {_ag_assigned} 檔未分類股票已歸入子題材 (min_corr={_auto_group_min_corr})")

    print(f"\n⏳ 回測中...")

    for day_idx, curr_date in enumerate(trading_dates, start=initial_day_idx):
        date_str = curr_date.strftime('%Y-%m-%d')

        # V14: 每週買入計數 — 週一重置
        if _weekly_max_buy > 0:
            _cur_week = curr_date.isocalendar()[1]
            if _cur_week != _last_week_num:
                _weekly_buy_count = 0
                _last_week_num = _cur_week

        # V4.7: 今日動作紀錄
        day_actions_buy = []    # "2330(BUY@150.0)" or "2330(ADD@150.0)"
        day_actions_sell = []   # "2330(SELL@180.0,+5.2%)"
        day_actions_swap = []   # "OUT:1234→IN:5678"

        # V8: 每日題材動量計算 (在 scoring 前算好, 整天共用)
        if _theme_enabled:
            _all_price_dfs = {tk: sd['df'] for tk, sd in stock_data.items() if 'df' in sd}
            _pool_tickers = list(_all_price_dfs.keys())
            _theme_scores = compute_all_theme_scores(
                _all_price_dfs, date_str, _pool_tickers,
                lookback=_theme_lookback, min_stocks=_theme_min_stocks,
            )
            # V8.2: 計算題材絕對報酬 (方向過濾用)
            if _theme_direction_filter:
                _theme_returns = compute_theme_returns(
                    _all_price_dfs, date_str,
                    lookback=_theme_lookback, min_stocks=_theme_min_stocks,
                )
            # V11: 每日決定允許的題材 (Theme Rotation)
            if _theme_rotation_enabled and _theme_scores:
                _sorted_themes = sorted(_theme_scores.items(), key=lambda x: -x[1])
                _allowed_themes = set()
                for _rank, (_t_name, _t_score) in enumerate(_sorted_themes):
                    if _rank < _theme_rotation_min_themes:
                        _allowed_themes.add(_t_name)  # floor: 至少允許 top N
                    elif _t_score >= _theme_rotation_min_score and len(_allowed_themes) < _theme_rotation_max_themes:
                        _allowed_themes.add(_t_name)
                    else:
                        break  # 已排序，後面更低
                # Bear/Panic 時關閉 rotation (允許全部)
                if _theme_market_adaptive:
                    _mkt_info = market_map.get(date_str, {})
                    _mkt_panic = _mkt_info.get('panic', False)
                    _mkt_trend = _mkt_info.get('trend', 'neutral')
                    if _mkt_panic or _mkt_trend == 'bear':
                        _allowed_themes = set(_theme_scores.keys())
            elif _theme_rotation_enabled:
                _allowed_themes = set()  # 無 scores 時不過濾

        # V7: NAV-based position sizing + V4.21 波動率倉位控制
        if _budget_pct > 0:
            _base_budget = int(prev_nav * _budget_pct / 100)
        else:
            _base_budget = budget_per_trade
        _today_budget = _base_budget
        if _vol_sizing_enabled and len(_vol_rets) >= _vol_lookback:
            _recent_vol = float(np.std(_vol_rets[-_vol_lookback:]))
            if _recent_vol > _vol_target and _recent_vol > 0:
                _vol_scale = max(_vol_floor, _vol_target / _recent_vol)
                _today_budget = int(_base_budget * _vol_scale)

        # V6.5: 趨勢預算 — 根據大盤趨勢調整今日 budget
        if _trend_budget_enabled:
            _tb_mkt = market_map.get(date_str)
            _tb_trend = _tb_mkt.get('twii', {}).get('trend', 'neutral') if _tb_mkt else 'neutral'
            _today_budget = _trend_budget_map.get(_tb_trend, budget_per_trade)

        # V24: 進場確認延遲 — 檢查昨天暫存的候選, 今日收盤 > 信號日收盤才確認
        if _entry_confirm_enabled and _entry_confirm_pending:
            _confirm_remove = []
            for _cf_ticker, _cf_data in _entry_confirm_pending.items():
                if _cf_data['day_idx'] < day_idx - 1:
                    _confirm_remove.append(_cf_ticker)  # 超過1天沒確認, 取消
                    continue
                if _cf_data['day_idx'] == day_idx - 1:
                    # 檢查今天的收盤是否 > 信號日收盤
                    if _cf_ticker in stock_data and curr_date in stock_data[_cf_ticker]['df'].index:
                        _cf_today_close = float(stock_data[_cf_ticker]['df'].loc[curr_date, 'Close'])
                        if _cf_today_close > _cf_data['close']:
                            pending[_cf_ticker] = _cf_data['order']  # 確認, 加入 pending (T+2 執行)
                    _confirm_remove.append(_cf_ticker)
            for _cf_t in _confirm_remove:
                del _entry_confirm_pending[_cf_t]

        # V17: 漸進式 DD 預算縮放 — DD 越大, 單筆金額越小 (不停買, 只縮量)
        if _dd_budget_enabled and peak_nav > 0 and prev_nav < peak_nav:
            _dd_cur = (peak_nav - prev_nav) / peak_nav * 100
            if _dd_cur >= _dd_budget_lv3_pct:
                _today_budget = int(_today_budget * _dd_budget_lv3_scale)
            elif _dd_cur >= _dd_budget_lv2_pct:
                _today_budget = int(_today_budget * _dd_budget_lv2_scale)
            elif _dd_cur >= _dd_budget_lv1_pct:
                _today_budget = int(_today_budget * _dd_budget_lv1_scale)

        # -----------------------------------------
        # T+1 開盤: 執行昨天的掛單
        # V4.14: 先賣後買 — sell/reduce 先回收現金，再執行 buy
        #        避免換股時 buy 先執行導致 cash 不足
        # -----------------------------------------
        if exec_mode in ('next_open', 'close_open') and pending:
            tickers_to_clear = []
            _limit_up_new_buy_skipped = []  # V4.18: 本日新買入被漲停跳過的 (供遞補)

            # 分兩輪: sell/reduce 先, buy 後
            _sell_orders = [(t, o) for t, o in pending.items() if o['action'] in ('sell', 'reduce')]
            _buy_orders  = [(t, o) for t, o in pending.items() if o['action'] == 'buy']

            for ticker, order in _sell_orders + _buy_orders:
                if ticker not in stock_data:
                    continue  # 缺資料 → 保留 pending, 下個交易日再試
                sdf = stock_data[ticker]['df']
                if curr_date not in sdf.index:
                    continue  # 缺資料 → 保留 pending, 下個交易日再試

                row = sdf.loc[curr_date]
                exec_price = float(row['Open'])
                sname = stock_data[ticker]['name']

                if order['action'] == 'buy':
                    # V4.19: 漲停跳過 — 開盤 OR 收盤 >= 前收 * 1.095 視為漲停買不到
                    #   零股交易撮合不連續, 即使開盤未漲停, 若收盤鎖漲停也搓不到
                    if _skip_limit_up:
                        _prev_days = sdf.index[sdf.index < curr_date]
                        if len(_prev_days) > 0:
                            _prev_close = float(sdf.loc[_prev_days[-1], 'Close'])
                            _today_close = float(sdf.loc[curr_date, 'Close'])
                            if (exec_price >= _prev_close * _limit_up_threshold
                                    or _today_close >= _prev_close * _limit_up_threshold):
                                _limit_up_skipped.append(ticker)
                                # 記錄漲停跳過到 trade_log
                                trade_log.append({
                                    'date': date_str, 'ticker': ticker,
                                    'name': stock_data[ticker].get('name', ticker) if ticker in stock_data else ticker,
                                    'type': 'SKIP_LIMIT_UP',
                                    'price': round(exec_price, 2), 'shares': 0,
                                    'fee': 0, 'profit': None,
                                    'note': (f'⛔ 漲停跳過 (open={exec_price:.1f}, '
                                             f'close={_today_close:.1f}, '
                                             f'prev_close={_prev_close:.1f}, '
                                             f'limit={_prev_close * _limit_up_threshold:.1f})'),
                                })
                                # V4.18: 記錄是否為新買入 (非加碼), 供遞補使用
                                if ticker not in positions:
                                    _limit_up_new_buy_skipped.append(ticker)
                                # V4.20: 漲停遞延 — 保留到隔天重試
                                _retry_count = order.get('_retry_count', 0)
                                if _limit_up_retry and _retry_count < _limit_up_max_retry:
                                    # 不清除 pending, 更新重試計數
                                    order['_retry_count'] = _retry_count + 1
                                    # 不加入 tickers_to_clear → pending 保留到隔天
                                else:
                                    tickers_to_clear.append(ticker)
                                continue

                    _is_add = (ticker in positions)
                    _is_force = order.get('_force_buy', False)  # close_open T+1 追加不受加碼限制

                    # V4.11: 加碼控制 (close_open 強制買不受限)
                    if _is_add and not _enable_add and not _is_force:
                        tickers_to_clear.append(ticker)
                        continue
                    if _is_add and positions[ticker]['buy_count'] >= _max_add_per_stock and not _is_force:
                        tickers_to_clear.append(ticker)
                        continue
                    # V26: 加碼要求獲利 — 持倉未獲利不加碼
                    if _add_require_profit and _is_add and not _is_force:
                        _arp_pos = positions[ticker]
                        _arp_pnl = calculate_net_pnl(ticker, _arp_pos['avg_cost'], exec_price, _arp_pos['shares'])
                        if _arp_pnl['net_pnl_pct'] < _add_require_profit_pct:
                            tickers_to_clear.append(ticker)
                            continue

                    # V41: 限價單模擬 — 開盤價追高太多時不買
                    if _limit_order_enabled and not _is_force:
                        # exec_price = 開盤價, 需要比較跟昨日收盤
                        if ticker in stock_data and curr_date in stock_data[ticker]['df'].index:
                            _lo_df = stock_data[ticker]['df']
                            _lo_idx = _lo_df.index.get_loc(curr_date)
                            if _lo_idx >= 1:
                                _lo_prev_close = float(_lo_df.iloc[_lo_idx - 1]['Close'])
                                if _lo_prev_close > 0 and exec_price > _lo_prev_close * (1 + _limit_order_premium):
                                    tickers_to_clear.append(ticker)
                                    continue  # 開盤追高太多，放棄

                    buy_price = exec_price * (1 + _slippage)  # V4.13 滑價
                    _buy_budget = int(_today_budget * order['_budget_ratio']) if order.get('_budget_ratio') else _today_budget
                    # V14: 金字塔加碼 — 根據已加碼次數決定本次投入倍率
                    if _pyramid_enabled:
                        _pyr_count = positions[ticker]['buy_count'] if _is_add else 0
                        _pyr_ratio = _pyramid_ratios[min(_pyr_count, len(_pyramid_ratios) - 1)]
                        _buy_budget = int(_buy_budget * _pyr_ratio)
                    shares_to_buy = int(_buy_budget / buy_price)  # V4.21: 波動率縮放
                    # V14: 金字塔模式下保證至少買 1 股（避免高價股被跳過）
                    if _pyramid_enabled and shares_to_buy <= 0:
                        shares_to_buy = 1
                    if shares_to_buy <= 0:
                        tickers_to_clear.append(ticker)
                        continue

                    buy_fee = calculate_fee(buy_price, shares_to_buy)
                    cost_this = shares_to_buy * buy_price + buy_fee

                    # V4.11: 現金檢查 (cash - 保留金 >= 買入成本)
                    if _cash_check and (cash - _cash_reserve) < cost_this:
                        tickers_to_clear.append(ticker)
                        continue

                    total_fees += buy_fee
                    cash -= cost_this
                    if cash < 0:
                        max_overdraft = max(max_overdraft, -cash)

                    if _is_add:
                        pos = positions[ticker]
                        old_val = pos['avg_cost'] * pos['shares']
                        pos['shares'] += shares_to_buy
                        pos['avg_cost'] = (old_val + shares_to_buy * buy_price) / pos['shares']
                        pos['cost_total'] += cost_this
                        pos['buy_count'] += 1
                        pos['last_buy_date_idx'] = day_idx
                        pos['buy_price'] = buy_price  # V6: 更新最近一次買入價
                        log_type = 'ADD'
                    else:
                        positions[ticker] = {
                            'shares': shares_to_buy,
                            'cost_total': cost_this,
                            'avg_cost': buy_price,
                            'buy_price': buy_price,         # V6: 最近一次買入價 (含滑價)
                            'buy_count': 1,
                            'last_buy_date_idx': day_idx,
                            'name': sname,
                            'reduce_stage': 0,              # V4.8
                            'last_reduce_date_idx': -99,    # V4.8
                            'peak_since_entry': exec_price, # V4.20: 建倉後最高價追蹤
                        }
                        log_type = 'BUY'
                        # V14: 每週買入計數 (只計新建倉, 不計加碼)
                        if _weekly_max_buy > 0:
                            _weekly_buy_count += 1

                    trade_log.append({
                        'date': date_str, 'ticker': ticker, 'name': sname,
                        'type': log_type, 'price': exec_price,  # trade_log 記錄原始市場價
                        'shares': shares_to_buy, 'fee': buy_fee,
                        'profit': None, 'roi': None,
                        'note': order['reason'],
                        'total_shares': positions[ticker]['shares'],
                    })
                    # V4.7: 記錄買入動作
                    _is_swap_buy = '[換股]' in order.get('reason', '')
                    if _is_swap_buy:
                        day_actions_swap.append(f"IN:{sname}")
                    else:
                        day_actions_buy.append(f"{sname}({log_type} {shares_to_buy}股@{buy_price:.1f})")

                # --- V4.8: 減碼 (部分賣出) ---
                elif order['action'] == 'reduce' and ticker in positions:
                    pos = positions[ticker]
                    reduce_ratio = order.get('reduce_ratio', 0.5)
                    shares_to_sell = int(pos['shares'] * reduce_ratio)
                    if shares_to_sell < 1:
                        shares_to_sell = 1
                    if shares_to_sell >= pos['shares']:
                        shares_to_sell = pos['shares']  # 不夠分就全賣

                    sell_price = exec_price * (1 - _slippage)  # V4.13 滑價
                    sell_fee = calculate_fee(sell_price, shares_to_sell)
                    sell_tax = calculate_tax(ticker, sell_price, shares_to_sell)
                    revenue = shares_to_sell * sell_price - sell_fee - sell_tax
                    cost_of_sold = (shares_to_sell / pos['shares']) * pos['cost_total']
                    profit = revenue - cost_of_sold
                    profit_pct = (profit / cost_of_sold * 100) if cost_of_sold > 0 else 0

                    cash += revenue
                    realized_profit += profit
                    total_fees += sell_fee + sell_tax

                    remaining = pos['shares'] - shares_to_sell
                    trade_log.append({
                        'date': date_str, 'ticker': ticker, 'name': sname,
                        'type': 'REDUCE', 'price': exec_price,  # trade_log 記錄原始市場價
                        'shares': shares_to_sell, 'fee': sell_fee + sell_tax,
                        'profit': profit, 'roi': profit_pct,
                        'note': order['reason'],
                        'total_shares': remaining,
                    })
                    day_actions_sell.append(
                        f"{sname}(減碼{shares_to_sell}股@{exec_price:.1f},{profit_pct:+.1f}%)"
                    )

                    if remaining <= 0:
                        # 全部賣完 (極端情況)
                        trade_count += 1
                        if profit > 0: win_count += 1
                        else: loss_count += 1
                        del positions[ticker]
                    else:
                        pos['shares'] = remaining
                        pos['cost_total'] -= cost_of_sold
                        pos['reduce_stage'] = order.get('reduce_stage', 1)
                        pos['last_reduce_date_idx'] = day_idx

                elif order['action'] == 'sell' and ticker in positions:
                    pos = positions[ticker]
                    sell_price = exec_price * (1 - _slippage)  # V4.13 滑價
                    sell_fee = calculate_fee(sell_price, pos['shares'])
                    sell_tax = calculate_tax(ticker, sell_price, pos['shares'])
                    revenue = pos['shares'] * sell_price - sell_fee - sell_tax
                    profit = revenue - pos['cost_total']
                    profit_pct = (profit / pos['cost_total'] * 100) if pos['cost_total'] > 0 else 0

                    cash += revenue
                    realized_profit += profit
                    total_fees += sell_fee + sell_tax
                    trade_count += 1
                    if profit > 0:
                        win_count += 1
                    else:
                        loss_count += 1

                    trade_log.append({
                        'date': date_str, 'ticker': ticker, 'name': sname,
                        'type': 'SELL', 'price': exec_price,  # trade_log 記錄原始市場價
                        'shares': pos['shares'], 'fee': sell_fee + sell_tax,
                        'profit': profit, 'roi': profit_pct,
                        'note': order['reason'],
                        'total_shares': 0,
                    })
                    # V4.7: 記錄賣出/換股動作
                    _is_swap_sell = '換股淘汰' in order.get('reason', '')
                    if _is_swap_sell:
                        day_actions_swap.append(f"OUT:{sname}({profit_pct:+.1f}%)")
                    else:
                        day_actions_sell.append(f"{sname}(SELL@{exec_price:.1f},{profit_pct:+.1f}%)")
                    del positions[ticker]

                tickers_to_clear.append(ticker)

            for t in tickers_to_clear:
                pending.pop(t, None)

            # -----------------------------------------
            # V4.18: 漲停遞補 — 新買入被漲停跳過 → 從備選佇列遞補
            #   - 只遞補新買入 (加碼漲停 → 暫緩, 不遞補)
            #   - 遞補候選也要檢查: 有股價資料 + 非漲停 + 現金足夠
            # -----------------------------------------
            if _enable_backup_fill and _limit_up_new_buy_skipped and _backup_queue:
                _n_to_fill = len(_limit_up_new_buy_skipped)
                _filled_count = 0
                _backup_filled = []  # 記錄遞補成功的 (供 log)
                for _bk_cand in _backup_queue:
                    if _filled_count >= _n_to_fill:
                        break
                    _bk_ticker = _bk_cand['ticker']
                    # 已持有 or 已有掛單 → 跳過
                    if _bk_ticker in positions or _bk_ticker in pending:
                        continue
                    # 無股價資料 → 跳過
                    if _bk_ticker not in stock_data:
                        continue
                    _bk_sdf = stock_data[_bk_ticker]['df']
                    if curr_date not in _bk_sdf.index:
                        continue
                    _bk_open = float(_bk_sdf.loc[curr_date, 'Open'])
                    _bk_name = stock_data[_bk_ticker]['name']
                    # V4.19: 遞補候選本身也漲停 → 跳過 (開盤 OR 收盤)
                    _bk_prev_days = _bk_sdf.index[_bk_sdf.index < curr_date]
                    if len(_bk_prev_days) > 0:
                        _bk_prev_close = float(_bk_sdf.loc[_bk_prev_days[-1], 'Close'])
                        _bk_today_close = float(_bk_sdf.loc[curr_date, 'Close'])
                        if (_bk_open >= _bk_prev_close * _limit_up_threshold
                                or _bk_today_close >= _bk_prev_close * _limit_up_threshold):
                            continue
                    # 計算成本
                    _bk_buy_price = _bk_open * (1 + _slippage)
                    _bk_shares = int(_today_budget / _bk_buy_price)  # V4.21: 波動率縮放
                    if _bk_shares <= 0:
                        continue
                    _bk_fee = calculate_fee(_bk_buy_price, _bk_shares)
                    _bk_cost = _bk_shares * _bk_buy_price + _bk_fee
                    # 現金檢查
                    if _cash_check and (cash - _cash_reserve) < _bk_cost:
                        continue
                    # ✅ 遞補買入
                    total_fees += _bk_fee
                    cash -= _bk_cost
                    if cash < 0:
                        max_overdraft = max(max_overdraft, -cash)
                    positions[_bk_ticker] = {
                        'shares': _bk_shares,
                        'cost_total': _bk_cost,
                        'avg_cost': _bk_buy_price,
                        'buy_price': _bk_buy_price,    # V6: 最近一次買入價 (含滑價)
                        'buy_count': 1,
                        'last_buy_date_idx': day_idx,
                        'name': _bk_name,
                        'reduce_stage': 0,
                        'last_reduce_date_idx': -99,
                        'peak_since_entry': _bk_open,  # V4.22 fix: 與正常 BUY 一致
                    }
                    trade_log.append({
                        'date': date_str, 'ticker': _bk_ticker, 'name': _bk_name,
                        'type': 'BUY', 'price': _bk_open,
                        'shares': _bk_shares, 'fee': _bk_fee,
                        'profit': None, 'roi': None,
                        'note': f'🔄 漲停遞補 (原 {_limit_up_new_buy_skipped[_filled_count]} 漲停)',
                        'total_shares': _bk_shares,
                    })
                    day_actions_buy.append(f"{_bk_name}(遞補BUY {_bk_shares}股@{_bk_buy_price:.1f})")
                    _backup_filled.append(_bk_ticker)
                    _filled_count += 1

        # -----------------------------------------
        # 盤後分析: 對每檔股票產生訊號
        # -----------------------------------------
        current_market = market_map.get(date_str, {
            'is_unsafe': False, 'is_overheated': False, 'is_panic': False
        })

        # --- 收集新建倉候選 (排序後再決定誰上車) ---
        new_buy_candidates = []
        add_candidates = []  # V8: 加碼候選 (排序後選取 top N)
        new_buys_today = 0

        for ticker, sdata in stock_data.items():
            sdf = sdata['df']
            if curr_date not in sdf.index:
                continue

            # 已有掛單 → 跳過
            if ticker in pending:
                continue

            row = sdf.loc[curr_date]
            close_price = float(row['Close'])

            history_df = sdf.loc[:curr_date]
            if len(history_df) < MIN_DATA_DAYS + 1:
                continue

            info = build_info_dict(history_df)
            if info is None:
                continue

            is_held = ticker in positions

            if is_held:
                # --- Daily Scanner 邏輯: 帶庫存檢查 ---
                # ✅ Bug 3 修正: 已持有的股票不再過量能/價格門檻
                pos = positions[ticker]
                # V4.20: 每日更新建倉後最高價
                _cur_peak = pos.get('peak_since_entry', pos['avg_cost'])
                pos['peak_since_entry'] = max(_cur_peak, close_price)
                _ticker_cfg = _get_ticker_config(ticker, industry_map, per_industry_config, config_override)
                # V4.21: 動態停損 — 大盤弱勢時收緊 hard_stop_net
                if _dyn_stop_enabled:
                    _today_trend_s = current_market.get('twii', {}).get('trend', 'neutral')
                    if _today_trend_s in _hard_stop_override:
                        _ticker_cfg = dict(_ticker_cfg) if _ticker_cfg else {}
                        _ticker_cfg['hard_stop_net'] = _hard_stop_override[_today_trend_s]
                # V37: 階梯式 zombie — 加碼越多, zombie 天數越長
                if _zombie_ladder:
                    _zl_bc = pos.get('buy_count', 1)
                    _zl_extra = min((_zl_bc - 1) * _zombie_ladder_extra, _zombie_ladder_cap - zombie_days)
                    if _zl_extra > 0:
                        _ticker_cfg = dict(_ticker_cfg) if _ticker_cfg else {}
                        _ticker_cfg['zombie_hold_days'] = zombie_days + _zl_extra

                # V38: 贏家升級 — 獲利+高加碼 → 切換長線參數
                if _winner_upgrade:
                    _wu_bc = pos.get('buy_count', 1)
                    _wu_pnl = calculate_net_pnl(ticker, pos['avg_cost'], close_price, pos['shares'])
                    if _wu_pnl['net_pnl_pct'] > _winner_min_profit and _wu_bc >= _winner_min_buys:
                        _ticker_cfg = dict(_ticker_cfg) if _ticker_cfg else {}
                        _ticker_cfg['tier_a_net'] = max(_ticker_cfg.get('tier_a_net', 80), _winner_tier_a)
                        _ticker_cfg['zombie_hold_days'] = max(_ticker_cfg.get('zombie_hold_days', 10), _winner_zombie_days)
                        _ticker_cfg['zombie_net_range'] = _winner_zombie_range

                # V35: 外資連續賣超 → 收緊 tier_b
                if _fi_enabled and _fi_data:
                    _fi_htid = ticker.replace('.TW', '').replace('.TWO', '')
                    _fi_hstock = _fi_data.get(_fi_htid)
                    if _fi_hstock:
                        _fi_h_consec_sell = 0
                        for _fi_h_off in range(1, _fi_consec_sell_tighten + 2):
                            _fi_h_date = (curr_date - pd.Timedelta(days=_fi_h_off)).strftime('%Y-%m-%d')
                            _fi_h_net = _fi_hstock.get(_fi_h_date)
                            if _fi_h_net is not None and _fi_h_net < 0:
                                _fi_h_consec_sell += 1
                            else:
                                break
                        if _fi_h_consec_sell >= _fi_consec_sell_tighten:
                            _ticker_cfg = dict(_ticker_cfg) if _ticker_cfg else {}
                            _ticker_cfg['tier_b_drawdown'] = min(
                                _ticker_cfg.get('tier_b_drawdown', 0.6), _fi_tighten_dd)

                # V31: 多時間框架 — 週線空頭時收緊持倉出場
                if _mtf_enabled and _mtf_weekly_tighten:
                    _mtf_wk = current_market.get('weekly_bullish', True)
                    if not _mtf_wk:
                        _ticker_cfg = dict(_ticker_cfg) if _ticker_cfg else {}
                        _ticker_cfg['tier_a_net'] = min(_ticker_cfg.get('tier_a_net', 80), _mtf_tighten_tier_a)
                        _ticker_cfg['zombie_hold_days'] = min(_ticker_cfg.get('zombie_hold_days', 10), _mtf_tighten_zombie)

                # V28: 市場狀態自適應參數 — 不同 regime 用不同 tier/zombie/s2
                if _regime_param_enabled:
                    _rp_trend = current_market.get('twii', {}).get('trend', 'neutral')
                    _rp_override = _regime_param_map.get(_rp_trend)
                    if _rp_override:
                        _ticker_cfg = dict(_ticker_cfg) if _ticker_cfg else {}
                        _ticker_cfg.update(_rp_override)

                # V20: 營收動能 — 連續營收衰退時收緊 tier_b
                if _val_rev_enabled and _val_rev_data:
                    _vr_id = ticker.replace('.TW', '').replace('.TWO', '')
                    _vr_stock = _val_rev_data.get(_vr_id)
                    if _vr_stock:
                        # 找當前月份往前 N 個月的 YoY
                        _vr_cm = curr_date.month
                        _vr_cy = curr_date.year
                        _vr_consecutive_decline = 0
                        for _vr_offset in range(1, _val_rev_decline_months + 2):
                            _vr_m = _vr_cm - _vr_offset
                            _vr_y = _vr_cy
                            if _vr_m <= 0:
                                _vr_m += 12; _vr_y -= 1
                            _vr_yoy = _vr_stock.get((_vr_y, _vr_m))
                            if _vr_yoy is not None and _vr_yoy < 0:
                                _vr_consecutive_decline += 1
                            else:
                                break
                        if _vr_consecutive_decline >= _val_rev_decline_months:
                            _ticker_cfg = dict(_ticker_cfg) if _ticker_cfg else {}
                            _ticker_cfg['tier_b_drawdown'] = min(
                                _ticker_cfg.get('tier_b_drawdown', 0.6),
                                _val_rev_tighten_dd)

                # V18: 乖離率分層出場 — 股價離 MA60 越遠, tier_b 越緊 (貴了快走)
                if _val_bias_enabled and info.get('ma60', 0) > 0:
                    _vb_ratio = close_price / info['ma60']
                    _ticker_cfg = dict(_ticker_cfg) if _ticker_cfg else {}
                    if _vb_ratio >= _val_bias_lv3_ratio:
                        _ticker_cfg['tier_b_drawdown'] = _val_bias_lv3_dd
                    elif _vb_ratio >= _val_bias_lv2_ratio:
                        _ticker_cfg['tier_b_drawdown'] = _val_bias_lv2_dd
                    elif _vb_ratio <= _val_bias_cheap_ratio:
                        _ticker_cfg['tier_b_drawdown'] = _val_bias_cheap_dd

                # V19: Peer RS 持倉調整 — 族群內漲太多的收緊, 偏弱的快砍
                if _val_peer_hold_enabled and ticker in stock_data:
                    _vp_sdf = stock_data[ticker]['df']
                    if curr_date in _vp_sdf.index:
                        _vp_idx = _vp_sdf.index.get_loc(curr_date)
                        if _vp_idx >= _val_peer_lookback:
                            _vp_now = float(_vp_sdf.iloc[_vp_idx]['Close'])
                            _vp_past = float(_vp_sdf.iloc[_vp_idx - _val_peer_lookback]['Close'])
                            if _vp_past > 0:
                                _vp_ret = _vp_now / _vp_past - 1
                                # 計算族群 RS 分布
                                # V19b: 用 THEME_MAP 分群 or 全產業
                                _vp_all = []
                                if _val_peer_use_theme and ticker in _val_peer_theme_map:
                                    _vp_peers = _val_peer_theme_map[ticker]
                                    for _vp_t in _vp_peers:
                                        if _vp_t in stock_data:
                                            _vp_tdf = stock_data[_vp_t]['df']
                                            if curr_date in _vp_tdf.index:
                                                _vp_ti = _vp_tdf.index.get_loc(curr_date)
                                                if _vp_ti >= _val_peer_lookback:
                                                    _vp_tp = float(_vp_tdf.iloc[_vp_ti]['Close'])
                                                    _vp_tpp = float(_vp_tdf.iloc[_vp_ti - _val_peer_lookback]['Close'])
                                                    if _vp_tpp > 0:
                                                        _vp_all.append(_vp_tp / _vp_tpp - 1)
                                else:
                                    for _vp_t, _vp_sd in stock_data.items():
                                        _vp_tdf = _vp_sd['df']
                                        if curr_date in _vp_tdf.index:
                                            _vp_ti = _vp_tdf.index.get_loc(curr_date)
                                            if _vp_ti >= _val_peer_lookback:
                                                _vp_tp = float(_vp_tdf.iloc[_vp_ti]['Close'])
                                                _vp_tpp = float(_vp_tdf.iloc[_vp_ti - _val_peer_lookback]['Close'])
                                                if _vp_tpp > 0:
                                                    _vp_all.append(_vp_tp / _vp_tpp - 1)
                                _vp_min_peers = 3 if _val_peer_use_theme else 5
                                if len(_vp_all) >= _vp_min_peers:
                                    _vp_mean = np.mean(_vp_all)
                                    _vp_std = np.std(_vp_all)
                                    if _vp_std > 0:
                                        _vp_z = (_vp_ret - _vp_mean) / _vp_std
                                        _ticker_cfg = dict(_ticker_cfg) if _ticker_cfg else {}
                                        if _vp_z >= _val_peer_expensive_z:
                                            # 族群內最貴 → 收緊 tier_b
                                            _ticker_cfg['tier_b_drawdown'] = min(
                                                _ticker_cfg.get('tier_b_drawdown', 0.6),
                                                _val_peer_expensive_dd)
                                        elif _vp_z <= _val_peer_weak_z:
                                            # 族群內最弱 → 收緊 zombie
                                            _cur_zd = _ticker_cfg.get('zombie_hold_days', zombie_days)
                                            _ticker_cfg['zombie_hold_days'] = max(5, _cur_zd - _val_peer_weak_zombie_cut)

                # V6.9: 信念持股 — 放寬 stop-loss 和 tier_a
                if _conviction_enabled and pos.get('buy_count', 1) >= _conviction_min_buys:
                    _ticker_cfg = dict(_ticker_cfg) if _ticker_cfg else {}
                    _cur_stop = _ticker_cfg.get('hard_stop_net', cfg.get('hard_stop_net', -15))
                    _ticker_cfg['hard_stop_net'] = _cur_stop - _conviction_stop_extra
                    _cur_tier_a = _ticker_cfg.get('tier_a_net', cfg.get('tier_a_net', 45))
                    _ticker_cfg['tier_a_net'] = _cur_tier_a + _conviction_tier_a_extra

                # V6.10: Regime-Adaptive — unsafe 時收緊 tier_b
                if _regime_adaptive and current_market.get('is_unsafe', False):
                    _ticker_cfg = dict(_ticker_cfg) if _ticker_cfg else {}
                    _ticker_cfg['tier_b_net'] = _regime_unsafe_tier_b

                _sig_fn = signal_func or check_strategy_signal
                signal = _sig_fn(
                    ticker, info, pos['avg_cost'], pos['shares'],
                    market_status=current_market,
                    history_df=history_df,
                    config=_ticker_cfg,
                    reduce_stage=pos.get('reduce_stage', 0),            # V4.8
                    last_reduce_day_idx=pos.get('last_reduce_date_idx', -99),  # V4.8
                    current_day_idx=day_idx,                            # V4.8
                    peak_price_since_entry=pos.get('peak_since_entry'), # V4.20
                )
                action = signal['action']
                reason = signal['reason']

                # 最小持有天數 (sell 和 reduce 都受限)
                if action in ('sell', 'reduce'):
                    days_held = day_idx - pos['last_buy_date_idx']
                    if days_held < min_hold_days:
                        continue

                # --- S4: 殭屍倉位清除 (V3.9 + V13 RS自適應) ---
                # 策略說 hold 但持有太久且不賺不賠 → 強制出場騰空間
                # V13: RS 強勢股延長等待天數 + 放寬淨利範圍
                if action == 'hold' and enable_zombie:
                    days_held = day_idx - pos['last_buy_date_idx']
                    _z_days = zombie_days
                    _z_range = zombie_range
                    _z_rs_tag = ''

                    # V6.9: 信念持股 — 加碼多次的股票放寬 zombie
                    if _conviction_enabled and pos.get('buy_count', 1) >= _conviction_min_buys:
                        _z_days += _conviction_zombie_extra
                        _z_range += _conviction_zombie_range
                        _z_rs_tag += f' [信念持股🔥 buy×{pos["buy_count"]}]'

                    # V6.10: Regime-Adaptive — unsafe 時加速 zombie
                    if _regime_adaptive and current_market.get('is_unsafe', False):
                        _z_days = min(_z_days, _regime_unsafe_zombie)

                    # V13: 計算個股 RS，判斷是否為慢爬強勢股
                    if _zombie_rs_adaptive and ticker in stock_data:
                        _z_sdf = stock_data[ticker]['df']
                        if curr_date in _z_sdf.index:
                            _z_idx = _z_sdf.index.get_loc(curr_date)
                            if _z_idx >= _zombie_rs_lookback:
                                _z_price_now = float(_z_sdf.iloc[_z_idx]['Close'])
                                _z_price_past = float(_z_sdf.iloc[_z_idx - _zombie_rs_lookback]['Close'])
                                if _z_price_past > 0:
                                    _z_rs = (_z_price_now / _z_price_past - 1)
                                    # 計算全場 RS 分位數 (用已持有+候選的所有股票)
                                    _z_all_rs = []
                                    for _z_t, _z_sd in stock_data.items():
                                        _z_tdf = _z_sd['df']
                                        if curr_date in _z_tdf.index:
                                            _z_ti = _z_tdf.index.get_loc(curr_date)
                                            if _z_ti >= _zombie_rs_lookback:
                                                _z_tp = float(_z_tdf.iloc[_z_ti]['Close'])
                                                _z_tpp = float(_z_tdf.iloc[_z_ti - _zombie_rs_lookback]['Close'])
                                                if _z_tpp > 0:
                                                    _z_all_rs.append(_z_tp / _z_tpp - 1)
                                    if _z_all_rs:
                                        _z_cutoff = sorted(_z_all_rs)[max(0, int(len(_z_all_rs) * (100 - _zombie_rs_top_pct) / 100))]
                                        if _z_rs >= _z_cutoff:
                                            _z_days = zombie_days + _zombie_rs_extra_days
                                            _z_range = zombie_range + _zombie_rs_extra_range
                                            _z_rs_tag = f' [RS強勢🦾 延長至{_z_days}天/±{_z_range}%]'

                    if days_held >= _z_days:
                        pnl_detail = calculate_net_pnl(ticker, pos['avg_cost'], close_price, pos['shares'])
                        z_net_pct = pnl_detail['net_pnl_pct']
                        if -_z_range <= z_net_pct <= _z_range:
                            action = 'sell'
                            reason = (f'🧟 殭屍清除 (持有{days_held}天, '
                                      f'淨利{z_net_pct:+.1f}%在±{_z_range}%內){_z_rs_tag}')

                    # V15: 非對稱殭屍 — 虧損股和獲利股分開計算天數
                    #   zombie_loss_days: 虧損持倉超過N天 → 強制砍 (快砍輸家)
                    #   zombie_profit_days: 獲利但不多的持倉超過N天且獲利<X% → 砍 (效率清除)
                    #   設 zombie_loss_days=0 表示不啟用
                    if action == 'hold' and _z_asym_enabled:
                        pnl_detail = calculate_net_pnl(ticker, pos['avg_cost'], close_price, pos['shares'])
                        z_net_pct = pnl_detail['net_pnl_pct']
                        if z_net_pct < 0 and days_held >= _z_asym_loss_days:
                            action = 'sell'
                            reason = (f'🧟 虧損殭屍 (持有{days_held}天≥{_z_asym_loss_days}, '
                                      f'淨利{z_net_pct:+.1f}%)')
                        elif 0 <= z_net_pct < _z_asym_profit_min and days_held >= _z_asym_profit_days:
                            action = 'sell'
                            reason = (f'🧟 效率殭屍 (持有{days_held}天≥{_z_asym_profit_days}, '
                                      f'淨利{z_net_pct:+.1f}%<{_z_asym_profit_min}%)')

                # V16b: 趨勢保護型 zombie (原版, 預設關閉)
                if _zombie_trend_protect and action == 'sell' and '殭屍' in reason:
                    _tp_ma20 = info.get('ma20', 0)
                    _tp_ma60 = info.get('ma60', 0)
                    if (_tp_ma20 > 0 and _tp_ma60 > 0 and
                            _tp_ma20 > _tp_ma60 and close_price > _tp_ma20):
                        action = 'hold'
                        reason = ''

                # V36: 趨勢保護 zombie（改良版）— 獲利+加碼+趨勢 → 不砍
                if _zombie_trend_protect_v2 and action == 'sell' and '殭屍' in reason:
                    _ztp2_pnl = calculate_net_pnl(ticker, pos['avg_cost'], close_price, pos['shares'])
                    _ztp2_net = _ztp2_pnl['net_pnl_pct']
                    _ztp2_bc = pos.get('buy_count', 1)
                    _ztp2_ma20 = info.get('ma20', 0)
                    _ztp2_ma60 = info.get('ma60', 0)
                    if (_ztp2_net > _ztp2_min_profit and          # 獲利中
                            _ztp2_bc >= _ztp2_min_buys and        # 有加碼確認
                            _ztp2_ma20 > 0 and _ztp2_ma60 > 0 and
                            close_price > _ztp2_ma20 and          # 短期趨勢上行
                            _ztp2_ma20 > _ztp2_ma60):             # 中期趨勢上行
                        action = 'hold'
                        reason = ''

                # V27: 無加碼快砍 — 持倉N天未被加碼 + 虧損 → 快砍
                if _no_add_fast_cut and action == 'hold':
                    _nafc_days = day_idx - pos['last_buy_date_idx']
                    _nafc_adds = pos.get('buy_count', 1) - 1  # buy_count=1是初始買入
                    if _nafc_adds == 0 and _nafc_days >= _no_add_fast_cut_days:
                        _nafc_pnl = calculate_net_pnl(ticker, pos['avg_cost'], close_price, pos['shares'])
                        if _nafc_pnl['net_pnl_pct'] <= _no_add_fast_cut_loss:
                            action = 'sell'
                            reason = (f'⚡ 無加碼快砍 ({_nafc_days}天未加碼, '
                                      f'淨利{_nafc_pnl["net_pnl_pct"]:+.1f}%≤{_no_add_fast_cut_loss}%)')

                # --- V6.5 S4: 營收衰退退出 ---
                # 連續2月營收年減 + 股價跌破 MA60 → 強制退場
                if action == 'hold' and _revenue_blacklist:
                    _rev_ym = (curr_date.year, curr_date.month)
                    _rev_bl = _revenue_blacklist.get(_rev_ym)
                    if _rev_bl is None:
                        _rev_prev_m = curr_date.month - 1 if curr_date.month > 1 else 12
                        _rev_prev_y = curr_date.year if curr_date.month > 1 else curr_date.year - 1
                        _rev_bl = _revenue_blacklist.get((_rev_prev_y, _rev_prev_m))
                    if _rev_bl is not None and ticker in _rev_bl:
                        # 額外條件: 股價在 MA60 以下才觸發 (雙確認)
                        _ma60_val = info.get('ma60', 0)
                        if _ma60_val > 0 and close_price < _ma60_val:
                            action = 'sell'
                            reason = f'📉 S4營收衰退 (連2月年減+低於MA60)'

                # --- V8: 加碼候選收集 (排序後選取 top N) ---
                if action == 'buy':
                    if not _enable_add:
                        continue
                    if pos['buy_count'] >= _max_add_per_stock:
                        continue
                    _add_bias = (close_price - info['ma20']) / info['ma20'] * 100 if info.get('ma20', 0) > 0 else 99
                    _add_vol_r = info['volume'] / info['vol_ma5'] if info.get('vol_ma5', 0) > 0 else 0
                    _add_net_pnl = (close_price - pos['avg_cost']) / pos['avg_cost'] * 100 if pos['avg_cost'] > 0 else 0
                    _add_score = (min(_add_vol_r, 3.0) * 0.4                   # 量比 (動能確認, 40%)
                                  + min(max(_add_net_pnl, -5), 10) / 10 * 0.3  # 帳面報酬 (追強不追弱, 30%)
                                  - max(0, _add_bias - 5) / 15 * 0.3)          # 乖離懲罰 (避免追高, 30%)
                    add_candidates.append({
                        'ticker': ticker, 'name': sdata['name'],
                        'close_price': close_price, 'reason': reason,
                        'bias_pct': _add_bias, 'vol_ratio': _add_vol_r,
                        'net_pnl': _add_net_pnl, 'score': _add_score,
                    })
                    continue

                if exec_mode in ('next_open', 'close_open'):
                    if action == 'sell':
                        pending[ticker] = {'action': 'sell', 'reason': reason}
                    elif action == 'reduce':
                        pending[ticker] = {
                            'action': 'reduce', 'reason': reason,
                            'reduce_ratio': signal.get('reduce_ratio', 0.5),
                            'reduce_stage': signal.get('reduce_stage', 1),
                        }
                    # V8: elif action == 'buy' 已移至上方 add_candidates 收集
                elif exec_mode == 'same_close':
                    if action == 'sell':
                        pos = positions[ticker]
                        sell_price_sc = close_price * (1 - _slippage)  # V4.13 滑價
                        sell_fee = calculate_fee(sell_price_sc, pos['shares'])
                        sell_tax = calculate_tax(ticker, sell_price_sc, pos['shares'])
                        revenue = pos['shares'] * sell_price_sc - sell_fee - sell_tax
                        profit = revenue - pos['cost_total']
                        profit_pct = (profit / pos['cost_total'] * 100) if pos['cost_total'] > 0 else 0
                        cash += revenue
                        realized_profit += profit
                        total_fees += sell_fee + sell_tax
                        trade_count += 1
                        if profit > 0: win_count += 1
                        else: loss_count += 1
                        trade_log.append({
                            'date': date_str, 'ticker': ticker, 'name': sdata['name'],
                            'type': 'SELL', 'price': close_price,
                            'shares': pos['shares'], 'fee': sell_fee + sell_tax,
                            'profit': profit, 'roi': profit_pct,
                            'note': reason, 'total_shares': 0,
                        })
                        day_actions_sell.append(f"{sdata['name']}(SELL@{close_price:.1f},{profit_pct:+.1f}%)")
                        del positions[ticker]
                    elif action == 'reduce':
                        # V4.8: same_close 減碼
                        pos = positions[ticker]
                        reduce_ratio = signal.get('reduce_ratio', 0.5)
                        shares_to_sell = int(pos['shares'] * reduce_ratio)
                        if shares_to_sell < 1:
                            shares_to_sell = 1
                        if shares_to_sell >= pos['shares']:
                            shares_to_sell = pos['shares']

                        sell_price_sc = close_price * (1 - _slippage)  # V4.13 滑價
                        sell_fee = calculate_fee(sell_price_sc, shares_to_sell)
                        sell_tax = calculate_tax(ticker, sell_price_sc, shares_to_sell)
                        revenue = shares_to_sell * sell_price_sc - sell_fee - sell_tax
                        cost_of_sold = (shares_to_sell / pos['shares']) * pos['cost_total']
                        profit = revenue - cost_of_sold
                        profit_pct = (profit / cost_of_sold * 100) if cost_of_sold > 0 else 0

                        cash += revenue
                        realized_profit += profit
                        total_fees += sell_fee + sell_tax

                        remaining = pos['shares'] - shares_to_sell
                        trade_log.append({
                            'date': date_str, 'ticker': ticker, 'name': sdata['name'],
                            'type': 'REDUCE', 'price': close_price,  # trade_log 記錄原始市場價
                            'shares': shares_to_sell, 'fee': sell_fee + sell_tax,
                            'profit': profit, 'roi': profit_pct,
                            'note': reason, 'total_shares': remaining,
                        })
                        day_actions_sell.append(
                            f"{sdata['name']}(減碼{shares_to_sell}股@{close_price:.1f},{profit_pct:+.1f}%)"
                        )

                        if remaining <= 0:
                            trade_count += 1
                            if profit > 0: win_count += 1
                            else: loss_count += 1
                            del positions[ticker]
                        else:
                            pos['shares'] = remaining
                            pos['cost_total'] -= cost_of_sold
                            pos['reduce_stage'] = signal.get('reduce_stage', 1)
                            pos['last_reduce_date_idx'] = day_idx
                    # V8: elif action == 'buy' 已移至上方 add_candidates 收集

            else:
                # --- Group Scanner 邏輯: 純買進條件 (傳 0, 0) ---
                # ✅ 篩選門檻只套用在新建倉 (與 group_scanner 一致)

                # V6.5: 營收白名單 pre-filter (只限制新建倉, 不影響已持有)
                if _revenue_whitelist:
                    _rev_ym = (curr_date.year, curr_date.month)
                    _rev_wl = _revenue_whitelist.get(_rev_ym)
                    if _rev_wl is None:
                        # 當月還沒更新, 用上月
                        _rev_prev_m = curr_date.month - 1 if curr_date.month > 1 else 12
                        _rev_prev_y = curr_date.year if curr_date.month > 1 else curr_date.year - 1
                        _rev_wl = _revenue_whitelist.get((_rev_prev_y, _rev_prev_m))
                    if _rev_wl is not None and ticker not in _rev_wl:
                        continue  # 營收不合格, 跳過新建倉

                if close_price < MIN_PRICE:
                    continue
                vol_ma5 = float(history_df['Volume'].tail(5).mean())
                if vol_ma5 < MIN_VOLUME_SHARES:
                    continue
                turnover_ma5 = float((history_df['Close'].tail(5) * history_df['Volume'].tail(5)).mean())
                if turnover_ma5 < MIN_TURNOVER:
                    continue

                _ticker_cfg = _get_ticker_config(ticker, industry_map, per_industry_config, config_override)
                _sig_fn2 = signal_func or check_strategy_signal
                signal = _sig_fn2(
                    ticker, info, 0, 0,
                    market_status=current_market,
                    history_df=history_df,
                    config=_ticker_cfg,
                )

                # --- V12: B8/B9 拉回買入 / 急跌掃貨 (B1-B7 未觸發時的備選) ---
                if signal['action'] != 'buy' and (cfg.get('enable_pullback_buy') or cfg.get('enable_dip_buy')):
                    _pb_result = _evaluate_pullback_buy(
                        close_price, info.get('ma20', 0), info.get('ma60', 0),
                        info.get('volume', 0), info.get('vol_ma5', 0),
                        info.get('prev_close', 0),
                        current_market, cfg, history_df)
                    if _pb_result['passed']:
                        signal = {
                            'action': 'buy', 'reason': _pb_result['tag'],
                            'consecutive': 0, 'is_fish_tail': False,
                            'reduce_ratio': 0, 'reduce_stage': 0,
                        }

                if signal['action'] == 'buy':
                    # --- V6.1: 品質預篩 (Quality Pre-filter) ---
                    if _qf_enabled:
                        _qf_pass = True
                        # 流動性: 過去60天有交易天數 / 60 >= 門檻
                        _qf_trade_days = len(history_df.tail(60).dropna(subset=['Close']))
                        if _qf_trade_days / 60.0 < _qf_min_trade_ratio:
                            _qf_pass = False
                        # 量能穩定性: 20日量 std/mean
                        if _qf_pass:
                            _qf_vol_20 = history_df['Volume'].tail(20)
                            _qf_vol_mean = float(_qf_vol_20.mean())
                            _qf_vol_std = float(_qf_vol_20.std())
                            if _qf_vol_mean > 0:
                                _qf_cv = _qf_vol_std / _qf_vol_mean
                                if _qf_cv > _qf_vol_stab_max:
                                    _qf_pass = False
                        # 20日均成交額門檻
                        if _qf_pass:
                            _qf_turnover_20 = float((history_df['Close'].tail(20) * history_df['Volume'].tail(20)).mean())
                            if _qf_turnover_20 < _qf_min_turnover_20d:
                                _qf_pass = False
                        if not _qf_pass:
                            continue  # 品質不過關, 跳過

                    # V11: Theme Rotation — 封鎖冷門題材
                    if _theme_rotation_enabled and _allowed_themes:
                        _cand_theme_check = get_stock_theme(ticker)
                        if _cand_theme_check is not None:
                            if _cand_theme_check not in _allowed_themes:
                                continue  # 冷門題材, 跳過
                        else:
                            if _theme_rotation_unmapped == 'block':
                                continue  # 未分類股票封鎖

                    # ✅ 收集候選 + 計算品質分數
                    bias_pct = (close_price - info['ma20']) / info['ma20'] * 100 if info['ma20'] > 0 else 99
                    vol_ratio = info['volume'] / info['vol_ma5'] if info['vol_ma5'] > 0 else 0
                    pct_change = (close_price - info['prev_close']) / info['prev_close'] * 100 if info['prev_close'] > 0 else 0

                    # V3.9 品質分數: 量價齊揚得高分, 乖離太高扣分
                    score = (min(vol_ratio, 3.0) * 0.4          # 量比 (cap 3x, 權重 40%)
                             + min(pct_change, 5.0) / 5 * 0.3   # 漲幅 (cap 5%, 權重 30%)
                             - max(0, bias_pct - 5) / 15 * 0.3) # 乖離懲罰 (>5%開始扣, 權重 30%)

                    # --- V8: 子題材動量 Boost ---
                    _cand_theme = None
                    _cand_theme_score = 0.0
                    _cand_theme_boost = 0.0
                    if _theme_enabled and _theme_scores:
                        # V8.1: 市場自適應 — 依大盤狀態調節 boost 倍率
                        _adaptive_mult = 1.0
                        if _theme_market_adaptive:
                            _mkt_trend = current_market.get('twii', {}).get('trend', 'neutral')
                            _mkt_panic = current_market.get('is_panic', False)
                            if _mkt_panic or _mkt_trend == 'bear':
                                _adaptive_mult = 0.0   # 空頭/恐慌 → 關閉
                            elif _mkt_trend == 'bull':
                                _adaptive_mult = 0.5   # 純多頭 → 減半
                            elif _mkt_trend == 'weak':
                                _adaptive_mult = 1.0   # 偏弱分化 → 全開
                            else:  # neutral
                                _adaptive_mult = 1.0   # 中性分化 → 全開

                        _cand_theme = get_stock_theme(ticker)
                        if _cand_theme and _cand_theme in _theme_scores:
                            _cand_theme_score = _theme_scores[_cand_theme]

                            # V8.2: 方向過濾 — 題材 20 日報酬 < 0 則不 boost
                            if _theme_direction_filter:
                                _t_ret = _theme_returns.get(_cand_theme, 0.0)
                                if _t_ret < 0:
                                    _cand_theme_score = 0.0  # 題材在跌，不追

                            _cand_theme_boost = _cand_theme_score * _theme_boost_max * _adaptive_mult
                            score += _cand_theme_boost

                        # V11: 未分類股票扣分
                        if _theme_rotation_enabled and _cand_theme is None and _theme_rotation_unmapped == 'penalty':
                            score += _theme_rotation_unmapped_penalty

                    # --- V42: 信號前型態 — 前N天走勢影響信號品質 ---
                    if _pattern_enabled and history_df is not None and len(history_df) >= _pattern_chase_days + 2:
                        _pt_closes = history_df['Close'].iloc[-(1 + _pattern_chase_days):-1]  # 不含今天
                        if len(_pt_closes) >= _pattern_lookback:
                            # 前 lookback 天全部下跌 → 回檔突破，加分
                            _pt_recent = _pt_closes.tail(_pattern_lookback)
                            _pt_down_days = sum(1 for i in range(1, len(_pt_recent))
                                                if _pt_recent.iloc[i] < _pt_recent.iloc[i-1])
                            if _pt_down_days >= _pattern_lookback - 1:
                                score += _pattern_pullback_bonus  # 前幾天下跌後突破

                            # 前 chase_days 天全部上漲 → 追高突破，扣分
                            _pt_chase = _pt_closes.tail(_pattern_chase_days)
                            _pt_up_days = sum(1 for i in range(1, len(_pt_chase))
                                              if _pt_chase.iloc[i] > _pt_chase.iloc[i-1])
                            if _pt_up_days >= _pattern_chase_days - 1:
                                score += _pattern_chase_penalty  # 連漲後追高

                    # --- V43: 量能型態 — 成交量趨勢 ---
                    if _vol_pattern_enabled and history_df is not None and len(history_df) >= 6:
                        _vp_vols = history_df['Volume'].iloc[-5:].values  # 最近5天量（含今天）
                        if len(_vp_vols) == 5 and all(v > 0 for v in _vp_vols):
                            # 量持續放大：每天量 > 前天量（至少3/4天）
                            _vp_expanding = sum(1 for i in range(1, 5) if _vp_vols[i] > _vp_vols[i-1])
                            if _vp_expanding >= 3:
                                score += _vol_pattern_expanding_bonus

                            # 只有今天爆量：今天量 > 前4天平均的2倍，但前4天量沒有趨勢
                            _vp_prev_avg = np.mean(_vp_vols[:4])
                            _vp_prev_expanding = sum(1 for i in range(1, 4) if _vp_vols[i] > _vp_vols[i-1])
                            if _vp_vols[4] > _vp_prev_avg * 2 and _vp_prev_expanding <= 1:
                                score += _vol_pattern_spike_penalty  # 一日爆量

                    # --- V35: 外資買賣超 — 連續買超加分 ---
                    if _fi_enabled and _fi_data:
                        _fi_tid = ticker.replace('.TW', '').replace('.TWO', '')
                        _fi_stock = _fi_data.get(_fi_tid)
                        if _fi_stock:
                            _fi_consec = 0
                            for _fi_off in range(1, _fi_consec_buy_bonus + 2):
                                _fi_check_date = (curr_date - pd.Timedelta(days=_fi_off)).strftime('%Y-%m-%d')
                                # 找最近的交易日
                                _fi_net = _fi_stock.get(_fi_check_date)
                                if _fi_net is not None and _fi_net > 0:
                                    _fi_consec += 1
                                else:
                                    break
                            if _fi_consec >= _fi_consec_buy_bonus:
                                score += _fi_bonus

                    # --- V34: 動能加速度 — 5日報酬 vs 20日報酬 ---
                    if _accel_enabled and history_df is not None and len(history_df) >= 21:
                        _ac_close = history_df['Close']
                        _ac_5d = float(_ac_close.iloc[-1] / _ac_close.iloc[-6] - 1) if len(_ac_close) >= 6 else 0
                        _ac_20d = float(_ac_close.iloc[-1] / _ac_close.iloc[-21] - 1) if len(_ac_close) >= 21 else 0
                        if _ac_20d != 0:
                            if _ac_5d > _ac_20d and _ac_5d > 0:
                                score += _accel_bonus    # 加速中
                            elif _ac_5d < _ac_20d * 0.5 and _ac_20d > 0:
                                score += _accel_penalty  # 明顯減速

                    # --- V30: 因子模型預篩 ---
                    if _factor_enabled:
                        _f_tid = ticker.replace('.TW', '').replace('.TWO', '')
                        _f_skip = False

                        # 營收因子: 排除連續衰退 + 成長加分
                        if _factor_rev_data and _f_tid in _factor_rev_data:
                            _f_rev = _factor_rev_data[_f_tid]
                            _f_cm, _f_cy = curr_date.month, curr_date.year
                            # 檢查連續衰退 (往前看N個月)
                            _f_decline = 0
                            for _f_off in range(1, _factor_rev_exclude + 2):
                                _f_m = _f_cm - _f_off
                                _f_y = _f_cy
                                if _f_m <= 0:
                                    _f_m += 12; _f_y -= 1
                                _f_key = f'{_f_y}-{_f_m}'
                                _f_entry = _f_rev.get(_f_key)
                                if _f_entry and _f_entry.get('yoy') is not None and _f_entry['yoy'] < 0:
                                    _f_decline += 1
                                else:
                                    break
                            if _f_decline >= _factor_rev_exclude:
                                _f_skip = True  # 連續營收衰退 → 排除

                            # 成長加分
                            if not _f_skip and _factor_rev_bonus > 0:
                                _f_last_key = f'{_f_cy}-{_f_cm - 1}' if _f_cm > 1 else f'{_f_cy-1}-12'
                                _f_last = _f_rev.get(_f_last_key)
                                if _f_last and _f_last.get('yoy') is not None:
                                    if _f_last['yoy'] > _factor_rev_bonus_threshold:
                                        score += _factor_rev_bonus

                        if _f_skip:
                            continue  # 排除此候選

                        # RS 因子: 排除底部 + 頂部加分
                        if (_factor_rs_exclude_pct > 0 or _factor_rs_bonus_pct > 0) and ticker in stock_data:
                            _f_sdf = stock_data[ticker]['df']
                            if curr_date in _f_sdf.index:
                                _f_idx = _f_sdf.index.get_loc(curr_date)
                                if _f_idx >= _factor_rs_lookback:
                                    _f_now = float(_f_sdf.iloc[_f_idx]['Close'])
                                    _f_past = float(_f_sdf.iloc[_f_idx - _factor_rs_lookback]['Close'])
                                    if _f_past > 0:
                                        _f_rs = _f_now / _f_past - 1
                                        # 計算全場 RS 分位
                                        _f_all_rs = []
                                        for _f_t, _f_sd in stock_data.items():
                                            _f_tdf = _f_sd['df']
                                            if curr_date in _f_tdf.index:
                                                _f_ti = _f_tdf.index.get_loc(curr_date)
                                                if _f_ti >= _factor_rs_lookback:
                                                    _f_tp = float(_f_tdf.iloc[_f_ti]['Close'])
                                                    _f_tpp = float(_f_tdf.iloc[_f_ti - _factor_rs_lookback]['Close'])
                                                    if _f_tpp > 0:
                                                        _f_all_rs.append(_f_tp / _f_tpp - 1)
                                        if _f_all_rs:
                                            _f_sorted = sorted(_f_all_rs)
                                            _f_rank_pct = sum(1 for x in _f_sorted if x <= _f_rs) / len(_f_sorted) * 100
                                            if _factor_rs_exclude_pct > 0 and _f_rank_pct < _factor_rs_exclude_pct:
                                                continue  # RS 底部 → 排除
                                            if _factor_rs_bonus_pct > 0 and _f_rank_pct >= (100 - _factor_rs_bonus_pct):
                                                score += _factor_rs_bonus

                    # --- V21: 子題材進場加分 — 同題材有持倉時加分 ---
                    if _theme_entry_boost_enabled:
                        _te_theme = _theme_entry_map.get(ticker)
                        if _te_theme:
                            # 計算同題材目前持倉數
                            _te_count = 0
                            for _te_t in positions:
                                if _theme_entry_map.get(_te_t) == _te_theme:
                                    _te_count += 1
                            if _te_count >= _theme_entry_min_hot:
                                score += _theme_entry_bonus

                    # --- V6.2: 相對強度 (RS) 計算 ---
                    _cand_rs_return = None
                    if _rs_enabled and len(history_df) > _rs_lookback:
                        _rs_close_now = close_price
                        _rs_close_past = float(history_df['Close'].iloc[-_rs_lookback - 1])
                        if _rs_close_past > 0:
                            _cand_rs_return = (_rs_close_now / _rs_close_past - 1)

                    new_buy_candidates.append({
                        'ticker': ticker, 'name': sdata['name'],
                        'close_price': close_price, 'reason': signal['reason'],
                        'bias_pct': bias_pct, 'vol_ratio': vol_ratio,
                        'pct_change': pct_change, 'score': score,
                        'consecutive': signal.get('consecutive', 0),
                        'rs_return': _cand_rs_return,  # V6.2: 供RS排序用
                        'theme': _cand_theme,           # V8: 所屬子題材
                        'theme_score': _cand_theme_score, # V8: 題材熱度 (0~1)
                        'theme_boost': _cand_theme_boost, # V8: 實際加分
                    })

        # -----------------------------------------
        # V8: 加碼排序 + 每日上限 (top N)
        # -----------------------------------------
        if add_candidates:
            add_candidates.sort(key=lambda x: (-x['score'], x['ticker']))
            _add_selected = add_candidates[:_max_add_per_day]

            for ac in _add_selected:
                _ac_ticker = ac['ticker']
                if _ac_ticker not in positions:
                    continue
                if _ac_ticker in pending:
                    continue
                _ac_pos = positions[_ac_ticker]
                _ac_close = ac['close_price']
                _ac_reason = ac['reason']

                if exec_mode in ('next_open', 'close_open'):
                    if exec_mode == 'close_open':
                        # V7: 收盤先加碼半額 + T+1再加碼半額
                        _co_budget = _today_budget // 2
                        _ac_buy_price = _ac_close * (1 + _slippage)
                        _ac_shares = int(_co_budget / _ac_buy_price)
                        if _ac_shares > 0:
                            _ac_fee = calculate_fee(_ac_buy_price, _ac_shares)
                            _ac_cost = _ac_shares * _ac_buy_price + _ac_fee
                            if _cash_check and (cash - _cash_reserve) < _ac_cost:
                                continue
                            total_fees += _ac_fee
                            cash -= _ac_cost
                            if cash < 0:
                                max_overdraft = max(max_overdraft, -cash)
                            _ac_old_val = _ac_pos['avg_cost'] * _ac_pos['shares']
                            _ac_pos['shares'] += _ac_shares
                            _ac_pos['avg_cost'] = (_ac_old_val + _ac_shares * _ac_buy_price) / _ac_pos['shares']
                            _ac_pos['cost_total'] += _ac_cost
                            _ac_pos['buy_count'] += 1
                            _ac_pos['last_buy_date_idx'] = day_idx
                            _ac_pos['buy_price'] = _ac_buy_price
                            trade_log.append({
                                'date': date_str, 'ticker': _ac_ticker, 'name': ac['name'],
                                'type': 'ADD', 'price': _ac_close,
                                'shares': _ac_shares, 'fee': _ac_fee,
                                'profit': None, 'roi': None,
                                'note': _ac_reason + ' [收盤加碼]', 'total_shares': _ac_pos['shares'],
                            })
                            day_actions_buy.append(f"{ac['name']}(ADD {_ac_shares}股@{_ac_buy_price:.1f} 收盤)")
                            pending[_ac_ticker] = {'action': 'buy', 'reason': _ac_reason + ' [T+1追加]', '_force_buy': True, '_budget_ratio': 0.5}
                    else:
                        _ac_pending = {'action': 'buy', 'reason': _ac_reason}
                        # V10: 加碼也依訊號強度調整倉位
                        if _score_sizing_enabled:
                            _ac_s = ac.get('score', 0)
                            if _ac_s >= _score_high_threshold:
                                _ac_pending['_budget_ratio'] = _score_high_ratio
                            elif _ac_s < _score_low_threshold:
                                _ac_pending['_budget_ratio'] = _score_low_ratio
                            else:
                                _ac_pending['_budget_ratio'] = _score_mid_ratio
                        pending[_ac_ticker] = _ac_pending
                elif exec_mode == 'same_close':
                    _ac_buy_price = _ac_close * (1 + _slippage)
                    _ac_shares = int(_today_budget / _ac_buy_price)
                    if _ac_shares > 0:
                        _ac_fee = calculate_fee(_ac_buy_price, _ac_shares)
                        _ac_cost = _ac_shares * _ac_buy_price + _ac_fee
                        if _cash_check and (cash - _cash_reserve) < _ac_cost:
                            continue
                        total_fees += _ac_fee
                        cash -= _ac_cost
                        if cash < 0:
                            max_overdraft = max(max_overdraft, -cash)
                        _ac_old_val = _ac_pos['avg_cost'] * _ac_pos['shares']
                        _ac_pos['shares'] += _ac_shares
                        _ac_pos['avg_cost'] = (_ac_old_val + _ac_shares * _ac_buy_price) / _ac_pos['shares']
                        _ac_pos['cost_total'] += _ac_cost
                        _ac_pos['buy_count'] += 1
                        _ac_pos['last_buy_date_idx'] = day_idx
                        _ac_pos['buy_price'] = _ac_buy_price
                        trade_log.append({
                            'date': date_str, 'ticker': _ac_ticker, 'name': ac['name'],
                            'type': 'ADD', 'price': _ac_close,
                            'shares': _ac_shares, 'fee': _ac_fee,
                            'profit': None, 'roi': None,
                            'note': _ac_reason, 'total_shares': _ac_pos['shares'],
                        })
                        day_actions_buy.append(f"{ac['name']}(ADD {_ac_shares}股@{_ac_buy_price:.1f})")

        # -----------------------------------------
        # V6.2: RS 過濾 — 過濾最弱 N%, 加分最強 N%
        # -----------------------------------------
        if _rs_enabled and new_buy_candidates:
            _rs_valid = [c for c in new_buy_candidates if c.get('rs_return') is not None]
            if len(_rs_valid) >= 3:  # 至少3個候選才做排序
                _rs_returns = sorted([c['rs_return'] for c in _rs_valid])
                _rs_cutoff_val = _rs_returns[max(0, int(len(_rs_returns) * _rs_cutoff_pct / 100) - 1)]
                _rs_bonus_val = _rs_returns[min(len(_rs_returns) - 1, int(len(_rs_returns) * (100 - _rs_bonus_top_pct) / 100))]

                _rs_filtered = []
                for c in new_buy_candidates:
                    _cr = c.get('rs_return')
                    if _cr is not None and _cr <= _rs_cutoff_val:
                        continue  # 過濾掉底部
                    if _cr is not None and _cr >= _rs_bonus_val:
                        c['score'] += _rs_bonus_score  # 頂部加分
                    _rs_filtered.append(c)
                new_buy_candidates = _rs_filtered

        # -----------------------------------------
        # V6.3: 產業動能 — 計算產業整體動能, 調整候選分數
        # -----------------------------------------
        if _sm_enabled and new_buy_candidates:
            # 計算所有有資料的股票的近N日平均報酬
            _sm_returns = []
            for _sm_ticker, _sm_sdata in stock_data.items():
                _sm_sdf = _sm_sdata['df']
                if curr_date in _sm_sdf.index and len(_sm_sdf.loc[:curr_date]) > _sm_lookback:
                    _sm_idx = _sm_sdf.index.get_loc(curr_date)
                    if _sm_idx >= _sm_lookback:
                        _sm_c_now = float(_sm_sdf.iloc[_sm_idx]['Close'])
                        _sm_c_past = float(_sm_sdf.iloc[_sm_idx - _sm_lookback]['Close'])
                        if _sm_c_past > 0:
                            _sm_returns.append(_sm_c_now / _sm_c_past - 1)

            if _sm_returns:
                _sm_avg = float(np.mean(_sm_returns))  # 產業平均動能

                for c in new_buy_candidates:
                    if _sm_avg > _sm_strong_th:
                        c['score'] += _sm_strong_bonus  # 產業強勁
                    elif _sm_avg > 0:
                        c['score'] += _sm_pos_bonus     # 產業正向
                    elif _sm_avg < _sm_weak_th:
                        c['score'] += _sm_weak_penalty   # 產業疲弱
                    elif _sm_avg < 0:
                        c['score'] += _sm_neg_penalty    # 產業負向

        # -----------------------------------------
        # V6.8: 族群相對估值 (Peer Z-score)
        #   每日計算各族群內所有股票的 N日報酬 → Z-score
        #   偏貴 (z>1.5) 扣分, 偏宜 (z<-1.0) 加分, 極貴 (z>2.5) 直接擋
        # -----------------------------------------
        if _peer_z_enabled and new_buy_candidates:
            # 延遲建立 peer maps (需要 stock_data 已就緒)
            if not _peer_maps_built:
                for _pt_name, _pt_codes in _PEER_THEME_MAP.items():
                    _pt_tks = []
                    for _pt_code in _pt_codes:
                        for _pt_sfx in ['.TW', '.TWO']:
                            _pt_tk = _pt_code + _pt_sfx
                            if _pt_tk in stock_data:
                                _peer_ticker_theme[_pt_tk] = _pt_name
                                _pt_tks.append(_pt_tk)
                                break
                    _peer_theme_tickers[_pt_name] = _pt_tks
                _peer_maps_built = True

            # 計算每個族群的 N日報酬分布
            _peer_theme_rets = {}  # theme → {ticker: ret}
            for _pz_theme, _pz_tickers in _peer_theme_tickers.items():
                _pz_rets = {}
                for _pz_tk in _pz_tickers:
                    if _pz_tk not in stock_data:
                        continue
                    _pz_df = stock_data[_pz_tk]['df']
                    if curr_date not in _pz_df.index:
                        continue
                    _pz_idx = _pz_df.index.get_loc(curr_date)
                    if _pz_idx >= _peer_z_lookback:
                        _pz_now = float(_pz_df.iloc[_pz_idx]['Close'])
                        _pz_past = float(_pz_df.iloc[_pz_idx - _peer_z_lookback]['Close'])
                        if _pz_past > 0:
                            _pz_rets[_pz_tk] = _pz_now / _pz_past - 1
                if len(_pz_rets) >= _peer_z_min:
                    _peer_theme_rets[_pz_theme] = _pz_rets

            # 對每個候選計算 Z-score
            _pz_filtered = []
            for c in new_buy_candidates:
                _pz_tk = c['ticker']
                _pz_theme = _peer_ticker_theme.get(_pz_tk)
                if _pz_theme and _pz_theme in _peer_theme_rets:
                    _pz_group = _peer_theme_rets[_pz_theme]
                    if _pz_tk in _pz_group:
                        _pz_vals = list(_pz_group.values())
                        _pz_mean = float(np.mean(_pz_vals))
                        _pz_std = float(np.std(_pz_vals))
                        if _pz_std > 0:
                            _pz_z = (_pz_group[_pz_tk] - _pz_mean) / _pz_std
                            if _pz_z > _peer_z_block:
                                continue  # 極端偏貴 → 不買
                            elif _pz_z > _peer_z_expensive:
                                c['score'] += _peer_z_penalty
                            elif _pz_z < _peer_z_cheap:
                                c['score'] += _peer_z_bonus
                _pz_filtered.append(c)
            new_buy_candidates = _pz_filtered

        # -----------------------------------------
        # V6.6: EWT Score Boost — 用隔夜 EWT 漲跌調整候選分數
        #   market_map[T日].ewt = US T-1 收盤 (= 台灣 T日凌晨已知)
        #   對 T日產生的 candidates 做分數加減
        #   market_adaptive: 空頭/恐慌時自動降低或關閉 boost
        # -----------------------------------------
        _ewt_boost_mult = 1.0  # adaptive 倍率 (也供 can_buy 使用)
        if _ewt_boost_enabled and new_buy_candidates:
            # 市場自適應: 根據大盤狀態調節 boost 倍率
            if _ewt_boost_adaptive:
                _ewt_mkt_trend = current_market.get('twii', {}).get('trend', 'neutral')
                _ewt_mkt_panic = current_market.get('is_panic', False)
                if _ewt_mkt_panic or _ewt_mkt_trend == 'bear':
                    _ewt_boost_mult = 0.0   # 空頭/恐慌 → 完全關閉
                elif _ewt_mkt_trend == 'weak':
                    _ewt_boost_mult = 0.5   # 偏弱 → 減半
                # bull/neutral → 1.0 (全開)

            _ewt_data = current_market.get('ewt')
            if _ewt_data is not None and _ewt_boost_mult > 0:
                _ewt_chg = _ewt_data.get('daily_chg', 0)
                _ewt_score_adj = 0
                if _ewt_chg >= _ewt_boost_strong_up:
                    _ewt_score_adj = _ewt_boost_score_strong_up
                elif _ewt_chg >= _ewt_boost_up:
                    _ewt_score_adj = _ewt_boost_score_up
                elif _ewt_chg <= _ewt_boost_strong_down:
                    _ewt_score_adj = _ewt_boost_score_strong_down
                elif _ewt_chg <= _ewt_boost_down:
                    _ewt_score_adj = _ewt_boost_score_down

                _ewt_score_adj *= _ewt_boost_mult  # 套用 adaptive 倍率

                if _ewt_score_adj != 0:
                    for c in new_buy_candidates:
                        c['score'] += _ewt_score_adj

        # -----------------------------------------
        # V4.21: 動態曝險管理 — 大盤弱勢時降低持倉上限
        #   - 根據 TWII trend + is_panic 決定今日最大持倉數
        #   - 超額持倉 → 強制賣出最弱持倉 (hold_score 最低)
        #   - exec_mode=next_open: 加入 pending 隔日賣出
        #   - exec_mode=same_close: 立即賣出
        # -----------------------------------------
        _effective_max_pos = max_positions  # 預設用靜態值
        if _dyn_enabled:
            _today_panic = current_market.get('is_panic', False)
            _today_trend = current_market.get('twii', {}).get('trend', 'neutral')

            if _today_panic:
                _effective_max_pos = _dyn_max_panic
            else:
                _effective_max_pos = _dyn_max_map.get(_today_trend, max_positions)

            # 計算有效持倉 (扣除已有 pending sell/reduce 的)
            _pending_sell_tickers = {t for t, o in pending.items() if o['action'] in ('sell', 'reduce')}
            _effective_hold = len(positions) - len(_pending_sell_tickers)
            _excess = _effective_hold - _effective_max_pos

            if _excess > 0:
                # 計算所有持倉的 hold_score (排除已有 pending 的)
                _dyn_scored = []
                for _dt, _dp in positions.items():
                    if _dt in pending:
                        continue
                    if _dt not in stock_data or curr_date not in stock_data[_dt]['df'].index:
                        continue
                    _d_cp = float(stock_data[_dt]['df'].loc[curr_date, 'Close'])
                    _d_pnl = calculate_net_pnl(_dt, _dp['avg_cost'], _d_cp, _dp['shares'])
                    _d_net_pct = _d_pnl['net_pnl_pct']
                    _d_days = day_idx - _dp['last_buy_date_idx']
                    _d_hold_score = _d_net_pct / 100 - max(0, _d_days - 10) * 0.01
                    _dyn_scored.append((_dt, _d_hold_score, _dp.get('name', _dt), _d_cp))

                _dyn_scored.sort(key=lambda x: x[1])  # 最差在前

                _dyn_sell_count = 0
                for _d_ticker, _d_score, _d_name, _d_close in _dyn_scored:
                    if _dyn_sell_count >= _excess:
                        break
                    _dyn_reason = (f'📉 動態曝險 ({_today_trend}'
                                   f'{"🚨恐慌" if _today_panic else ""}'
                                   f'→上限{_effective_max_pos}, '
                                   f'超額{_excess}檔, 價值{_d_score:.2f})')

                    if exec_mode in ('next_open', 'close_open'):
                        pending[_d_ticker] = {'action': 'sell', 'reason': _dyn_reason}
                    elif exec_mode == 'same_close':
                        _d_pos = positions[_d_ticker]
                        _d_sell_price = _d_close * (1 - _slippage)
                        _d_sell_fee = calculate_fee(_d_sell_price, _d_pos['shares'])
                        _d_sell_tax = calculate_tax(_d_ticker, _d_sell_price, _d_pos['shares'])
                        _d_revenue = _d_pos['shares'] * _d_sell_price - _d_sell_fee - _d_sell_tax
                        _d_profit = _d_revenue - _d_pos['cost_total']
                        _d_profit_pct = (_d_profit / _d_pos['cost_total'] * 100) if _d_pos['cost_total'] > 0 else 0
                        cash += _d_revenue
                        realized_profit += _d_profit
                        total_fees += _d_sell_fee + _d_sell_tax
                        trade_count += 1
                        if _d_profit > 0:
                            win_count += 1
                        else:
                            loss_count += 1
                        trade_log.append({
                            'date': date_str, 'ticker': _d_ticker, 'name': _d_name,
                            'type': 'SELL', 'price': _d_close,
                            'shares': _d_pos['shares'], 'fee': _d_sell_fee + _d_sell_tax,
                            'profit': _d_profit, 'roi': _d_profit_pct,
                            'note': _dyn_reason, 'total_shares': 0,
                        })
                        day_actions_sell.append(f"{_d_name}(曝險SELL@{_d_close:.1f},{_d_profit_pct:+.1f}%)")
                        del positions[_d_ticker]

                    _dyn_sell_count += 1
                    _dyn_force_sell_count += 1

        # --- V12: N 日新高過濾 (只買突破期間內最高價的股票) ---
        if _require_period_high and new_buy_candidates:
            _high_filtered = []
            for c in new_buy_candidates:
                _h_ticker = c['ticker']
                if _h_ticker not in stock_data or curr_date not in stock_data[_h_ticker]['df'].index:
                    continue
                _h_df = stock_data[_h_ticker]['df']
                _h_idx = _h_df.index.get_loc(curr_date)
                if _h_idx < _period_high_lookback:
                    _high_filtered.append(c)  # 資料不足，不過濾
                    continue
                _h_window = _h_df.iloc[_h_idx - _period_high_lookback:_h_idx]
                _h_prev_high = float(_h_window['High'].max())
                if c['close_price'] >= _h_prev_high:
                    _high_filtered.append(c)
                # else: 過濾掉 — 沒有突破期間新高
            new_buy_candidates = _high_filtered

        # --- V39: VIX 過濾 — 恐慌指數高時限制買入 ---
        if _vix_enabled and new_buy_candidates and _vix_data:
            # VIX 用前一天的值（美國收盤 = 台灣開盤前已知）
            _vix_yesterday = None
            for _vix_off in range(1, 5):
                _vix_check = (curr_date - pd.Timedelta(days=_vix_off)).strftime('%Y-%m-%d')
                if _vix_check in _vix_data:
                    _vix_yesterday = _vix_data[_vix_check]
                    break
            if _vix_yesterday is not None:
                if _vix_yesterday >= _vix_block_threshold:
                    new_buy_candidates = []  # VIX 極高，全停
                elif _vix_yesterday >= _vix_caution_threshold:
                    pass  # 在 can_buy 限制中處理

        # --- V40: USD/TWD 過濾 — 台幣急貶時限制買入 ---
        if _fx_enabled and new_buy_candidates and _fx_data:
            _fx_today = None
            _fx_20d_ago = None
            for _fx_off in range(0, 5):
                _fx_check = (curr_date - pd.Timedelta(days=_fx_off)).strftime('%Y-%m-%d')
                if _fx_check in _fx_data:
                    _fx_today = _fx_data[_fx_check]
                    break
            for _fx_off in range(20, 30):
                _fx_check = (curr_date - pd.Timedelta(days=_fx_off)).strftime('%Y-%m-%d')
                if _fx_check in _fx_data:
                    _fx_20d_ago = _fx_data[_fx_check]
                    break
            if _fx_today and _fx_20d_ago and _fx_20d_ago > 0:
                _fx_chg = (_fx_today - _fx_20d_ago) / _fx_20d_ago * 100  # 正值=台幣貶值
                if _fx_chg > _fx_depreciation_threshold:
                    pass  # 在 can_buy 限制中處理

        # --- V33: 市場廣度過濾 — 族群中 > MA20 的比例太低時限制買入 ---
        if _breadth_enabled and new_buy_candidates:
            _br_above = 0
            _br_total = 0
            for _br_t, _br_sd in stock_data.items():
                _br_df = _br_sd['df']
                if curr_date in _br_df.index:
                    _br_idx = _br_df.index.get_loc(curr_date)
                    if _br_idx >= 20:
                        _br_close = float(_br_df.iloc[_br_idx]['Close'])
                        _br_ma20 = float(_br_df['Close'].iloc[_br_idx-19:_br_idx+1].mean())
                        _br_total += 1
                        if _br_close > _br_ma20:
                            _br_above += 1
            if _br_total > 0:
                _br_pct = _br_above / _br_total * 100
                if _br_pct < _breadth_block_threshold:
                    new_buy_candidates = []  # 族群太弱，全停
                elif _br_pct < _breadth_caution_threshold:
                    # 限制每日買入數
                    pass  # 透過下面的 can_buy 限制

        # --- V31: 多時間框架 — 週線空頭時不開新倉 ---
        if _mtf_enabled and _mtf_weekly_block and new_buy_candidates:
            _mtf_weekly_ok = current_market.get('weekly_bullish', True)
            if not _mtf_weekly_ok:
                new_buy_candidates = []  # 週線空頭，全部不買

        # --- V23A: 量增門檻過濾 — vol_ratio < min_vol_multiplier 的候選剔除 ---
        if _entry_min_vol_mult > 1.0 and new_buy_candidates:
            _before = len(new_buy_candidates)
            new_buy_candidates = [c for c in new_buy_candidates if c.get('vol_ratio', 0) >= _entry_min_vol_mult]

        # --- V23B: 大盤趨勢停手 — TWII MA20 < MA60 時不開新倉 ---
        if _entry_twii_trend_gate and new_buy_candidates:
            _twii_data = current_market.get('twii', {})
            _twii_ma20 = _twii_data.get('ma20', 0)
            _twii_ma60 = _twii_data.get('ma60', 0)
            if _twii_ma20 > 0 and _twii_ma60 > 0 and _twii_ma20 < _twii_ma60:
                new_buy_candidates = []  # 大盤趨勢向下，全部不買

        # --- V23C: 波動率門檻 — TWII 20日實現波動率 > 閾值時不買 ---
        if _entry_vol_gate and new_buy_candidates:
            # 用市場歷史數據計算 TWII 波動率
            _vg_twii_rets = []
            for _vg_i in range(max(0, day_idx - _entry_vol_gate_lookback), day_idx):
                if _vg_i < len(trading_dates):
                    _vg_ds = trading_dates[_vg_i].strftime('%Y-%m-%d')
                    _vg_mkt = market_map.get(_vg_ds, {})
                    _vg_chg = _vg_mkt.get('twii', {}).get('change_pct', 0)
                    if _vg_chg != 0:
                        _vg_twii_rets.append(_vg_chg)
            if len(_vg_twii_rets) >= 10:
                _vg_vol = float(np.std(_vg_twii_rets))
                if _vg_vol > _entry_vol_gate_threshold:
                    new_buy_candidates = []

        # --- ✅ V3.9 上車: 品質排序 + 換股機制 ---
        if new_buy_candidates:
            # 排序: 依品質分數高→低 (V3.9 預設) / 乖離低 / 量比高
            if entry_sort_by == 'volume':
                new_buy_candidates.sort(key=lambda x: (-x['vol_ratio'], x['ticker']))
            elif entry_sort_by == 'bias':
                new_buy_candidates.sort(key=lambda x: (x['bias_pct'], -x['vol_ratio'], x['ticker']))
            else:  # 'score' (V3.9 default)
                new_buy_candidates.sort(key=lambda x: (-x['score'], x['ticker']))

            # 計算今日可新建倉幾檔 (V4.21: 動態曝險時用 _effective_max_pos)
            # ADD pending 計入倉位數 → 加碼活躍時自動壓縮新買名額
            current_hold_count = len(positions) + len(
                [o for o in pending.values() if o['action'] == 'buy'])
            slots_available = max(0, _effective_max_pos - current_hold_count)
            can_buy = min(slots_available, max_new_buy_per_day)

            # V4.21: 動態限買 — 大盤弱勢時降低每日新買上限
            if _dyn_buy_enabled:
                _today_panic_buy = current_market.get('is_panic', False)
                _today_trend_buy = current_market.get('twii', {}).get('trend', 'neutral')
                if _today_panic_buy:
                    _dyn_buy_limit = _dyn_buy_panic
                else:
                    _dyn_buy_limit = _dyn_buy_map.get(_today_trend_buy, max_new_buy_per_day)
                can_buy = min(can_buy, _dyn_buy_limit)

            # V6.10: Regime-Adaptive — unsafe 時限制每日買入
            if _regime_adaptive and current_market.get('is_unsafe', False):
                can_buy = min(can_buy, _regime_unsafe_max_buy)

            # V14: 每週買入上限
            if _weekly_max_buy > 0:
                _weekly_remaining = max(0, _weekly_max_buy - _weekly_buy_count)
                can_buy = min(can_buy, _weekly_remaining)

            # V6.6: EWT 隔夜 can_buy 調整 — 大漲多買、大跌少買
            #   adaptive 時空頭/恐慌自動跳過 (_ewt_boost_mult=0)
            if _ewt_boost_enabled and _ewt_boost_mult > 0:
                _ewt_data_buy = current_market.get('ewt')
                if _ewt_data_buy is not None:
                    _ewt_chg_buy = _ewt_data_buy.get('daily_chg', 0)
                    if _ewt_chg_buy >= _ewt_boost_strong_up:
                        can_buy = can_buy + _ewt_boost_can_buy_bonus
                    elif _ewt_chg_buy <= _ewt_boost_strong_down:
                        can_buy = max(0, can_buy + _ewt_boost_can_buy_penalty)

            # V39: VIX caution zone
            if _vix_enabled and _vix_data and _vix_yesterday is not None:
                if _vix_yesterday >= _vix_caution_threshold and _vix_yesterday < _vix_block_threshold:
                    can_buy = min(can_buy, _vix_caution_max_buy)

            # V40: FX depreciation caution
            if _fx_enabled and _fx_data:
                if _fx_today and _fx_20d_ago and _fx_20d_ago > 0:
                    _fx_chg_buy = (_fx_today - _fx_20d_ago) / _fx_20d_ago * 100
                    if _fx_chg_buy > _fx_depreciation_threshold:
                        can_buy = min(can_buy, _fx_caution_max_buy)

            # V33: 市場廣度 — caution zone 時限制 can_buy
            if _breadth_enabled and _br_total > 0:
                _br_pct_buy = _br_above / _br_total * 100
                if _br_pct_buy < _breadth_caution_threshold and _br_pct_buy >= _breadth_block_threshold:
                    can_buy = min(can_buy, _breadth_caution_max_buy)

            # V16: Portfolio DD 斷路器 — NAV 回撤時暫停新買
            #   用 prev_nav (前日收盤NAV) vs peak_nav 判斷
            if _dd_breaker_enabled:
                if day_idx <= _dd_breaker_cooldown_until:
                    can_buy = 0  # 冷卻期內不買
                elif peak_nav > 0 and prev_nav < peak_nav:
                    _cur_dd = (peak_nav - prev_nav) / peak_nav * 100
                    if _cur_dd >= _dd_breaker_lv2_pct:
                        can_buy = 0
                        _dd_breaker_cooldown_until = day_idx + _dd_breaker_lv2_days
                    elif _cur_dd >= _dd_breaker_lv1_pct:
                        can_buy = 0
                        _dd_breaker_cooldown_until = day_idx + _dd_breaker_lv1_days

            # --- V2.11 配額制: 計算各產業目前持倉數 ---
            def _get_industry_hold_count():
                """回傳 {industry_name: int} 各產業目前持倉數"""
                counts = {}
                if industry_map:
                    for t in positions:
                        ind = industry_map.get(t, '其他')
                        counts[ind] = counts.get(ind, 0) + 1
                    # pending buy 也算
                    for t, o in pending.items():
                        if o['action'] == 'buy' and t not in positions:
                            ind = industry_map.get(t, '其他')
                            counts[ind] = counts.get(ind, 0) + 1
                return counts

            def _quota_ok(ticker_to_check):
                """配額制下，檢查該 ticker 的產業是否還有空位"""
                if not industry_quota or not industry_map:
                    return True  # 非配額模式，永遠 OK
                ind = industry_map.get(ticker_to_check, '其他')
                quota = industry_quota.get(ind, 0)
                if quota <= 0:
                    return False  # 該產業不在配額中
                hold_counts = _get_industry_hold_count()
                current = hold_counts.get(ind, 0)
                return current < quota

            # --- 正常買入 (先填空位) ---
            swapped_tickers = set()  # 已被換股使用的候選 ticker
            bought_tickers = set()   # 本輪正常買入的 ticker
            buy_count_today = 0
            for cand in new_buy_candidates:
                if buy_count_today >= can_buy:
                    break
                ticker = cand['ticker']
                if ticker in pending:
                    continue
                # V2.11: 配額制檢查
                if not _quota_ok(ticker):
                    continue
                close_price = cand['close_price']
                reason = cand['reason']

                # V32: 持倉相關性過濾 — 跟現有持倉高度相關則不買
                if _corr_filter_enabled and positions and ticker in stock_data:
                    _cf_sdf = stock_data[ticker]['df']
                    if curr_date in _cf_sdf.index:
                        _cf_idx = _cf_sdf.index.get_loc(curr_date)
                        if _cf_idx >= _corr_filter_lookback:
                            _cf_cand_rets = _cf_sdf['Close'].iloc[_cf_idx-_corr_filter_lookback:_cf_idx].pct_change().dropna()
                            _cf_high_corr = 0
                            for _cf_pt in positions:
                                if _cf_pt in stock_data and curr_date in stock_data[_cf_pt]['df'].index:
                                    _cf_pf = stock_data[_cf_pt]['df']
                                    _cf_pi = _cf_pf.index.get_loc(curr_date)
                                    if _cf_pi >= _corr_filter_lookback:
                                        _cf_pos_rets = _cf_pf['Close'].iloc[_cf_pi-_corr_filter_lookback:_cf_pi].pct_change().dropna()
                                        try:
                                            _cf_common = pd.concat([_cf_cand_rets.reset_index(drop=True),
                                                                     _cf_pos_rets.reset_index(drop=True)], axis=1).dropna()
                                            if len(_cf_common) >= 10:
                                                _cf_corr = _cf_common.iloc[:,0].corr(_cf_common.iloc[:,1])
                                                if _cf_corr >= _corr_filter_threshold:
                                                    _cf_high_corr += 1
                                        except:
                                            pass
                            if _cf_high_corr >= _corr_filter_max_similar:
                                continue  # 跟太多持倉高度相關，跳過

                # V29: 子題材持倉集中度限制
                if _theme_max_hold > 0 and _theme_entry_map:
                    _v29_theme = _theme_entry_map.get(ticker)
                    if _v29_theme:
                        _v29_count = sum(1 for _v29_t in positions
                                         if _theme_entry_map.get(_v29_t) == _v29_theme)
                        if _v29_count >= _theme_max_hold:
                            continue  # 同題材已滿，跳過

                if exec_mode == 'next_open':
                    _pending_order = {'action': 'buy', 'reason': reason}
                    # V10: 訊號強度加權倉位
                    if _score_sizing_enabled:
                        _cand_score = cand.get('score', 0)
                        if _cand_score >= _score_high_threshold:
                            _pending_order['_budget_ratio'] = _score_high_ratio
                        elif _cand_score < _score_low_threshold:
                            _pending_order['_budget_ratio'] = _score_low_ratio
                        else:
                            _pending_order['_budget_ratio'] = _score_mid_ratio
                    # V24: 進場確認延遲
                    if _entry_confirm_enabled:
                        _entry_confirm_pending[ticker] = {
                            'day_idx': day_idx, 'close': close_price,
                            'order': _pending_order,
                        }
                    else:
                        pending[ticker] = _pending_order
                elif exec_mode == 'close_open':
                    # V7: 收盤先買半額 + T+1開盤再買半額 (總額 = 1x budget)
                    _co_budget = _today_budget // 2
                    buy_price_co = close_price * (1 + _slippage)
                    shares_to_buy = int(_co_budget / buy_price_co)
                    if shares_to_buy > 0:
                        buy_fee = calculate_fee(buy_price_co, shares_to_buy)
                        cost_this = shares_to_buy * buy_price_co + buy_fee
                        if _cash_check and (cash - _cash_reserve) < cost_this:
                            continue
                        total_fees += buy_fee
                        cash -= cost_this
                        if cash < 0:
                            max_overdraft = max(max_overdraft, -cash)
                        positions[ticker] = {
                            'shares': shares_to_buy,
                            'cost_total': cost_this,
                            'avg_cost': buy_price_co,
                            'buy_price': buy_price_co,
                            'buy_count': 1,
                            'last_buy_date_idx': day_idx,
                            'name': cand['name'],
                            'reduce_stage': 0,
                            'last_reduce_date_idx': -99,
                            'peak_since_entry': close_price,
                        }
                        trade_log.append({
                            'date': date_str, 'ticker': ticker, 'name': cand['name'],
                            'type': 'BUY', 'price': close_price,
                            'shares': shares_to_buy, 'fee': buy_fee,
                            'profit': None, 'roi': None,
                            'note': reason + ' [收盤先買]', 'total_shares': shares_to_buy,
                        })
                        day_actions_buy.append(f"{cand['name']}(BUY {shares_to_buy}股@{buy_price_co:.1f} 收盤)")
                        # T+1 開盤再買半額
                        pending[ticker] = {'action': 'buy', 'reason': reason + ' [T+1追加]', '_force_buy': True, '_budget_ratio': 0.5}
                elif exec_mode == 'same_close':
                    buy_price_sc = close_price * (1 + _slippage)  # V4.13 滑價
                    shares_to_buy = int(_today_budget / buy_price_sc)  # V4.21: 波動率縮放
                    if shares_to_buy > 0:
                        buy_fee = calculate_fee(buy_price_sc, shares_to_buy)
                        cost_this = shares_to_buy * buy_price_sc + buy_fee
                        # V4.11: 現金檢查
                        if _cash_check and (cash - _cash_reserve) < cost_this:
                            continue
                        total_fees += buy_fee
                        cash -= cost_this
                        if cash < 0:
                            max_overdraft = max(max_overdraft, -cash)
                        positions[ticker] = {
                            'shares': shares_to_buy,
                            'cost_total': cost_this,
                            'avg_cost': buy_price_sc,
                            'buy_price': buy_price_sc,      # V6: 最近一次買入價 (含滑價)
                            'buy_count': 1,
                            'last_buy_date_idx': day_idx,
                            'name': cand['name'],
                            'reduce_stage': 0,              # V4.8
                            'last_reduce_date_idx': -99,    # V4.8
                            'peak_since_entry': close_price, # V4.20
                        }
                        trade_log.append({
                            'date': date_str, 'ticker': ticker, 'name': cand['name'],
                            'type': 'BUY', 'price': close_price,
                            'shares': shares_to_buy, 'fee': buy_fee,
                            'profit': None, 'roi': None,
                            'note': reason, 'total_shares': shares_to_buy,
                        })
                        day_actions_buy.append(f"{cand['name']}(BUY {shares_to_buy}股@{buy_price_sc:.1f})")
                bought_tickers.add(ticker)
                buy_count_today += 1

            # --- V4.3 換股機制: 填完空位後，剩餘候選 vs 最差持倉 ---
            remaining_candidates = [c for c in new_buy_candidates
                                    if c['ticker'] not in bought_tickers
                                    and c['ticker'] not in pending]
            if enable_swap and remaining_candidates and len(positions) >= _effective_max_pos:
                # 1. 計算所有持倉的 hold_score，排序（最差在前）
                scored_positions = []
                for t, p in positions.items():
                    if t in pending:
                        continue
                    if t in bought_tickers:  # 剛買的不換
                        continue
                    if t not in stock_data or curr_date not in stock_data[t]['df'].index:
                        continue
                    cp = float(stock_data[t]['df'].loc[curr_date, 'Close'])
                    p_pnl = calculate_net_pnl(t, p['avg_cost'], cp, p['shares'])
                    p_net_pct = p_pnl['net_pnl_pct']
                    # V25: 獲利持倉保護 — 不參與換股
                    if _swap_protect_profit and p_net_pct > _swap_protect_threshold:
                        continue
                    p_days = day_idx - p['last_buy_date_idx']
                    hold_score = p_net_pct / 100 - max(0, p_days - 10) * 0.01
                    scored_positions.append((t, hold_score))

                scored_positions.sort(key=lambda x: x[1])  # 最差在前

                # 2. 逐一配對: 最差持倉 vs 最佳剩餘候選
                swap_count = 0
                cand_idx = 0
                for worst_ticker, worst_hold_score in scored_positions:
                    if swap_count >= max_swap_per_day:
                        break
                    if cand_idx >= len(remaining_candidates):
                        break
                    # V2.11 配額制: 找到下一個配額合格的候選
                    cand = None
                    while cand_idx < len(remaining_candidates):
                        _c = remaining_candidates[cand_idx]
                        if industry_quota and industry_map:
                            # 配額制: 候選產業有空位，或與被換出者同產業（騰出名額）
                            cand_ind = industry_map.get(_c['ticker'], '其他')
                            worst_ind = industry_map.get(worst_ticker, '其他')
                            if cand_ind == worst_ind or _quota_ok(_c['ticker']):
                                cand = _c
                                break
                            else:
                                cand_idx += 1
                                continue
                        else:
                            cand = _c
                            break
                    if cand is None:
                        break
                    if cand['score'] <= worst_hold_score + swap_margin:
                        break  # 候選不夠好，後面更差，停止

                    wp = positions[worst_ticker]
                    swap_reason = (f'🔄 換股淘汰 (持有價值{worst_hold_score:.2f} < '
                                   f'新候選{cand["ticker"]}分數{cand["score"]:.2f})')

                    if exec_mode in ('next_open', 'close_open'):
                        pending[worst_ticker] = {'action': 'sell', 'reason': swap_reason}
                        pending[cand['ticker']] = {'action': 'buy', 'reason': cand['reason'] + ' [換股]'}
                    elif exec_mode == 'same_close':
                        # 賣出最差持倉
                        cp_w = float(stock_data[worst_ticker]['df'].loc[curr_date, 'Close'])
                        sell_price_sw = cp_w * (1 - _slippage)  # V4.13 滑價
                        sell_fee = calculate_fee(sell_price_sw, wp['shares'])
                        sell_tax = calculate_tax(worst_ticker, sell_price_sw, wp['shares'])
                        revenue = wp['shares'] * sell_price_sw - sell_fee - sell_tax
                        profit = revenue - wp['cost_total']
                        profit_pct = (profit / wp['cost_total'] * 100) if wp['cost_total'] > 0 else 0
                        cash += revenue
                        realized_profit += profit
                        total_fees += sell_fee + sell_tax
                        trade_count += 1
                        if profit > 0: win_count += 1
                        else: loss_count += 1
                        trade_log.append({
                            'date': date_str, 'ticker': worst_ticker, 'name': wp['name'],
                            'type': 'SELL', 'price': cp_w,
                            'shares': wp['shares'], 'fee': sell_fee + sell_tax,
                            'profit': profit, 'roi': profit_pct,
                            'note': swap_reason, 'total_shares': 0,
                        })
                        day_actions_swap.append(f"OUT:{wp['name']}({profit_pct:+.1f}%)")
                        del positions[worst_ticker]
                        # 立即買入新候選
                        swap_raw_price = cand['close_price']  # 原始市場價 (trade_log 用)
                        swap_buy_price = swap_raw_price * (1 + _slippage)  # V4.13 滑價
                        swap_shares = int(_today_budget / swap_buy_price)  # V4.21: 波動率縮放
                        if swap_shares > 0:
                            swap_buy_fee = calculate_fee(swap_buy_price, swap_shares)
                            swap_cost = swap_shares * swap_buy_price + swap_buy_fee
                            # V4.11: 換股買入也檢查現金 (但賣出回收後通常夠)
                            if _cash_check and (cash - _cash_reserve) < swap_cost:
                                swap_shares = 0  # 跳過買入 (只完成賣出)
                        if swap_shares > 0:
                            swap_buy_fee = calculate_fee(swap_buy_price, swap_shares)
                            swap_cost = swap_shares * swap_buy_price + swap_buy_fee
                            total_fees += swap_buy_fee
                            cash -= swap_cost
                            if cash < 0:
                                max_overdraft = max(max_overdraft, -cash)
                            positions[cand['ticker']] = {
                                'shares': swap_shares,
                                'cost_total': swap_cost,
                                'avg_cost': swap_buy_price,
                                'buy_price': swap_buy_price,    # V6: 最近一次買入價 (含滑價)
                                'buy_count': 1,
                                'last_buy_date_idx': day_idx,
                                'name': cand['name'],
                                'reduce_stage': 0,              # V4.8
                                'last_reduce_date_idx': -99,    # V4.8
                                'peak_since_entry': swap_raw_price, # V4.20
                            }
                            trade_log.append({
                                'date': date_str, 'ticker': cand['ticker'], 'name': cand['name'],
                                'type': 'BUY', 'price': swap_raw_price,  # trade_log 記錄原始市場價
                                'shares': swap_shares, 'fee': swap_buy_fee,
                                'profit': None, 'roi': None,
                                'note': cand['reason'] + ' [換股]',
                                'total_shares': swap_shares,
                            })
                            day_actions_swap.append(f"IN:{cand['name']}")

                    swapped_tickers.add(cand['ticker'])
                    swap_count += 1
                    cand_idx += 1

            # --- V4.18: 儲存備選候補佇列 (排序後未被選中、未被換股使用的新買入候選) ---
            if _enable_backup_fill and _skip_limit_up:
                _used = bought_tickers | swapped_tickers | set(pending.keys())
                _backup_queue = [c for c in new_buy_candidates
                                 if c['ticker'] not in _used
                                 and c['ticker'] not in positions]

        # --- 每日回撤追蹤 ---
        total_invested = sum(p['cost_total'] for p in positions.values())
        if total_invested > max_capital_invested:
            max_capital_invested = total_invested

        unrealized = 0
        for ticker, pos in positions.items():
            sdf = stock_data[ticker]['df']
            _ffill_total_lookups += 1
            if curr_date in sdf.index:
                cp = float(sdf.loc[curr_date, 'Close'])
            else:
                # V4.30: Forward-fill — 用最近一個有報價的交易日收盤價
                _valid = sdf.index[sdf.index < curr_date]
                if len(_valid) > 0:
                    cp = float(sdf.loc[_valid[-1], 'Close'])
                    _ffill_gap_count += 1
                else:
                    continue  # 完全無歷史資料, 跳過
            sf = calculate_fee(cp, pos['shares'])
            st = calculate_tax(ticker, cp, pos['shares'])
            unrealized += (pos['shares'] * cp - sf - st) - pos['cost_total']

        current_equity = realized_profit + unrealized
        equity_curve.append({'date': date_str, 'equity': current_equity, 'positions': len(positions), 'cash': cash})
        if current_equity > peak_equity:
            peak_equity = current_equity
        dd = peak_equity - current_equity
        if dd > max_drawdown:
            max_drawdown = dd

        # V4.7: 每日快照 (供 CSV 輸出)
        # 帳戶淨值 = 現金 + 持股淨變現值
        # V4.30: Forward-fill — 無報價日用前一日收盤價, 避免 NAV 虛降
        _stock_val_net = 0
        for _t, _p in positions.items():
            _sdf = stock_data[_t]['df']
            if curr_date in _sdf.index:
                _cp = float(_sdf.loc[curr_date, 'Close'])
            else:
                _valid_ff = _sdf.index[_sdf.index < curr_date]
                if len(_valid_ff) > 0:
                    _cp = float(_sdf.loc[_valid_ff[-1], 'Close'])
                else:
                    continue
            _stock_val_net += _p['shares'] * _cp - calculate_fee(_cp, _p['shares']) - calculate_tax(_t, _cp, _p['shares'])
        nav_today = cash + _stock_val_net
        daily_ret = ((nav_today / prev_nav) - 1) * 100 if prev_nav > 0 else 0
        cum_ret = ((nav_today / initial_capital) - 1) * 100 if initial_capital > 0 else 0
        # V4.15: 標準 drawdown 公式 — 分母用 peak_nav (帳戶淨值高水位)
        if nav_today > peak_nav:
            peak_nav = nav_today
        dd_nav = peak_nav - nav_today
        if dd_nav > max_drawdown_nav:
            max_drawdown_nav = dd_nav
        dd_pct = ((peak_nav - nav_today) / peak_nav * 100) if peak_nav > 0 else 0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

        # -----------------------------------------
        # V4.10: 持倉組合級恐慌偵測
        # -----------------------------------------
        _pp_daily_returns.append(daily_ret)
        if len(_pp_daily_returns) > 3:
            _pp_daily_returns.pop(0)
        _pp_cum_3d = sum(_pp_daily_returns[-3:]) if len(_pp_daily_returns) >= 3 else sum(_pp_daily_returns)
        _pp_triggered = False
        _pp_trigger_reason = ''

        if _pp_enabled and positions and exec_mode in ('next_open', 'close_open'):
            _pp_days_since = day_idx - _pp_last_trigger_idx
            if _pp_days_since >= _pp_cooldown:
                # 檢查觸發條件
                if daily_ret < _pp_day_th:
                    _pp_triggered = True
                    _pp_trigger_reason = f'📉組合單日{daily_ret:.1f}%<{_pp_day_th}%'
                elif len(_pp_daily_returns) >= 3 and _pp_cum_3d < _pp_3d_th:
                    _pp_triggered = True
                    _pp_trigger_reason = f'📉組合3日累計{_pp_cum_3d:.1f}%<{_pp_3d_th}%'

        _pp_sells_today = []
        if _pp_triggered:
            _pp_last_trigger_idx = day_idx
            # 找出要賣的持倉 (虧損倉或全部)
            for _pp_t, _pp_p in list(positions.items()):
                if _pp_t in pending:
                    continue  # 已有掛單不重複
                _pp_sdf = stock_data[_pp_t]['df']
                if curr_date not in _pp_sdf.index:
                    continue
                _pp_cp = float(_pp_sdf.loc[curr_date, 'Close'])
                _pp_pnl = calculate_net_pnl(_pp_t, _pp_p['avg_cost'], _pp_cp, _pp_p['shares'])
                _pp_net = _pp_pnl['net_pnl_pct']

                should_sell = False
                if _pp_action == 'sell_all':
                    should_sell = True
                else:  # sell_losers
                    should_sell = (_pp_net < _pp_loss_th)

                if should_sell:
                    pending[_pp_t] = {
                        'action': 'sell',
                        'reason': f'🚨 組合恐慌賣出 ({_pp_trigger_reason}, 淨利{_pp_net:+.1f}%)'
                    }
                    _pp_sells_today.append(_pp_p.get('name', _pp_t))

        # 大盤狀態
        _mkt = current_market
        _trend = _mkt.get('twii', {}).get('trend', '') if _mkt else ''
        _flags = []
        if _mkt.get('is_unsafe'):     _flags.append('偏空')
        if _mkt.get('is_overheated'): _flags.append('過熱')
        if _mkt.get('is_panic'):      _flags.append('恐慌')
        if _pp_triggered:             _flags.append('組合恐慌')
        if not _flags:                _flags.append('安全')

        daily_snapshots.append({
            'date': date_str,
            'market_trend': _trend,
            'market_flags': '/'.join(_flags),
            'positions_count': len(positions),
            'holdings': ', '.join(f"{p.get('name', t)}({p['shares']}股@{p['avg_cost']:.1f})" for t, p in sorted(positions.items(), key=lambda x: x[1].get('name', x[0]))),
            'day_buys': ', '.join(day_actions_buy) if day_actions_buy else '',
            'day_sells': ', '.join(day_actions_sell) if day_actions_sell else '',
            'day_swaps': ', '.join(day_actions_swap) if day_actions_swap else '',
            'cash': int(cash),
            'nav': int(nav_today),
            'daily_return_pct': round(daily_ret, 2),
            'cum_return_pct': round(cum_ret, 2),
            'drawdown_pct': round(dd_pct, 2),
            'realized_pnl': int(realized_profit),
            'unrealized_pnl': int(unrealized),
        })
        # V3.0 驗證: 每日 positions 深拷貝
        if _capture_positions:
            _positions_history.append({
                'date': date_str,
                'day_idx': day_idx,
                'positions': {t: dict(p) for t, p in positions.items()},
                'cash': cash,
                'pending': dict(pending),
            })
        prev_nav = nav_today

        # === CSV daily dump ===
        if csv_output_dir:
            # positions CSV
            _pos_path = os.path.join(csv_output_dir, f'positions_{date_str}.csv')
            with open(_pos_path, 'w', newline='', encoding='utf-8-sig') as _pf:
                _pw = csv.writer(_pf)
                _pw.writerow(['ticker', 'name', 'shares', 'avg_cost', 'cost_total',
                              'buy_price', 'buy_count', 'last_buy_date_idx',
                              'reduce_stage', 'last_reduce_date_idx', 'peak_since_entry'])
                for _ct, _cp_pos in sorted(positions.items()):
                    _pw.writerow([
                        _ct, _cp_pos.get('name', ''),
                        _cp_pos['shares'], round(_cp_pos['avg_cost'], 4),
                        round(_cp_pos.get('cost_total', _cp_pos['avg_cost'] * _cp_pos['shares']), 2),
                        round(_cp_pos.get('buy_price', _cp_pos['avg_cost']), 4),
                        _cp_pos.get('buy_count', 1),
                        _cp_pos.get('last_buy_date_idx', 0),
                        _cp_pos.get('reduce_stage', 0),
                        _cp_pos.get('last_reduce_date_idx', -99),
                        round(_cp_pos.get('peak_since_entry', _cp_pos['avg_cost']), 4),
                    ])
            # trades CSV (today only)
            _trades_path = os.path.join(csv_output_dir, f'trades_{date_str}.csv')
            _today_trades = [t for t in trade_log if t.get('date') == date_str]
            with open(_trades_path, 'w', newline='', encoding='utf-8-sig') as _tf:
                _tw = csv.writer(_tf)
                _tw.writerow(['date', 'ticker', 'name', 'type', 'price', 'shares',
                              'fee', 'profit', 'roi', 'note', 'total_shares'])
                for _tt in _today_trades:
                    _tw.writerow([
                        _tt.get('date', ''), _tt.get('ticker', ''), _tt.get('name', ''),
                        _tt.get('type', ''), _tt.get('price', ''), _tt.get('shares', ''),
                        round(_tt.get('fee', 0), 2) if _tt.get('fee') is not None else '',
                        round(_tt.get('profit', 0), 2) if _tt.get('profit') is not None else '',
                        round(_tt.get('roi', 0), 2) if _tt.get('roi') is not None else '',
                        _tt.get('note', ''), _tt.get('total_shares', ''),
                    ])
            # state JSON
            _state_path = os.path.join(csv_output_dir, f'state_{date_str}.json')
            _state_obj = {
                'date': date_str,
                'day_idx': day_idx,
                'cash': round(cash, 2),
                'realized_profit': round(realized_profit, 2),
                'total_fees': round(total_fees, 2),
                'trade_count': trade_count,
                'win_count': win_count,
                'loss_count': loss_count,
                'prev_nav': round(nav_today, 2),
                'peak_nav': round(peak_nav, 2),
                'vol_rets': [round(v, 6) for v in _vol_rets[-50:]],
                'pp_daily_returns': [round(v, 6) for v in _pp_daily_returns],
                'pp_last_trigger_idx': _pp_last_trigger_idx,
                'pending': {k: {**{pk: pv for pk, pv in v.items() if not pk.startswith('_')},
                                'name': (positions.get(k, {}).get('name', '')
                                         or stock_data.get(k, {}).get('name', k))}
                            for k, v in pending.items()},
                'nav': round(nav_today, 2),
                'equity': round(current_equity, 2),
                'positions_count': len(positions),
            }
            with open(_state_path, 'w', encoding='utf-8') as _sf:
                json.dump(_state_obj, _sf, ensure_ascii=False, indent=2)

        # V4.21: 波動率追蹤 (供下一日 vol sizing 使用)
        if _vol_sizing_enabled:
            _vol_rets.append(daily_ret)

    # =========================================
    # 5. 最終結算: 未平倉部位
    # =========================================
    final_unrealized = 0
    open_positions = []

    for ticker, pos in positions.items():
        sdf = stock_data[ticker]['df']
        last_rows = sdf.loc[sdf.index <= sim_end]
        if last_rows.empty:
            continue
        final_price = float(last_rows.iloc[-1]['Close'])
        sf = calculate_fee(final_price, pos['shares'])
        st = calculate_tax(ticker, final_price, pos['shares'])
        unreal = (pos['shares'] * final_price - sf - st) - pos['cost_total']
        final_unrealized += unreal

        open_positions.append({
            'ticker': ticker, 'name': pos['name'],
            'shares': pos['shares'], 'avg_cost': pos['avg_cost'],
            'current_price': final_price,
            'unrealized': unreal,
            'buy_count': pos['buy_count'],
        })

    total_pnl = realized_profit + final_unrealized
    roi = (total_pnl / max_capital_invested * 100) if max_capital_invested > 0 else 0

    # V4.30: Forward-fill 診斷
    if _ffill_total_lookups > 0:
        _ffill_pct = _ffill_gap_count / _ffill_total_lookups * 100
        print(f"   📊 Forward-fill 統計: {_ffill_gap_count}/{_ffill_total_lookups} 次 ({_ffill_pct:.1f}%) 使用前日收盤價")

    # V4.1: 帳戶結算 (算法 A: 實際投入現金報酬率)
    final_stock_value = 0       # 持股市值 (不扣 fee/tax)
    final_stock_value_net = 0   # 持股淨變現值 (扣 fee/tax)
    for op in open_positions:
        pos = positions.get(op['ticker'])
        if pos:
            final_stock_value += pos['shares'] * op['current_price']
            final_stock_value_net += pos['shares'] * op['current_price'] - \
                calculate_fee(op['current_price'], pos['shares']) - \
                calculate_tax(op['ticker'], op['current_price'], pos['shares'])

    final_total_value = cash + final_stock_value_net
    total_return_pct = ((final_total_value - initial_capital) / initial_capital * 100
                        if initial_capital > 0 else 0)

    # V4.2: 進階績效指標 (CAGR, MDD%, Sharpe, Calmar)
    # --- 年化報酬率 (CAGR) ---
    days_in_backtest = (pd.Timestamp(sim_end) - pd.Timestamp(sim_start)).days
    years = days_in_backtest / 365.25 if days_in_backtest > 0 else 1
    if initial_capital > 0 and final_total_value > 0:
        cagr = ((final_total_value / initial_capital) ** (1 / years) - 1) * 100
    else:
        cagr = 0.0

    # --- MDD% (V4.21: 用每日正確計算的 dd_pct 最大值, 修復舊版分母用最終peak的bug) ---
    mdd_pct = max_dd_pct

    # --- Sharpe Ratio (用每日帳戶總值變化計算) ---
    # equity = realized_profit + unrealized = 總損益 (起始=0)
    # 帳戶總值 = initial_capital + equity
    sharpe_ratio = 0.0
    if len(equity_curve) >= 2 and initial_capital > 0:
        daily_nav = [initial_capital + e['equity'] for e in equity_curve]  # 每日帳戶淨值
        daily_returns = []
        for i in range(1, len(daily_nav)):
            if daily_nav[i - 1] > 0:
                daily_returns.append(daily_nav[i] / daily_nav[i - 1] - 1)
        if daily_returns and np.std(daily_returns) > 0:
            # 台股一年約 245 個交易日, 無風險利率約 1.5%
            risk_free_daily = 0.015 / 245
            excess_returns = [r - risk_free_daily for r in daily_returns]
            annualized_excess = np.mean(excess_returns) * 245
            annualized_vol = np.std(daily_returns) * np.sqrt(245)
            sharpe_ratio = annualized_excess / annualized_vol

    # --- Calmar Ratio (CAGR / MDD%) ---
    calmar_ratio = (cagr / mdd_pct) if mdd_pct > 0 else 0.0

    return {
        'realized': realized_profit,
        'unrealized': final_unrealized,
        'total_pnl': total_pnl,
        'roi': roi,
        'fees': total_fees,
        'trades': trade_count,
        'wins': win_count,
        'losses': loss_count,
        'win_rate': (win_count / trade_count * 100) if trade_count > 0 else 0,
        'max_drawdown': max_drawdown,
        'max_capital': max_capital_invested,
        'trade_log': trade_log,
        'open_positions': open_positions,
        'equity_curve': equity_curve,
        'exec_mode': exec_mode,
        'slippage_pct': cfg.get('slippage_pct', 0.0),  # V4.13
        # V4.1: 現金帳戶
        'initial_capital': initial_capital,
        'final_cash': cash,
        'final_stock_value': final_stock_value,
        'final_stock_value_net': final_stock_value_net,
        'final_total_value': final_total_value,
        'total_return_pct': total_return_pct,
        'max_overdraft': max_overdraft,
        # V4.2: 進階績效指標
        'cagr': cagr,
        'mdd_pct': mdd_pct,
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio,
        'backtest_days': days_in_backtest,
        'backtest_years': years,
        # V4.7: 每日快照
        'daily_snapshots': daily_snapshots,
        # V3.0 daily: 引擎內部狀態 (daily_signal + 驗證用)
        '_raw_positions': {t: dict(p) for t, p in positions.items()},
        '_pending': dict(pending),
        '_limit_up_skipped': list(_limit_up_skipped),  # V4.16: 漲停跳過的 tickers
        '_last_candidates': list(new_buy_candidates),  # V4.16: 最後一天的候選清單
        '_last_add_candidates': list(add_candidates),  # V9: 加碼候選清單 (含未被選中的)
        '_backup_queue': list(_backup_queue),  # V4.18: 備選佇列 (遞補用)
        '_positions_history': _positions_history,  # None if not captured
        '_stock_data': stock_data,  # 供驗證模式重用
        # V9: 引擎狀態輸出 (DailyEngine 共用回測邏輯時使用)
        '_state': {
            'cash': cash,
            'prev_nav': prev_nav,
            'peak_nav': peak_nav,
            'vol_rets': list(_vol_rets),
            'pp_daily_returns': list(_pp_daily_returns),
            'pp_last_trigger_idx': _pp_last_trigger_idx,
            'realized_profit': realized_profit,
            'total_fees': total_fees,
            'trade_count': trade_count,
            'win_count': win_count,
            'loss_count': loss_count,
        },
        '_final_day_idx': day_idx + 1 if trading_dates else initial_day_idx,
        '_theme_rotation_status': {
            'allowed': sorted(_allowed_themes) if _theme_rotation_enabled else [],
            'scores': dict(_theme_scores) if _theme_enabled else {},
        },
    }


# ==========================================
# 📄 CSV 每日快照輸出 (V4.7)
# ==========================================
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def export_daily_csv(result, config_name, start_date, end_date):
    """
    將回測結果的每日快照輸出為 CSV，方便 Excel 檢視。
    檔名: reports/{config_name}_{start}_{end}.csv
    回傳: 檔案路徑 (成功) 或 None (失敗)
    """
    if result is None:
        return None
    snapshots = result.get('daily_snapshots', [])
    if not snapshots:
        return None

    os.makedirs(REPORT_DIR, exist_ok=True)
    s = start_date.replace('-', '')
    e = end_date.replace('-', '')
    filename = f"{config_name}_{s}_{e}.csv"
    filepath = os.path.join(REPORT_DIR, filename)

    import csv
    columns = [
        'date', 'market_trend', 'market_flags',
        'positions_count', 'holdings',
        'day_buys', 'day_sells', 'day_swaps',
        'cash', 'nav',
        'daily_return_pct', 'cum_return_pct', 'drawdown_pct',
        'realized_pnl', 'unrealized_pnl',
    ]
    headers = [
        '日期', '大盤趨勢', '大盤狀態',
        '持倉數', '持有股票',
        '今日買入', '今日賣出', '今日換股',
        '現金', '帳戶淨值',
        '當日報酬%', '累計報酬%', '回撤%',
        '已實現損益', '未實現損益',
    ]

    # BOM for Excel UTF-8 support
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for snap in snapshots:
            writer.writerow([snap.get(c, '') for c in columns])

    return filepath


# ==========================================
# 📁 每日部位快照輸出 (V5 — portfolio_strategy.csv 格式)
# ==========================================
def export_portfolio_snapshots(result, config_name, start_date, end_date,
                                industry_map=None, industry_label=None,
                                csv_prefix='sim_portfolio'):
    """
    從回測結果中提取每日持倉，輸出為 portfolio CSV 格式。
    每個檔案代表該日收盤後的部位狀態，可直接被 Mode 1/Mode 5 載入。
    輸出目錄: reports/positions_{label}_{start}_{end}/
    檔名: {csv_prefix}_{label}_{YYYYMMDD}.csv
      - 回測模擬: sim_portfolio_{label}_{YYYYMMDD}.csv
      - 真實部位: portfolio_{label}_{YYYYMMDD}.csv
    回傳: 輸出檔案路徑列表
    """
    positions_history = result.get('_positions_history')
    if not positions_history:
        return []

    _label = industry_label or config_name
    s = start_date.replace('-', '')
    e = end_date.replace('-', '')
    snapshot_dir = os.path.join(REPORT_DIR, f"positions_{_label}_{s}_{e}")
    os.makedirs(snapshot_dir, exist_ok=True)

    cols = _PORTFOLIO_COLS  # 完整欄位 (含 buy_price, peak_since_entry)
    files = []
    for snap in positions_history:
        date_str = snap['date']
        positions = snap['positions']
        if not positions:
            continue

        rows = []
        for ticker, pos in sorted(positions.items(), key=lambda x: x[1].get('name', x[0])):
            _ind = ''
            if industry_map and ticker in industry_map:
                _ind = industry_map[ticker]
            rows.append({
                'ticker': ticker,
                'name': pos.get('name', ticker),
                'industry': _ind,
                'shares': pos['shares'],
                'avg_cost': round(pos['avg_cost'], 2),
                'buy_price': round(pos.get('buy_price', pos['avg_cost']), 2),
                'peak_since_entry': round(pos.get('peak_since_entry', pos['avg_cost']), 2),
                'note': '',
            })

        df = pd.DataFrame(rows, columns=cols)
        filepath = os.path.join(snapshot_dir,
                    f"{csv_prefix}_{_label}_{date_str.replace('-', '')}.csv")
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        files.append(filepath)

    return files


# ==========================================
# 📊 真實部位分析 (V6: Mode 1B)
# ==========================================
def run_real_portfolio_analysis(selected_industries, industry_map, all_stocks,
                                 start_date, end_date, budget, initial_capital,
                                 market_map, config_override=None,
                                 per_industry_config=None, use_per_industry=False):
    """
    真實部位分析模式 (Mode 1 子模式 B)

    讀取使用者手動維護的 portfolio_{產業}_{日期}.csv 序列，分析績效 + 產生 T+1 訊號。

    流程:
    1. 自動找到日期範圍內的 portfolio CSV 檔案
    2. 逐日比對 CSV，偵測買入/賣出/加碼/減碼
    3. 自動計算 avg_cost (加權平均) 和 peak_since_entry
    4. 計算日期範圍績效
    5. 用最後一天部位跑引擎 → 產生 T+1 訊號 (像 Mode 5)
    """
    from strategy import DEFAULT_CONFIG as _dc_rp
    _ind_label = '_'.join(selected_industries)

    print(f"\n{'='*70}")
    print(f"📊 真實部位分析 (Mode 1B)")
    print(f"   分析期間: {start_date} ~ {end_date}")
    print(f"   產業: {' + '.join(selected_industries)}")
    print(f"   每筆預算: ${budget:,} | 初始資金: ${initial_capital:,}")
    print(f"{'='*70}")

    # --- 1. 搜尋日期範圍內的 portfolio CSV ---
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    _start_ts = pd.Timestamp(start_date)
    _end_ts = pd.Timestamp(end_date)

    # 搜尋 CSV 來源 — 只找 portfolio_* (不找 sim_portfolio / positions 子目錄)
    # 根目錄優先 (使用者手動放的真實部位), 不碰回測產出的 sim_portfolio
    import glob as _glob_rp
    _csv_patterns = [
        # 根目錄 (使用者手動維護的真實部位, 最高優先)
        os.path.join(_base_dir, f"portfolio_{_ind_label}_*.csv"),
        # reports/ 頂層 (如果使用者放在這裡)
        os.path.join(REPORT_DIR, f"portfolio_{_ind_label}_*.csv"),
    ]
    _all_csvs = {}  # {date_str: filepath}
    for pat in _csv_patterns:
        for fp in _glob_rp.glob(pat):
            bn = os.path.basename(fp)
            # 排除 sim_portfolio 開頭 (回測產出, 不是真實部位)
            if bn.startswith('sim_'):
                continue
            # 提取日期: portfolio_xxx_YYYYMMDD.csv
            parts = bn.replace('.csv', '').split('_')
            _date_part = parts[-1] if parts else ''
            if len(_date_part) == 8 and _date_part.isdigit():
                _dt = f"{_date_part[:4]}-{_date_part[4:6]}-{_date_part[6:]}"
                _dt_ts = pd.Timestamp(_dt)
                # 只搜尋 start_date 到 end_date
                if _start_ts <= _dt_ts <= _end_ts:
                    # 同日: 根目錄優先 (先找到的優先)
                    if _dt not in _all_csvs:
                        _all_csvs[_dt] = fp

    _sorted_dates = sorted(_all_csvs.keys())

    if not _sorted_dates:
        print(f"\n   ❌ 未找到日期範圍內的 portfolio CSV")
        print(f"   請準備: portfolio_{_ind_label}_YYYYMMDD.csv")
        print(f"   欄位: ticker, name, industry, shares, buy_price")
        return None

    print(f"\n   📂 找到 {len(_sorted_dates)} 個 CSV:")
    for _dt in _sorted_dates:
        _df_tmp = pd.read_csv(_all_csvs[_dt], dtype={'ticker': str})
        print(f"      {_dt}: {os.path.basename(_all_csvs[_dt])} ({len(_df_tmp)} 檔)")

    # --- 2. 第一天 CSV = 初始部位, 其餘 = 逐日比對 ---
    if not _sorted_dates:
        print(f"\n   ❌ 無法確定初始部位")
        return None
    _init_csv_date = _sorted_dates[0]
    _range_csv_dates = _sorted_dates[1:]

    # --- 3. 讀取初始部位 ---
    _init_df = pd.read_csv(_all_csvs[_init_csv_date], dtype={'ticker': str})
    _portfolio = {}  # ticker → {shares, avg_cost, buy_price, peak_since_entry, name, industry}
    for _, row in _init_df.iterrows():
        _t = str(row.get('ticker', '')).strip()
        if not _t or _t == 'nan':
            continue
        _bp_raw = row.get('buy_price')
        _ac_raw = row.get('avg_cost')
        _bp = float(_bp_raw) if (not pd.isna(_bp_raw) if _bp_raw is not None else False) else 0
        _ac = float(_ac_raw) if (not pd.isna(_ac_raw) if _ac_raw is not None else False) else _bp
        if _ac == 0:
            _ac = _bp
        _peak_raw = row.get('peak_since_entry')
        _peak = float(_peak_raw) if (_peak_raw is not None and not pd.isna(_peak_raw)) else _ac

        _portfolio[_t] = {
            'shares': int(row.get('shares', 0)),
            'avg_cost': _ac,
            'buy_price': _bp,
            'peak_since_entry': _peak,
            'name': str(row.get('name', _t)) if not pd.isna(row.get('name')) else _t,
            'industry': str(row.get('industry', '')) if not pd.isna(row.get('industry')) else '',
        }

    print(f"\n   📦 初始部位 ({_init_csv_date}, {len(_portfolio)} 檔):")
    for _t, _p in sorted(_portfolio.items(), key=lambda x: x[1]['name']):
        print(f"      {_t} {_p['name']} {_p['shares']}股 "
              f"成本${_p['avg_cost']:.1f} 買價${_p['buy_price']:.1f}")

    # 保存初始部位 (第一個 CSV 的部位, 供引擎多日跑使用)
    _init_portfolio = {t: dict(p) for t, p in _portfolio.items()}

    # --- 4. 逐日比對 CSV，偵測交易 + 更新 avg_cost / peak ---
    _trade_history = []  # 交易紀錄 for display
    _daily_mv = []       # 每日市值 for 績效

    # 下載股票資料 (需要收盤價來更新 peak_since_entry 和計算績效)
    _all_tickers_set = set(t for t, _ in all_stocks)
    for _t in _portfolio:
        if _t not in _all_tickers_set:
            _n = _portfolio[_t].get('name', _t)
            all_stocks.append((_t, _n))
            _all_tickers_set.add(_t)

    # 讀取 range 內 CSV 中出現的所有 ticker
    for _dt in _range_csv_dates:
        _df_tmp = pd.read_csv(_all_csvs[_dt], dtype={'ticker': str})
        for _, row in _df_tmp.iterrows():
            _t = str(row.get('ticker', '')).strip()
            if _t and _t != 'nan' and _t not in _all_tickers_set:
                _n = str(row.get('name', _t)) if not pd.isna(row.get('name')) else _t
                all_stocks.append((_t, _n))
                _all_tickers_set.add(_t)

    # 下載資料
    _sim_start = pd.Timestamp(start_date)
    _download_start = (_sim_start - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
    _download_end = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')

    print(f"\n   ⏳ 下載股票資料 ({len(all_stocks)} 檔)...")
    stock_data, _skipped = batch_download_stocks(
        all_stocks, _download_start, _download_end,
        min_data_days=5, force_refresh=False,
    )
    print(f"   ✅ 有效: {len(stock_data)} 檔")
    if _skipped.get('download_fail', 0) > 0:
        print(f"   ⚠️ 下載失敗: {_skipped['download_fail']} 檔")

    # 計算初始市值 + 更新 peak_since_entry
    _init_mv = 0
    for _t, _p in _portfolio.items():
        if _t in stock_data:
            sdf = stock_data[_t]['df']
            _vd = sdf.index[sdf.index <= pd.Timestamp(_init_csv_date)]
            if len(_vd) > 0:
                _cp = float(sdf.loc[_vd[-1], 'Close'])
                _init_mv += _cp * _p['shares']
                # 更新 peak_since_entry
                _cur_peak = _p.get('peak_since_entry', _p['avg_cost'])
                _portfolio[_t]['peak_since_entry'] = max(_cur_peak, _cp)
    _init_cost_total = sum(_p['avg_cost'] * _p['shares'] for _p in _portfolio.values())
    _cash = initial_capital - _init_cost_total

    print(f"\n   💰 初始狀態: 持股市值=${_init_mv:,.0f} | 成本=${_init_cost_total:,.0f} | 現金=${_cash:,.0f}")

    # V6: 直接更新初始 CSV (原地覆蓋, 補上 peak_since_entry)
    if _portfolio:
        _init_src_path = _all_csvs[_init_csv_date]
        _wb_init_rows = []
        for _t, _p in sorted(_portfolio.items(), key=lambda x: x[1]['name']):
            _wb_init_rows.append({
                'ticker': _t,
                'name': _p['name'],
                'industry': _p.get('industry', industry_map.get(_t, '')),
                'shares': _p['shares'],
                'avg_cost': round(_p['avg_cost'], 2),
                'buy_price': round(_p.get('buy_price', _p['avg_cost']), 2),
                'peak_since_entry': round(_p.get('peak_since_entry', _p['avg_cost']), 2),
                'note': '',
            })
        pd.DataFrame(_wb_init_rows, columns=_PORTFOLIO_COLS).to_csv(
            _init_src_path, index=False, encoding='utf-8-sig')
        print(f"   💾 更新: {os.path.basename(_init_src_path)} (補 peak_since_entry)")

    # --- 逐日處理 ---
    _prev_portfolio = dict(_portfolio)  # deep copy
    _prev_portfolio = {t: dict(p) for t, p in _portfolio.items()}

    for _dt in _range_csv_dates:
        _df_day = pd.read_csv(_all_csvs[_dt], dtype={'ticker': str})
        _day_portfolio = {}
        for _, row in _df_day.iterrows():
            _t = str(row.get('ticker', '')).strip()
            if not _t or _t == 'nan':
                continue
            _bp_raw = row.get('buy_price')
            _bp = float(_bp_raw) if (_bp_raw is not None and not pd.isna(_bp_raw)) else 0
            _day_portfolio[_t] = {
                'shares': int(row.get('shares', 0)),
                'buy_price': _bp,
                'name': str(row.get('name', _t)) if not pd.isna(row.get('name')) else _t,
                'industry': str(row.get('industry', '')) if not pd.isna(row.get('industry')) else '',
            }

        # 比對差異
        _prev_tickers = set(_prev_portfolio.keys())
        _curr_tickers = set(_day_portfolio.keys())

        # 新出現 = 買入
        for _t in _curr_tickers - _prev_tickers:
            _dp = _day_portfolio[_t]
            _portfolio[_t] = {
                'shares': _dp['shares'],
                'avg_cost': _dp['buy_price'] if _dp['buy_price'] > 0 else 0,
                'buy_price': _dp['buy_price'],
                'peak_since_entry': _dp['buy_price'],
                'name': _dp['name'],
                'industry': _dp['industry'],
            }
            _trade_history.append({
                'date': _dt, 'ticker': _t, 'name': _dp['name'],
                'type': '🟢買入', 'shares': _dp['shares'],
                'price': _dp['buy_price'],
            })

        # 消失 = 賣出
        for _t in _prev_tickers - _curr_tickers:
            _pp = _prev_portfolio[_t]
            # 計算賣出價 (用當日收盤估算)
            _sell_price = 0
            if _t in stock_data:
                sdf = stock_data[_t]['df']
                _vd = sdf.index[sdf.index <= pd.Timestamp(_dt)]
                if len(_vd) > 0:
                    _sell_price = float(sdf.loc[_vd[-1], 'Close'])
            _pnl_pct = ((_sell_price - _pp['avg_cost']) / _pp['avg_cost'] * 100
                        if _pp['avg_cost'] > 0 and _sell_price > 0 else 0)
            _trade_history.append({
                'date': _dt, 'ticker': _t, 'name': _pp['name'],
                'type': '🔴賣出', 'shares': _pp['shares'],
                'price': _sell_price, 'pnl_pct': _pnl_pct,
            })
            if _t in _portfolio:
                # 賣出回收現金 (估算)
                if _sell_price > 0:
                    _cash += _pp['shares'] * _sell_price
                del _portfolio[_t]

        # 兩邊都有 → 比較股數
        for _t in _curr_tickers & _prev_tickers:
            _dp = _day_portfolio[_t]
            _pp = _prev_portfolio[_t]
            _delta = _dp['shares'] - _pp['shares']

            if _delta > 0:
                # 加碼
                _old_avg = _portfolio[_t]['avg_cost']
                _old_shares = _pp['shares']
                _new_bp = _dp['buy_price'] if _dp['buy_price'] > 0 else _old_avg
                # 加權平均成本
                _portfolio[_t]['avg_cost'] = (
                    (_old_shares * _old_avg + _delta * _new_bp) / _dp['shares']
                )
                _portfolio[_t]['shares'] = _dp['shares']
                _portfolio[_t]['buy_price'] = _new_bp
                _trade_history.append({
                    'date': _dt, 'ticker': _t, 'name': _dp['name'],
                    'type': '💰加碼', 'shares': _delta,
                    'price': _new_bp,
                })
                # 加碼扣現金
                if _new_bp > 0:
                    _cash -= _delta * _new_bp

            elif _delta < 0:
                # 減碼
                _sell_price = 0
                if _t in stock_data:
                    sdf = stock_data[_t]['df']
                    _vd = sdf.index[sdf.index <= pd.Timestamp(_dt)]
                    if len(_vd) > 0:
                        _sell_price = float(sdf.loc[_vd[-1], 'Close'])
                _portfolio[_t]['shares'] = _dp['shares']
                _trade_history.append({
                    'date': _dt, 'ticker': _t, 'name': _dp['name'],
                    'type': '✂️減碼', 'shares': abs(_delta),
                    'price': _sell_price,
                })
                # 減碼回收現金
                if _sell_price > 0:
                    _cash += abs(_delta) * _sell_price

            # 更新 buy_price (若 CSV 有填新的)
            if _dp['buy_price'] > 0 and _delta == 0:
                _portfolio[_t]['buy_price'] = _dp['buy_price']

        # 更新 peak_since_entry (用當日收盤)
        for _t in _portfolio:
            if _t in stock_data:
                sdf = stock_data[_t]['df']
                _vd = sdf.index[sdf.index <= pd.Timestamp(_dt)]
                if len(_vd) > 0:
                    _cp = float(sdf.loc[_vd[-1], 'Close'])
                    _cur_peak = _portfolio[_t].get('peak_since_entry', _portfolio[_t]['avg_cost'])
                    _portfolio[_t]['peak_since_entry'] = max(_cur_peak, _cp)

        # V6: 原地更新 CSV (補上 avg_cost + peak_since_entry, 不產生備份)
        if _portfolio and _dt in _all_csvs:
            _wb_src_path = _all_csvs[_dt]
            _wb_rows = []
            for _t, _p in sorted(_portfolio.items(), key=lambda x: x[1]['name']):
                _wb_rows.append({
                    'ticker': _t,
                    'name': _p['name'],
                    'industry': _p.get('industry', industry_map.get(_t, '')),
                    'shares': _p['shares'],
                    'avg_cost': round(_p['avg_cost'], 2),
                    'buy_price': round(_p.get('buy_price', _p['avg_cost']), 2),
                    'peak_since_entry': round(_p.get('peak_since_entry', _p['avg_cost']), 2),
                    'note': '',
                })
            pd.DataFrame(_wb_rows, columns=_PORTFOLIO_COLS).to_csv(
                _wb_src_path, index=False, encoding='utf-8-sig')
            print(f"   💾 更新: {os.path.basename(_wb_src_path)} (avg_cost + peak)")

        # 計算當日市值
        _day_mv = 0
        for _t, _p in _portfolio.items():
            if _t in stock_data:
                sdf = stock_data[_t]['df']
                _vd = sdf.index[sdf.index <= pd.Timestamp(_dt)]
                if len(_vd) > 0:
                    _cp = float(sdf.loc[_vd[-1], 'Close'])
                    _day_mv += _cp * _p['shares']
        _daily_mv.append({'date': _dt, 'mv': _day_mv, 'cash': _cash})

        # 更新 prev_portfolio
        _prev_portfolio = {t: dict(p) for t, p in _portfolio.items()}

    # --- 5. 績效報告 ---
    print(f"\n{'═'*70}")
    print(f"📊 真實部位績效 ({start_date} ~ {end_date})")
    print(f"{'═'*70}")

    # 交易紀錄
    if _trade_history:
        print(f"\n   📝 交易紀錄 ({len(_trade_history)} 筆):")
        for _tr in _trade_history:
            _extra = ''
            if 'pnl_pct' in _tr:
                _extra = f" (損益 {_tr['pnl_pct']:+.1f}%)"
            print(f"      {_tr['date']} {_tr['type']} {_tr['ticker']} {_tr['name']} "
                  f"{_tr['shares']}股 @ ${_tr['price']:.1f}{_extra}")
    else:
        print(f"\n   ⏸️  期間無交易 (持倉不變)")

    # 最終部位
    _end_mv = 0
    _end_cost = 0
    print(f"\n   📋 最終部位 ({len(_portfolio)} 檔):")
    _pos_entries = []
    for _t, _p in sorted(_portfolio.items(), key=lambda x: x[1]['name']):
        _cp = 0
        if _t in stock_data:
            sdf = stock_data[_t]['df']
            _vd = sdf.index[sdf.index <= pd.Timestamp(end_date)]
            if len(_vd) > 0:
                _cp = float(sdf.loc[_vd[-1], 'Close'])
        _mv = _cp * _p['shares']
        _cost = _p['avg_cost'] * _p['shares']
        _pnl_pct = ((_cp - _p['avg_cost']) / _p['avg_cost'] * 100
                    if _p['avg_cost'] > 0 and _cp > 0 else 0)
        _end_mv += _mv
        _end_cost += _cost
        _pos_entries.append({
            'ticker': _t, 'name': _p['name'], 'industry': _p.get('industry', ''),
            'shares': _p['shares'], 'avg_cost': _p['avg_cost'],
            'buy_price': _p.get('buy_price', _p['avg_cost']),
            'close': _cp, 'mv': _mv, 'pnl_pct': _pnl_pct,
            'peak': _p.get('peak_since_entry', _p['avg_cost']),
        })

    print(f"   {'代碼':<12} {'名稱':<8} {'股數':>5} {'成本':>8} {'現價':>8} {'損益%':>7} {'市值':>10}")
    print(f"   {'─'*68}")
    for e in sorted(_pos_entries, key=lambda x: -x['pnl_pct']):
        print(f"   {e['ticker']:<12} {e['name']:<8} {e['shares']:>5} "
              f"${e['avg_cost']:>7.1f} ${e['close']:>7.1f} "
              f"{e['pnl_pct']:>+6.1f}%  ${e['mv']:>9,.0f}")

    # 績效摘要
    _total_ret_pct = ((_end_mv - _end_cost) / _end_cost * 100) if _end_cost > 0 else 0
    _total_pnl = _end_mv - _end_cost
    _nav = _cash + _end_mv
    _nav_ret = ((_nav / initial_capital) - 1) * 100

    print(f"\n   {'─'*68}")
    print(f"   💰 持股市值: ${_end_mv:,.0f} | 持股成本: ${_end_cost:,.0f} | 未實現損益: ${_total_pnl:+,.0f} ({_total_ret_pct:+.1f}%)")
    print(f"   💵 現金: ${_cash:,.0f} | NAV: ${_nav:,.0f} | 總報酬: {_nav_ret:+.1f}%")

    # --- 6. T+1 訊號 (用最後一天部位跑引擎) ---
    _t1_date = (pd.Timestamp(end_date) + pd.offsets.BDay(1)).strftime('%Y-%m-%d')
    print(f"\n{'═'*70}")
    print(f"📋 {_t1_date} 操作建議 (T+1, 基於 {end_date} 盤後部位)")
    print(f"{'═'*70}")

    # 準備 initial_positions for engine (使用第一個 CSV 的部位)
    # 引擎從 start_date 跑到 end_date, 累積組合狀態 (portfolio panic 等)
    _is_multiday = (start_date != end_date and len(_sorted_dates) > 1)
    # 多日模式: last_buy_date_idx=0 → 首日不賣 (模擬「剛買入」, 與回測一致)
    # 單日模式: last_buy_date_idx=-1 → 當天就能賣 (已持有的庫存)
    _lbd_idx = 0 if _is_multiday else -1
    _engine_init_pos = {}
    for _t, _p in _init_portfolio.items():
        _engine_init_pos[_t] = {
            'shares': _p['shares'],
            'avg_cost': _p['avg_cost'],
            'buy_price': _p.get('buy_price', _p['avg_cost']),
            'name': _p['name'],
            'buy_count': 1,
            'last_buy_date_idx': _lbd_idx,
            'reduce_stage': 0,
            'peak_since_entry': _p.get('peak_since_entry', _p['avg_cost']),
        }

    # 確保所有持倉 ticker 都在 stock pool 中
    _pool_tickers = set(t for t, _ in all_stocks)
    for _t in _engine_init_pos:
        if _t not in _pool_tickers:
            all_stocks.append((_t, _engine_init_pos[_t]['name']))

    # 跑引擎多天 (從 start_date 開始, 累積 portfolio panic / daily returns 等狀態)
    # 這樣 T+1 訊號才能與 Mode 1A 回測一致
    _engine_start = start_date
    _engine_end = end_date

    # 產業參數
    _block_b_override = config_override
    if use_per_industry and per_industry_config:
        # 多產業各自參數 — 合併為單一 config (取第一個有的)
        for _ind in selected_industries:
            if _ind in per_industry_config:
                _block_b_override = per_industry_config[_ind]
                break

    _engine_result = run_group_backtest(
        stock_list=all_stocks,
        start_date=_engine_start,
        end_date=_engine_end,
        budget_per_trade=budget,
        market_map=market_map,
        exec_mode='next_open',
        initial_capital=initial_capital,
        preloaded_data=stock_data,
        initial_positions=_engine_init_pos,
        config_override=_block_b_override,
    )

    if _engine_result is None:
        print(f"\n   ❌ 引擎執行失敗")
        return {'portfolio': _portfolio, 'trades': _trade_history}

    # 提取 pending (T+1 動作)
    _pending = _engine_result.get('_pending', {})
    _trade_log = _engine_result.get('trade_log', [])
    _raw_positions = _engine_result.get('_raw_positions', {})

    _slippage = _dc_rp.get('slippage_pct', 0.3) / 100

    _eng_sells = []
    _eng_buys = []

    for _t, order in _pending.items():
        action = order['action']
        reason = order.get('reason', '')

        # 賣出訊號: 只顯示用戶實際持有的股票 (引擎多日跑可能自行買賣, 過濾掉)
        if action in ('sell', 'reduce') and _t not in _portfolio:
            continue

        pos = _raw_positions.get(_t, {})
        name = pos.get('name', _t)
        # 使用用戶真實部位的 avg_cost/shares (優先於引擎內部計算值)
        _user_pos = _portfolio.get(_t, {})
        if _user_pos:
            _disp_avg = _user_pos.get('avg_cost', pos.get('avg_cost', 0))
            _disp_shares = _user_pos.get('shares', pos.get('shares', 0))
            _disp_name = _user_pos.get('name', name)
        else:
            _disp_avg = pos.get('avg_cost', 0)
            _disp_shares = pos.get('shares', 0)
            _disp_name = name

        if action in ('sell', 'reduce'):
            _cp = 0
            if _t in stock_data:
                sdf = stock_data[_t]['df']
                _vd = sdf.index[sdf.index <= pd.Timestamp(end_date)]
                if len(_vd) > 0:
                    _cp = float(sdf.loc[_vd[-1], 'Close'])
            _pnl_pct = 0
            if _disp_avg > 0 and _cp > 0:
                _pnl = calculate_net_pnl(_t, _disp_avg, _cp, _disp_shares)
                _pnl_pct = _pnl['net_pnl_pct']
            _eng_sells.append({
                'ticker': _t, 'name': _disp_name, 'shares': _disp_shares,
                'avg_cost': _disp_avg, 'close': _cp, 'pnl_pct': _pnl_pct,
                'reason': reason, 'action': action,
            })
        elif action == 'buy':
            _is_add = _t in _portfolio  # 用戶實際部位中已持有 = 加碼
            _cp = 0
            if _t in stock_data:
                sdf = stock_data[_t]['df']
                _vd = sdf.index[sdf.index <= pd.Timestamp(end_date)]
                if len(_vd) > 0:
                    _cp = float(sdf.loc[_vd[-1], 'Close'])
            _sim_price = _cp * (1 + _slippage) if _cp > 0 else 0
            _shares = int(budget / _sim_price) if _sim_price > 0 else 0
            _eng_buys.append({
                'ticker': _t, 'name': _disp_name, 'close': _cp,
                'sim_price': _sim_price, 'shares_to_buy': _shares,
                'reason': reason, 'is_add': _is_add,
                'existing_shares': _portfolio.get(_t, {}).get('shares', 0),
                'existing_avg_cost': _portfolio.get(_t, {}).get('avg_cost', 0),
            })

    # --- 輸出 T+1 訊號 ---
    if _eng_sells:
        print(f"\n   🔴 賣出/減碼訊號 ({len(_eng_sells)} 檔):")
        for e in _eng_sells:
            _act = '賣出' if e['action'] == 'sell' else '減碼'
            print(f"      {_act} {e['ticker']} {e['name']} {e['shares']}股 "
                  f"@ ${e['avg_cost']:.1f} → ${e['close']:.1f} ({e['pnl_pct']:+.1f}%)")
            print(f"         原因: {e['reason']}")

    if _eng_buys:
        print(f"\n   🟢 買入/加碼訊號 ({len(_eng_buys)} 檔):")
        for e in _eng_buys:
            _tag = '加碼' if e['is_add'] else '新買入'
            _sp = e.get('sim_price', e['close'])
            _cost = e['shares_to_buy'] * _sp if _sp > 0 else 0
            print(f"      [{_tag}] {e['ticker']} {e['name']} "
                  f"| {e['shares_to_buy']}股 × ${_sp:.1f} ≈ ${_cost:,.0f}")
            if e['is_add']:
                print(f"         已持 {e['existing_shares']}股 @ ${e['existing_avg_cost']:.1f}")
            print(f"         訊號: {e['reason']}")

    # 持有 (基於用戶實際 end_date 部位)
    _held_tickers = set(_portfolio.keys()) - set(e['ticker'] for e in _eng_sells) - set(e['ticker'] for e in _eng_buys)
    if _held_tickers:
        _held_names = [f"{_portfolio[t]['name']}({_portfolio[t]['shares']}股)" for t in sorted(_held_tickers)]
        print(f"\n   ⚪ 持有 ({len(_held_tickers)} 檔): {', '.join(_held_names)}")

    if not _eng_sells and not _eng_buys:
        print(f"\n   ⏸️  引擎無操作建議 (維持現狀)")

    # (各 CSV 已在逐日迴圈中原地更新, 不需額外輸出)

    return {
        'portfolio': _portfolio,
        'trades': _trade_history,
        'daily_mv': _daily_mv,
        'engine_result': _engine_result,
        'sells': _eng_sells,
        'buys': _eng_buys,
    }


# ==========================================
# 🖨️ 印出 Group 回測報告
# ==========================================
def print_group_report(result, industries, start_date, end_date, budget):
    if result is None:
        print("❌ 無結果")
        return

    _em = result.get('exec_mode', 'next_open')
    mode_label = "T+1 開盤" if _em == 'next_open' else ("收盤+開盤雙買" if _em == 'close_open' else "當日收盤")
    ind_label = " + ".join(industries) if len(industries) <= 3 else f"{len(industries)} 個產業"

    slip_pct = result.get('slippage_pct', 0.0)
    slip_label = f" | 滑價 {slip_pct}%" if slip_pct > 0 else ""

    print(f"\n{'='*100}")
    print(f"📊 Group Scanner 回測報告 [{mode_label}]")
    print(f"   產業: {ind_label}")
    print(f"   區間: {start_date} ~ {end_date} | 每次 ${budget:,}{slip_label}")
    print(f"{'='*100}")

    # --- 交易明細 ---
    log = result['trade_log']
    if log:
        print(f"\n📋 交易明細 ({len(log)} 筆):")
        print(f"{'日期':<12} {'代號':<10} {'名稱':<8} {'動作':<5} {'價格':<8} {'股數':<7} "
              f"{'費用':<7} {'損益($)':<10} {'報酬率':<8}")
        print("-" * 105)

        # 統計每檔的交易
        stock_trades = {}
        for entry in log:
            ticker = entry['ticker']
            if ticker not in stock_trades:
                stock_trades[ticker] = {'name': entry['name'], 'buys': 0, 'sells': 0, 'profit': 0}

            profit_str = f"{int(entry['profit']):+,}" if entry['profit'] is not None else ""
            roi_str = f"{entry['roi']:+.1f}%" if entry['roi'] is not None else ""

            print(f"{entry['date']:<12} {entry['ticker']:<10} {entry['name']:<8} "
                  f"{entry['type']:<5} {entry['price']:<8.1f} {entry['shares']:<7} "
                  f"{int(entry['fee']):<7} {profit_str:<10} {roi_str:<8}")

            if entry['type'] in ['BUY', 'ADD']:
                stock_trades[ticker]['buys'] += 1
            elif entry['type'] == 'SELL':
                stock_trades[ticker]['sells'] += 1
                if entry['profit'] is not None:
                    stock_trades[ticker]['profit'] += entry['profit']

        # --- 各股統計 ---
        print(f"\n📈 各股交易統計:")
        print(f"{'代號':<10} {'名稱':<8} {'買入次':<7} {'賣出次':<7} {'已實現損益':>12}")
        print("-" * 55)
        for ticker, st in sorted(stock_trades.items(), key=lambda x: x[1]['profit'], reverse=True):
            emoji = "🟢" if st['profit'] > 0 else "🔴" if st['profit'] < 0 else "⚪"
            print(f"{emoji} {ticker:<8} {st['name']:<8} {st['buys']:<7} {st['sells']:<7} "
                  f"{int(st['profit']):>+12,}")
    else:
        print("\n📋 回測期間無任何交易")

    # --- 未平倉 ---
    if result['open_positions']:
        print(f"\n📦 未平倉部位 ({len(result['open_positions'])} 檔):")
        print(f"{'代號':<10} {'名稱':<8} {'股數':<7} {'均價':<8} {'現價':<8} {'未實現':>10} {'買入次':<6}")
        print("-" * 65)
        for op in sorted(result['open_positions'], key=lambda x: x['unrealized'], reverse=True):
            emoji = "🟢" if op['unrealized'] > 0 else "🔴"
            print(f"{emoji} {op['ticker']:<8} {op['name']:<8} {op['shares']:<7} "
                  f"{op['avg_cost']:<8.1f} {op['current_price']:<8.1f} "
                  f"{int(op['unrealized']):>+10,} {op['buy_count']:<6}")

    # --- 總績效 ---
    print(f"\n{'-'*100}")
    print(f"💵 已實現損益: ${int(result['realized']):+,}")
    print(f"📦 未實現損益: ${int(result['unrealized']):+,}")
    print(f"💰 總淨損益:   ${int(result['total_pnl']):+,}")
    print(f"💸 累計稅費:   ${int(result['fees']):,}")
    print(f"📉 最大回撤:   ${int(result['max_drawdown']):,}")
    print(f"🏦 最大佔用本金: ${int(result['max_capital']):,}")
    print(f"📈 ROI (佔用本金): {result['roi']:.2f}%")

    # --- V4.1: 帳戶結算 ---
    if 'initial_capital' in result:
        print(f"\n{'─'*50}")
        print(f"💼 帳戶結算 (算法A: 實際投入報酬率)")
        print(f"   初始資金:       ${int(result['initial_capital']):>12,}")
        print(f"   結算日現金:     ${int(result['final_cash']):>12,}")
        print(f"   持股市值:       ${int(result['final_stock_value']):>12,}")
        print(f"   持股淨變現值:   ${int(result['final_stock_value_net']):>12,}")
        print(f"   {'─'*40}")
        print(f"   帳戶總值:       ${int(result['final_total_value']):>12,}")
        total_ret = result['total_return_pct']
        ret_emoji = "📈" if total_ret >= 0 else "📉"
        print(f"   {ret_emoji} 總報酬率:       {total_ret:>+12.2f}%")
        if result.get('max_overdraft', 0) > 0:
            print(f"   ⚠️ 最大透支:       ${int(result['max_overdraft']):>12,}")

        # V4.2: 進階績效指標
        print(f"\n{'─'*50}")
        print(f"📊 風險調整績效 (回測 {result.get('backtest_days', 0)} 天 ≈ {result.get('backtest_years', 0):.1f} 年)")
        cagr = result.get('cagr', 0)
        mdd_pct = result.get('mdd_pct', 0)
        sharpe = result.get('sharpe_ratio', 0)
        calmar = result.get('calmar_ratio', 0)
        cagr_emoji = "📈" if cagr >= 0 else "📉"
        print(f"   {cagr_emoji} 年化報酬 (CAGR):  {cagr:>+10.2f}%")
        print(f"   📉 最大回撤 (MDD%):  {mdd_pct:>10.2f}%")
        sharpe_emoji = "🟢" if sharpe >= 1.0 else "🟡" if sharpe >= 0.5 else "🔴"
        print(f"   {sharpe_emoji} Sharpe Ratio:      {sharpe:>10.2f}    ", end="")
        if sharpe >= 1.5:
            print("(優秀)")
        elif sharpe >= 1.0:
            print("(良好)")
        elif sharpe >= 0.5:
            print("(普通)")
        else:
            print("(偏差)")
        calmar_emoji = "🟢" if calmar >= 1.0 else "🟡" if calmar >= 0.5 else "🔴"
        print(f"   {calmar_emoji} Calmar Ratio:      {calmar:>10.2f}    ", end="")
        if calmar >= 2.0:
            print("(優秀)")
        elif calmar >= 1.0:
            print("(良好)")
        elif calmar >= 0.5:
            print("(普通)")
        else:
            print("(偏差)")

    if result['trades'] > 0:
        avg_pnl = result['realized'] / result['trades']
        print(f"\n🎯 交易統計:")
        print(f"   完成來回: {result['trades']} 次")
        print(f"   勝率: {result['win_rate']:.0f}% ({result['wins']}勝 {result['losses']}敗)")
        print(f"   平均損益: ${int(avg_pnl):+,}/筆")

    # --- 權益曲線摘要 ---
    ec = result['equity_curve']
    if ec:
        max_pos = max(e['positions'] for e in ec)
        avg_pos = np.mean([e['positions'] for e in ec])
        print(f"\n📊 部位統計:")
        print(f"   同時最多持有: {max_pos} 檔")
        print(f"   平均持有: {avg_pos:.1f} 檔")

    print(f"{'='*100}")

    # --- 持有天數分布 ---
    print_holding_days_distribution(result)


# ==========================================
# 📊 持有天數分布統計 (V4.12)
# ==========================================
def analyze_holding_days(result):
    """
    從交易日誌分析每筆完整來回交易的持有天數分布。

    做法:
      1. 找出每檔股票的「首次 BUY 日期」和「最終 SELL 日期」
      2. 計算持有天數 (交易日)
      3. 統計分布 (含未平倉的持有中天數)

    Args:
        result: run_group_backtest() 的回傳值

    Returns:
        dict with:
            'closed_trades':    已平倉交易 [{ticker, name, buy_date, sell_date,
                                             hold_days, profit, roi, sell_reason}, ...]
            'open_trades':      未平倉持有 [{ticker, name, buy_date, hold_days}, ...]
            'distribution':     持有天數分布 {bucket_label: count}
            'stats':            統計摘要 {mean, median, min, max, std}
            'by_outcome':       依盈虧分組 {'winners': {...stats}, 'losers': {...stats}}
    """
    if result is None:
        return None

    log = result.get('trade_log', [])
    if not log:
        return None

    # --- 1. 重建每檔股票的交易歷程 ---
    # 追蹤每檔的首次買入日期 (BUY, 不算 ADD)
    # 遇到 SELL (total_shares=0) 時結算一個完整來回
    active_buys = {}   # ticker → first_buy_date
    closed_trades = []

    for entry in log:
        ticker = entry['ticker']
        trade_type = entry['type']
        date_str = entry['date']

        if trade_type == 'BUY' and ticker not in active_buys:
            # 首次建倉 (或前一筆已結清後重新買入)
            active_buys[ticker] = date_str

        elif trade_type == 'SELL' and entry.get('total_shares', 0) == 0:
            # 完整賣出 → 結算一個來回
            buy_date = active_buys.pop(ticker, date_str)
            buy_dt = pd.Timestamp(buy_date)
            sell_dt = pd.Timestamp(date_str)
            # 用交易日計算 (簡化: 用日曆天 / 1.4 估算交易日, 或直接用日曆天)
            cal_days = (sell_dt - buy_dt).days
            # 取交易日更精確: 約 5/7 的日曆天
            hold_days = max(1, cal_days)

            closed_trades.append({
                'ticker': ticker,
                'name': entry.get('name', ticker),
                'buy_date': buy_date,
                'sell_date': date_str,
                'hold_days': hold_days,
                'profit': entry.get('profit', 0),
                'roi': entry.get('roi', 0),
                'sell_reason': entry.get('note', ''),
            })

        elif trade_type == 'REDUCE' and entry.get('total_shares', 0) == 0:
            # 減碼到 0 股也算完整結清
            buy_date = active_buys.pop(ticker, date_str)
            buy_dt = pd.Timestamp(buy_date)
            sell_dt = pd.Timestamp(date_str)
            hold_days = max(1, (sell_dt - buy_dt).days)

            closed_trades.append({
                'ticker': ticker,
                'name': entry.get('name', ticker),
                'buy_date': buy_date,
                'sell_date': date_str,
                'hold_days': hold_days,
                'profit': entry.get('profit', 0),
                'roi': entry.get('roi', 0),
                'sell_reason': entry.get('note', ''),
            })

    # --- 2. 未平倉持有天數 ---
    open_trades = []
    last_date_str = log[-1]['date'] if log else ''
    for ticker, buy_date in active_buys.items():
        buy_dt = pd.Timestamp(buy_date)
        last_dt = pd.Timestamp(last_date_str) if last_date_str else buy_dt
        hold_days = max(1, (last_dt - buy_dt).days)
        # 找名稱
        name = ticker
        for entry in log:
            if entry['ticker'] == ticker:
                name = entry.get('name', ticker)
                break
        open_trades.append({
            'ticker': ticker,
            'name': name,
            'buy_date': buy_date,
            'hold_days': hold_days,
        })

    # --- 3. 分布統計 ---
    all_hold_days = [t['hold_days'] for t in closed_trades]

    if not all_hold_days:
        return {
            'closed_trades': closed_trades,
            'open_trades': open_trades,
            'distribution': {},
            'stats': {},
            'by_outcome': {},
        }

    # 分桶: 1-3天, 4-7天, 8-14天, 15-30天, 31-60天, 60+天
    buckets = [
        ('1-3 天',    1, 3),
        ('4-7 天',    4, 7),
        ('8-14 天',   8, 14),
        ('15-30 天',  15, 30),
        ('31-60 天',  31, 60),
        ('61-90 天',  61, 90),
        ('91-180 天', 91, 180),
        ('180+ 天',   181, 99999),
    ]

    distribution = {}
    for label, lo, hi in buckets:
        count = sum(1 for d in all_hold_days if lo <= d <= hi)
        if count > 0:
            distribution[label] = count

    # 統計摘要
    arr = np.array(all_hold_days)
    stats = {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'min': int(np.min(arr)),
        'max': int(np.max(arr)),
        'std': float(np.std(arr)),
        'total': len(arr),
    }

    # --- 4. 依盈虧分組 ---
    winner_days = [t['hold_days'] for t in closed_trades if (t.get('profit') or 0) > 0]
    loser_days = [t['hold_days'] for t in closed_trades if (t.get('profit') or 0) <= 0]

    by_outcome = {}
    if winner_days:
        w_arr = np.array(winner_days)
        by_outcome['winners'] = {
            'count': len(w_arr),
            'mean': float(np.mean(w_arr)),
            'median': float(np.median(w_arr)),
            'min': int(np.min(w_arr)),
            'max': int(np.max(w_arr)),
        }
    if loser_days:
        l_arr = np.array(loser_days)
        by_outcome['losers'] = {
            'count': len(l_arr),
            'mean': float(np.mean(l_arr)),
            'median': float(np.median(l_arr)),
            'min': int(np.min(l_arr)),
            'max': int(np.max(l_arr)),
        }

    return {
        'closed_trades': closed_trades,
        'open_trades': open_trades,
        'distribution': distribution,
        'stats': stats,
        'by_outcome': by_outcome,
    }


def print_holding_days_distribution(result):
    """印出持有天數分布統計"""
    analysis = analyze_holding_days(result)
    if analysis is None or not analysis.get('stats'):
        return

    stats = analysis['stats']
    dist = analysis['distribution']
    by_outcome = analysis['by_outcome']
    open_trades = analysis['open_trades']

    print(f"\n📅 持有天數分布 (已平倉 {stats['total']} 筆):")
    print(f"   {'─' * 55}")

    # 分布直方圖 (ASCII)
    if dist:
        max_count = max(dist.values())
        bar_width = 30

        for label, count in dist.items():
            bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
            bar = '█' * bar_len
            pct = count / stats['total'] * 100
            print(f"   {label:>10s} │{bar:<{bar_width}s}│ {count:>4d} 筆 ({pct:>5.1f}%)")

    # 統計摘要
    print(f"   {'─' * 55}")
    print(f"   平均: {stats['mean']:.1f} 天 | 中位數: {stats['median']:.0f} 天 | "
          f"最短: {stats['min']} 天 | 最長: {stats['max']} 天")

    # 盈虧分組比較
    if 'winners' in by_outcome and 'losers' in by_outcome:
        w = by_outcome['winners']
        l = by_outcome['losers']
        print(f"\n   📊 盈虧比較:")
        print(f"   {'':>10s}  {'筆數':>6s}  {'平均':>8s}  {'中位數':>8s}  {'最短':>6s}  {'最長':>6s}")
        print(f"   {'🟢 獲利':>10s}  {w['count']:>6d}  {w['mean']:>7.1f}天  {w['median']:>7.0f}天  {w['min']:>5d}天  {w['max']:>5d}天")
        print(f"   {'🔴 虧損':>10s}  {l['count']:>6d}  {l['mean']:>7.1f}天  {l['median']:>7.0f}天  {l['min']:>5d}天  {l['max']:>5d}天")

        # 判讀
        if w['mean'] > l['mean'] * 1.3:
            print(f"   💡 獲利交易平均持有較久 → 策略有「讓利潤奔跑」的特性 ✅")
        elif l['mean'] > w['mean'] * 1.3:
            print(f"   ⚠️ 虧損交易平均持有較久 → 可能有「抱虧損不放」的問題")
        else:
            print(f"   💡 盈虧持有天數接近 → 進出場節奏一致")

    # 未平倉
    if open_trades:
        print(f"\n   📦 未平倉 ({len(open_trades)} 檔, 持有中):")
        for ot in sorted(open_trades, key=lambda x: -x['hold_days']):
            print(f"      {ot['ticker']} {ot['name']}: 已持有 {ot['hold_days']} 天 (自 {ot['buy_date']})")


# ==========================================
# 🧪 Group Ablation 設定
#
# 策略參數 + 上車限制 組合測試
# key → (config_dict, description)
# ==========================================
GROUP_ABLATION_CONFIGS = {
    # --- 基準線 (V3.9: V3.8 + 殭屍清除 + 品質排序 + 換股) ---
    'baseline':       ({},                                     '✅ V3.9 完整版 (基準線)'),

    # --- V3.6 舊版基準 (用於 AB 對照) ---
    'v36_baseline':   ({'market_filter_mode': 'strict',
                        'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                        'tier_b_net': 10, 'tier_b_drawdown': 0.5},
                                                               '🔄 V3.6 舊版 (strict + S1原始門檻)'),

    # --- 策略模組 ablation ---
    'no_filter':      ({'enable_market_filter': False},         '❌ 關閉大盤濾網 (F1+F2+F3)'),
    'no_tiered':      ({'enable_tiered_stops': False},          '❌ 關閉階梯停利 (只留 S2+S3)'),
    'no_breakout':    ({'enable_breakout': False},              '❌ 關閉突破濾網 (B6)'),
    'no_fish_tail':   ({'enable_fish_tail': False},             '❌ 關閉魚尾偵測 (B5)'),
    'no_shooting':    ({'enable_shooting_star': False},         '❌ 關閉避雷針 (B7)'),
    'no_bias':        ({'enable_bias_limit': False},            '❌ 關閉乖離上限 (B4)'),
    'no_defense':     ({'enable_bias_limit': False,
                        'enable_fish_tail': False,
                        'enable_shooting_star': False},         '❌ 關閉所有防禦 (B4+B5+B7)'),

    # --- 上車限制 ablation ---
    'pos5_buy2':      ({'max_positions': 5,
                        'max_new_buy_per_day': 2},             '🔧 保守: 最多5檔/日新建2'),
    'pos15_buy5':     ({'max_positions': 15,
                        'max_new_buy_per_day': 5},             '🔧 積極: 最多15檔/日新建5'),
    'pos99_buy99':    ({'max_positions': 99,
                        'max_new_buy_per_day': 99},            '🔧 無限制: 有訊號就買'),

    # --- V4.3 倉位網格搜索 (找最佳 pos × buy × swap 組合) ---
    'p8_b3_s1':       ({'max_positions': 8,  'max_new_buy_per_day': 3, 'max_swap_per_day': 1},
                                                               '🔍 8檔/日買3/換1'),
    'p8_b3_s2':       ({'max_positions': 8,  'max_new_buy_per_day': 3, 'max_swap_per_day': 2},
                                                               '🔍 8檔/日買3/換2'),
    'p10_b3_s1':      ({'max_positions': 10, 'max_new_buy_per_day': 3, 'max_swap_per_day': 1},
                                                               '🔍 10檔/日買3/換1'),
    'p10_b3_s2':      ({'max_positions': 10, 'max_new_buy_per_day': 3, 'max_swap_per_day': 2},
                                                               '🔍 10檔/日買3/換2'),
    'p10_b5_s2':      ({'max_positions': 10, 'max_new_buy_per_day': 5, 'max_swap_per_day': 2},
                                                               '🔍 10檔/日買5/換2'),
    'p12_b3_s2':      ({'max_positions': 12, 'max_new_buy_per_day': 3, 'max_swap_per_day': 2},
                                                               '🔍 12檔/日買3/換2'),
    'p12_b5_s2':      ({'max_positions': 12, 'max_new_buy_per_day': 5, 'max_swap_per_day': 2},
                                                               '🔍 12檔/日買5/換2'),
    'p12_b5_s3':      ({'max_positions': 12, 'max_new_buy_per_day': 5, 'max_swap_per_day': 3},
                                                               '🔍 12檔/日買5/換3'),
    'p15_b3_s2':      ({'max_positions': 15, 'max_new_buy_per_day': 3, 'max_swap_per_day': 2},
                                                               '🔍 15檔/日買3/換2'),
    'p15_b5_s3':      ({'max_positions': 15, 'max_new_buy_per_day': 5, 'max_swap_per_day': 3},
                                                               '🔍 15檔/日買5/換3'),
    'p20_b5_s3':      ({'max_positions': 20, 'max_new_buy_per_day': 5, 'max_swap_per_day': 3},
                                                               '🔍 20檔/日買5/換3'),
    'p20_b8_s3':      ({'max_positions': 20, 'max_new_buy_per_day': 8, 'max_swap_per_day': 3},
                                                               '🔍 20檔/日買8/換3'),

    # --- 排序方式 ablation ---
    'sort_volume':    ({'entry_sort_by': 'volume'},            '🔧 排序: 量比高優先 (預設乖離低優先)'),

    # --- 參數調整 ablation ---
    'breakout_5d':    ({'breakout_lookback': 5},               '🔧 突破回看 5天 (預設10)'),
    'breakout_20d':   ({'breakout_lookback': 20},              '🔧 突破回看 20天 (更嚴)'),
    'fish_3d':        ({'fish_tail_lookback': 3},              '🔧 魚尾回看 3天 (預設5, 更寬鬆)'),
    'fish_7d':        ({'fish_tail_lookback': 7},              '🔧 魚尾回看 7天 (更嚴格)'),
    'shooting_2x':    ({'shooting_star_ratio': 2.0},           '🔧 避雷針 2倍 (預設3, 更敏感)'),

    # --- B4 乖離上限細化 ablation (V4.3: 找最佳門檻) ---
    'bias_15':        ({'bias_limit_bull': 15, 'bias_limit_neutral': 15, 'bias_limit_bear': 15},
                                                               '🔧 B4 乖離上限 15% (V4.3前舊值)'),
    'bias_25':        ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25},
                                                               '🔧 B4 乖離上限 25%'),
    'bias_30':        ({'bias_limit_bull': 30, 'bias_limit_neutral': 30, 'bias_limit_bear': 30},
                                                               '🔧 B4 乖離上限 30%'),

    # --- F1 大盤濾網嚴格度 ablation ---
    'f1_strict':      ({'market_filter_mode': 'strict'},       '🔧 F1 嚴格 (bear+weak+crash全擋, V4.11以前default)'),
    'f1_moderate':    ({'market_filter_mode': 'moderate'},      '🔧 F1 中等 (bear+crash擋, weak放行)'),
    'f1_relaxed':     ({'market_filter_mode': 'relaxed'},      '🔧 F1 寬鬆 (只擋bear, =V4.12 default)'),

    # --- S1 階梯停利參數 ablation (V3.8 新增) ---
    's1_tight':       ({'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                        'tier_b_net': 10, 'tier_b_drawdown': 0.5},
                                                               '🔧 S1 原始 (TierA=20/0.98, TierB=10/0.5)'),
    's1_loose':       ({'tier_a_net': 40, 'tier_a_ma_buf': 0.95,
                        'tier_b_net': 20, 'tier_b_drawdown': 0.7},
                                                               '🔧 S1 超寬 (=V4.12 default, TierA=40/0.95)'),

    # --- V4.12 舊版 baseline 對照 ---
    'v411_baseline':  ({'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                        'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                        'market_filter_mode': 'strict',
                        'max_new_buy_per_day': 3, 'max_swap_per_day': 1},
                                                               '🔄 V4.11 舊baseline (S1嚴/F1strict/buy3/swap1)'),
    'v412_baseline':  ({'max_new_buy_per_day': 3, 'max_swap_per_day': 1},
                                                               '🔄 V4.12 舊baseline (buy3/swap1)'),

    # --- V3.9 倉位品質管理 ablation ---
    'no_zombie':      ({'enable_zombie_cleanup': False},         '❌ 關閉殭屍清除 (S4)'),
    'no_swap':        ({'enable_position_swap': False},          '❌ 關閉換股機制'),
    'sort_bias':      ({'entry_sort_by': 'bias'},               '🔧 排序: 乖離低優先 (V3.8舊版)'),
    'no_quality':     ({'enable_zombie_cleanup': False,
                        'enable_position_swap': False,
                        'entry_sort_by': 'bias'},               '❌ 關閉全部V3.9倉位管理'),
    'zombie_10d':     ({'zombie_hold_days': 10},                '🔧 殭屍門檻 10天 (預設15)'),
    'zombie_20d':     ({'zombie_hold_days': 20},                '🔧 殭屍門檻 20天 (更寬)'),
    'swap_low':       ({'swap_score_margin': 0.1},              '🔧 換股門檻 0.1 (預設0.5, 更易換)'),
    'swap_high':      ({'swap_score_margin': 0.8},              '🔧 換股門檻 0.8 (預設0.5, 更難換)'),
    'swap_2d':        ({'max_swap_per_day': 2},                 '🔧 每日換股上限 2檔 (預設1)'),
    'swap_3d':        ({'max_swap_per_day': 3},                 '🔧 每日換股上限 3檔'),

    # --- V3.8 基準 (無V3.9功能, 用於AB對照) ---
    'v38_baseline':   ({'enable_zombie_cleanup': False,
                        'enable_position_swap': False,
                        'entry_sort_by': 'bias'},               '🔄 V3.8 舊版 (無殭屍/換股/品質分數)'),

    # --- V4.0 動態防禦 ablation (方案3: 只收緊空頭B4) ---
    'static_b4b7':    ({'bias_limit_bull': 15, 'bias_limit_neutral': 15, 'bias_limit_bear': 15},
                                                               '🔧 B4 全靜態15% (還原V3.9, 對照空頭收緊)'),

    # --- V4.4 交叉 ablation: B7 × swap_margin (bias 固定20%, pos 固定10/3/1) ---
    # 目的: 找出 B7 On/Off + swap 0.3/0.5 的最佳「組合」
    # (之前個別 ablation 各自最佳, 但合在一起可能有交互效應)
    'b7on_sw03':      ({'enable_shooting_star': True,  'swap_score_margin': 0.3},
                                                               '🔬 B7開+swap0.3 (舊版)'),
    'b7on_sw05':      ({'enable_shooting_star': True,  'swap_score_margin': 0.5},
                                                               '🔬 B7開+swap0.5'),
    'b7off_sw03':     ({'enable_shooting_star': False, 'swap_score_margin': 0.3},
                                                               '🔬 B7關+swap0.3'),
    'b7off_sw05':     ({'enable_shooting_star': False, 'swap_score_margin': 0.5},
                                                               '🔬 B7關+swap0.5 (目前default)'),

    # --- V4.5 交叉 ablation: bias_limit × swap_margin (找最佳組合) ---
    'bi25_sw05':      ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                        'swap_score_margin': 0.5},
                                                               '🔬 bias25+swap0.5 (只改bias)'),
    'bi25_sw08':      ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                        'swap_score_margin': 0.8},
                                                               '🔬 bias25+swap0.8 (候選新default)'),
    'bi20_sw08':      ({'bias_limit_bull': 20, 'bias_limit_neutral': 20, 'bias_limit_bear': 20,
                        'swap_score_margin': 0.8},
                                                               '🔬 bias20+swap0.8 (只改swap)'),
    'bi30_sw08':      ({'bias_limit_bull': 30, 'bias_limit_neutral': 30, 'bias_limit_bear': 30,
                        'swap_score_margin': 0.8},
                                                               '🔬 bias30+swap0.8 (激進)'),
    # --- V4.5 動態 bias: bull/neutral 放寬, bear 收緊 ---
    'bi_dyn_25_20':   ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 20,
                        'swap_score_margin': 0.8},
                                                               '🔬 動態bias(牛25/空20)+swap0.8'),
    'bi_dyn_25_15':   ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 15,
                        'swap_score_margin': 0.8},
                                                               '🔬 動態bias(牛25/空15)+swap0.8'),
    'bi_dyn_30_20':   ({'bias_limit_bull': 30, 'bias_limit_neutral': 25, 'bias_limit_bear': 20,
                        'swap_score_margin': 0.8},
                                                               '🔬 動態bias(牛30/中25/空20)+swap0.8'),

    # === V4.6 Ablation A: 動態 bias 精細化 ===
    # 牛市 bias 30 確認有效，測不同的中性/空頭組合
    'dyn_30_25_15':   ({'bias_limit_bull': 30, 'bias_limit_neutral': 25, 'bias_limit_bear': 15,
                        'swap_score_margin': 0.8},
                                                               '🔬A 動態(牛30/中25/空15)'),
    'dyn_30_25_10':   ({'bias_limit_bull': 30, 'bias_limit_neutral': 25, 'bias_limit_bear': 10,
                        'swap_score_margin': 0.8},
                                                               '🔬A 動態(牛30/中25/空10)'),
    'dyn_30_30_15':   ({'bias_limit_bull': 30, 'bias_limit_neutral': 30, 'bias_limit_bear': 15,
                        'swap_score_margin': 0.8},
                                                               '🔬A 動態(牛30/中30/空15)'),
    'dyn_30_30_20':   ({'bias_limit_bull': 30, 'bias_limit_neutral': 30, 'bias_limit_bear': 20,
                        'swap_score_margin': 0.8},
                                                               '🔬A 動態(牛30/中30/空20)'),
    # 對照: bi_dyn_30_20 = 牛30/中25/空20 已有

    # === V4.6 Ablation A2: zombie 搭配 swap 0.8 ===
    # swap 0.8 換股變少 → 殭屍清除可能需要更積極
    'dyn30_z10':      ({'bias_limit_bull': 30, 'bias_limit_neutral': 25, 'bias_limit_bear': 20,
                        'swap_score_margin': 0.8, 'zombie_hold_days': 10},
                                                               '🔬A dyn30_20+zombie10天(更積極)'),
    'dyn30_z20':      ({'bias_limit_bull': 30, 'bias_limit_neutral': 25, 'bias_limit_bear': 20,
                        'swap_score_margin': 0.8, 'zombie_hold_days': 20},
                                                               '🔬A dyn30_20+zombie20天(更寬鬆)'),
    'bi25_z10':       ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                        'swap_score_margin': 0.8, 'zombie_hold_days': 10},
                                                               '🔬A bi25_sw08+zombie10天'),
    'bi25_z20':       ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                        'swap_score_margin': 0.8, 'zombie_hold_days': 20},
                                                               '🔬A bi25_sw08+zombie20天'),

    # === V4.6 Ablation B: swap 0.8 高檔修正弱點修補 ===
    # swap 0.8 在高檔修正虧最多 → 搭配 zombie 收緊 + swap_per_day 放寬
    'dyn30_z10_sw2':  ({'bias_limit_bull': 30, 'bias_limit_neutral': 25, 'bias_limit_bear': 20,
                        'swap_score_margin': 0.8, 'zombie_hold_days': 10,
                        'max_swap_per_day': 2},
                                                               '🔬B dyn30+zombie10+每日換2檔'),
    'bi25_z10_sw2':   ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                        'swap_score_margin': 0.8, 'zombie_hold_days': 10,
                        'max_swap_per_day': 2},
                                                               '🔬B bi25+zombie10+每日換2檔'),
    'dyn30_zr3':      ({'bias_limit_bull': 30, 'bias_limit_neutral': 25, 'bias_limit_bear': 20,
                        'swap_score_margin': 0.8, 'zombie_net_range': 3.0},
                                                               '🔬B dyn30+殭屍範圍±3%(更敏感)'),
    'bi25_zr3':       ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                        'swap_score_margin': 0.8, 'zombie_net_range': 3.0},
                                                               '🔬B bi25+殭屍範圍±3%(更敏感)'),

    # === V4.7 Ablation: 持倉/換倉/加碼金額 網格搜索 ===
    # --- 持倉數量 (max_positions), max_new_buy_per_day 按比例配 ---
    'pos10':          ({'max_positions': 10, 'max_new_buy_per_day': 3},
                                                               '🔍 持倉10檔/日買3 (default)'),
    'pos15':          ({'max_positions': 15, 'max_new_buy_per_day': 3},
                                                               '🔍 持倉15檔/日買3'),
    'pos20':          ({'max_positions': 20, 'max_new_buy_per_day': 4},
                                                               '🔍 持倉20檔/日買4'),
    'pos25':          ({'max_positions': 25, 'max_new_buy_per_day': 5},
                                                               '🔍 持倉25檔/日買5'),
    'pos30':          ({'max_positions': 30, 'max_new_buy_per_day': 6},
                                                               '🔍 持倉30檔/日買6'),

    # --- 每日換倉數量 (max_swap_per_day) ---
    'swap_off':       ({'enable_position_swap': False},         '🔍 不換倉'),
    'swap_1d':        ({'max_swap_per_day': 1},                 '🔍 每日換1檔 (default)'),
    'swap_2d':        ({'max_swap_per_day': 2},                 '🔍 每日換2檔'),
    'swap_3d_v47':    ({'max_swap_per_day': 3},                 '🔍 每日換3檔'),
    'swap_5d':        ({'max_swap_per_day': 5},                 '🔍 每日換5檔'),

    # --- 每次加碼金額 (_budget 特殊key, 覆蓋 budget_per_trade) ---
    'budget_1w':      ({'_budget': 10000},                      '🔍 每次1萬'),
    'budget_2w':      ({'_budget': 20000},                      '🔍 每次2萬 (default)'),
    'budget_5w':      ({'_budget': 50000},                      '🔍 每次5萬'),
    'budget_10w':     ({'_budget': 100000},                     '🔍 每次10萬'),
    'budget_20w':     ({'_budget': 200000},                     '🔍 每次20萬'),

    # === V4.7 Ablation C: 交叉驗證 (pos × swap × budget 最佳組合) ===
    # 第一輪結果: pos15 + swap2 + budget2w 各自最佳, 交叉確認
    'p15_sw2':        ({'max_positions': 15, 'max_new_buy_per_day': 3,
                        'max_swap_per_day': 2},
                                                               '🔬C 15檔+換2 (候選最佳)'),
    'p15_sw1':        ({'max_positions': 15, 'max_new_buy_per_day': 3,
                        'max_swap_per_day': 1},
                                                               '🔬C 15檔+換1 (對照)'),
    'p15_sw3':        ({'max_positions': 15, 'max_new_buy_per_day': 3,
                        'max_swap_per_day': 3},
                                                               '🔬C 15檔+換3 (對照)'),
    'p10_sw2':        ({'max_positions': 10, 'max_new_buy_per_day': 3,
                        'max_swap_per_day': 2},
                                                               '🔬C 10檔+換2 (對照)'),
    'p20_sw2':        ({'max_positions': 20, 'max_new_buy_per_day': 4,
                        'max_swap_per_day': 2},
                                                               '🔬C 20檔+換2 (對照)'),
    # budget 交叉 (搭配 pos15+sw2)
    'p15s2_b1w':      ({'max_positions': 15, 'max_new_buy_per_day': 3,
                        'max_swap_per_day': 2, '_budget': 10000},
                                                               '🔬C 15檔+換2+每次1萬'),
    'p15s2_b2w':      ({'max_positions': 15, 'max_new_buy_per_day': 3,
                        'max_swap_per_day': 2, '_budget': 20000},
                                                               '🔬C 15檔+換2+每次2萬'),
    'p15s2_b3w':      ({'max_positions': 15, 'max_new_buy_per_day': 3,
                        'max_swap_per_day': 2, '_budget': 30000},
                                                               '🔬C 15檔+換2+每次3萬'),
    'p15s2_b5w':      ({'max_positions': 15, 'max_new_buy_per_day': 3,
                        'max_swap_per_day': 2, '_budget': 50000},
                                                               '🔬C 15檔+換2+每次5萬'),

    # === V4.7 Ablation D: p15_sw1 最佳組合 budget 精細化 ===
    'p15s1_b15k':     ({'max_positions': 15, 'max_new_buy_per_day': 3,
                        'max_swap_per_day': 1, '_budget': 15000},
                                                               '🔬D 15檔+換1+每次1.5萬'),
    'p15s1_b20k':     ({'max_positions': 15, 'max_new_buy_per_day': 3,
                        'max_swap_per_day': 1, '_budget': 20000},
                                                               '🔬D 15檔+換1+每次2萬'),

    # === V4.8 Ablation E: 減碼 (分批停利) 測試 ===
    'reduce_off':     ({'enable_reduce': False},
                                                               '🔬E 減碼關閉(對照組)'),
    'reduce_t20r50':  ({'enable_reduce': True, 'reduce_tier1_net': 20,
                        'reduce_tier1_ratio': 0.5},
                                                               '🔬E R1=20%減50%'),
    'reduce_t15r50':  ({'enable_reduce': True, 'reduce_tier1_net': 15,
                        'reduce_tier1_ratio': 0.5},
                                                               '🔬E R1=15%減50%'),
    'reduce_t25r50':  ({'enable_reduce': True, 'reduce_tier1_net': 25,
                        'reduce_tier1_ratio': 0.5},
                                                               '🔬E R1=25%減50%'),
    'reduce_t20r33':  ({'enable_reduce': True, 'reduce_tier1_net': 20,
                        'reduce_tier1_ratio': 0.33},
                                                               '🔬E R1=20%減33%'),
    'reduce_2tier':   ({'enable_reduce': True, 'reduce_tier1_net': 20,
                        'reduce_tier1_ratio': 0.5, 'reduce_tier2_net': 40,
                        'reduce_tier2_ratio': 0.5},
                                                               '🔬E 雙層R1=20%+R2=40%'),

    # === V4.9 Ablation F: 大盤偵測門檻 (crash/panic 敏感度調整) ===
    'panic_default': ({},                                      '🔬F 預設門檻(對照組)'),
    # --- 單日跌幅門檻 (crash) ---
    'crash_d20':     ({'crash_day_threshold': -0.020},         '🔬F crash日跌-2.0%'),
    'crash_d15':     ({'crash_day_threshold': -0.015},         '🔬F crash日跌-1.5%'),
    # --- 恐慌門檻 (panic) ---
    'panic_d25':     ({'panic_day_threshold': -0.025},         '🔬F panic日跌-2.5%'),
    'panic_d20':     ({'panic_day_threshold': -0.020},         '🔬F panic日跌-2.0%'),
    'panic_3d40':    ({'panic_3d_threshold': -0.040},          '🔬F panic3日-4.0%'),
    'panic_3d35':    ({'panic_3d_threshold': -0.035},          '🔬F panic3日-3.5%'),
    # --- 組合: crash + panic 同時收緊 ---
    'sens_mid':      ({'crash_day_threshold': -0.020,
                       'panic_day_threshold': -0.025,
                       'panic_3d_threshold': -0.040},          '🔬F 中敏感(crash-2/panic-2.5/3d-4)'),
    'sens_high':     ({'crash_day_threshold': -0.015,
                       'panic_day_threshold': -0.020,
                       'panic_3d_threshold': -0.035},          '🔬F 高敏感(crash-1.5/panic-2/3d-3.5)'),

    # === V4.10 Ablation G: 持倉組合級恐慌偵測 ===
    'pp_off':        ({'enable_portfolio_panic': False},        '🔬G 組合恐慌關閉(對照)'),
    # --- 單日門檻 ---
    'pp_d3':         ({'portfolio_panic_day_pct': -3.0},       '🔬G 組合日跌-3%'),
    'pp_d4':         ({'portfolio_panic_day_pct': -4.0},       '🔬G 組合日跌-4%(預設)'),
    'pp_d5':         ({'portfolio_panic_day_pct': -5.0},       '🔬G 組合日跌-5%'),
    # --- 3日累計門檻 ---
    'pp_3d5':        ({'portfolio_panic_3d_pct': -5.0},        '🔬G 組合3日累跌-5%'),
    'pp_3d7':        ({'portfolio_panic_3d_pct': -7.0},        '🔬G 組合3日累跌-7%(預設)'),
    'pp_3d10':       ({'portfolio_panic_3d_pct': -10.0},       '🔬G 組合3日累跌-10%'),
    # --- 賣出範圍 ---
    'pp_sell_all':   ({'portfolio_panic_action': 'sell_all',
                       'portfolio_panic_day_pct': -4.0},       '🔬G 組合恐慌全賣'),
    'pp_loss_n5':    ({'portfolio_panic_loss_threshold': -5.0}, '🔬G 只賣虧>5%的倉'),
    # --- 組合: 敏感 + 寬鬆 ---
    'pp_sens':       ({'portfolio_panic_day_pct': -3.0,
                       'portfolio_panic_3d_pct': -5.0},        '🔬G 敏感(日-3%/3日-5%)'),
    'pp_relaxed':    ({'portfolio_panic_day_pct': -5.0,
                       'portfolio_panic_3d_pct': -10.0},       '🔬G 寬鬆(日-5%/3日-10%)'),
    # --- 搭配 V4.9 最佳門檻 (panic_3d35 已是預設) ---
    'pp_d3_3d5':     ({'portfolio_panic_day_pct': -3.0,
                       'portfolio_panic_3d_pct': -5.0,
                       'portfolio_panic_cooldown': 5},         '🔬G 敏感+冷卻5天'),

    # === V4.11 Ablation H: 現金管理 (禁止透支 + 加碼控制) ===
    # --- 診斷: 加碼的價值 ---
    'cash_no_add_unlimited': ({'enable_cash_check': False, 'enable_add': False},
                                                               '🔬H 禁止加碼(不限現金,診斷用)'),
    'cash_no_add':    ({'enable_add': False},                  '🔬H 禁止加碼+現金限制'),
    # --- 現金限制 ---
    'cash_unlimited': ({'enable_cash_check': False},           '🔬H 允許透支(舊邏輯,對照)'),
    'cash_strict':    ({'enable_cash_check': True,
                        'cash_reserve_pct': 0.0},              '🔬H 嚴格現金(=baseline預設)'),
    'cash_reserve_10': ({'cash_reserve_pct': 10.0},            '🔬H 保留10%資金'),
    'cash_reserve_20': ({'cash_reserve_pct': 20.0},            '🔬H 保留20%資金'),
    # --- 加碼次數限制 ---
    'cash_add_limit_3': ({'max_add_per_stock': 3},             '🔬H 每檔最多加碼3次'),
    'cash_add_limit_5': ({'max_add_per_stock': 5},             '🔬H 每檔最多加碼5次'),

    # === V4.11 Ablation H2: 交叉驗證 (reserve × add_limit) ===
    # --- reserve 精細化 ---
    'cash_reserve_5':  ({'cash_reserve_pct': 5.0},             '🔬H2 保留5%資金'),
    'cash_reserve_15': ({'cash_reserve_pct': 15.0},            '🔬H2 保留15%資金'),
    # --- add_limit 精細化 ---
    'cash_add_limit_7': ({'max_add_per_stock': 7},             '🔬H2 每檔最多加碼7次'),
    'cash_add_limit_10': ({'max_add_per_stock': 10},           '🔬H2 每檔最多加碼10次'),
    # --- 交叉: reserve × add_limit ---
    'r5_a5':   ({'cash_reserve_pct': 5.0,  'max_add_per_stock': 5},  '🔬H2 保留5%+限加碼5次'),
    'r5_a7':   ({'cash_reserve_pct': 5.0,  'max_add_per_stock': 7},  '🔬H2 保留5%+限加碼7次'),
    'r10_a5':  ({'cash_reserve_pct': 10.0, 'max_add_per_stock': 5},  '🔬H2 保留10%+限加碼5次'),
    'r10_a7':  ({'cash_reserve_pct': 10.0, 'max_add_per_stock': 7},  '🔬H2 保留10%+限加碼7次'),
    'r10_a10': ({'cash_reserve_pct': 10.0, 'max_add_per_stock': 10}, '🔬H2 保留10%+限加碼10次'),
    'r15_a5':  ({'cash_reserve_pct': 15.0, 'max_add_per_stock': 5},  '🔬H2 保留15%+限加碼5次'),
    'r15_a7':  ({'cash_reserve_pct': 15.0, 'max_add_per_stock': 7},  '🔬H2 保留15%+限加碼7次'),

    # === V4.12 Ablation I: 停利×濾網×資金 組合驗證 ===
    # --- I1: S1 中間值 (baseline 30/0.97/15/0.6 ↔ loose 40/0.95/20/0.7 的折中) ---
    's1_mid':         ({'tier_a_net': 35, 'tier_a_ma_buf': 0.96,
                        'tier_b_net': 18, 'tier_b_drawdown': 0.65},
                                                               '🔬I S1中間 (TierA=35/0.96, TierB=18/0.65)'),

    # --- I2: S1 × F1 交叉 (停利寬鬆度 × 濾網寬鬆度) ---
    's1_loose_f1_mod':  ({'tier_a_net': 40, 'tier_a_ma_buf': 0.95,
                          'tier_b_net': 20, 'tier_b_drawdown': 0.7,
                          'market_filter_mode': 'moderate'},
                                                               '🔬I s1_loose+F1中等'),
    's1_loose_f1_rel':  ({'tier_a_net': 40, 'tier_a_ma_buf': 0.95,
                          'tier_b_net': 20, 'tier_b_drawdown': 0.7,
                          'market_filter_mode': 'relaxed'},
                                                               '🔬I s1_loose+F1寬鬆'),
    's1_mid_f1_mod':    ({'tier_a_net': 35, 'tier_a_ma_buf': 0.96,
                          'tier_b_net': 18, 'tier_b_drawdown': 0.65,
                          'market_filter_mode': 'moderate'},
                                                               '🔬I s1_mid+F1中等'),
    's1_mid_f1_rel':    ({'tier_a_net': 35, 'tier_a_ma_buf': 0.96,
                          'tier_b_net': 18, 'tier_b_drawdown': 0.65,
                          'market_filter_mode': 'relaxed'},
                                                               '🔬I s1_mid+F1寬鬆'),
    'base_f1_mod':      ({'market_filter_mode': 'moderate'},
                                                               '🔬I baseline+F1中等 (=f1_moderate對照)'),

    # --- I3: S1 × cash_reserve 交叉 ---
    's1_loose_r5':      ({'tier_a_net': 40, 'tier_a_ma_buf': 0.95,
                          'tier_b_net': 20, 'tier_b_drawdown': 0.7,
                          'cash_reserve_pct': 5.0},
                                                               '🔬I s1_loose+保留5%'),
    's1_mid_r5':        ({'tier_a_net': 35, 'tier_a_ma_buf': 0.96,
                          'tier_b_net': 18, 'tier_b_drawdown': 0.65,
                          'cash_reserve_pct': 5.0},
                                                               '🔬I s1_mid+保留5%'),

    # --- I4: 三因子組合 (S1 × F1 × reserve) ---
    's1_loose_f1_mod_r5': ({'tier_a_net': 40, 'tier_a_ma_buf': 0.95,
                            'tier_b_net': 20, 'tier_b_drawdown': 0.7,
                            'market_filter_mode': 'moderate',
                            'cash_reserve_pct': 5.0},
                                                               '🔬I s1_loose+F1中等+保留5%'),
    's1_mid_f1_mod_r5':   ({'tier_a_net': 35, 'tier_a_ma_buf': 0.96,
                            'tier_b_net': 18, 'tier_b_drawdown': 0.65,
                            'market_filter_mode': 'moderate',
                            'cash_reserve_pct': 5.0},
                                                               '🔬I s1_mid+F1中等+保留5%'),

    # === V4.13 Ablation J: 滑價壓力測試 ===
    'slip_0':     ({'slippage_pct': 0.0},    '🔬J 無滑價 (=baseline)'),
    'slip_01':    ({'slippage_pct': 0.1},    '🔬J 滑價0.1% (整股交易)'),
    'slip_03':    ({'slippage_pct': 0.3},    '🔬J 滑價0.3% (零股合理)'),
    'slip_05':    ({'slippage_pct': 0.5},    '🔬J 滑價0.5% (零股保守)'),
    'slip_10':    ({'slippage_pct': 1.0},    '🔬J 滑價1.0% (極端壓力)'),

    # === V4.13 Ablation K: 交易參數最佳化 (90萬固定資金) ===
    # K1: 持倉數量 (budget=15K 不變, 測分散度)
    'k_pos8':     ({'max_positions': 8,  'max_new_buy_per_day': 2},  '🔬K1 持倉8檔 (集中)'),
    'k_pos10':    ({'max_positions': 10, 'max_new_buy_per_day': 2},  '🔬K1 持倉10檔'),
    'k_pos12':    ({'max_positions': 12, 'max_new_buy_per_day': 3},  '🔬K1 持倉12檔'),
    # baseline = pos15, buy3
    'k_pos20':    ({'max_positions': 20, 'max_new_buy_per_day': 4},  '🔬K1 持倉20檔 (分散)'),
    'k_pos25':    ({'max_positions': 25, 'max_new_buy_per_day': 5},  '🔬K1 持倉25檔 (超分散)'),

    # K2: 每次下單金額 (pos=15 不變, 測單筆大小)
    'k_bud10k':   ({'_budget': 10000},                               '🔬K2 每次1萬 (小額)'),
    # baseline = 15K
    'k_bud15k':   ({'_budget': 15000},                               '🔬K2 每次1.5萬 (daily預設)'),
    'k_bud20k':   ({'_budget': 20000},                               '🔬K2 每次2萬'),
    'k_bud25k':   ({'_budget': 25000},                               '🔬K2 每次2.5萬'),
    'k_bud30k':   ({'_budget': 30000},                               '🔬K2 每次3萬'),
    'k_bud45k':   ({'_budget': 45000},                               '🔬K2 每次4.5萬 (大額)'),
    'k_bud60k':   ({'_budget': 60000},                               '🔬K2 每次6萬 (滿倉=90萬)'),

    # K3: 每日換股上限
    'k_swap0':    ({'max_swap_per_day': 0},                          '🔬K3 禁止換股'),
    # baseline = swap 1
    'k_swap2':    ({'max_swap_per_day': 2},                          '🔬K3 每日換2檔'),
    'k_swap3':    ({'max_swap_per_day': 3},                          '🔬K3 每日換3檔'),

    # K4: 每日建倉上限
    'k_buy1':     ({'max_new_buy_per_day': 1},                       '🔬K4 每日最多建倉1檔'),
    'k_buy2':     ({'max_new_buy_per_day': 2},                       '🔬K4 每日最多建倉2檔'),
    # baseline = buy 3
    'k_buy5':     ({'max_new_buy_per_day': 5},                       '🔬K4 每日最多建倉5檔'),

    # K5: 資金利用率組合 (budget×pos ≈ 同水位, 測粗細粒度)
    'k_20k_30p':  ({'_budget': 20000, 'max_positions': 30, 'max_new_buy_per_day': 5},
                                                                     '🔬K5 2萬×30檔 (超分散)'),
    'k_30k_20p':  ({'_budget': 30000, 'max_positions': 20, 'max_new_buy_per_day': 4},
                                                                     '🔬K5 3萬×20檔 (均衡)'),
    'k_45k_15p':  ({'_budget': 45000, 'max_positions': 15, 'max_new_buy_per_day': 3},
                                                                     '🔬K5 4.5萬×15檔 (集中)'),
    'k_60k_10p':  ({'_budget': 60000, 'max_positions': 10, 'max_new_buy_per_day': 2},
                                                                     '🔬K5 6萬×10檔 (高集中)'),
    'k_90k_8p':   ({'_budget': 90000, 'max_positions': 8,  'max_new_buy_per_day': 2},
                                                                     '🔬K5 9萬×8檔 (極集中)'),

    # K6: 殭屍清除參數
    'k_zomb_off': ({'enable_zombie_cleanup': False},                 '🔬K6 關閉殭屍清除'),
    'k_zomb10d':  ({'zombie_hold_days': 10},                         '🔬K6 殭屍10天 (積極)'),
    # baseline = 15天
    'k_zomb20d':  ({'zombie_hold_days': 20},                         '🔬K6 殭屍20天 (寬鬆)'),
    'k_zomb30d':  ({'zombie_hold_days': 30},                         '🔬K6 殭屍30天 (很寬鬆)'),

    # K7: 換股門檻 (swap_score_margin)
    'k_sm03':     ({'swap_score_margin': 0.3},                       '🔬K7 換股門檻0.3 (積極換)'),
    'k_sm05':     ({'swap_score_margin': 0.5},                       '🔬K7 換股門檻0.5'),
    # baseline = 0.8
    'k_sm12':     ({'swap_score_margin': 1.2},                       '🔬K7 換股門檻1.2 (保守換)'),
    'k_sm15':     ({'swap_score_margin': 1.5},                       '🔬K7 換股門檻1.5 (很少換)'),

    # === V4.13 Ablation K8: 組合驗證 (K3/K4/K6 冠軍交叉) ===
    'k8_buy2':          ({'max_new_buy_per_day': 2},
                                                                     '🔬K8 buy2 (單改建倉)'),
    'k8_buy2_swap2':    ({'max_new_buy_per_day': 2, 'max_swap_per_day': 2},
                                                                     '🔬K8 buy2+swap2'),
    'k8_buy2_z10':      ({'max_new_buy_per_day': 2, 'zombie_hold_days': 10},
                                                                     '🔬K8 buy2+殭屍10天'),
    'k8_buy2_s2_z10':   ({'max_new_buy_per_day': 2, 'max_swap_per_day': 2, 'zombie_hold_days': 10},
                                                                     '🔬K8 buy2+swap2+殭屍10天'),
    'k8_z10':           ({'zombie_hold_days': 10},
                                                                     '🔬K8 殭屍10天 (單改)'),
    'k8_swap2':         ({'max_swap_per_day': 2},
                                                                     '🔬K8 swap2 (單改換股)'),

    # === V4.14 跨產業參數搜索 (分層漏斗) ===
    # Layer 1: 單因子掃描 — 每個維度各自測試，找出對該產業最敏感的參數

    # L1-A: B4 乖離上限 (不同股性乖離天花板差異最大)
    'ind_bias10':   ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10},
                                                                     '🏭L1 B4 乖離上限 10%'),
    'ind_bias15':   ({'bias_limit_bull': 15, 'bias_limit_neutral': 15, 'bias_limit_bear': 15},
                                                                     '🏭L1 B4 乖離上限 15%'),
    'ind_bias20':   ({'bias_limit_bull': 20, 'bias_limit_neutral': 20, 'bias_limit_bear': 20},
                                                                     '🏭L1 B4 乖離上限 20%'),
    'ind_bias25':   ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25},
                                                                     '🏭L1 B4 乖離上限 25%'),
    'ind_bias30':   ({'bias_limit_bull': 30, 'bias_limit_neutral': 30, 'bias_limit_bear': 30},
                                                                     '🏭L1 B4 乖離上限 30%'),

    # L1-B: S1 停利參數 (震盪股 vs 趨勢股需要不同停利寬度)
    'ind_s1_tight': ({'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                      'tier_b_net': 10, 'tier_b_drawdown': 0.5},
                                                                     '🏭L1 S1 停利緊 (A=20/B=10)'),
    'ind_s1_mid':   ({'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                      'tier_b_net': 15, 'tier_b_drawdown': 0.6},
                                                                     '🏭L1 S1 停利中 (A=30/B=15)'),
    'ind_s1_loose': ({'tier_a_net': 40, 'tier_a_ma_buf': 0.95,
                      'tier_b_net': 20, 'tier_b_drawdown': 0.7},
                                                                     '🏭L1 S1 停利寬 (A=40/B=20)'),

    # L1-C: 殭屍清除天數 (慢牛產業需要更多耐心)
    'ind_z10':      ({'zombie_hold_days': 10},                       '🏭L1 殭屍 10天 (積極)'),
    'ind_z15':      ({'zombie_hold_days': 15},                       '🏭L1 殭屍 15天 (=default)'),
    'ind_z20':      ({'zombie_hold_days': 20},                       '🏭L1 殭屍 20天 (寬鬆)'),
    'ind_z30':      ({'zombie_hold_days': 30},                       '🏭L1 殭屍 30天 (很寬鬆)'),

    # L1-D: F1 大盤濾網嚴格度 (有些產業跟大盤連動低)
    'ind_f1_strict':  ({'market_filter_mode': 'strict'},             '🏭L1 F1 嚴格 (bear+weak+crash全擋)'),
    'ind_f1_mod':     ({'market_filter_mode': 'moderate'},           '🏭L1 F1 中等 (bear+crash擋)'),
    'ind_f1_relaxed': ({'market_filter_mode': 'relaxed'},            '🏭L1 F1 寬鬆 (只擋bear, =default)'),

    # L1-E: 每次下單金額 (不同產業股價分布不同)
    'ind_bud10k':   ({'_budget': 10000},                             '🏭L1 每次1萬'),
    'ind_bud15k':   ({'_budget': 15000},                             '🏭L1 每次1.5萬 (=default)'),
    'ind_bud20k':   ({'_budget': 20000},                             '🏭L1 每次2萬'),
    'ind_bud30k':   ({'_budget': 30000},                             '🏭L1 每次3萬'),

    # L1-F: B6 突破回看天數 (不同產業盤整週期不同)
    'ind_brk5':     ({'breakout_lookback': 5},                       '🏭L1 突破回看 5天 (短)'),
    'ind_brk10':    ({'breakout_lookback': 10},                      '🏭L1 突破回看 10天 (=default)'),
    'ind_brk20':    ({'breakout_lookback': 20},                      '🏭L1 突破回看 20天 (長)'),

    # === V4.15 其他電子業 分區間驗證 ===
    # L2 冠軍: bias=10, s1=mid(A30/B15), zombie=30
    # 核心組: L2 冠軍 + 微調對照
    'oe_champ':     ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                      'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                      'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                      'zombie_hold_days': 30},
                                                                     '🏭OE L2冠軍 b10/s1mid/z30'),
    'oe_champ_nz':  ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                      'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                      'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                      'enable_zombie_cleanup': False},
                                                                     '🏭OE L2冠軍+關殭屍 (L3最佳)'),
    # bias 微調 (±5)
    'oe_b15_mid_z30': ({'bias_limit_bull': 15, 'bias_limit_neutral': 15, 'bias_limit_bear': 15,
                        'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                        'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                        'zombie_hold_days': 30},
                                                                     '🏭OE b15/s1mid/z30 (bias放寬)'),
    # s1 微調
    'oe_b10_tight_z30': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                          'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                          'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                          'zombie_hold_days': 30},
                                                                     '🏭OE b10/s1tight/z30 (停利收緊)'),
    'oe_b10_loose_z30': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                          'tier_a_net': 40, 'tier_a_ma_buf': 0.95,
                          'tier_b_net': 20, 'tier_b_drawdown': 0.7,
                          'zombie_hold_days': 30},
                                                                     '🏭OE b10/s1loose/z30 (停利放寬)'),
    # zombie 微調
    'oe_b10_mid_z20': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                        'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                        'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                        'zombie_hold_days': 20},
                                                                     '🏭OE b10/s1mid/z20 (殭屍收緊)'),
    'oe_b10_mid_z15': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                        'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                        'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                        'zombie_hold_days': 15},
                                                                     '🏭OE b10/s1mid/z15 (殭屍=default)'),
    # 防禦模組驗證 (基於冠軍)
    'oe_no_filter': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                      'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                      'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                      'zombie_hold_days': 30,
                      'enable_market_filter': False},
                                                                     '❌OE 冠軍+關F1F2F3'),
    'oe_no_breakout': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                        'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                        'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                        'zombie_hold_days': 30,
                        'enable_breakout': False},
                                                                     '❌OE 冠軍+關突破B6'),
    'oe_no_bias':   ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                      'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                      'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                      'zombie_hold_days': 30,
                      'enable_bias_limit': False},
                                                                     '❌OE 冠軍+關乖離B4'),
    'oe_no_fish':   ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                      'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                      'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                      'zombie_hold_days': 30,
                      'enable_fish_tail': False},
                                                                     '❌OE 冠軍+關魚尾B5'),
    'oe_no_s1':     ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                      'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                      'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                      'zombie_hold_days': 30,
                      'enable_tiered_stops': False},
                                                                     '❌OE 冠軍+關S1停利'),

    # === V4.15 電腦及週邊設備業 分區間驗證 ===
    # L2 冠軍: bias=10, s1=mid(A30/B15), zombie=25, brk=default(10)
    # L1 冠軍特徵: bias=10, brk=5(L1最佳), s1=mid, zombie=20
    # L3 最佳: L3_no_filter (Sharpe 1.07) — F1 濾網反而有害, 需分區間確認

    # 核心組: L2 冠軍
    'cp_champ':     ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                      'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                      'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                      'zombie_hold_days': 25},
                                                                     '🏭CP L2冠軍 b10/s1mid/z25'),
    # L2冠軍 + brk=5 (L1突破冠軍, Sharpe 1.18)
    'cp_champ_brk5': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                       'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                       'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                       'zombie_hold_days': 25,
                       'breakout_lookback': 5},
                                                                     '🏭CP L2冠軍+brk5 (L1突破冠軍)'),
    # bias 微調
    'cp_b15_mid_z25': ({'bias_limit_bull': 15, 'bias_limit_neutral': 15, 'bias_limit_bear': 15,
                        'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                        'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                        'zombie_hold_days': 25},
                                                                     '🏭CP b15/s1mid/z25 (bias放寬)'),
    # s1 微調
    'cp_b10_tight_z25': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                          'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                          'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                          'zombie_hold_days': 25},
                                                                     '🏭CP b10/s1tight/z25 (停利收緊)'),
    'cp_b10_loose_z25': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                          'tier_a_net': 40, 'tier_a_ma_buf': 0.95,
                          'tier_b_net': 20, 'tier_b_drawdown': 0.7,
                          'zombie_hold_days': 25},
                                                                     '🏭CP b10/s1loose/z25 (停利放寬)'),
    # zombie 微調
    'cp_b10_mid_z20': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                        'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                        'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                        'zombie_hold_days': 20},
                                                                     '🏭CP b10/s1mid/z20 (殭屍收緊)'),
    'cp_b10_mid_z30': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                        'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                        'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                        'zombie_hold_days': 30},
                                                                     '🏭CP b10/s1mid/z30 (殭屍放寬)'),
    'cp_b10_mid_z15': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                        'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                        'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                        'zombie_hold_days': 15},
                                                                     '🏭CP b10/s1mid/z15 (殭屍=default)'),
    # 防禦模組驗證 (基於冠軍) — 特別關注 F1 (全期L3最佳=關F1)
    'cp_no_filter': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                      'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                      'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                      'zombie_hold_days': 25,
                      'enable_market_filter': False},
                                                                     '❌CP 冠軍+關F1F2F3 (全期L3最佳!)'),
    'cp_no_breakout': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                        'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                        'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                        'zombie_hold_days': 25,
                        'enable_breakout': False},
                                                                     '❌CP 冠軍+關突破B6'),
    'cp_no_bias':   ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                      'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                      'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                      'zombie_hold_days': 25,
                      'enable_bias_limit': False},
                                                                     '❌CP 冠軍+關乖離B4'),
    'cp_no_fish':   ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                      'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                      'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                      'zombie_hold_days': 25,
                      'enable_fish_tail': False},
                                                                     '❌CP 冠軍+關魚尾B5'),
    'cp_no_s1':     ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                      'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                      'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                      'zombie_hold_days': 25,
                      'enable_tiered_stops': False},
                                                                     '❌CP 冠軍+關S1停利'),
    'cp_no_zombie': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                      'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                      'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                      'zombie_hold_days': 25,
                      'enable_zombie_cleanup': False},
                                                                     '❌CP 冠軍+關殭屍S4'),

    # === V4.15 通信網路業 分區間驗證 ===
    # L2 冠軍: bias=25, s1=tight(A20/B10), zombie=20, f1=moderate, brk=20, budget=30k
    # L1 冠軍: bias=20, s1=tight, zombie=20, f1=moderate, budget=30k, brk=20
    # L3 最佳: L3_best (Sharpe 1.02) — 所有防禦模組都有效
    # 注意: 6 個維度全部偏離 baseline, 客製化程度最高

    # 核心組: L2 冠軍 (完整 6 維度)
    'cn_champ':     ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                      'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                      'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                      'zombie_hold_days': 20,
                      'market_filter_mode': 'moderate',
                      'breakout_lookback': 20,
                      '_budget': 30000},
                                                                     '🏭CN L2冠軍 b25/tight/z20/f1mod/brk20/30k'),
    # L1 冠軍 bias=20 (vs L2 的 25)
    'cn_b20_tight_z20': ({'bias_limit_bull': 20, 'bias_limit_neutral': 20, 'bias_limit_bear': 20,
                          'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                          'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                          'zombie_hold_days': 20,
                          'market_filter_mode': 'moderate',
                          'breakout_lookback': 20,
                          '_budget': 30000},
                                                                     '🏭CN b20/tight/z20 (L1 bias冠軍)'),
    # bias 微調
    'cn_b30_tight_z20': ({'bias_limit_bull': 30, 'bias_limit_neutral': 30, 'bias_limit_bear': 30,
                          'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                          'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                          'zombie_hold_days': 20,
                          'market_filter_mode': 'moderate',
                          'breakout_lookback': 20,
                          '_budget': 30000},
                                                                     '🏭CN b30/tight/z20 (bias放寬=default)'),
    # s1 微調
    'cn_b25_mid_z20':   ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                          'tier_a_net': 30, 'tier_a_ma_buf': 0.97,
                          'tier_b_net': 15, 'tier_b_drawdown': 0.6,
                          'zombie_hold_days': 20,
                          'market_filter_mode': 'moderate',
                          'breakout_lookback': 20,
                          '_budget': 30000},
                                                                     '🏭CN b25/s1mid/z20 (停利放寬)'),
    'cn_b25_loose_z20': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                          'tier_a_net': 40, 'tier_a_ma_buf': 0.95,
                          'tier_b_net': 20, 'tier_b_drawdown': 0.7,
                          'zombie_hold_days': 20,
                          'market_filter_mode': 'moderate',
                          'breakout_lookback': 20,
                          '_budget': 30000},
                                                                     '🏭CN b25/s1loose/z20 (停利=default)'),
    # zombie 微調
    'cn_b25_tight_z15': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                          'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                          'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                          'zombie_hold_days': 15,
                          'market_filter_mode': 'moderate',
                          'breakout_lookback': 20,
                          '_budget': 30000},
                                                                     '🏭CN b25/tight/z15 (殭屍=default)'),
    'cn_b25_tight_z30': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                          'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                          'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                          'zombie_hold_days': 30,
                          'market_filter_mode': 'moderate',
                          'breakout_lookback': 20,
                          '_budget': 30000},
                                                                     '🏭CN b25/tight/z30 (殭屍放寬)'),
    # budget 微調 (L1 冠軍=30k, 對照 15k default)
    'cn_champ_bud15k':  ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                          'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                          'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                          'zombie_hold_days': 20,
                          'market_filter_mode': 'moderate',
                          'breakout_lookback': 20},
                                                                     '🏭CN L2冠軍+budget=15k (=default)'),
    # brk 微調 (L1 冠軍=20, 對照 10 default)
    'cn_champ_brk10':   ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                          'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                          'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                          'zombie_hold_days': 20,
                          'market_filter_mode': 'moderate',
                          'breakout_lookback': 10,
                          '_budget': 30000},
                                                                     '🏭CN L2冠軍+brk10 (=default)'),
    # f1 微調 (L1 冠軍=moderate, 對照 relaxed/strict)
    'cn_champ_f1relax': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                          'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                          'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                          'zombie_hold_days': 20,
                          'market_filter_mode': 'relaxed',
                          'breakout_lookback': 20,
                          '_budget': 30000},
                                                                     '🏭CN L2冠軍+F1relaxed (=default)'),
    # 防禦模組驗證 (基於冠軍) — 全期所有模組都有效
    'cn_no_filter': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                      'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                      'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                      'zombie_hold_days': 20,
                      'breakout_lookback': 20,
                      '_budget': 30000,
                      'enable_market_filter': False},
                                                                     '❌CN 冠軍+關F1F2F3'),
    'cn_no_breakout': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                        'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                        'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                        'zombie_hold_days': 20,
                        'market_filter_mode': 'moderate',
                        '_budget': 30000,
                        'enable_breakout': False},
                                                                     '❌CN 冠軍+關突破B6'),
    'cn_no_bias':   ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                      'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                      'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                      'zombie_hold_days': 20,
                      'market_filter_mode': 'moderate',
                      'breakout_lookback': 20,
                      '_budget': 30000,
                      'enable_bias_limit': False},
                                                                     '❌CN 冠軍+關乖離B4'),
    'cn_no_fish':   ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                      'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                      'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                      'zombie_hold_days': 20,
                      'market_filter_mode': 'moderate',
                      'breakout_lookback': 20,
                      '_budget': 30000,
                      'enable_fish_tail': False},
                                                                     '❌CN 冠軍+關魚尾B5'),
    'cn_no_s1':     ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                      'zombie_hold_days': 20,
                      'market_filter_mode': 'moderate',
                      'breakout_lookback': 20,
                      '_budget': 30000,
                      'enable_tiered_stops': False},
                                                                     '❌CN 冠軍+關S1停利'),
    'cn_no_zombie': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                      'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                      'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                      'market_filter_mode': 'moderate',
                      'breakout_lookback': 20,
                      '_budget': 30000,
                      'enable_zombie_cleanup': False},
                                                                     '❌CN 冠軍+關殭屍S4'),

    # ═══════════════════════════════════════════════════════════════
    # V4.16 L2 交叉驗證 — 基於 L1 冠軍, 6 個產業
    # ═══════════════════════════════════════════════════════════════

    # === [24] 半導體業 L2 交叉 ===
    # L1 冠軍: bias30/loose/z10/f1_strict/brk10
    # 交叉: bias(25,30) × s1(loose) × z(10,15) + F1=strict 固定
    'sc_b25_loose_z10': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                          'zombie_hold_days': 10,
                          'market_filter_mode': 'strict'},
                                                                     '🔬SC L2 b25/loose/z10/strict'),
    'sc_b25_loose_z15': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                          'zombie_hold_days': 15,
                          'market_filter_mode': 'strict'},
                                                                     '🔬SC L2 b25/loose/z15/strict'),
    'sc_b30_loose_z10': ({'bias_limit_bull': 30, 'bias_limit_neutral': 30, 'bias_limit_bear': 30,
                          'zombie_hold_days': 10,
                          'market_filter_mode': 'strict'},
                                                                     '🔬SC L2 b30/loose/z10/strict (L1冠軍)'),
    'sc_b30_loose_z15': ({'bias_limit_bull': 30, 'bias_limit_neutral': 30, 'bias_limit_bear': 30,
                          'zombie_hold_days': 15,
                          'market_filter_mode': 'strict'},
                                                                     '🔬SC L2 b30/loose/z15/strict'),
    # 額外: 不加 strict 對照 (確認 F1 效果)
    'sc_b30_loose_z10_rel': ({'bias_limit_bull': 30, 'bias_limit_neutral': 30, 'bias_limit_bear': 30,
                              'zombie_hold_days': 10},
                                                                     '🔬SC L2 b30/loose/z10/relaxed (F1對照)'),

    # === [25] 其他電子業 L2 交叉 ===
    # L1 冠軍: bias10/loose/z30/relaxed/brk5
    # 交叉: bias(10,15) × s1(loose) × z(25,30) + brk5 加掛確認
    'oe2_b10_loose_z25': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                           'zombie_hold_days': 25},
                                                                     '🔬OE2 L2 b10/loose/z25'),
    'oe2_b10_loose_z30': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                           'zombie_hold_days': 30},
                                                                     '🔬OE2 L2 b10/loose/z30 (L1冠軍)'),
    'oe2_b15_loose_z25': ({'bias_limit_bull': 15, 'bias_limit_neutral': 15, 'bias_limit_bear': 15,
                           'zombie_hold_days': 25},
                                                                     '🔬OE2 L2 b15/loose/z25'),
    'oe2_b15_loose_z30': ({'bias_limit_bull': 15, 'bias_limit_neutral': 15, 'bias_limit_bear': 15,
                           'zombie_hold_days': 30},
                                                                     '🔬OE2 L2 b15/loose/z30'),
    # brk5 加掛: 最佳 bias×z 組合 + brk5
    'oe2_b10_loose_z30_brk5': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                                'zombie_hold_days': 30,
                                'breakout_lookback': 5},
                                                                     '🔬OE2 L2 b10/loose/z30+brk5'),
    'oe2_b10_loose_z25_brk5': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                                'zombie_hold_days': 25,
                                'breakout_lookback': 5},
                                                                     '🔬OE2 L2 b10/loose/z25+brk5'),

    # === [26] 通信網路業 L2 交叉 ===
    # L1 冠軍: bias20/tight/z20/relaxed/brk20
    # 交叉: bias(20,25) × s1(tight,loose) × z(15,20)
    'cn2_b20_tight_z15': ({'bias_limit_bull': 20, 'bias_limit_neutral': 20, 'bias_limit_bear': 20,
                           'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                           'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                           'zombie_hold_days': 15,
                           'breakout_lookback': 20},
                                                                     '🔬CN2 L2 b20/tight/z15/brk20'),
    'cn2_b20_tight_z20': ({'bias_limit_bull': 20, 'bias_limit_neutral': 20, 'bias_limit_bear': 20,
                           'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                           'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                           'zombie_hold_days': 20,
                           'breakout_lookback': 20},
                                                                     '🔬CN2 L2 b20/tight/z20/brk20 (L1冠軍)'),
    'cn2_b20_loose_z15': ({'bias_limit_bull': 20, 'bias_limit_neutral': 20, 'bias_limit_bear': 20,
                           'zombie_hold_days': 15,
                           'breakout_lookback': 20},
                                                                     '🔬CN2 L2 b20/loose/z15/brk20'),
    'cn2_b20_loose_z20': ({'bias_limit_bull': 20, 'bias_limit_neutral': 20, 'bias_limit_bear': 20,
                           'zombie_hold_days': 20,
                           'breakout_lookback': 20},
                                                                     '🔬CN2 L2 b20/loose/z20/brk20'),
    'cn2_b25_tight_z15': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                           'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                           'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                           'zombie_hold_days': 15,
                           'breakout_lookback': 20},
                                                                     '🔬CN2 L2 b25/tight/z15/brk20'),
    'cn2_b25_tight_z20': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                           'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                           'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                           'zombie_hold_days': 20,
                           'breakout_lookback': 20},
                                                                     '🔬CN2 L2 b25/tight/z20/brk20'),
    'cn2_b25_loose_z15': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                           'zombie_hold_days': 15,
                           'breakout_lookback': 20},
                                                                     '🔬CN2 L2 b25/loose/z15/brk20'),
    'cn2_b25_loose_z20': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                           'zombie_hold_days': 20,
                           'breakout_lookback': 20},
                                                                     '🔬CN2 L2 b25/loose/z20/brk20'),

    # === [27] 電子零組件業 L2 交叉 ===
    # L1 冠軍: bias10/loose/z20/relaxed/brk10
    # 交叉: bias(10,15) × s1(loose) × z(15,20,25)
    'ec_b10_loose_z15': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                          'zombie_hold_days': 15},
                                                                     '🔬EC L2 b10/loose/z15 (=baseline)'),
    'ec_b10_loose_z20': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                          'zombie_hold_days': 20},
                                                                     '🔬EC L2 b10/loose/z20 (L1冠軍)'),
    'ec_b10_loose_z25': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                          'zombie_hold_days': 25},
                                                                     '🔬EC L2 b10/loose/z25'),
    'ec_b15_loose_z15': ({'bias_limit_bull': 15, 'bias_limit_neutral': 15, 'bias_limit_bear': 15,
                          'zombie_hold_days': 15},
                                                                     '🔬EC L2 b15/loose/z15'),
    'ec_b15_loose_z20': ({'bias_limit_bull': 15, 'bias_limit_neutral': 15, 'bias_limit_bear': 15,
                          'zombie_hold_days': 20},
                                                                     '🔬EC L2 b15/loose/z20'),
    'ec_b15_loose_z25': ({'bias_limit_bull': 15, 'bias_limit_neutral': 15, 'bias_limit_bear': 15,
                          'zombie_hold_days': 25},
                                                                     '🔬EC L2 b15/loose/z25'),

    # === [28] 電機機械 L2 交叉 ===
    # L1 冠軍: bias15/loose/z15/f1_mod/brk5
    # 交叉: bias(10,15,20) × s1(loose) × z(10,15) + F1=moderate 固定
    # brk5 加掛確認
    'em_b10_loose_z10': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                          'zombie_hold_days': 10,
                          'market_filter_mode': 'moderate'},
                                                                     '🔬EM L2 b10/loose/z10/mod'),
    'em_b10_loose_z15': ({'bias_limit_bull': 10, 'bias_limit_neutral': 10, 'bias_limit_bear': 10,
                          'zombie_hold_days': 15,
                          'market_filter_mode': 'moderate'},
                                                                     '🔬EM L2 b10/loose/z15/mod'),
    'em_b15_loose_z10': ({'bias_limit_bull': 15, 'bias_limit_neutral': 15, 'bias_limit_bear': 15,
                          'zombie_hold_days': 10,
                          'market_filter_mode': 'moderate'},
                                                                     '🔬EM L2 b15/loose/z10/mod'),
    'em_b15_loose_z15': ({'bias_limit_bull': 15, 'bias_limit_neutral': 15, 'bias_limit_bear': 15,
                          'zombie_hold_days': 15,
                          'market_filter_mode': 'moderate'},
                                                                     '🔬EM L2 b15/loose/z15/mod (L1冠軍)'),
    'em_b20_loose_z10': ({'bias_limit_bull': 20, 'bias_limit_neutral': 20, 'bias_limit_bear': 20,
                          'zombie_hold_days': 10,
                          'market_filter_mode': 'moderate'},
                                                                     '🔬EM L2 b20/loose/z10/mod'),
    'em_b20_loose_z15': ({'bias_limit_bull': 20, 'bias_limit_neutral': 20, 'bias_limit_bear': 20,
                          'zombie_hold_days': 15,
                          'market_filter_mode': 'moderate'},
                                                                     '🔬EM L2 b20/loose/z15/mod'),
    # brk5 加掛
    'em_b15_loose_z15_brk5': ({'bias_limit_bull': 15, 'bias_limit_neutral': 15, 'bias_limit_bear': 15,
                               'zombie_hold_days': 15,
                               'market_filter_mode': 'moderate',
                               'breakout_lookback': 5},
                                                                     '🔬EM L2 b15/loose/z15/mod+brk5 (L1冠軍)'),
    'em_b15_loose_z10_brk5': ({'bias_limit_bull': 15, 'bias_limit_neutral': 15, 'bias_limit_bear': 15,
                               'zombie_hold_days': 10,
                               'market_filter_mode': 'moderate',
                               'breakout_lookback': 5},
                                                                     '🔬EM L2 b15/loose/z10/mod+brk5'),

    # === [29] 電腦及週邊設備業 L2 交叉 ===
    # L1 冠軍: bias25/tight/z20/relaxed/brk5
    # 交叉: bias(20,25,30) × s1(tight,loose) × z(15,20)
    # brk5 加掛確認
    'cp2_b20_tight_z15': ({'bias_limit_bull': 20, 'bias_limit_neutral': 20, 'bias_limit_bear': 20,
                           'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                           'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                           'zombie_hold_days': 15},
                                                                     '🔬CP2 L2 b20/tight/z15'),
    'cp2_b20_tight_z20': ({'bias_limit_bull': 20, 'bias_limit_neutral': 20, 'bias_limit_bear': 20,
                           'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                           'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                           'zombie_hold_days': 20},
                                                                     '🔬CP2 L2 b20/tight/z20'),
    'cp2_b20_loose_z15': ({'bias_limit_bull': 20, 'bias_limit_neutral': 20, 'bias_limit_bear': 20,
                           'zombie_hold_days': 15},
                                                                     '🔬CP2 L2 b20/loose/z15'),
    'cp2_b20_loose_z20': ({'bias_limit_bull': 20, 'bias_limit_neutral': 20, 'bias_limit_bear': 20,
                           'zombie_hold_days': 20},
                                                                     '🔬CP2 L2 b20/loose/z20'),
    'cp2_b25_tight_z15': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                           'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                           'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                           'zombie_hold_days': 15},
                                                                     '🔬CP2 L2 b25/tight/z15'),
    'cp2_b25_tight_z20': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                           'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                           'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                           'zombie_hold_days': 20},
                                                                     '🔬CP2 L2 b25/tight/z20 (L1冠軍)'),
    'cp2_b25_loose_z15': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                           'zombie_hold_days': 15},
                                                                     '🔬CP2 L2 b25/loose/z15'),
    'cp2_b25_loose_z20': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                           'zombie_hold_days': 20},
                                                                     '🔬CP2 L2 b25/loose/z20'),
    'cp2_b30_tight_z15': ({'bias_limit_bull': 30, 'bias_limit_neutral': 30, 'bias_limit_bear': 30,
                           'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                           'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                           'zombie_hold_days': 15},
                                                                     '🔬CP2 L2 b30/tight/z15'),
    'cp2_b30_tight_z20': ({'bias_limit_bull': 30, 'bias_limit_neutral': 30, 'bias_limit_bear': 30,
                           'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                           'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                           'zombie_hold_days': 20},
                                                                     '🔬CP2 L2 b30/tight/z20'),
    'cp2_b30_loose_z15': ({'bias_limit_bull': 30, 'bias_limit_neutral': 30, 'bias_limit_bear': 30,
                           'zombie_hold_days': 15},
                                                                     '🔬CP2 L2 b30/loose/z15'),
    'cp2_b30_loose_z20': ({'bias_limit_bull': 30, 'bias_limit_neutral': 30, 'bias_limit_bear': 30,
                           'zombie_hold_days': 20},
                                                                     '🔬CP2 L2 b30/loose/z20'),
    # brk5 加掛 (L1 最強因子)
    'cp2_b25_tight_z20_brk5': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                                'tier_a_net': 20, 'tier_a_ma_buf': 0.98,
                                'tier_b_net': 10, 'tier_b_drawdown': 0.5,
                                'zombie_hold_days': 20,
                                'breakout_lookback': 5},
                                                                     '🔬CP2 L2 b25/tight/z20+brk5 (L1冠軍)'),
    'cp2_b25_loose_z20_brk5': ({'bias_limit_bull': 25, 'bias_limit_neutral': 25, 'bias_limit_bear': 25,
                                'zombie_hold_days': 20,
                                'breakout_lookback': 5},
                                                                     '🔬CP2 L2 b25/loose/z20+brk5'),

    # === V4.20 Ablation M: 漲停遞延 (T+1買不到 → T+2再試) ===
    'lu_skip':        ({'skip_limit_up': True, 'limit_up_retry': False},
                                                               '🔬M 漲停跳過(=baseline)'),
    'lu_retry1':      ({'skip_limit_up': True, 'limit_up_retry': True, 'limit_up_max_retry': 1},
                                                               '🔬M 漲停遞延T+2 (重試1次)'),
    'lu_retry2':      ({'skip_limit_up': True, 'limit_up_retry': True, 'limit_up_max_retry': 2},
                                                               '🔬M 漲停遞延T+3 (重試2次)'),
    'lu_no_skip':     ({'skip_limit_up': False},
                                                               '🔬M 不跳漲停(理想成交,對照)'),

    # === V4.20 Ablation N: S2 跌破季線緩衝 ===
    's2_no_buf':      ({'s2_buffer_enabled': False},
                                                               '🔬N S2無緩衝(=baseline)'),
    's2_buf_10_2d':   ({'s2_buffer_enabled': True, 's2_buffer_net_pct': 10, 's2_buffer_days': 2},
                                                               '🔬N S2緩衝(淨利>10%+連2日)'),
    's2_buf_10_3d':   ({'s2_buffer_enabled': True, 's2_buffer_net_pct': 10, 's2_buffer_days': 3},
                                                               '🔬N S2緩衝(淨利>10%+連3日)'),
    's2_buf_5_2d':    ({'s2_buffer_enabled': True, 's2_buffer_net_pct': 5, 's2_buffer_days': 2},
                                                               '🔬N S2緩衝(淨利>5%+連2日)'),
    's2_buf_15_2d':   ({'s2_buffer_enabled': True, 's2_buffer_net_pct': 15, 's2_buffer_days': 2},
                                                               '🔬N S2緩衝(淨利>15%+連2日)'),
    's2_buf_20_2d':   ({'s2_buffer_enabled': True, 's2_buffer_net_pct': 20, 's2_buffer_days': 2},
                                                               '🔬N S2緩衝(淨利>20%+連2日)'),

    # === V4.20 Ablation O: Tier B 建倉後peak追蹤 ===
    'tb_20d_peak':    ({'tier_b_use_entry_peak': False},
                                                               '🔬O TierB用近20日高(=baseline)'),
    'tb_entry_peak':  ({'tier_b_use_entry_peak': True},
                                                               '🔬O TierB用建倉後最高價'),

    # === V4.20 Ablation P: 組合測試 (M+N+O 交叉) ===
    'v420_retry_s2buf': ({'limit_up_retry': True, 'limit_up_max_retry': 1,
                          's2_buffer_enabled': True, 's2_buffer_net_pct': 10, 's2_buffer_days': 2},
                                                               '🔬P 漲停遞延+S2緩衝'),
    'v420_retry_peak':  ({'limit_up_retry': True, 'limit_up_max_retry': 1,
                          'tier_b_use_entry_peak': True},
                                                               '🔬P 漲停遞延+TierB建倉peak'),
    'v420_full':        ({'limit_up_retry': True, 'limit_up_max_retry': 1,
                          's2_buffer_enabled': True, 's2_buffer_net_pct': 10, 's2_buffer_days': 2,
                          'tier_b_use_entry_peak': True},
                                                               '🔬P 漲停遞延+S2緩衝+TierB建倉peak'),

    # === V4.20 Ablation Q: OTC 大盤濾網模式 ===
    # _otc_mode 是特殊 key，ablation loop 會用不同 market_map
    'otc_or':           ({'_otc_mode': 'or'},
                                                               '🔬Q OTC∪TWII (現行, 最嚴格)'),
    'otc_and':          ({'_otc_mode': 'and'},
                                                               '🔬Q OTC∩TWII (共識法, 較寬鬆)'),
    'otc_off':          ({'_otc_mode': 'off'},
                                                               '🔬Q 只看TWII (v2.11相容, 無OTC)'),
    'otc_and_ep_off':   ({'_otc_mode': 'and', 'tier_b_use_entry_peak': False},
                                                               '🔬Q OTC共識+關entry_peak'),
    'otc_off_ep_off':   ({'_otc_mode': 'off', 'tier_b_use_entry_peak': False},
                                                               '🔬Q 無OTC+關entry_peak (=v2.11)'),

    # === V4.21 Ablation R: 動態曝險管理 (大盤弱勢降低持倉上限) ===
    # 命名規則: dyn_{bull}_{neutral}_{weak}_{bear}_{panic}
    # baseline = 靜態 15 檔不變
    'dyn_A': ({'enable_dynamic_exposure': True,
               'dyn_max_bull': 15, 'dyn_max_neutral': 15,
               'dyn_max_weak': 12, 'dyn_max_bear': 8, 'dyn_max_panic': 5},
              '🔬R 動態曝險A (15/15/12/8/5, 溫和減倉)'),
    'dyn_B': ({'enable_dynamic_exposure': True,
               'dyn_max_bull': 15, 'dyn_max_neutral': 15,
               'dyn_max_weak': 10, 'dyn_max_bear': 5, 'dyn_max_panic': 3},
              '🔬R 動態曝險B (15/15/10/5/3, 積極減倉)'),
    'dyn_C': ({'enable_dynamic_exposure': True,
               'dyn_max_bull': 15, 'dyn_max_neutral': 12,
               'dyn_max_weak': 8, 'dyn_max_bear': 5, 'dyn_max_panic': 3},
              '🔬R 動態曝險C (15/12/8/5/3, 全面收緊)'),
    'dyn_D': ({'enable_dynamic_exposure': True,
               'dyn_max_bull': 15, 'dyn_max_neutral': 15,
               'dyn_max_weak': 15, 'dyn_max_bear': 10, 'dyn_max_panic': 5},
              '🔬R 動態曝險D (15/15/15/10/5, 只減空頭+恐慌)'),
    'dyn_E': ({'enable_dynamic_exposure': True,
               'dyn_max_bull': 15, 'dyn_max_neutral': 15,
               'dyn_max_weak': 15, 'dyn_max_bear': 8, 'dyn_max_panic': 0},
              '🔬R 動態曝險E (15/15/15/8/0, 空頭砍+恐慌清倉)'),
    'dyn_F': ({'enable_dynamic_exposure': True,
               'dyn_max_bull': 15, 'dyn_max_neutral': 15,
               'dyn_max_weak': 10, 'dyn_max_bear': 8, 'dyn_max_panic': 5},
              '🔬R 動態曝險F (15/15/10/8/5, 偏弱開始砍)'),
    'dyn_G': ({'enable_dynamic_exposure': True,
               'dyn_max_bull': 15, 'dyn_max_neutral': 13,
               'dyn_max_weak': 10, 'dyn_max_bear': 6, 'dyn_max_panic': 3},
              '🔬R 動態曝險G (15/13/10/6/3, 階梯式)'),
    'dyn_H': ({'enable_dynamic_exposure': True,
               'dyn_max_bull': 15, 'dyn_max_neutral': 15,
               'dyn_max_weak': 12, 'dyn_max_bear': 6, 'dyn_max_panic': 0},
              '🔬R 動態曝險H (15/15/12/6/0, 空頭重砍+恐慌清倉)'),

    # === V4.21 Ablation S: 動態限買 (大盤弱勢降低每日新買上限, 不強制賣出) ===
    # 命名: buy_{bull}/{neutral}/{weak}/{bear}/{panic}
    'buy_A': ({'enable_dyn_buy_limit': True,
               'dyn_buy_bull': 4, 'dyn_buy_neutral': 4,
               'dyn_buy_weak': 2, 'dyn_buy_bear': 1, 'dyn_buy_panic': 0},
              '🔬S 動態限買A (4/4/2/1/0, 溫和限買)'),
    'buy_B': ({'enable_dyn_buy_limit': True,
               'dyn_buy_bull': 4, 'dyn_buy_neutral': 3,
               'dyn_buy_weak': 1, 'dyn_buy_bear': 0, 'dyn_buy_panic': 0},
              '🔬S 動態限買B (4/3/1/0/0, 積極限買)'),
    'buy_C': ({'enable_dyn_buy_limit': True,
               'dyn_buy_bull': 4, 'dyn_buy_neutral': 4,
               'dyn_buy_weak': 3, 'dyn_buy_bear': 2, 'dyn_buy_panic': 1},
              '🔬S 動態限買C (4/4/3/2/1, 保守限買)'),
    'buy_D': ({'enable_dyn_buy_limit': True,
               'dyn_buy_bull': 4, 'dyn_buy_neutral': 4,
               'dyn_buy_weak': 4, 'dyn_buy_bear': 0, 'dyn_buy_panic': 0},
              '🔬S 動態限買D (4/4/4/0/0, 只停空頭+恐慌)'),

    # === V4.21 Ablation T: 動態停損 (大盤弱勢時收緊 hard_stop_net) ===
    # baseline hard_stop_net = -15, 恐慌時 strategy.py 內部再 +5 = -10
    'stop_A': ({'enable_dyn_stop': True,
                'hard_stop_weak': -12, 'hard_stop_bear': -10},
               '🔬T 動態停損A (weak=-12, bear=-10)'),
    'stop_B': ({'enable_dyn_stop': True,
                'hard_stop_weak': -15, 'hard_stop_bear': -10},
               '🔬T 動態停損B (weak不變, bear=-10)'),
    'stop_C': ({'enable_dyn_stop': True,
                'hard_stop_weak': -12, 'hard_stop_bear': -8},
               '🔬T 動態停損C (weak=-12, bear=-8)'),
    'stop_D': ({'enable_dyn_stop': True,
                'hard_stop_weak': -10, 'hard_stop_bear': -8},
               '🔬T 動態停損D (weak=-10, bear=-8, 最激進)'),

    # === V4.21 Ablation U: 波動率倉位控制 (高波動時縮小單筆金額) ===
    'vol_A': ({'enable_vol_sizing': True,
               'vol_lookback': 20, 'vol_target_pct': 1.5, 'vol_scale_floor_pct': 50},
              '🔬U 波動率A (target=1.5%, floor=50%, 20日)'),
    'vol_B': ({'enable_vol_sizing': True,
               'vol_lookback': 20, 'vol_target_pct': 1.0, 'vol_scale_floor_pct': 50},
              '🔬U 波動率B (target=1.0%, floor=50%, 積極縮放)'),
    'vol_C': ({'enable_vol_sizing': True,
               'vol_lookback': 20, 'vol_target_pct': 2.0, 'vol_scale_floor_pct': 60},
              '🔬U 波動率C (target=2.0%, floor=60%, 保守)'),
    'vol_D': ({'enable_vol_sizing': True,
               'vol_lookback': 10, 'vol_target_pct': 1.5, 'vol_scale_floor_pct': 50},
              '🔬U 波動率D (target=1.5%, floor=50%, 10日快速反應)'),

    # === V4.21 Ablation V: 參數精煉 (S1/zombie/swap/cash 微調) ===
    'ref_A': ({'tier_a_net': 50, 'tier_b_drawdown': 0.75},
              '🔬V 精煉A (停利放寬: TierA=50, TierB回撤=0.75)'),
    'ref_B': ({'tier_a_net': 30, 'tier_b_drawdown': 0.65},
              '🔬V 精煉B (停利收緊: TierA=30, TierB回撤=0.65)'),
    'ref_C': ({'swap_score_margin': 0.5, 'max_swap_per_day': 3},
              '🔬V 精煉C (積極換股: margin=0.5, swap=3)'),
    'ref_D': ({'swap_score_margin': 1.0, 'max_swap_per_day': 1},
              '🔬V 精煉D (保守換股: margin=1.0, swap=1)'),
    'ref_E': ({'cash_reserve_pct': 5.0},
              '🔬V 精煉E (低現金保留: 5%)'),
    'ref_F': ({'cash_reserve_pct': 15.0},
              '🔬V 精煉F (高現金保留: 15%)'),

    # === V4.21 Ablation W: 組合交叉驗證 (stop_B × pos12 × ref_D × vol_C) ===
    'combo_stopB': ({'enable_dyn_stop': True, 'hard_stop_bear': -10},
                    '🔬W stop_B 單獨 (bear=-10)'),
    'combo_pos12': ({'max_positions': 12},
                    '🔬W pos12 單獨 (optimizer冠軍)'),
    'combo_sB_p12': ({'enable_dyn_stop': True, 'hard_stop_bear': -10,
                      'max_positions': 12},
                     '🔬W stop_B + pos12'),
    'combo_sB_rD': ({'enable_dyn_stop': True, 'hard_stop_bear': -10,
                     'swap_score_margin': 1.0, 'max_swap_per_day': 1},
                    '🔬W stop_B + 保守換股'),
    'combo_sB_p12_rD': ({'enable_dyn_stop': True, 'hard_stop_bear': -10,
                         'max_positions': 12,
                         'swap_score_margin': 1.0, 'max_swap_per_day': 1},
                        '🔬W stop_B + pos12 + 保守換股'),
    'combo_p12_rD': ({'max_positions': 12,
                      'swap_score_margin': 1.0, 'max_swap_per_day': 1},
                     '🔬W pos12 + 保守換股'),
    'combo_sB_p12_vC': ({'enable_dyn_stop': True, 'hard_stop_bear': -10,
                         'max_positions': 12,
                         'enable_vol_sizing': True, 'vol_target_pct': 2.0, 'vol_scale_floor_pct': 60},
                        '🔬W stop_B + pos12 + vol_C'),
    'combo_full': ({'enable_dyn_stop': True, 'hard_stop_bear': -10,
                    'max_positions': 12,
                    'swap_score_margin': 1.0, 'max_swap_per_day': 1,
                    'enable_vol_sizing': True, 'vol_target_pct': 2.0, 'vol_scale_floor_pct': 60},
                   '🔬W 全配 (stop_B+pos12+保守換股+vol_C)'),
    'combo_sB_p12_b9': ({'enable_dyn_stop': True, 'hard_stop_bear': -9,
                         'max_positions': 12},
                        '🔬W bear=-9 + pos12 (微調)'),
    'combo_sB_p12_b11': ({'enable_dyn_stop': True, 'hard_stop_bear': -11,
                          'max_positions': 12},
                         '🔬W bear=-11 + pos12 (微調)'),
    'combo_sB_p12_b35k': ({'enable_dyn_stop': True, 'hard_stop_bear': -10,
                           'max_positions': 12,
                           '_budget': 35_000},
                          '🔬W stop_B + pos12 + bud35K'),

    # === V4.21 Ablation X: pos12+保守換股 微調 (ablation 39 冠軍精細搜索) ===
    # 基底: max_positions=12, swap_score_margin=1.0, max_swap_per_day=1 (Sharpe 1.29)

    # --- X1: 倉位數微調 ---
    'tune_p10_s10': ({'max_positions': 10, 'swap_score_margin': 1.0, 'max_swap_per_day': 1},
                     '🔬X pos10 + margin=1.0 + swap=1'),
    'tune_p11_s10': ({'max_positions': 11, 'swap_score_margin': 1.0, 'max_swap_per_day': 1},
                     '🔬X pos11 + margin=1.0 + swap=1'),
    'tune_p12_s10': ({'max_positions': 12, 'swap_score_margin': 1.0, 'max_swap_per_day': 1},
                     '🔬X pos12 + margin=1.0 + swap=1 (=冠軍)'),
    'tune_p13_s10': ({'max_positions': 13, 'swap_score_margin': 1.0, 'max_swap_per_day': 1},
                     '🔬X pos13 + margin=1.0 + swap=1'),
    'tune_p14_s10': ({'max_positions': 14, 'swap_score_margin': 1.0, 'max_swap_per_day': 1},
                     '🔬X pos14 + margin=1.0 + swap=1'),

    # --- X2: 換股門檻微調 (pos12 固定) ---
    'tune_p12_s08': ({'max_positions': 12, 'swap_score_margin': 0.8, 'max_swap_per_day': 1},
                     '🔬X pos12 + margin=0.8 + swap=1 (原始margin)'),
    'tune_p12_s09': ({'max_positions': 12, 'swap_score_margin': 0.9, 'max_swap_per_day': 1},
                     '🔬X pos12 + margin=0.9 + swap=1'),
    'tune_p12_s11': ({'max_positions': 12, 'swap_score_margin': 1.1, 'max_swap_per_day': 1},
                     '🔬X pos12 + margin=1.1 + swap=1'),
    'tune_p12_s12': ({'max_positions': 12, 'swap_score_margin': 1.2, 'max_swap_per_day': 1},
                     '🔬X pos12 + margin=1.2 + swap=1'),

    # --- X3: 完全不換股 ---
    'tune_p12_noswap': ({'max_positions': 12, 'enable_position_swap': False},
                        '🔬X pos12 + 完全不換股'),
    'tune_p11_noswap': ({'max_positions': 11, 'enable_position_swap': False},
                        '🔬X pos11 + 完全不換股'),

    # --- X4: 冠軍 + cash/buy 微調 ---
    'tune_p12_s10_c5': ({'max_positions': 12, 'swap_score_margin': 1.0, 'max_swap_per_day': 1,
                         'cash_reserve_pct': 5.0},
                        '🔬X 冠軍 + cash=5%'),
    'tune_p12_s10_b3': ({'max_positions': 12, 'swap_score_margin': 1.0, 'max_swap_per_day': 1,
                         'max_new_buy_per_day': 3},
                        '🔬X 冠軍 + buy=3/日'),
    'tune_p12_s10_b5': ({'max_positions': 12, 'swap_score_margin': 1.0, 'max_swap_per_day': 1,
                         'max_new_buy_per_day': 5},
                        '🔬X 冠軍 + buy=5/日'),
}


# ==========================================
# 🖨️ Ablation 比較報告
# ==========================================
def print_ablation_report(all_results, configs_run, industries, start_date, end_date, budget, exec_mode):
    mode_label = "T+1 開盤" if exec_mode == 'next_open' else ("收盤+開盤雙買" if exec_mode == 'close_open' else "當日收盤")
    ind_label = " + ".join(industries) if len(industries) <= 3 else f"{len(industries)} 個產業"

    print(f"\n{'='*120}")
    print(f"🧪 Group Ablation 比較報告 [{mode_label}]")
    print(f"   產業: {ind_label}")
    print(f"   區間: {start_date} ~ {end_date} | 每次 ${budget:,}")
    print(f"{'='*120}")

    # --- 彙總表 (V4.6: 加入 Calmar) ---
    print(f"\n{'策略':<18} {'說明':<34} {'總損益':>10} {'報酬率%':>7} {'CAGR%':>7} "
          f"{'MDD%':>6} {'Sharpe':>7} {'Calmar':>7} {'勝率':>6} {'交易':>5} {'持倉':>5}")
    print("-" * 140)

    summary_rows = []
    for config_name, (_, desc) in configs_run.items():
        r = all_results.get(config_name)
        if r is None:
            continue
        ec = r.get('equity_curve', [])
        max_pos = max((e['positions'] for e in ec), default=0)
        ret_pct = r.get('total_return_pct', r['roi'])
        summary_rows.append({
            'name': config_name, 'desc': desc,
            'pnl': r['total_pnl'], 'ret_pct': ret_pct,
            'cagr': r.get('cagr', 0), 'mdd_pct': r.get('mdd_pct', 0),
            'sharpe': r.get('sharpe_ratio', 0),
            'calmar': r.get('calmar_ratio', 0),
            'win_rate': r['win_rate'], 'trades': r['trades'],
            'mdd': r['max_drawdown'], 'max_pos': max_pos,
        })

    summary_rows.sort(key=lambda x: x['pnl'], reverse=True)

    for row in summary_rows:
        marker = " *" if row['name'] == 'baseline' else ""
        print(f"{row['name']:<18} {row['desc'][:32]:<34} "
              f"{int(row['pnl']):>10,} {row['ret_pct']:>7.1f} {row['cagr']:>7.1f} "
              f"{row['mdd_pct']:>6.1f} {row['sharpe']:>7.2f} {row['calmar']:>7.2f} "
              f"{row['win_rate']:>5.0f}% {row['trades']:>5} "
              f"{row['max_pos']:>5}{marker}")

    print("-" * 130)

    # --- 差異分析 (V4.6: 加入 Calmar 差異) ---
    baseline_row = next((r for r in summary_rows if r['name'] == 'baseline'), None)
    baseline_pnl = baseline_row['pnl'] if baseline_row else 0
    baseline_mdd = baseline_row['mdd'] if baseline_row else 0
    baseline_sharpe = baseline_row['sharpe'] if baseline_row else 0
    baseline_calmar = baseline_row['calmar'] if baseline_row else 0

    if baseline_pnl != 0:
        print(f"\n📊 模組/參數貢獻度 (相對 baseline):")
        print(f"{'策略':<18} {'損益差距':>12} {'MDD差距':>12} {'Sharpe差':>8} {'Calmar差':>8} {'解讀'}")
        print("-" * 110)

        for row in summary_rows:
            if row['name'] == 'baseline':
                continue
            pnl_diff = row['pnl'] - baseline_pnl
            mdd_diff = row['mdd'] - baseline_mdd
            sharpe_diff = row['sharpe'] - baseline_sharpe
            calmar_diff = row['calmar'] - baseline_calmar

            if pnl_diff > 0 and mdd_diff <= 0:
                interp = "🟢 關掉/調整反而更好"
            elif pnl_diff < 0 and mdd_diff > 0:
                interp = "🔴 損益降+回撤增 (模組有效保護)"
            elif pnl_diff < 0 and mdd_diff <= 0:
                interp = "🟡 損益降但回撤也降 (需權衡)"
            elif pnl_diff > 0 and mdd_diff > 0:
                interp = "🟡 獲利增但回撤也增 (風險更大)"
            else:
                interp = "⚪ 差異不明顯"

            print(f"{row['name']:<18} {int(pnl_diff):>+12,} {int(mdd_diff):>+12,} "
                  f"{sharpe_diff:>+8.2f} {calmar_diff:>+8.2f} {interp}")

    print(f"\n{'='*130}")


# ==========================================
# 🎮 主程式
# ==========================================
def _select_industries():
    """共用: 選擇產業 + 收集股票清單"""
    print("\n⏳ 載入產業資料庫...")
    df_all = get_all_companies()
    industries = list_industries(df_all)

    if not industries:
        print("❌ 無法載入產業資料")
        return None, None, None, None

    print(f"\n🏭 產業列表 ({len(industries)} 個):")
    for i, ind in enumerate(industries):
        print(f"  [{i+1:02d}] {ind:<12}", end="")
        if (i + 1) % 4 == 0:
            print()
    print()

    print("\n👉 請輸入要掃描的產業 (逗號分隔編號或名稱，例如: 1,3,5 或 半導體業,光電業)")
    sel_input = input("👉 選擇: ").strip()

    selected_industries = []
    for part in sel_input.split(','):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(industries):
                selected_industries.append(industries[idx])
        else:
            if part in industries:
                selected_industries.append(part)

    if not selected_industries:
        print("❌ 未選擇任何產業")
        return None, None, None, None

    print(f"\n✅ 已選擇 {len(selected_industries)} 個產業:")
    all_stocks = []
    seen_tickers = set()
    industry_map = {}  # {ticker: industry_name}

    for ind in selected_industries:
        stocks = get_stocks_by_industry(ind)
        new_count = 0
        for ticker, name in stocks:
            if ticker not in seen_tickers:
                all_stocks.append((ticker, name))
                seen_tickers.add(ticker)
                industry_map[ticker] = ind
                new_count += 1
        print(f"   {ind}: {new_count} 檔")

    print(f"   合計: {len(all_stocks)} 檔 (去重後)")
    return selected_industries, all_stocks, industries, industry_map


# ==========================================
# 📊 Mode 5: 每日實盤訊號 — utility functions
# ==========================================

def load_portfolio(filepath):
    """讀取庫存 CSV"""
    if not os.path.exists(filepath):
        print(f"   📂 庫存檔案不存在，建立空白: {os.path.basename(filepath)}")
        df = pd.DataFrame(columns=_PORTFOLIO_COLS)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return df
    df = pd.read_csv(filepath, dtype={'ticker': str})
    return df


def _append_trade_log(records, filepath=None):
    """追加交易紀錄到 trade_log.csv"""
    if filepath is None:
        filepath = TRADE_LOG_FILE
    if not records:
        return
    new_df = pd.DataFrame(records)
    if os.path.exists(filepath):
        old_df = pd.read_csv(filepath, dtype={'ticker': str})
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"   📝 交易紀錄: +{len(records)} 筆 → {os.path.basename(filepath)}")


def _append_performance_log(date_str, initial_capital, strategy_positions, stock_data, engine_result):
    """
    V4.15: 追加每日績效到 performance_log.csv
    記錄策略內 (portfolio_strategy) 的 NAV、報酬率、回撤。
    """
    filepath = PERFORMANCE_LOG_FILE

    # 計算策略內市值 (淨變現值: 扣手續費+稅)
    stock_val_net = 0
    for ticker, pos in strategy_positions.items():
        if ticker in stock_data:
            sdf = stock_data[ticker]['df']
            vd = sdf.index[sdf.index <= pd.Timestamp(date_str)]
            if len(vd) > 0:
                cp = float(sdf.loc[vd[-1], 'Close'])
                sf = calculate_fee(cp, pos['shares'])
                st = calculate_tax(ticker, cp, pos['shares'])
                stock_val_net += pos['shares'] * cp - sf - st

    cash = engine_result.get('final_cash', initial_capital)
    nav = cash + stock_val_net
    n_positions = len(strategy_positions)

    # 讀取歷史記錄以計算累計報酬和回撤
    prev_nav = initial_capital
    peak_nav = initial_capital
    if os.path.exists(filepath):
        try:
            hist = pd.read_csv(filepath)
            if len(hist) > 0:
                prev_nav = float(hist.iloc[-1]['NAV'])
                peak_nav = float(hist['NAV'].max())
        except Exception:
            pass

    if nav > peak_nav:
        peak_nav = nav

    daily_ret = ((nav / prev_nav) - 1) * 100 if prev_nav > 0 else 0
    cum_ret = ((nav / initial_capital) - 1) * 100 if initial_capital > 0 else 0
    dd_pct = ((peak_nav - nav) / peak_nav * 100) if peak_nav > 0 else 0

    realized = engine_result.get('realized', 0)
    unrealized = int(stock_val_net - sum(
        p.get('cost_total', p['avg_cost'] * p['shares'])
        for p in strategy_positions.values()
    )) if strategy_positions else 0

    new_row = pd.DataFrame([{
        '日期': date_str,
        '持倉數': n_positions,
        '現金': int(cash),
        '持股淨值': int(stock_val_net),
        'NAV': int(nav),
        '當日報酬%': round(daily_ret, 2),
        '累計報酬%': round(cum_ret, 2),
        '回撤%': round(dd_pct, 2),
        '已實現損益': int(realized),
        '未實現損益': unrealized,
    }])

    if os.path.exists(filepath):
        old_df = pd.read_csv(filepath)
        # 避免重複寫入同一天
        old_df = old_df[old_df['日期'] != date_str]
        combined = pd.concat([old_df, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"   📈 績效日誌: {date_str} NAV=${nav:,.0f} 報酬={cum_ret:+.2f}% 回撤={dd_pct:.2f}%")


def _strategy_df_to_engine_positions(strategy_df, today_str):
    """從 portfolio_strategy.csv 提取持股，轉為引擎的 initial_positions 格式
    last_buy_date_idx: 自動用分析日 (today_str) 當基準, 假設都是「今天持有」→ idx=0
    reduce_stage: enable_reduce=False, 固定 0
    """
    def _safe_int(val, default=0):
        """NaN-safe int conversion"""
        if pd.isna(val):
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    initial_positions = {}
    for _, row in strategy_df.iterrows():
        ticker = str(row.get('ticker', '')).strip()
        if not ticker or ticker == 'nan':
            continue

        _avg = float(row['avg_cost']) if not pd.isna(row.get('avg_cost')) else 0.0
        # V4.20: peak_since_entry 從 CSV 讀取，舊 CSV 沒有此欄則用 avg_cost
        _peak_raw = row.get('peak_since_entry')
        _peak = float(_peak_raw) if (not pd.isna(_peak_raw) if _peak_raw is not None else False) else _avg

        # V6: buy_price 從 CSV 讀取，舊 CSV 沒有此欄則用 avg_cost
        _bp_raw = row.get('buy_price')
        _bp = float(_bp_raw) if (_bp_raw is not None and not pd.isna(_bp_raw)) else _avg

        initial_positions[ticker] = {
            'shares': _safe_int(row.get('shares', 0)),
            'avg_cost': _avg,
            'buy_price': _bp,          # V6: 最近一次買入價
            'name': str(row['name']) if not pd.isna(row.get('name')) else ticker,
            'buy_count': 1,            # max_add_per_stock=99, 不影響
            'last_buy_date_idx': -1,   # 假設昨天已持有 (避免殭屍股誤判第一天就賣)
            'reduce_stage': 0,         # enable_reduce=False, 固定 0
            'peak_since_entry': _peak, # V4.20: Tier B entry peak 追蹤
        }
    return initial_positions


def _save_portfolio_csv(next_rows, filepath, today_str, label):
    """將 next_rows dict 寫入 CSV (含備份)"""
    cols = _PORTFOLIO_COLS
    rows_list = []
    for t, row in next_rows.items():
        r = {col: row.get(col, '') for col in cols}
        r['ticker'] = t
        rows_list.append(r)

    next_df = pd.DataFrame(rows_list, columns=cols)

    # 備份
    basename = os.path.splitext(os.path.basename(filepath))[0]
    backup_name = f"{basename}_{today_str.replace('-', '')}.csv"
    backup_path = os.path.join(os.path.dirname(filepath), backup_name)
    if os.path.exists(filepath):
        shutil.copy2(filepath, backup_path)
        print(f"   📋 備份: {backup_name}")

    next_df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"   💾 已更新: {os.path.basename(filepath)} ({len(next_df)} 檔)")


def _update_strategy_portfolio(strategy_df, today_str,
                                engine_sells, engine_buys,
                                budget, pool_industry_map, global_industry_map,
                                stock_data, initial_positions):
    """根據引擎訊號更新 portfolio_strategy.csv"""
    trade_records = []
    next_rows = {}
    for _, row in strategy_df.iterrows():
        ticker = str(row['ticker']).strip()
        next_rows[ticker] = row.to_dict()

    # 移除: 引擎賣出
    for e in engine_sells:
        t = e['ticker']
        if t in next_rows:
            del next_rows[t]
            tag = '換股' if e['is_swap'] else ('殭屍' if '殭屍' in e['reason'] else '策略')
            print(f"   ➖ [策略內] 移除 {t} {e['name']} ({tag}賣出)")
            trade_records.append({
                'date': today_str, 'portfolio': 'strategy',
                'ticker': t, 'name': e['name'], 'industry': e['industry'],
                'action': 'SELL' if e['action'] == 'sell' else 'REDUCE',
                'shares': e['shares'], 'price': e['close'],
                'avg_cost': e['avg_cost'], 'pnl_pct': round(e['net_pnl_pct'], 2),
                'reason': e['reason'],
            })

    # 新增/加碼: 引擎買入
    for e in engine_buys:
        t = e['ticker']
        if e['is_add'] and t in next_rows:
            old = next_rows[t]
            old_shares = int(old['shares'])
            old_cost = float(old['avg_cost'])
            add_shares = e['shares_to_buy']
            if add_shares <= 0:
                continue
            new_shares = old_shares + add_shares
            new_avg = (old_cost * old_shares + e['close'] * add_shares) / new_shares
            next_rows[t]['shares'] = new_shares
            next_rows[t]['avg_cost'] = round(new_avg, 2)
            next_rows[t]['buy_price'] = round(e['close'], 2)  # V6: 最近一次買入價
            print(f"   📝 [策略內] 加碼 {t} {e['name']} +{add_shares}股 "
                  f"(總{new_shares}股, 均價${new_avg:.2f})")
            trade_records.append({
                'date': today_str, 'portfolio': 'strategy',
                'ticker': t, 'name': e['name'], 'industry': e['industry'],
                'action': 'ADD',
                'shares': add_shares, 'price': e['close'],
                'avg_cost': round(new_avg, 2), 'pnl_pct': 0,
                'reason': e['reason'],
            })
        else:
            shares = e['shares_to_buy']
            if shares <= 0:
                continue
            ind = pool_industry_map.get(t, global_industry_map.get(t, ''))
            tag = '換股買入' if e['is_swap'] else '新買入'
            next_rows[t] = {
                'ticker': t, 'name': e['name'], 'industry': ind,
                'shares': shares, 'avg_cost': e['close'],
                'buy_price': round(e['close'], 2),  # V6: 最近一次買入價
                'peak_since_entry': e['close'],  # V4.20: 新買入 peak = 買入價
                'note': tag,
            }
            print(f"   ➕ [策略內] 加入 {t} {e['name']} {shares}股 @ ${e['close']:.1f} ({tag})")
            trade_records.append({
                'date': today_str, 'portfolio': 'strategy',
                'ticker': t, 'name': e['name'], 'industry': ind,
                'action': 'BUY' if not e['is_swap'] else 'SWAP_BUY',
                'shares': shares, 'price': e['close'],
                'avg_cost': e['close'], 'pnl_pct': 0,
                'reason': e['reason'],
            })

    # V4.20: 更新所有持倉的 peak_since_entry (含今日收盤)
    for t, row in next_rows.items():
        _old_peak_raw = row.get('peak_since_entry')
        _avg = float(row.get('avg_cost', 0))
        try:
            _old_peak = float(_old_peak_raw) if _old_peak_raw and not pd.isna(_old_peak_raw) else _avg
        except (ValueError, TypeError):
            _old_peak = _avg
        # 用今日收盤更新 peak
        _today_close = _avg  # fallback
        if stock_data and t in stock_data:
            _sd = stock_data[t]
            _sdf = _sd.get('df') if isinstance(_sd, dict) else _sd
            if _sdf is not None and len(_sdf) > 0:
                _today_close = float(_sdf['Close'].iloc[-1])
        row['peak_since_entry'] = round(max(_old_peak, _today_close), 2)

    _save_portfolio_csv(next_rows, PORTFOLIO_STRATEGY_FILE, today_str, '策略內')

    n_removed = len(engine_sells)
    n_added = sum(1 for e in engine_buys if not e['is_add'] and e['shares_to_buy'] > 0)
    n_updated = sum(1 for e in engine_buys if e['is_add'] and e['shares_to_buy'] > 0)
    print(f"   📊 策略內異動: 移除 {n_removed} / 新增 {n_added} / 加碼 {n_updated}")

    return trade_records


def _update_other_portfolio(other_df, today_str, out_sell):
    """根據策略外賣出訊號更新 portfolio_other.csv (只有賣出/持有)"""
    trade_records = []
    next_rows = {}
    for _, row in other_df.iterrows():
        ticker = str(row['ticker']).strip()
        next_rows[ticker] = row.to_dict()

    for e in out_sell:
        t = e['ticker']
        if t in next_rows:
            del next_rows[t]
            print(f"   ➖ [策略外] 移除 {t} {e['name']} (L2 訊號賣出)")
            trade_records.append({
                'date': today_str, 'portfolio': 'other',
                'ticker': t, 'name': e['name'], 'industry': e['industry'],
                'action': 'SELL',
                'shares': e['shares'], 'price': e['close'],
                'avg_cost': e['avg_cost'], 'pnl_pct': round(e['net_pnl_pct'], 2),
                'reason': e['signal']['reason'] if 'signal' in e else '',
            })

    _save_portfolio_csv(next_rows, PORTFOLIO_OTHER_FILE, today_str, '策略外')

    if out_sell:
        print(f"   📊 策略外異動: 移除 {len(out_sell)}")
    else:
        print(f"   📊 策略外: 無異動")

    return trade_records


def _build_daily_stock_pool(industries):
    """建立每日訊號的標的池 + 全局產業 mapping"""
    from industry_manager import get_stocks_by_industry, get_all_companies
    all_stocks = []
    seen = set()
    industry_map = {}
    for ind in industries:
        stocks = get_stocks_by_industry(ind)
        for ticker, name in stocks:
            if ticker not in seen:
                all_stocks.append((ticker, name))
                seen.add(ticker)
                industry_map[ticker] = ind

    # 全局 mapping (所有產業)
    df = get_all_companies()
    global_industry_map = {}
    if not df.empty:
        global_industry_map = dict(zip(df['Ticker'], df['Industry']))

    return all_stocks, industry_map, global_industry_map


def _select_daily_industries():
    """互動式選擇目標產業 (mode 5 專用)"""
    from industry_manager import get_all_companies, list_industries
    print("\n⏳ 載入產業資料庫...")
    df_all = get_all_companies()
    industries = list_industries(df_all)
    if not industries:
        print("❌ 無法載入產業資料")
        return None

    print(f"\n🏭 產業列表 ({len(industries)} 個):")
    for i, ind in enumerate(industries):
        tag = ' ★' if ind in INDUSTRY_CONFIGS and INDUSTRY_CONFIGS[ind].get('config') else ''
        elim = ' ⛔' if ind in INDUSTRY_CONFIGS and '淘汰' in INDUSTRY_CONFIGS[ind].get('desc', '') else ''
        print(f"  [{i+1:02d}] {ind:<14}{tag}{elim}", end="")
        if (i + 1) % 3 == 0:
            print()
    print()
    print(f"   ★ = 有 L2 最優參數   ⛔ = 已淘汰")
    print(f"   🏆 v2.11 預設: {' + '.join(DAILY_DEFAULT_INDUSTRIES)}")
    print(f"\n👉 輸入目標產業 (逗號分隔編號, Enter=使用預設)")
    sel_input = input("👉 選擇: ").strip()

    if not sel_input:
        return DAILY_DEFAULT_INDUSTRIES

    selected = []
    for part in sel_input.split(','):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(industries):
                selected.append(industries[idx])
        elif part in industries:
            selected.append(part)
    return selected or DAILY_DEFAULT_INDUSTRIES


def _find_best_snapshot(base_filepath, target_date_str):
    """找最適合的快照檔 (≤ target_date 的最近快照)

    例: base = portfolio_strategy.csv, target = 2026-02-21
    會搜尋 portfolio_strategy_YYYYMMDD.csv, 找 ≤ 20260221 的最近一個。
    回傳完整路徑, 或 None (無快照)。
    """
    import glob as _glob
    base_dir = os.path.dirname(base_filepath)
    base_name = os.path.splitext(os.path.basename(base_filepath))[0]  # portfolio_strategy
    pattern = os.path.join(base_dir, f"{base_name}_*.csv")
    snapshots = _glob.glob(pattern)

    if not snapshots:
        return None

    target_num = target_date_str.replace('-', '')  # '20260221'

    # 提取日期, 過濾 ≤ target
    candidates = []
    for fp in snapshots:
        fname = os.path.basename(fp)
        # portfolio_strategy_20260211.csv → 20260211
        date_part = fname.replace(f"{base_name}_", '').replace('.csv', '').strip('.')
        if date_part.isdigit() and len(date_part) == 8 and date_part <= target_num:
            candidates.append((date_part, fp))

    if not candidates:
        return None

    # 取最近的 (日期最大的)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ==========================================
# 📊 Mode 5: 每日實盤訊號 — 主函數
# ==========================================

def run_daily_signal_mode(target_date=None, industries=None, budget=None,
                          initial_capital=None, auto_mode=False,
                          max_positions=None, max_new_buy=None, max_swap=None):
    """
    每日實盤訊號 (Mode 5) — 整合進 group_backtest

    使用回測引擎的現金管理 (cash_reserve_pct=10%) 控制預算上限。
    - portfolio_strategy.csv → 引擎統一處理 (buy/sell/swap/zombie/add)
    - portfolio_other.csv   → 各產業 L2 訊號 (sell/hold only)
    - 交易紀錄統一寫入 trade_log.csv
    """
    # V4.15: Block B 參數與 Mode 1/7 一致 (DEFAULT_CONFIG)
    from strategy import DEFAULT_CONFIG as _dc
    if budget is None:
        budget = DAILY_BUDGET
    if max_positions is None:
        max_positions = _dc.get('max_positions', 12)
    if max_new_buy is None:
        max_new_buy = _dc.get('max_new_buy_per_day', 4)
    if max_swap is None:
        max_swap = _dc.get('max_swap_per_day', 1)
    if initial_capital is None:
        initial_capital = 900_000

    today_str = target_date or datetime.date.today().strftime('%Y-%m-%d')

    # --- 0. 選擇目標產業 ---
    if industries is None:
        if auto_mode:
            industries = DAILY_DEFAULT_INDUSTRIES
        else:
            industries = _select_daily_industries()
            if industries is None:
                return None

    # per_industry_config — 允許選擇 L3 或 V5
    per_industry_config = {}
    _m5_cfg_labels = {}  # {ind: label} for display

    if not auto_mode:
        # 互動模式: 檢查有無 V5 可選
        _v5_industries = [ind for ind in industries
                          if ind in INDUSTRY_CONFIGS_V5 and INDUSTRY_CONFIGS_V5[ind].get('config')]

        if _v5_industries:
            print(f"\n🏭 策略參數選擇:")
            for ind in industries:
                if ind in INDUSTRY_CONFIGS and INDUSTRY_CONFIGS[ind]['config']:
                    _l3_cfg = INDUSTRY_CONFIGS[ind]
                    print(f"\n   [{ind}]")
                    print(f"   ★ L3: {_l3_cfg['desc']}")
                    if ind in _v5_industries:
                        _v5_cfg = INDUSTRY_CONFIGS_V5[ind]
                        print(f"   🧪 V5: {_v5_cfg['desc']}")

            print(f"\n   A. 使用 L3 最佳參數 (推薦, 已驗證)")
            print(f"   B. 🧪 使用 V5 冠軍參數 (實驗性, 低MDD)")
            print(f"   C. 使用 DEFAULT_CONFIG (無產業優化)")
            _param_choice = input("   👉 選擇 (Enter=A): ").strip().upper()

            if _param_choice == 'B':
                for ind in industries:
                    if ind in INDUSTRY_CONFIGS_V5 and INDUSTRY_CONFIGS_V5[ind].get('config'):
                        per_industry_config[ind] = INDUSTRY_CONFIGS_V5[ind]['config']
                        _m5_cfg_labels[ind] = 'V5'
                    elif ind in INDUSTRY_CONFIGS and INDUSTRY_CONFIGS[ind]['config']:
                        per_industry_config[ind] = INDUSTRY_CONFIGS[ind]['config']
                        _m5_cfg_labels[ind] = 'L3'
                print(f"   → 使用 V5 冠軍參數 🧪")
            elif _param_choice == 'C':
                for ind in industries:
                    _m5_cfg_labels[ind] = 'DEFAULT'
                print(f"   → 使用 DEFAULT_CONFIG")
            else:
                for ind in industries:
                    if ind in INDUSTRY_CONFIGS and INDUSTRY_CONFIGS[ind]['config']:
                        per_industry_config[ind] = INDUSTRY_CONFIGS[ind]['config']
                        _m5_cfg_labels[ind] = 'L3'
                print(f"   → 使用 L3 參數 ✅")
        else:
            # 沒有 V5 可選, 直接用 L3
            for ind in industries:
                if ind in INDUSTRY_CONFIGS and INDUSTRY_CONFIGS[ind]['config']:
                    per_industry_config[ind] = INDUSTRY_CONFIGS[ind]['config']
                    _m5_cfg_labels[ind] = 'L3'
    else:
        # auto_mode: 默認 L3
        for ind in industries:
            if ind in INDUSTRY_CONFIGS and INDUSTRY_CONFIGS[ind]['config']:
                per_industry_config[ind] = INDUSTRY_CONFIGS[ind]['config']
                _m5_cfg_labels[ind] = 'L3'

    _cfg_tag = '/'.join(sorted(set(_m5_cfg_labels.values()))) if _m5_cfg_labels else 'DEFAULT'

    print("\n" + "=" * 70)
    print(f"📊 每日實盤訊號 (Mode 5 — 策略: {_cfg_tag})")
    print(f"   日期: {today_str}")
    print(f"   策略產業: {' + '.join(industries)}")
    print(f"   倉位上限: {max_positions} | 每筆: ${budget:,.0f}")
    print(f"   初始資金: ${initial_capital:,.0f} | 保留: ${int(initial_capital * 0.1):,.0f} (10%)")
    print(f"   每日新買: {max_new_buy} | 每日換股: {max_swap}")
    for ind in industries:
        _pf_name = os.path.basename(INDUSTRY_PORTFOLIO_FILES.get(ind, PORTFOLIO_STRATEGY_FILE))
        _lbl = _m5_cfg_labels.get(ind, 'DEFAULT')
        print(f"   📂 {ind} → {_pf_name} [{_lbl}]")
    print("=" * 70)

    # --- 1. 讀取庫存 (智慧選擇: 今天→主檔, 過去→快照) ---
    _actual_today = datetime.date.today().strftime('%Y-%m-%d')
    _is_today = (today_str == _actual_today)

    # V2.15: 各產業可指定獨立 CSV (通信網路業 → portfolio_telecom.csv)
    _portfolio_files = {}  # {filepath: [industry_name, ...]}
    for ind in industries:
        _pf = INDUSTRY_PORTFOLIO_FILES.get(ind, PORTFOLIO_STRATEGY_FILE)
        if _pf not in _portfolio_files:
            _portfolio_files[_pf] = []
        _portfolio_files[_pf].append(ind)

    _csv_sources = []
    _all_strategy_dfs = []

    for _pf, _pf_inds in sorted(_portfolio_files.items()):
        if _is_today:
            _use_file = _pf
            _csv_sources.append(f'{os.path.basename(_pf)} (最新)')
            _df = load_portfolio(_use_file)
        else:
            _snap = _find_best_snapshot(_pf, today_str)
            if _snap:
                _base_name = os.path.splitext(os.path.basename(_pf))[0]
                _snap_date = os.path.basename(_snap).replace(f'{_base_name}_', '').replace('.csv', '')
                _csv_sources.append(f'{os.path.basename(_pf)} 快照 {_snap_date}')
                _df = load_portfolio(_snap)
            else:
                # 過去日期 + 無快照 → 視為空倉 (主檔是今日資料, 不適用於歷史日期)
                _csv_sources.append(f'{os.path.basename(_pf)} (無快照→空倉)')
                _df = pd.DataFrame(columns=_PORTFOLIO_COLS)

        _all_strategy_dfs.append(_df)

    if len(_all_strategy_dfs) == 1:
        _raw_strategy_df = _all_strategy_dfs[0]
    else:
        _raw_strategy_df = pd.concat(_all_strategy_dfs, ignore_index=True)
        _raw_strategy_df = _raw_strategy_df.drop_duplicates(
            subset='ticker', keep='first'
        ).reset_index(drop=True)

    _csv_source = ' + '.join(_csv_sources)

    # other (策略外) 固定用 portfolio_other.csv
    if _is_today:
        other_df = load_portfolio(PORTFOLIO_OTHER_FILE)
    else:
        _other_snap = _find_best_snapshot(PORTFOLIO_OTHER_FILE, today_str)
        if _other_snap:
            other_df = load_portfolio(_other_snap)
        else:
            # 過去日期 + 無快照 → 視為空倉
            other_df = pd.DataFrame(columns=_PORTFOLIO_COLS)

    # --- 1a. 依所選產業過濾: 策略 CSV 中不屬於目標產業的 → 移到策略外 ---
    _industry_set = set(industries)
    _mask_in = _raw_strategy_df['industry'].isin(_industry_set)
    strategy_df = _raw_strategy_df[_mask_in].copy().reset_index(drop=True)
    _moved_out = _raw_strategy_df[~_mask_in].copy().reset_index(drop=True)

    if len(_moved_out) > 0:
        other_df = pd.concat([other_df, _moved_out], ignore_index=True)
        _moved_names = ', '.join(f"{r['name']}({r['industry']})" for _, r in _moved_out.iterrows())

    n_in = len(strategy_df)
    n_out = len(other_df)
    n_total = n_in + n_out

    print(f"\n📦 庫存概覽 [{_csv_source}]:")
    print(f"   總持股: {n_total} 檔")
    print(f"   ✅ 策略內 ({' + '.join(industries)}): {n_in} 檔")
    print(f"   ⚠️ 策略外 (其他產業): {n_out} 檔")
    if len(_moved_out) > 0:
        print(f"   🔀 策略CSV→策略外: {len(_moved_out)} 檔 ({_moved_names})")
    print(f"   倉位: {n_in}/{max_positions} (策略內佔倉)")

    # V5: 盤中/盤後模式提示
    if _is_today:
        import datetime as _dt_mod
        _now = _dt_mod.datetime.now()
        _market_open = _now.replace(hour=9, minute=0, second=0)
        _market_close = _now.replace(hour=13, minute=30, second=0)
        if _now < _market_open:
            print(f"   ⏰ 盤前模式 — 使用最新庫存, 盤前數據 (尚未開盤)")
        elif _now < _market_close:
            print(f"   ⏰ 盤中模式 — 使用最新庫存 + 即時數據 (尚未收盤)")
        else:
            print(f"   ✅ 盤後模式 — 使用今日收盤數據")

    # --- 1b. 建立標的池 ---
    stock_pool, pool_industry_map, global_industry_map = _build_daily_stock_pool(industries)
    pool_tickers = {t for t, _ in stock_pool}

    # 策略內 CSV → initial_positions
    initial_positions = _strategy_df_to_engine_positions(strategy_df, today_str)

    # --- 2. 取得大盤狀態 ---
    print(f"\n⏳ 取得大盤狀態...")
    market_status = get_market_status(target_date_str=today_str)
    print_market_status(market_status, today_str)

    # --- 3. 下載資料 ---
    extra_out = []
    if not other_df.empty:
        for _, row in other_df.iterrows():
            extra_out.append((str(row['ticker']).strip(), str(row['name'])))

    # 策略內持股也需要下載 (可能跟選的產業不同)
    extra_strategy = [(t, p['name']) for t, p in initial_positions.items()]
    download_list = stock_pool + extra_out + extra_strategy
    seen_dl = set()
    unique_dl = []
    for t, n in download_list:
        if t not in seen_dl:
            unique_dl.append((t, n))
            seen_dl.add(t)

    download_start = (pd.Timestamp(today_str) - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
    download_end = (pd.Timestamp(today_str) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')

    print(f"\n⏳ 下載 {len(unique_dl)} 檔股票資料...")
    stock_data, skipped = batch_download_stocks(
        unique_dl, download_start, download_end,
        min_data_days=5, force_refresh=False, cache_ttl_hours=12
    )
    print(f"   ✅ 有效: {len(stock_data)} 檔")

    # --- 3b. 驗證今日有交易資料 ---
    _today_ts = pd.Timestamp(today_str)
    _has_today_data = any(
        _today_ts in sdata['df'].index for sdata in stock_data.values()
    )
    if not _has_today_data:
        # Mode 5 是即時訊號, 快取可能過期 → 自動強制刷新再試一次
        print(f"   ⚠️ 快取中無 {today_str} 資料, 嘗試強制重新下載...")
        stock_data, skipped = batch_download_stocks(
            unique_dl, download_start, download_end,
            min_data_days=5, force_refresh=True
        )
        print(f"   ✅ 重新下載完成: {len(stock_data)} 檔")
        _has_today_data = any(
            _today_ts in sdata['df'].index for sdata in stock_data.values()
        )
        if not _has_today_data:
            print(f"   ⚠️ {today_str} 確實非交易日或尚無收盤資料，無法產生訊號")
            print(f"   💡 台股收盤後 (13:30 後) 再試, 或確認今天是否為交易日")
            return None

    # V5 fix: 保存引擎用資料 (download_end = today_str + 5 天)
    #   yfinance 調整後價格 (Adj Close) 會因後續除息/除權而回溯修改歷史值。
    #   如果之後為了即時損益重新下載 (end=actual_today), 歷史價格會不同,
    #   導致 MA20/bias/score 計算結果跟 Mode 1 不一致。
    #   因此: 引擎用原始範圍資料 (與 Mode 1 一致), 即時損益用最新資料。
    _engine_stock_data = stock_data

    # --- 3c. 最新收盤日期 (用於即時損益顯示) ---
    _latest_price_dates = [sdata['df'].index.max() for sdata in stock_data.values() if len(sdata['df']) > 0]
    _latest_price_date = max(_latest_price_dates).strftime('%Y-%m-%d') if _latest_price_dates else today_str

    # 快取可能過舊 → 強制刷新取最新收盤 (例: 輸入3/3但今天3/4, 快取只到3/3)
    if _latest_price_date < _actual_today:
        _realtime_end = (pd.Timestamp(_actual_today) + pd.Timedelta(days=3)).strftime('%Y-%m-%d')
        print(f"   📡 快取僅到 {_latest_price_date}, 強制刷新取 {_actual_today} 收盤...")
        stock_data, skipped = batch_download_stocks(
            unique_dl, download_start, _realtime_end,
            min_data_days=5, force_refresh=True, cache_ttl_hours=12
        )
        print(f"   ✅ 刷新完成: {len(stock_data)} 檔")
        _latest_price_dates = [sdata['df'].index.max() for sdata in stock_data.values() if len(sdata['df']) > 0]
        _latest_price_date = max(_latest_price_dates).strftime('%Y-%m-%d') if _latest_price_dates else today_str

    _use_realtime = (_latest_price_date != today_str)
    if _use_realtime:
        print(f"   📡 即時損益: 使用 {_latest_price_date} 收盤 (訊號仍基於 {today_str})")

    # ============================================================
    # === SECTION A: 策略外持股分析 (直接用 check_strategy_signal) ===
    # ============================================================
    out_sell = []
    out_hold = []
    out_skip = []

    if n_out > 0:
        print(f"\n{'═'*70}")
        print(f"📋 區塊A: 策略外持股分析 ({n_out} 檔)")
        print(f"{'═'*70}")

        for _, row in other_df.iterrows():
            ticker = str(row['ticker']).strip()
            name = str(row['name'])
            shares = int(row['shares'])
            avg_cost = float(row['avg_cost'])
            reduce_stage = 0  # enable_reduce=False, 固定 0
            industry = str(row.get('industry', ''))

            # 取得 L2 參數
            _ticker_cfg = None
            if industry and industry in INDUSTRY_CONFIGS:
                _cfg = INDUSTRY_CONFIGS[industry].get('config')
                if _cfg:
                    _ticker_cfg = _cfg

            if _ticker_cfg is None:
                # 無 L2 參數 → 不分析
                close_price = 0
                if ticker in stock_data:
                    sdf = stock_data[ticker]['df']
                    if len(sdf) > 0:
                        close_price = float(sdf.iloc[-1]['Close'])  # 最新收盤
                mv = close_price * shares if close_price > 0 else avg_cost * shares
                out_skip.append({
                    'ticker': ticker, 'name': name, 'industry': industry,
                    'shares': shares, 'avg_cost': avg_cost,
                    'close': close_price, 'market_value': mv,
                })
                continue

            # 取得股票資料
            if ticker not in stock_data:
                out_skip.append({
                    'ticker': ticker, 'name': name, 'industry': industry,
                    'shares': shares, 'avg_cost': avg_cost,
                    'close': 0, 'market_value': avg_cost * shares,
                })
                continue

            sdf = stock_data[ticker]['df']
            vd = sdf.index[sdf.index <= pd.Timestamp(today_str)]
            if len(vd) < 5:
                continue

            history_df = sdf.loc[:vd[-1]]
            close_price = float(sdf.iloc[-1]['Close'])  # 最新收盤 (即時損益)
            info = build_info_dict(history_df)  # V4.15: fix — 移除多餘的 ticker 參數

            if info is None:
                mv = close_price * shares if close_price > 0 else avg_cost * shares
                out_skip.append({
                    'ticker': ticker, 'name': name, 'industry': industry,
                    'shares': shares, 'avg_cost': avg_cost,
                    'close': close_price, 'market_value': mv,
                })
                continue

            signal = check_strategy_signal(
                ticker, info,
                held_cost=avg_cost,
                held_shares=shares,
                market_status=market_status,
                history_df=history_df,
                config=_ticker_cfg,
                reduce_stage=reduce_stage,
            )

            pnl_pct = 0
            if avg_cost > 0 and close_price > 0:
                pnl = calculate_net_pnl(ticker, avg_cost, close_price, shares)
                pnl_pct = pnl['net_pnl_pct']

            mv = close_price * shares

            entry = {
                'ticker': ticker, 'name': name, 'industry': industry,
                'shares': shares, 'avg_cost': avg_cost,
                'close': close_price, 'net_pnl_pct': pnl_pct,
                'market_value': mv, 'signal': signal,
            }

            if signal['action'] == 'sell':
                out_sell.append(entry)
            else:
                out_hold.append(entry)

        # 策略外輸出
        if out_sell:
            print(f"\n   🔴 策略外賣出建議 ({len(out_sell)} 檔):")
            for e in out_sell:
                print(f"   {e['ticker']} {e['name']} [{e['industry']}] "
                      f"| {e['shares']}股 @ ${e['avg_cost']:.1f} → ${e['close']:.1f} "
                      f"| 損益 {e['net_pnl_pct']:+.1f}%")
                print(f"      原因: {e['signal']['reason']}")

        if out_hold:
            print(f"\n   🟡 策略外持有 ({len(out_hold)} 檔):")
            for e in sorted(out_hold, key=lambda x: -x['net_pnl_pct']):
                print(f"   {e['ticker']} {e['name']} [{e['industry']}] "
                      f"| {e['shares']}股 | 損益 {e['net_pnl_pct']:+.1f}%")

        if out_skip:
            print(f"\n   ⚪ 無L2參數 ({len(out_skip)} 檔): "
                  f"{', '.join(e['ticker'] for e in out_skip[:5])}"
                  f"{'...' if len(out_skip) > 5 else ''}")

    # ============================================================
    # === SECTION B/C/D: 策略內 — 呼叫回測引擎 ===
    # ============================================================
    print(f"\n{'═'*70}")
    print(f"📋 區塊B/C/D: 策略內 — 引擎分析")
    print(f"{'═'*70}")

    # V4.14: 只跑今天這一天 — 用真實部位產生明日訊號
    #   引擎只跑 1 個交易日，initial_positions = CSV 部位
    #   pending 就是明日操作 (與回測單日迴圈 100% 一致)
    engine_start = today_str
    engine_end = today_str

    market_map = reconstruct_market_history(engine_start, engine_end)

    # V4.15b: 同 Mode 1 — 單一產業自動套用產業專屬參數
    # V5 fix: 優先使用 per_industry_config (已含使用者的 V5/L3 選擇), 否則回退 INDUSTRY_CONFIGS
    _block_b_override = None
    if len(industries) == 1:
        _ind_name = industries[0]
        if _ind_name in per_industry_config and per_industry_config[_ind_name]:
            _block_b_override = per_industry_config[_ind_name]
        elif _ind_name in INDUSTRY_CONFIGS:
            _ind_cfg = INDUSTRY_CONFIGS[_ind_name]
            _block_b_override = _ind_cfg['config'] if _ind_cfg['config'] else None
        if _block_b_override:
            _cfg_label = _m5_cfg_labels.get(_ind_name, 'L3')
            print(f"\n   🏭 Block B 產業專屬參數 [{_ind_name}] ({_cfg_label}): {', '.join(f'{k}={v}' for k,v in _block_b_override.items())}")

    # 確保策略內持股也在 stock_pool 裡
    pool_set = {t for t, _ in stock_pool}
    augmented_pool = list(stock_pool)
    for ticker, pos in initial_positions.items():
        if ticker not in pool_set:
            augmented_pool.append((ticker, pos['name']))
            pool_set.add(ticker)
            ind = global_industry_map.get(ticker, '')
            if ind:
                pool_industry_map[ticker] = ind

    # preloaded_data — 優先使用引擎專用資料 (download_end = today_str + 5 天)
    #   避免即時刷新的 adjusted prices 影響引擎訊號計算
    #   Fallback: 引擎資料缺少的股票 (下載失敗/資料不足) → 用即時資料避免 crash
    preloaded = {}
    for t, sdata in _engine_stock_data.items():
        if t in pool_set:
            preloaded[t] = sdata
    # Fallback: initial_positions 或 pool 中有但引擎資料沒有的 → 用 stock_data
    for t in pool_set:
        if t not in preloaded and t in stock_data:
            preloaded[t] = stock_data[t]

    # 計算持倉成本 (僅策略內部位影響預算)
    strategy_cost = sum(p['avg_cost'] * p['shares'] for p in initial_positions.values())
    other_cost = sum(float(r['avg_cost']) * int(r['shares']) for _, r in other_df.iterrows()) if not other_df.empty else 0
    available_cash = initial_capital - strategy_cost
    cash_reserve = int(initial_capital * 0.1)
    usable_cash = available_cash - cash_reserve

    print(f"\n   💰 資金狀況 (預算僅限策略內部位):")
    print(f"      初始資金:       ${initial_capital:,.0f}")
    print(f"      策略內持倉成本: ${strategy_cost:,.0f} ({n_in} 檔)")
    print(f"      策略外持倉成本: ${other_cost:,.0f} ({n_out} 檔) ← 不影響引擎預算")
    print(f"      帳面現金:       ${available_cash:,.0f}")
    print(f"      保留金 (10%):   ${cash_reserve:,.0f}")
    print(f"      可用現金:       ${usable_cash:,.0f}")
    if usable_cash < budget:
        print(f"      ⚠️ 可用現金不足一筆買入 (${budget:,.0f}), 引擎不會建議新買入")

    print(f"\n   ⏳ 呼叫回測引擎 (initial_positions={n_in} 檔, pool={len(augmented_pool)} 檔)...")

    # V4.15b: Block B 同 Mode 1 — 含產業專屬 config_override (不傳 per_industry/industry_map)
    engine_result = run_group_backtest(
        stock_list=augmented_pool,
        start_date=engine_start,
        end_date=engine_end,
        budget_per_trade=budget,
        market_map=market_map,
        exec_mode='next_open',
        initial_capital=initial_capital,
        preloaded_data=preloaded,
        initial_positions=initial_positions,
        config_override=_block_b_override,
    )

    if engine_result is None:
        print("   ❌ 引擎執行失敗")
        return None

    # --- 從引擎結果提取明日操作 ---
    pending = engine_result.get('_pending', {})
    trade_log = engine_result.get('trade_log', [])
    raw_positions = engine_result.get('_raw_positions', {})

    from strategy import DEFAULT_CONFIG as _dc_pre
    _sim_slippage = _dc_pre.get('slippage_pct', 0.3) / 100  # 提前取得滑價比例

    engine_sells = []
    engine_buys = []

    for ticker, order in pending.items():
        action = order['action']
        reason = order.get('reason', '')
        pos = raw_positions.get(ticker, {})
        name = (stock_data.get(ticker, {}).get('name', '')
                or pos.get('name', ticker))

        if action == 'sell' or action == 'reduce':
            p_shares = pos.get('shares', 0)
            p_avg = pos.get('avg_cost', 0)
            close_price = 0
            if ticker in stock_data:
                sdf = stock_data[ticker]['df']
                if len(sdf) > 0:
                    close_price = float(sdf.iloc[-1]['Close'])  # 最新收盤
            pnl_pct = 0
            if p_avg > 0 and close_price > 0:
                pnl = calculate_net_pnl(ticker, p_avg, close_price, p_shares)
                pnl_pct = pnl['net_pnl_pct']

            is_swap = '換股' in reason
            industry = pool_industry_map.get(ticker, global_industry_map.get(ticker, ''))

            engine_sells.append({
                'ticker': ticker, 'name': name, 'industry': industry,
                'shares': p_shares, 'avg_cost': p_avg,
                'close': close_price, 'net_pnl_pct': pnl_pct,
                'reason': reason, 'is_swap': is_swap,
                'action': action,
            })

        elif action == 'buy':
            is_add = ticker in initial_positions
            is_swap = '換股' in reason
            close_price = 0
            if ticker in stock_data:
                sdf = stock_data[ticker]['df']
                if len(sdf) > 0:
                    # 取訊號日收盤 (而非最新收盤, 避免用到隔天數據)
                    _signal_dates = sdf.index[sdf.index <= pd.Timestamp(today_str)]
                    if len(_signal_dates) > 0:
                        close_price = float(sdf.loc[_signal_dates[-1], 'Close'])
                    else:
                        close_price = float(sdf.iloc[-1]['Close'])
            sim_price = close_price * (1 + _sim_slippage) if close_price > 0 else 0  # 模擬執行價
            shares_to_buy = int(budget / sim_price) if sim_price > 0 else 0
            industry = pool_industry_map.get(ticker, global_industry_map.get(ticker, ''))

            existing = initial_positions.get(ticker, {})
            _sname = (stock_data.get(ticker, {}).get('name', '')
                      or raw_positions.get(ticker, {}).get('name', '')
                      or existing.get('name', ticker))
            engine_buys.append({
                'ticker': ticker,
                'name': _sname,
                'industry': industry,
                'close': close_price, 'sim_price': sim_price,
                'shares_to_buy': shares_to_buy,
                'reason': reason, 'is_add': is_add, 'is_swap': is_swap,
                'existing_shares': existing.get('shares', 0),
                'existing_avg_cost': existing.get('avg_cost', 0),
            })

    # --- V4.15: 模擬次日現金檢查 (sell/reduce 先回收 → buy 逐筆扣款) ---
    #   Mode 5 只跑 1 天，pending 不會被執行，所以這裡模擬引擎的
    #   next_open 執行邏輯，標記哪些 buy 會因現金不足被取消。
    _sim_cash = engine_result.get('final_cash', available_cash)
    # _sim_slippage 已在上方定義
    _sim_cash_reserve = cash_reserve
    _cancelled_buys = set()  # ticker set: 被現金擋掉的 buy

    # 先模擬 sell/reduce 回收現金 (用今日收盤估算明日開盤)
    for e in engine_sells:
        if e['close'] > 0 and e['shares'] > 0:
            sim_sell_price = e['close'] * (1 - _sim_slippage)
            sim_sell_shares = e['shares']
            if e['action'] == 'reduce':
                reduce_ratio = pending.get(e['ticker'], {}).get('reduce_ratio', 0.5)
                sim_sell_shares = int(e['shares'] * reduce_ratio)
                if sim_sell_shares < 1:
                    sim_sell_shares = 1
            sf = calculate_fee(sim_sell_price, sim_sell_shares)
            st = calculate_tax(e['ticker'], sim_sell_price, sim_sell_shares)
            _sim_cash += sim_sell_shares * sim_sell_price - sf - st

    # 再模擬 buy 逐筆扣款
    for e in engine_buys:
        if e.get('sim_price', 0) > 0 and e['shares_to_buy'] > 0:
            sim_buy_price = e['sim_price']  # 直接用預計算的模擬價
            sim_buy_shares = e['shares_to_buy']  # 已用 sim_price 計算過
            if sim_buy_shares <= 0:
                _cancelled_buys.add(e['ticker'])
                continue
            bf = calculate_fee(sim_buy_price, sim_buy_shares)
            sim_cost = sim_buy_shares * sim_buy_price + bf
            if (_sim_cash - _sim_cash_reserve) < sim_cost:
                _cancelled_buys.add(e['ticker'])
            else:
                _sim_cash -= sim_cost

    # --- 輸出 B: 策略內持股狀態 ---
    if n_in > 0:
        print(f"\n   📊 策略內持倉 ({n_in} 檔):")
        in_entries = []
        for ticker, pos in initial_positions.items():
            close_price = 0
            if ticker in stock_data:
                sdf = stock_data[ticker]['df']
                if len(sdf) > 0:
                    close_price = float(sdf.iloc[-1]['Close'])  # 最新收盤
            pnl_pct = 0
            if pos['avg_cost'] > 0 and close_price > 0:
                pnl = calculate_net_pnl(ticker, pos['avg_cost'], close_price, pos['shares'])
                pnl_pct = pnl['net_pnl_pct']

            action_tag = '持有'
            if ticker in pending:
                pa = pending[ticker]['action']
                if pa == 'sell':
                    action_tag = '🔴賣出'
                elif pa == 'buy':
                    if ticker in _cancelled_buys:
                        action_tag = '❌加碼(現金不足)'
                    else:
                        action_tag = '💰加碼'
                elif pa == 'reduce':
                    action_tag = '✂️減碼'
            industry = pool_industry_map.get(ticker, global_industry_map.get(ticker, ''))

            in_entries.append({
                'ticker': ticker, 'name': pos['name'], 'industry': industry,
                'shares': pos['shares'], 'avg_cost': pos['avg_cost'],
                'close': close_price, 'pnl_pct': pnl_pct,
                'action_tag': action_tag,
            })

        _price_label = f"現價({_latest_price_date[5:]})" if _use_realtime else "現價"
        print(f"   {'代碼':<12} {'名稱':<8} {'股數':>5} {'成本':>8} {_price_label:>8} {'損益%':>7} {'明日動作'}")
        print(f"   {'─'*70}")
        for e in sorted(in_entries, key=lambda x: -x['pnl_pct']):
            print(f"   {e['ticker']:<12} {e['name']:<8} {e['shares']:>5} "
                  f"${e['avg_cost']:>7.1f} ${e['close']:>7.1f} "
                  f"{e['pnl_pct']:>+6.1f}%  {e['action_tag']}")

        in_total_mv = sum(e['close'] * e['shares'] for e in in_entries if e['close'] > 0)
        print(f"\n   📊 策略內小計: 市值 ${in_total_mv:,.0f} | 持股 {n_in}/{max_positions}")
    else:
        in_total_mv = 0

    # --- 輸出 C: 賣出明細 ---
    if engine_sells:
        print(f"\n   🔴 引擎賣出訊號 ({len(engine_sells)} 檔) → 明天開盤執行:")
        for e in engine_sells:
            tag = '[換股]' if e['is_swap'] else '[策略]'
            print(f"   {tag} {e['ticker']} {e['name']} [{e['industry']}] "
                  f"| {e['shares']}股 @ ${e['avg_cost']:.1f} → ${e['close']:.1f} "
                  f"| 損益 {e['net_pnl_pct']:+.1f}%")
            print(f"      原因: {e['reason']}")

    # --- 輸出 D: 買入明細 (V4.15: 顯示現金檢查結果) ---
    _valid_buys = [e for e in engine_buys if e['ticker'] not in _cancelled_buys]
    _failed_buys = [e for e in engine_buys if e['ticker'] in _cancelled_buys]

    if _valid_buys:
        print(f"\n   🟢 引擎買入訊號 ({len(_valid_buys)} 檔) → 明天開盤執行:")
        for e in _valid_buys:
            if e['is_add']:
                tag = '[加碼]'
                extra = f"已持 {e['existing_shares']}股 @ ${e['existing_avg_cost']:.1f}"
            elif e['is_swap']:
                tag = '[換股]'
                extra = ''
            else:
                tag = '[新買]'
                extra = ''
            _sp = e.get('sim_price', e['close'])
            cost = e['shares_to_buy'] * _sp if _sp > 0 else 0
            print(f"   {tag} {e['ticker']} {e['name']} [{e['industry']}] "
                  f"| {e['shares_to_buy']}股 × ${_sp:.1f} ≈ ${cost:,.0f}")
            if extra:
                print(f"      {extra}")
            print(f"      訊號: {e['reason']}")

    if _failed_buys:
        print(f"\n   ❌ 現金不足取消 ({len(_failed_buys)} 檔):")
        for e in _failed_buys:
            tag = '加碼' if e['is_add'] else ('換股' if e['is_swap'] else '新買')
            _sp = e.get('sim_price', e['close'])
            cost = e['shares_to_buy'] * _sp if _sp > 0 else 0
            print(f"   [{tag}] {e['ticker']} {e['name']} "
                  f"| {e['shares_to_buy']}股 ≈ ${cost:,.0f} → 現金不足取消")
            print(f"      訊號: {e['reason']}")

    if not engine_sells and not engine_buys:
        print(f"\n   ⏸️  引擎無操作建議 (維持現狀)")

    # --- trade_log 最後幾天 (供參考) ---
    if trade_log:
        recent = [t for t in trade_log if t['date'] == today_str]
        if recent:
            print(f"\n   📜 引擎 trade_log ({today_str}):")
            for t in recent:
                print(f"      {t['type']} {t['ticker']} {t['name']} "
                      f"{t['shares']}股 @ ${t['price']:.1f}")

    # ============================================================
    # === SECTION E: 行動摘要 + 更新雙 CSV + 交易紀錄 ===
    # ============================================================
    print(f"\n{'═'*70}")
    print(f"📋 區塊E: 明日行動摘要")
    print(f"{'═'*70}")

    all_actions = []
    for e in out_sell:
        all_actions.append(f"   🔴 [策略外] 賣出 {e['ticker']} {e['name']} {e['shares']}股 "
                          f"(損益 {e['net_pnl_pct']:+.1f}%)")
    for e in engine_sells:
        tag = '換股賣' if e['is_swap'] else '策略內'
        all_actions.append(f"   🔴 [{tag}] 賣出 {e['ticker']} {e['name']} {e['shares']}股 "
                          f"(損益 {e['net_pnl_pct']:+.1f}%)")
    # V4.16+V5: 漲停檢查 — 統一 Mode 1 邏輯
    #   過去日期 (有次日數據): 實際檢查次日 Open/Close vs 今日 Close (同 Mode 1)
    #   今天 (無次日數據): 預測性警告 (今日收盤 vs 前日收盤)
    from strategy import DEFAULT_CONFIG as _dc_e
    _lu_threshold = _dc_e.get('limit_up_threshold', 1.095)
    _limit_up_actual = set()  # 實際漲停 (過去日期, 有次日數據可驗證)
    _limit_up_warn = set()    # 預估漲停 (今天, 無次日數據)
    for e in engine_buys:
        t = e['ticker']
        if t not in stock_data:
            continue
        _sdf = stock_data[t]['df']
        _vd = _sdf.index[_sdf.index <= pd.Timestamp(today_str)]
        if len(_vd) < 1:
            continue
        _today_close = float(_sdf.loc[_vd[-1], 'Close'])

        # 檢查是否有次日交易數據
        _next_days = _sdf.index[_sdf.index > pd.Timestamp(today_str)]
        if len(_next_days) > 0:
            # 過去日期: 用 Mode 1 相同邏輯 — 次日開盤或收盤 >= 今日收盤 * 1.095
            _next_day = _next_days[0]
            _next_open = float(_sdf.loc[_next_day, 'Open'])
            _next_close = float(_sdf.loc[_next_day, 'Close'])
            if (_next_open >= _today_close * _lu_threshold
                    or _next_close >= _today_close * _lu_threshold):
                _limit_up_actual.add(t)
        else:
            # 今天: 預測性警告 — 今日已漲停 → 明天可能也漲停
            if len(_vd) >= 2:
                _prev_close = float(_sdf.loc[_vd[-2], 'Close'])
                if _prev_close > 0 and _today_close >= _prev_close * _lu_threshold:
                    _limit_up_warn.add(t)

    # V4.17: 52週高點距離 — 幫助判斷是否在「爬坑」中
    _52w_info = {}  # ticker → {'high52': float, 'gap_pct': float}
    for e in engine_buys:
        t = e['ticker']
        if t in stock_data:
            _sdf = stock_data[t]['df']
            _vd = _sdf.index[_sdf.index <= pd.Timestamp(today_str)]
            if len(_vd) >= 5:
                # 取最近 252 個交易日 (≈52 週) 的最高價
                _n52 = min(252, len(_vd))
                _high52 = float(_sdf.loc[_vd[-_n52:], 'High'].max())
                _curr = float(_sdf.loc[_vd[-1], 'Close'])
                if _high52 > 0:
                    _gap = (_curr - _high52) / _high52 * 100
                    _52w_info[t] = {'high52': _high52, 'gap_pct': _gap}

    for e in engine_buys:
        if e['ticker'] in _cancelled_buys:
            continue  # V4.15: 現金不足取消的不顯示在行動摘要
        if e['ticker'] in _limit_up_actual:
            continue  # V5: 實際漲停 → 不顯示在行動摘要 (確定買不到)
        if e['is_add']:
            tag = '加碼'
        elif e['is_swap']:
            tag = '換股買'
        else:
            tag = '新買入'
        _sp = e.get('sim_price', e['close'])
        cost = e['shares_to_buy'] * _sp if _sp > 0 else 0
        # 漲停標記: 實際(過去)=⛔ / 預估(今天)=⚠️
        warn = ''
        if e['ticker'] in _limit_up_warn:
            warn = ' ⚠️漲停(預估)'
        # 52週高標籤
        _52tag = ''
        if e['ticker'] in _52w_info:
            _g = _52w_info[e['ticker']]['gap_pct']
            if _g >= -1:
                _52tag = f' 🔥52週高近在咫尺({_g:+.1f}%)'
            elif _g >= -5:
                _52tag = f' 🟢近52週高({_g:+.1f}%)'
            elif _g >= -20:
                _52tag = f' (距52週高 {_g:+.1f}%)'
            else:
                _52tag = f' ⚠️距52週高遠({_g:+.1f}%)'
        all_actions.append(f"   🟢 [{tag}] 買入 {e['ticker']} {e['name']} "
                          f"{e['shares_to_buy']}股 ≈ ${cost:,.0f}{warn}{_52tag}")

    # V5: 顯示漲停被擋的股票 (實際)
    _lu_actual_buys = [e for e in engine_buys
                       if e['ticker'] in _limit_up_actual and e['ticker'] not in _cancelled_buys]
    if _lu_actual_buys:
        for e in _lu_actual_buys:
            tag = '加碼' if e['is_add'] else ('換股買' if e['is_swap'] else '新買入')
            _sp = e.get('sim_price', e['close'])
            cost = e['shares_to_buy'] * _sp if _sp > 0 else 0
            all_actions.append(f"   ⛔ [{tag}] {e['ticker']} {e['name']} "
                              f"{e['shares_to_buy']}股 ≈ ${cost:,.0f} → 漲停買不到(實際)")

    if all_actions:
        for a in all_actions:
            print(a)
    else:
        print(f"   ⏸️  無操作")

    # V4.16: 備選股票 — 引擎排名但未被選中的候選
    _all_cands = engine_result.get('_last_candidates', [])
    _selected = set(e['ticker'] for e in engine_buys)
    _backup_cands = [c for c in _all_cands if c['ticker'] not in _selected
                     and c['ticker'] not in _cancelled_buys]
    # 有買入建議時, 一律顯示備選區塊 (即使 0 檔也要讓使用者知道)
    if engine_buys:
        # 計算加碼 vs 新買入
        _n_add = sum(1 for e in engine_buys if e.get('type') == 'BUY' and e['ticker'] in initial_positions)
        _n_new = sum(1 for e in engine_buys if e.get('type') == 'BUY' and e['ticker'] not in initial_positions)
        if _backup_cands:
            n_show = min(3, len(_backup_cands))
            print(f"\n   📋 備選 (萬一漲停買不到可考慮, 新買入候選共 {len(_all_cands)} 檔):")
            for c in _backup_cands[:n_show]:
                # 漲停預警
                _bk_warn = ''
                if c['ticker'] in _limit_up_warn:
                    _bk_warn = ' ⚠️漲停'
                # 52週高
                _bk_52 = ''
                _bt = c['ticker']
                if _bt in stock_data:
                    _bsdf = stock_data[_bt]['df']
                    _bvd = _bsdf.index[_bsdf.index <= pd.Timestamp(today_str)]
                    if len(_bvd) >= 5:
                        _bn52 = min(252, len(_bvd))
                        _bh52 = float(_bsdf.loc[_bvd[-_bn52:], 'High'].max())
                        _bcurr = float(_bsdf.loc[_bvd[-1], 'Close'])
                        if _bh52 > 0:
                            _bg = (_bcurr - _bh52) / _bh52 * 100
                            _bk_52 = f' [{_bg:+.1f}%]'
                _bk_cost = int(budget / c['close_price']) * c['close_price'] if c['close_price'] > 0 else 0
                _bk_shares = int(budget / c['close_price']) if c['close_price'] > 0 else 0
                print(f"      🔸 {c['ticker']} {c['name']} "
                      f"{_bk_shares}股 ≈ ${_bk_cost:,.0f} "
                      f"(分數 {c['score']:.2f}){_bk_warn}{_bk_52}")
        else:
            print(f"\n   📋 備選: 無 (新買入候選共 {len(_all_cands)} 檔, 全部已選入)")

    # V4.18: 漲停遞補提示 — 加碼漲停→暫緩 / 新買入漲停→遞補備選
    _lu_add_tickers = [e for e in engine_buys
                       if e['ticker'] in _limit_up_warn and e['is_add']
                       and e['ticker'] not in _cancelled_buys]
    _lu_new_tickers = [e for e in engine_buys
                       if e['ticker'] in _limit_up_warn and not e['is_add'] and not e['is_swap']
                       and e['ticker'] not in _cancelled_buys]
    if _lu_add_tickers or _lu_new_tickers:
        print(f"\n   🚦 漲停因應:")
        for e in _lu_add_tickers:
            print(f"      ⏸️  {e['ticker']} {e['name']} 加碼漲停 → 暫緩, 不遞補")
        for e in _lu_new_tickers:
            # 找備選中的第一個可用候選
            _fill_hint = '無備選'
            for c in _backup_cands:
                if c['ticker'] not in _limit_up_warn:
                    _fill_hint = f"遞補 → {c['ticker']} {c['name']} (分數 {c['score']:.2f})"
                    break
            print(f"      🔄 {e['ticker']} {e['name']} 新買入漲停 → {_fill_hint}")

    # V4.17: 52 週高圖例
    if _52w_info:
        print(f"\n   📏 52週高: 🔥<-1%近在咫尺 | 🟢-1~-5%接近 | -5~-20%中等 | ⚠️>-20%還在爬坑")

    # V4.15b: 不再自動更新 CSV / log (使用者可能買不到, 需手動調整)
    # 改為只顯示建議異動, 不寫入任何檔案
    _effective_buys = [e for e in engine_buys if e['ticker'] not in _cancelled_buys]
    all_trade_records = []  # 保留結構供 return 使用

    # 帳戶總覽
    out_mv = sum(e['market_value'] for e in out_sell + out_hold) if (out_sell or out_hold) else 0
    skip_mv = sum(e['market_value'] for e in out_skip) if out_skip else 0
    out_mv_total = out_mv + skip_mv
    grand_total_mv = in_total_mv + out_mv_total
    grand_total_cost = 0
    for _, r in strategy_df.iterrows():
        grand_total_cost += float(r['avg_cost']) * int(r['shares'])
    for _, r in other_df.iterrows():
        grand_total_cost += float(r['avg_cost']) * int(r['shares'])

    print(f"\n{'═'*70}")
    if grand_total_cost > 0:
        grand_pnl = grand_total_mv - grand_total_cost
        print(f"🏦 帳戶總覽: 市值 ${grand_total_mv:,.0f} | 本金 ${grand_total_cost:,.0f} "
              f"| 損益 ${grand_pnl:+,.0f} ({grand_pnl/grand_total_cost*100:+.1f}%)")
    print(f"💡 訊號基於 {today_str} 收盤 + 回測引擎，明天開盤執行")
    print(f"{'═'*70}")

    # V4.15b: 不再自動追加績效日誌 (改用獨立程式 analyze_performance.py)

    return {
        'out_sell': out_sell, 'out_hold': out_hold, 'out_skip': out_skip,
        'engine_sells': engine_sells, 'engine_buys': engine_buys,
        'engine_result': engine_result,
        'trade_records': all_trade_records,
    }


# ==========================================
# ✅ 引擎一致性驗證 (--verify)
# ==========================================

def run_verify(industries=None, budget=None, max_positions=None,
               max_new_buy=None, max_swap=None):
    """
    驗證 initial_positions 機制與標準回測的一致性。
    1. 標準回測 6 個月 (空倉, _capture_positions=True)
    2. 取第 60 個交易日的 positions 快照
    3. 從第 60 天開始帶入 initial_positions 跑到結束
    4. 比對 trade_log 是否完全一致
    """
    if industries is None:
        industries = DAILY_DEFAULT_INDUSTRIES
    if budget is None:
        budget = DAILY_BUDGET
    if max_positions is None:
        max_positions = DAILY_MAX_POSITIONS
    if max_new_buy is None:
        max_new_buy = DAILY_MAX_NEW_BUY
    if max_swap is None:
        max_swap = DAILY_MAX_SWAP

    print("=" * 70)
    print("✅ 引擎一致性驗證 (initial_positions)")
    print(f"   產業: {' + '.join(industries)}")
    print("=" * 70)

    stock_pool, pool_industry_map, _ = _build_daily_stock_pool(industries)
    per_industry_config = {}
    for ind in industries:
        if ind in INDUSTRY_CONFIGS:
            per_industry_config[ind] = INDUSTRY_CONFIGS[ind]['config']

    end_date = datetime.date.today().strftime('%Y-%m-%d')
    start_date = (datetime.date.today() - datetime.timedelta(days=180)).strftime('%Y-%m-%d')

    config_override = {
        'max_positions': max_positions,
        'max_new_buy_per_day': max_new_buy,
        'max_swap_per_day': max_swap,
    }

    # --- Step 1: 標準回測 (全程) ---
    print(f"\n📊 Step 1: 標準回測 ({start_date} ~ {end_date})...")
    market_map = reconstruct_market_history(start_date, end_date)

    result_full = run_group_backtest(
        stock_list=stock_pool,
        start_date=start_date,
        end_date=end_date,
        budget_per_trade=budget,
        market_map=market_map,
        exec_mode='next_open',
        config_override=config_override,
        industry_map=pool_industry_map,
        per_industry_config=per_industry_config,
        _capture_positions=True,
    )

    if result_full is None:
        print("❌ 標準回測失敗")
        return False

    log_full = result_full['trade_log']
    history = result_full.get('_positions_history', [])
    shared_stock_data = result_full.get('_stock_data', None)

    print(f"   ✅ 標準回測完成: {len(log_full)} 筆交易, {len(history)} 天快照")

    if len(history) < 80:
        print("❌ 快照不足 80 天，無法驗證")
        return False

    # --- Step 2: 取中間點 ---
    split_idx = 60
    snapshot = history[split_idx]
    split_date = snapshot['date']
    split_day_idx = snapshot['day_idx']
    split_positions = snapshot['positions']
    split_cash = snapshot['cash']
    split_pending = snapshot.get('pending', {})

    print(f"\n📊 Step 2: 取第 {split_idx} 天快照 ({split_date})")
    print(f"   持倉: {len(split_positions)} 檔, 現金: ${split_cash:,.0f}")
    if split_pending:
        print(f"   掛單: {len(split_pending)} 筆 → {list(split_pending.keys())}")

    offset = split_day_idx + 1

    initial_pos = {}
    for ticker, pos in split_positions.items():
        initial_pos[ticker] = {
            'shares': pos['shares'],
            'avg_cost': pos['avg_cost'],
            'cost_total': pos.get('cost_total', pos['avg_cost'] * pos['shares']),
            'name': pos['name'],
            'buy_count': pos['buy_count'],
            'last_buy_date_idx': pos['last_buy_date_idx'] - offset,
            'reduce_stage': pos.get('reduce_stage', 0),
            'last_reduce_date_idx': pos.get('last_reduce_date_idx', -99) - offset,
        }

    initial_pend = {}
    if split_pending:
        initial_pend = {k: dict(v) for k, v in split_pending.items()}

    # --- Step 3: 從中間點的下一天跑到結束 ---
    from datetime import timedelta
    partial_start_dt = datetime.datetime.strptime(split_date, '%Y-%m-%d').date() + timedelta(days=1)
    partial_start = partial_start_dt.strftime('%Y-%m-%d')

    init_cost = sum(
        p.get('cost_total', p['avg_cost'] * p['shares'])
        for p in initial_pos.values()
    )
    partial_capital = split_cash + init_cost

    full_initial_capital = int(budget * max_positions * 1.5)
    from strategy import DEFAULT_CONFIG as _dc
    full_cash_reserve_pct = _dc.get('cash_reserve_pct', 0.0)
    full_cash_reserve = int(full_initial_capital * full_cash_reserve_pct / 100)

    partial_config = {**config_override}

    print(f"\n📊 Step 3: 從 {partial_start} 帶入 initial_positions 跑到 {end_date}...")

    result_partial = run_group_backtest(
        stock_list=stock_pool,
        start_date=partial_start,
        end_date=end_date,
        budget_per_trade=budget,
        market_map=market_map,
        exec_mode='next_open',
        config_override=partial_config,
        initial_capital=partial_capital,
        industry_map=pool_industry_map,
        per_industry_config=per_industry_config,
        initial_positions=initial_pos,
        initial_pending=initial_pend,
        preloaded_data=shared_stock_data,
        _cash_reserve_override=full_cash_reserve,
    )

    if result_partial is None:
        print("❌ 分段回測失敗")
        return False

    log_partial = result_partial['trade_log']

    print(f"   ✅ 分段回測完成: {len(log_partial)} 筆交易")

    # --- Step 4: 比對 ---
    print(f"\n📊 Step 4: 比對 trade_log")

    log_full_after = [t for t in log_full if t['date'] > split_date]

    print(f"   標準回測 (split後): {len(log_full_after)} 筆")
    print(f"   分段回測:           {len(log_partial)} 筆")

    match_count = 0
    diff_count = 0
    diffs = []

    max_compare = max(len(log_full_after), len(log_partial))
    for i in range(max_compare):
        a = log_full_after[i] if i < len(log_full_after) else None
        b = log_partial[i] if i < len(log_partial) else None

        if a is None or b is None:
            diff_count += 1
            diffs.append((i, a, b))
            continue

        same = (a['date'] == b['date'] and
                a['ticker'] == b['ticker'] and
                a['type'] == b['type'] and
                a['shares'] == b['shares'])

        if same:
            match_count += 1
        else:
            diff_count += 1
            diffs.append((i, a, b))

    print(f"\n   {'='*50}")
    if diff_count == 0:
        print(f"   ✅ 完全一致! {match_count}/{match_count} 筆交易吻合")
    else:
        print(f"   ⚠️  差異: {diff_count} 筆 (一致 {match_count} 筆)")
        for idx, a, b in diffs[:10]:
            print(f"\n   差異 #{idx}:")
            if a:
                print(f"      FULL:    {a['date']} {a['ticker']} {a['type']} {a['shares']}股")
            else:
                print(f"      FULL:    (無)")
            if b:
                print(f"      PARTIAL: {b['date']} {b['ticker']} {b['type']} {b['shares']}股")
            else:
                print(f"      PARTIAL: (無)")

    # 比對最終持倉
    raw_full = result_full.get('_raw_positions', {})
    raw_partial = result_partial.get('_raw_positions', {})

    full_tickers = set(raw_full.keys())
    partial_tickers = set(raw_partial.keys())

    if full_tickers == partial_tickers:
        print(f"\n   ✅ 最終持倉一致: {len(full_tickers)} 檔")
    else:
        only_full = full_tickers - partial_tickers
        only_partial = partial_tickers - full_tickers
        print(f"\n   ⚠️  最終持倉差異:")
        if only_full:
            print(f"      只在 FULL: {only_full}")
        if only_partial:
            print(f"      只在 PARTIAL: {only_partial}")

    cash_diff = abs(result_full['final_cash'] - result_partial['final_cash'])
    if cash_diff < 1:
        print(f"   ✅ 最終現金一致: ${result_full['final_cash']:,.0f}")
    else:
        print(f"   ⚠️  現金差異: FULL ${result_full['final_cash']:,.0f} "
              f"vs PARTIAL ${result_partial['final_cash']:,.0f} (差 ${cash_diff:,.0f})")

    print(f"\n{'='*70}")
    return diff_count == 0


# ==========================================
# ✅ Mode 7: 逐日訊號 vs 回測一致性驗證 (V4.14)
#
#   證明「每天帶真實部位跑 1 天引擎」的結果
#   與「標準回測一次跑完」完全一致。
#
#   做法:
#     A. 標準回測 (空倉 → 跑 N 天, _capture_positions=True)
#     B. 逐日模擬: 每天取前一天的 positions+pending,
#        跑 1 天引擎, 收集 trade_log
#     C. 比對 A 和 B 的 trade_log
# ==========================================

def run_daily_vs_backtest_verify(industries=None, budget=None,
                                  start_date_override=None, end_date_override=None):
    """
    驗證逐日 1-day 引擎 vs 標準回測的一致性。
    V5.0: 參數與 Mode 1 標準回測完全一致 (L3最佳參數)

    Args:
        industries:    目標產業 (None=使用預設)
        budget:        每筆金額
        start_date_override: 指定起始日 (YYYY-MM-DD)
        end_date_override:   指定結束日 (YYYY-MM-DD)
    """
    if industries is None:
        industries = DAILY_DEFAULT_INDUSTRIES
    if budget is None:
        budget = DAILY_BUDGET

    # V4.15: 直接從 DEFAULT_CONFIG 讀取 — 與 Mode 1 完全一致
    from strategy import DEFAULT_CONFIG as _dc
    max_positions = _dc.get('max_positions', 12)
    max_new_buy = _dc.get('max_new_buy_per_day', 4)
    max_swap = _dc.get('max_swap_per_day', 1)

    # V4.15b+V4.17: 產業專屬參數自動偵測 (同 Mode 1)
    _industry_override = None
    _cfg_desc = "DEFAULT_CONFIG"
    _use_per_industry_m7 = False
    _per_industry_config_m7 = {}

    if len(industries) == 1:
        _ind_name = industries[0]
        if _ind_name in INDUSTRY_CONFIGS:
            _ind_cfg = INDUSTRY_CONFIGS[_ind_name]
            _industry_override = _ind_cfg['config'] if _ind_cfg['config'] else None
            _has_v5 = _ind_name in INDUSTRY_CONFIGS_V5 and INDUSTRY_CONFIGS_V5[_ind_name].get('config')
            if _industry_override:
                print(f"\n🏭 產業參數選擇 [{_ind_name}]:")
                print(f"   ★ L3: {_ind_cfg['desc']}")
                print(f"   覆蓋項目: {', '.join(_industry_override.keys())}")
                if _has_v5:
                    _v5_cfg = INDUSTRY_CONFIGS_V5[_ind_name]
                    print(f"   🧪 V5: {_v5_cfg['desc']}")
                    print(f"\n   A. 使用 L3 最佳參數 (推薦, 已驗證)")
                    print(f"   B. 🧪 使用 V5 冠軍參數 (實驗性)")
                    print(f"   C. 使用 DEFAULT_CONFIG (無產業優化)")
                    _param_choice = input("   👉 選擇 (Enter=A): ").strip().upper()
                    if _param_choice == 'B':
                        _industry_override = _v5_cfg['config']
                        _cfg_desc = f"V5冠軍 ({', '.join(f'{k}={v}' for k,v in _industry_override.items())})"
                        print(f"   → 使用 V5 冠軍參數 🧪")
                    elif _param_choice == 'C':
                        _industry_override = None
                        _cfg_desc = "DEFAULT_CONFIG (使用者選擇)"
                        print(f"   → 使用 DEFAULT_CONFIG")
                    else:
                        _cfg_desc = f"L3 ({', '.join(f'{k}={v}' for k,v in _industry_override.items())})"
                        print(f"   → 使用 L3 參數 ✅")
                else:
                    print(f"\n   A. 使用 L3 最佳參數 (推薦)")
                    print(f"   B. 使用 DEFAULT_CONFIG")
                    _param_choice = input("   👉 選擇 (Enter=A): ").strip().upper()
                    if _param_choice == 'B':
                        _industry_override = None
                        _cfg_desc = "DEFAULT_CONFIG (使用者選擇)"
                        print(f"   → 使用 DEFAULT_CONFIG")
                    else:
                        _cfg_desc = f"L3 ({', '.join(f'{k}={v}' for k,v in _industry_override.items())})"
                        print(f"   → 使用 L3 參數 ✅")
    elif len(industries) > 1:
        # V4.17: 多產業讓使用者選擇
        _configs_found_m7 = []
        for _ind in industries:
            if _ind in INDUSTRY_CONFIGS and INDUSTRY_CONFIGS[_ind]['config']:
                _configs_found_m7.append((_ind, INDUSTRY_CONFIGS[_ind]['config']))
                _per_industry_config_m7[_ind] = INDUSTRY_CONFIGS[_ind]['config']
            else:
                _configs_found_m7.append((_ind, None))

        _n_with_l2 = sum(1 for _, c in _configs_found_m7 if c is not None)
        if _n_with_l2 > 0:
            _l2_industries_m7 = [(i, c) for i, c in _configs_found_m7 if c is not None]

            print(f"\n🏭 多產業參數選擇 ({_n_with_l2}/{len(industries)} 個產業有 L2):")
            for _ind, _cfg in _configs_found_m7:
                if _cfg:
                    print(f"   ★ {_ind}: {INDUSTRY_CONFIGS[_ind]['desc']}")
                else:
                    print(f"   · {_ind}: DEFAULT_CONFIG")
            print(f"\n   A. 各產業用各自 L2 最佳參數 (無L2的用DEFAULT)")
            print(f"   B. 全部統一用 DEFAULT_CONFIG")
            print(f"   C. 全部統一用某一產業的 L2 (擴大選股池, 參數不變)")
            _param_choice = input("   👉 選擇 (Enter=A): ").strip().upper()
            if _param_choice == 'C':
                if len(_l2_industries_m7) == 1:
                    _chosen_ind, _chosen_cfg = _l2_industries_m7[0]
                else:
                    print(f"\n   選擇要統一使用哪個產業的 L2:")
                    for _ci, (_cind, _ccfg) in enumerate(_l2_industries_m7, 1):
                        print(f"      {_ci}. {_cind}")
                    _c_choice = input(f"      👉 選擇 (Enter=1): ").strip()
                    _c_idx = int(_c_choice) - 1 if _c_choice.isdigit() else 0
                    _c_idx = max(0, min(_c_idx, len(_l2_industries_m7) - 1))
                    _chosen_ind, _chosen_cfg = _l2_industries_m7[_c_idx]
                _industry_override = _chosen_cfg
                _per_industry_config_m7 = {}
                _use_per_industry_m7 = False
                _cfg_desc = f"統一用 {_chosen_ind} L2 ({', '.join(f'{k}={v}' for k, v in _chosen_cfg.items())})"
                print(f"   → 全部股票統一用 {_chosen_ind} L2 參數 ✅")
            elif _param_choice == 'B':
                _per_industry_config_m7 = {}
                _cfg_desc = "DEFAULT_CONFIG (使用者選擇)"
                print(f"   → 統一使用 DEFAULT_CONFIG")
            else:
                _use_per_industry_m7 = True
                _industry_override = None
                _cfg_desc = "各產業 L2 參數"
                print(f"   → 各產業套用各自 L2 參數 ✅")
        else:
            _cfg_desc = "DEFAULT_CONFIG (無產業有 L2 參數)"

    print("=" * 70)
    print("✅ 逐日訊號 vs 回測一致性驗證 (V4.15b — 同 Mode 1 含產業專屬參數)")
    print(f"   產業: {' + '.join(industries)}")
    print(f"   倉位: {max_positions} | 每筆: ${budget:,} | 新買: {max_new_buy} | 換股: {max_swap}")
    print(f"   初始資金: $900,000 | 參數: {_cfg_desc}")
    print("=" * 70)

    stock_pool, pool_industry_map, _ = _build_daily_stock_pool(industries)

    # V4.15: 支援指定日期範圍
    if start_date_override and end_date_override:
        start_date = start_date_override
        end_date = end_date_override
    else:
        end_date = datetime.date.today().strftime('%Y-%m-%d')
        start_date = (datetime.date.today() - datetime.timedelta(days=120)).strftime('%Y-%m-%d')

    # V4.17: 快取檢查 (同 Mode 1, 確保兩邊用同一份資料)
    _m7_sim_start = pd.Timestamp(start_date)
    _m7_dl_start = (_m7_sim_start - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
    _m7_dl_end = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')

    print(f"\n🔍 檢查資料快取...")
    cache_info = check_cache_vs_latest(stock_pool, _m7_dl_start, _m7_dl_end)
    print_cache_check(cache_info)

    _m7_force_refresh = False
    if cache_info is not None and not cache_info['data_match']:
        print(f"\n   ⚠️ 快取與最新數據有差異！")
        refresh_choice = input("   👉 要更新資料嗎？(Y=重新下載 / Enter=繼續用快取): ").strip().upper()
        _m7_force_refresh = (refresh_choice == 'Y')
    elif cache_info is not None:
        print(f"   ✅ 快取數據與最新一致，直接使用")

    # --- Step 1: 標準回測 (全程, 空倉起步) ---
    print(f"\n📊 Step 1: 標準回測 ({start_date} ~ {end_date})...")
    market_map = reconstruct_market_history(start_date, end_date)

    _m7_engine_kwargs = dict(
        stock_list=stock_pool,
        start_date=start_date,
        end_date=end_date,
        budget_per_trade=budget,
        market_map=market_map,
        exec_mode='next_open',
        initial_capital=900_000,
        _capture_positions=True,
        config_override=_industry_override,
        force_refresh=_m7_force_refresh,
    )
    if _use_per_industry_m7:
        _m7_engine_kwargs['industry_map'] = pool_industry_map
        _m7_engine_kwargs['per_industry_config'] = _per_industry_config_m7
        _m7_engine_kwargs['config_override'] = None  # 策略參數由 per_industry 負責

    result_full = run_group_backtest(**_m7_engine_kwargs)

    if result_full is None:
        print("❌ 標準回測失敗")
        return False

    log_full = result_full['trade_log']
    history = result_full.get('_positions_history', [])
    shared_stock_data = result_full.get('_stock_data', None)
    initial_capital = result_full['initial_capital']

    from strategy import DEFAULT_CONFIG as _dc
    _cash_reserve_pct = _dc.get('cash_reserve_pct', 0.0)
    _cash_reserve = int(initial_capital * _cash_reserve_pct / 100)

    print(f"   ✅ 標準回測完成: {len(log_full)} 筆交易, {len(history)} 天快照")

    # 輸出完整績效報告 (同 Mode 1)
    print_group_report(result_full, industries, start_date, end_date, budget)

    if len(history) < 2:
        print("❌ 交易日不足 (至少需要 2 天)，無法驗證")
        return False

    # --- Step 2: 逐日模擬 ---
    # snapshot[i] 記錄 Day i 結束時的 positions/cash/pending
    # pending 是 Day i 的訊號 → 應在 Day i+1 開盤執行
    # 所以: 用 snapshot[i] 的 positions+pending, 以 snapshot[i+1] 的日期跑引擎
    n_steps = len(history) - 1  # 最後一天沒有下一天可跑
    print(f"\n📊 Step 2: 逐日 1-day 引擎模擬 ({n_steps} 步, 涵蓋 {len(history)} 個交易日)...")

    daily_trade_log = []  # 累積的 trade_log

    for i in range(n_steps):
        snapshot = history[i]
        next_date = history[i + 1]['date']  # pending 執行日
        day_positions = snapshot['positions']
        day_cash = snapshot['cash']
        day_pending = snapshot.get('pending', {})
        day_idx_offset = snapshot['day_idx']

        # 重建 initial_positions (調整 date_idx offset)
        init_pos = {}
        for ticker, pos in day_positions.items():
            init_pos[ticker] = {
                'shares': pos['shares'],
                'avg_cost': pos['avg_cost'],
                'cost_total': pos.get('cost_total', pos['avg_cost'] * pos['shares']),
                'name': pos.get('name', ticker),
                'buy_count': pos.get('buy_count', 1),
                'last_buy_date_idx': pos.get('last_buy_date_idx', -10) - day_idx_offset,
                'reduce_stage': pos.get('reduce_stage', 0),
                'last_reduce_date_idx': pos.get('last_reduce_date_idx', -99) - day_idx_offset,
                'peak_since_entry': pos.get('peak_since_entry', pos['avg_cost']),  # V4.20: Tier B entry peak
            }

        # 重建 initial_pending
        init_pend = {k: dict(v) for k, v in day_pending.items()} if day_pending else {}

        # 計算資金: cash + 持倉成本
        init_cost = sum(
            p.get('cost_total', p['avg_cost'] * p['shares'])
            for p in init_pos.values()
        )
        partial_capital = day_cash + init_cost

        # 跑 1 天引擎: 以下一交易日為目標日 (V4.17: 同 Step 1 參數)
        _m7_day_kwargs = dict(
            stock_list=stock_pool,
            start_date=next_date,
            end_date=next_date,
            budget_per_trade=budget,
            market_map=market_map,
            exec_mode='next_open',
            initial_capital=partial_capital,
            initial_positions=init_pos,
            initial_pending=init_pend,
            preloaded_data=shared_stock_data,
            _cash_reserve_override=_cash_reserve,
            config_override=_industry_override,
        )
        if _use_per_industry_m7:
            _m7_day_kwargs['industry_map'] = pool_industry_map
            _m7_day_kwargs['per_industry_config'] = _per_industry_config_m7
            _m7_day_kwargs['config_override'] = None

        result_day = run_group_backtest(**_m7_day_kwargs)

        if result_day is None:
            continue

        # 收集這一天的 trade_log
        for t in result_day['trade_log']:
            daily_trade_log.append(t)

        # 進度提示 (每 20 步報一次)
        if (i + 1) % 20 == 0 or i == n_steps - 1:
            print(f"   📅 {i+1}/{n_steps} ({next_date}) — 累計 {len(daily_trade_log)} 筆交易")

    print(f"\n   ✅ 逐日模擬完成: {len(daily_trade_log)} 筆交易")

    # --- Step 3: 比對 ---
    print(f"\n📊 Step 3: 比對 trade_log")
    print(f"   標準回測: {len(log_full)} 筆")
    print(f"   逐日模擬: {len(daily_trade_log)} 筆")

    match_count = 0
    diff_count = 0
    diffs = []

    max_compare = max(len(log_full), len(daily_trade_log))
    for i in range(max_compare):
        a = log_full[i] if i < len(log_full) else None
        b = daily_trade_log[i] if i < len(daily_trade_log) else None

        if a is None or b is None:
            diff_count += 1
            diffs.append((i, a, b))
            continue

        same = (a['date'] == b['date'] and
                a['ticker'] == b['ticker'] and
                a['type'] == b['type'] and
                a['shares'] == b['shares'])

        if same:
            match_count += 1
        else:
            diff_count += 1
            diffs.append((i, a, b))

    print(f"\n   {'='*50}")
    if diff_count == 0:
        print(f"   ✅ 完全一致! {match_count}/{match_count} 筆交易吻合")
        print(f"   → 逐日帶部位跑引擎 = 標準回測，邏輯 100% 一致")
    else:
        print(f"   ⚠️  差異: {diff_count} 筆 (一致 {match_count} 筆)")
        for idx, a, b in diffs[:15]:
            print(f"\n   差異 #{idx}:")
            if a:
                print(f"      BACKTEST: {a['date']} {a['ticker']} {a['type']} {a['shares']}股 "
                      f"@ ${a['price']:.1f}")
            else:
                print(f"      BACKTEST: (無)")
            if b:
                print(f"      DAILY:   {b['date']} {b['ticker']} {b['type']} {b['shares']}股 "
                      f"@ ${b['price']:.1f}")
            else:
                print(f"      DAILY:   (無)")
        if len(diffs) > 15:
            print(f"\n   ... 還有 {len(diffs) - 15} 個差異未顯示")

    # 比對最終績效
    print(f"\n   📊 績效比較:")
    print(f"      {'':>18} {'標準回測':>12} {'逐日模擬':>12}")
    print(f"      {'已實現損益':>18} ${result_full['realized']:>11,.0f}  (逐日模擬不計算完整績效)")
    print(f"      {'交易次數':>18} {result_full['trades']:>12}  {len(daily_trade_log):>12} (筆)")

    # --- Step 4: 輸出 CSV 報告 ---
    os.makedirs(REPORT_DIR, exist_ok=True)
    s_tag = start_date.replace('-', '')
    e_tag = end_date.replace('-', '')

    # 4a: 標準回測每日快照 (reuse existing export)
    csv_backtest_daily = export_daily_csv(result_full, 'verify_backtest',
                                          start_date, end_date)

    # 4b: 交易明細 CSV — 回測 vs 逐日, 方便 Excel 逐筆對照
    import csv as _csv
    _trade_columns = ['日期', '股票代碼', '股票名稱', '類型', '價格', '股數',
                      '手續費/稅', '損益', '報酬率%', '原因', '持有張數']

    def _write_trade_csv(filepath, trade_list):
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            w = _csv.writer(f)
            w.writerow(_trade_columns)
            for t in trade_list:
                w.writerow([
                    t.get('date', ''),
                    t.get('ticker', ''),
                    t.get('name', ''),
                    t.get('type', ''),
                    f"{t['price']:.2f}" if t.get('price') else '',
                    t.get('shares', ''),
                    f"{t['fee']:.0f}" if t.get('fee') else '',
                    f"{t['profit']:.0f}" if t.get('profit') is not None else '',
                    f"{t['roi']:.2f}" if t.get('roi') is not None else '',
                    t.get('note', ''),
                    t.get('total_shares', ''),
                ])

    csv_bt = os.path.join(REPORT_DIR, f'verify_trades_backtest_{s_tag}_{e_tag}.csv')
    csv_daily = os.path.join(REPORT_DIR, f'verify_trades_daily_{s_tag}_{e_tag}.csv')
    _write_trade_csv(csv_bt, log_full)
    _write_trade_csv(csv_daily, daily_trade_log)

    # 4c: 合併比對 CSV — 同一列顯示回測 vs 逐日, 差異標記
    csv_compare = os.path.join(REPORT_DIR, f'verify_compare_{s_tag}_{e_tag}.csv')
    with open(csv_compare, 'w', newline='', encoding='utf-8-sig') as f:
        w = _csv.writer(f)
        w.writerow(['#', '一致',
                     'BT日期', 'BT代碼', 'BT名稱', 'BT類型', 'BT價格', 'BT股數',
                     'DY日期', 'DY代碼', 'DY名稱', 'DY類型', 'DY價格', 'DY股數'])
        for i in range(max_compare):
            a = log_full[i] if i < len(log_full) else {}
            b = daily_trade_log[i] if i < len(daily_trade_log) else {}
            same = (a.get('date') == b.get('date') and
                    a.get('ticker') == b.get('ticker') and
                    a.get('type') == b.get('type') and
                    a.get('shares') == b.get('shares'))
            w.writerow([
                i + 1, '✓' if same else '✗',
                a.get('date', ''), a.get('ticker', ''), a.get('name', ''),
                a.get('type', ''), f"{a['price']:.2f}" if a.get('price') else '',
                a.get('shares', ''),
                b.get('date', ''), b.get('ticker', ''), b.get('name', ''),
                b.get('type', ''), f"{b['price']:.2f}" if b.get('price') else '',
                b.get('shares', ''),
            ])

    print(f"\n   📁 報告已輸出:")
    if csv_backtest_daily:
        print(f"      回測每日快照: {csv_backtest_daily}")
    print(f"      回測交易明細: {csv_bt}")
    print(f"      逐日交易明細: {csv_daily}")
    print(f"      合併比對表:   {csv_compare}")

    print(f"\n{'='*70}")
    return diff_count == 0


# ==========================================
# 🔍 Mode 6: 純產業掃描 (不帶部位)
# ==========================================

def run_scan_mode(target_date=None, industries=None, budget=None,
                  initial_capital=None, auto_mode=False):
    """
    純掃描模式：不帶任何現有部位，掃描某產業所有股票，
    用該產業的 L2 最優參數，輸出今天的買入候選清單。

    不讀取也不更新任何 CSV。
    """
    if budget is None:
        budget = DAILY_BUDGET
    if initial_capital is None:
        initial_capital = 900_000

    today_str = target_date or datetime.date.today().strftime('%Y-%m-%d')

    # --- 選擇產業 ---
    if industries is None:
        if auto_mode:
            industries = DAILY_DEFAULT_INDUSTRIES
        else:
            industries = _select_daily_industries()
            if industries is None:
                return None

    per_industry_config = {}
    for ind in industries:
        if ind in INDUSTRY_CONFIGS:
            per_industry_config[ind] = INDUSTRY_CONFIGS[ind]['config']

    print("\n" + "=" * 70)
    print(f"🔍 產業掃描模式 (Mode 6 — 不帶部位)")
    print(f"   日期: {today_str}")
    print(f"   掃描產業: {' + '.join(industries)}")
    print(f"   每筆: ${budget:,.0f} | 初始資金: ${initial_capital:,.0f}")
    print("=" * 70)

    # --- 建立標的池 ---
    stock_pool, pool_industry_map, _ = _build_daily_stock_pool(industries)
    print(f"\n   📊 標的池: {len(stock_pool)} 檔")

    # --- 下載資料 ---
    download_start = (pd.Timestamp(today_str) - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
    download_end = (pd.Timestamp(today_str) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')

    print(f"   ⏳ 下載 {len(stock_pool)} 檔股票資料...")
    stock_data, skipped = batch_download_stocks(
        stock_pool, download_start, download_end,
        min_data_days=5, force_refresh=False
    )
    print(f"   ✅ 有效: {len(stock_data)} 檔")

    # --- 取得大盤狀態 ---
    print(f"   ⏳ 取得大盤狀態...")
    market_status = get_market_status(target_date_str=today_str)
    print_market_status(market_status, today_str)

    # --- 空倉跑引擎 (最近 60 天，含 MA 前置期) ---
    engine_start = (pd.Timestamp(today_str) - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
    engine_end = today_str

    market_map = reconstruct_market_history(engine_start, engine_end)

    config_override = {}  # 使用 DEFAULT_CONFIG (15/4/2)

    preloaded = {t: sdata for t, sdata in stock_data.items()}

    print(f"\n   ⏳ 空倉跑引擎 ({engine_start} ~ {engine_end}, {len(stock_pool)} 檔)...")

    engine_result = run_group_backtest(
        stock_list=stock_pool,
        start_date=engine_start,
        end_date=engine_end,
        budget_per_trade=budget,
        market_map=market_map,
        exec_mode='next_open',
        config_override=config_override,
        initial_capital=initial_capital,
        preloaded_data=preloaded,
        industry_map=pool_industry_map,
        per_industry_config=per_industry_config,
    )

    if engine_result is None:
        print("   ❌ 引擎執行失敗")
        return None

    # --- 提取結果 ---
    pending = engine_result.get('_pending', {})
    trade_log = engine_result.get('trade_log', [])
    raw_positions = engine_result.get('_raw_positions', {})

    # 明日買入候選 (pending 中 action=buy 的)
    buy_candidates = []
    for ticker, order in pending.items():
        if order['action'] == 'buy':
            reason = order.get('reason', '')
            close_price = 0
            if ticker in stock_data:
                sdf = stock_data[ticker]['df']
                vd = sdf.index[sdf.index <= pd.Timestamp(today_str)]
                if len(vd) > 0:
                    close_price = float(sdf.loc[vd[-1], 'Close'])
            shares_to_buy = int(budget / close_price) if close_price > 0 else 0
            cost = shares_to_buy * close_price
            industry = pool_industry_map.get(ticker, '')
            name = raw_positions.get(ticker, {}).get('name', ticker)

            buy_candidates.append({
                'ticker': ticker, 'name': name, 'industry': industry,
                'close': close_price, 'shares': shares_to_buy, 'cost': cost,
                'reason': reason,
            })

    # 目前持倉 (引擎在這 10 天買入的)
    current_holdings = []
    for ticker, pos in raw_positions.items():
        close_price = 0
        if ticker in stock_data:
            sdf = stock_data[ticker]['df']
            vd = sdf.index[sdf.index <= pd.Timestamp(today_str)]
            if len(vd) > 0:
                close_price = float(sdf.loc[vd[-1], 'Close'])
        pnl_pct = 0
        if pos['avg_cost'] > 0 and close_price > 0:
            pnl = calculate_net_pnl(ticker, pos['avg_cost'], close_price, pos['shares'])
            pnl_pct = pnl['net_pnl_pct']

        # 檢查是否有賣出 pending
        action_tag = '持有'
        if ticker in pending:
            pa = pending[ticker]['action']
            if pa == 'sell':
                action_tag = '🔴賣出'
            elif pa == 'buy':
                action_tag = '💰加碼'

        industry = pool_industry_map.get(ticker, '')
        current_holdings.append({
            'ticker': ticker, 'name': pos['name'], 'industry': industry,
            'shares': pos['shares'], 'avg_cost': pos['avg_cost'],
            'close': close_price, 'pnl_pct': pnl_pct,
            'action_tag': action_tag,
        })

    # --- 輸出 ---
    print(f"\n{'═'*70}")
    print(f"📊 掃描結果 ({today_str})")
    print(f"{'═'*70}")

    # 引擎在 10 天內建立的部位
    if current_holdings:
        print(f"\n   📦 引擎 10 天模擬持倉 ({len(current_holdings)} 檔):")
        print(f"   {'代碼':<12} {'名稱':<8} {'股數':>5} {'成本':>8} {'現價':>8} {'損益%':>7} {'明日動作'}")
        print(f"   {'─'*70}")
        for e in sorted(current_holdings, key=lambda x: -x['pnl_pct']):
            print(f"   {e['ticker']:<12} {e['name']:<8} {e['shares']:>5} "
                  f"${e['avg_cost']:>7.1f} ${e['close']:>7.1f} "
                  f"{e['pnl_pct']:>+6.1f}%  {e['action_tag']}")

    # 明日買入候選
    if buy_candidates:
        print(f"\n   🟢 明日買入候選 ({len(buy_candidates)} 檔):")
        for e in buy_candidates:
            print(f"   {e['ticker']} {e['name']} [{e['industry']}] "
                  f"| {e['shares']}股 × ${e['close']:.1f} ≈ ${e['cost']:,.0f}")
            print(f"      訊號: {e['reason']}")
    else:
        print(f"\n   ⏸️  今日無新買入候選")

    # 近期交易紀錄
    if trade_log:
        recent = [t for t in trade_log if t['date'] >= engine_start]
        if recent:
            print(f"\n   📜 近 10 天交易紀錄 ({len(recent)} 筆):")
            for t in recent[-10:]:
                print(f"      {t['date']} {t['type']:<5} {t['ticker']} {t['name']} "
                      f"{t['shares']}股 @ ${t['price']:.1f}")

    # 統計
    print(f"\n{'═'*70}")
    print(f"📊 統計: {len(stock_pool)} 檔掃描 | "
          f"{len(current_holdings)} 檔模擬持倉 | "
          f"{len(buy_candidates)} 檔明日候選")
    print(f"💡 此為空倉模擬，僅供選股參考，不更新任何 CSV")
    print(f"{'═'*70}")

    return {
        'buy_candidates': buy_candidates,
        'current_holdings': current_holdings,
        'trade_log': trade_log,
        'engine_result': engine_result,
    }


# ==========================================
# 🆚 Mode 8b: 策略內 vs 策略外 績效比較 (V4.20)
# ==========================================

def run_strategy_vs_other_comparison():
    """
    比較策略內 (strategy) vs 策略外 (other) 投資組合績效
    純基於快照推算, 不使用引擎回測
    """
    import numpy as np
    import math
    import datetime as _dt_mod
    import glob as _glob_mod
    from analyze_performance import (
        find_portfolio_dates, collect_all_tickers, download_price_data,
        analyze_strategy, analyze_other,
        _read_source_holdings, get_close_price,
    )

    print("\n" + "=" * 70)
    print("🆚 策略內 vs 策略外 績效比較 (Mode 8b)")
    print("=" * 70)

    # --- 0. 找共同快照日期 ---
    strat_dates = find_portfolio_dates(_BASE_DIR)
    _other_files = _glob_mod.glob(os.path.join(_BASE_DIR, 'portfolio_other_*.csv'))
    other_dates = []
    for f in _other_files:
        m = re.search(r'portfolio_other_(\d{8})\.csv', os.path.basename(f))
        if m:
            try:
                other_dates.append(datetime.datetime.strptime(m.group(1), '%Y%m%d').date())
            except ValueError:
                continue
    other_dates.sort()
    common_dates = sorted(set(strat_dates) & set(other_dates))

    if len(common_dates) < 2:
        print(f"\n   ❌ 共同快照不足 (需至少 2 個)")
        print(f"      策略快照: {len(strat_dates)} 個")
        print(f"      其他快照: {len(other_dates)} 個")
        print(f"      共同日期: {len(common_dates)} 個")
        return None

    print(f"\n   📁 共同快照 ({len(common_dates)} 個):")
    for i, d in enumerate(common_dates):
        _s_h = _read_source_holdings(_BASE_DIR, d, 'portfolio_strategy', 'strategy')
        _o_h = _read_source_holdings(_BASE_DIR, d, 'portfolio_other', 'other')
        print(f"      {i+1}. {d.strftime('%Y-%m-%d')}  策略:{len(_s_h)}檔  其他:{len(_o_h)}檔")

    # --- 1. 選日期範圍 ---
    print(f"\n   建議: {common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}")
    _start_in = input(f"📅 起始日 (Enter={common_dates[0].strftime('%Y-%m-%d')}): ").strip()
    start_date = _start_in if _start_in else common_dates[0].strftime('%Y-%m-%d')
    _end_in = input(f"📅 結束日 (Enter={common_dates[-1].strftime('%Y-%m-%d')}): ").strip()
    end_date = _end_in if _end_in else common_dates[-1].strftime('%Y-%m-%d')

    _sd = _dt_mod.datetime.strptime(start_date, '%Y-%m-%d').date()
    _ed = _dt_mod.datetime.strptime(end_date, '%Y-%m-%d').date()
    dates_in_range = [d for d in common_dates if _sd <= d <= _ed]
    if len(dates_in_range) < 2:
        print(f"\n   ❌ 範圍內共同快照不足")
        return None

    # --- 2. 設定 ---
    initial_capital = 900_000
    print(f"\n   📊 比較設定:")
    print(f"      區間: {start_date} ~ {end_date}")
    print(f"      共同快照: {len(dates_in_range)} 個")
    print(f"      策略初始資金: ${initial_capital:,}")
    for i in range(1, len(dates_in_range)):
        gap = (dates_in_range[i] - dates_in_range[i-1]).days
        if gap > 7:
            print(f"      ⚠️ {dates_in_range[i-1].strftime('%m/%d')} → "
                  f"{dates_in_range[i].strftime('%m/%d')} 間隔 {gap} 天")

    # --- 3. 下載股價 ---
    print(f"\n   ⏳ 下載股價資料...")
    all_tickers = collect_all_tickers(dates_in_range, _BASE_DIR)
    stock_data = download_price_data(all_tickers, dates_in_range)

    # --- 4. 用 analyze 函數取快照日數據 ---
    strat_results, strat_trades = analyze_strategy(
        dates_in_range, _BASE_DIR, stock_data, initial_capital)
    other_results, other_trades = analyze_other(
        dates_in_range, _BASE_DIR, stock_data)

    # --- 5. 建每日 NAV (快照之間用收盤價插值) ---
    market_map = reconstruct_market_history(start_date, end_date)
    trading_days = sorted(d for d in market_map.keys() if start_date <= d <= end_date)
    _snap_dates_str = [dt.strftime('%Y-%m-%d') for dt in dates_in_range]

    # 5a. 策略 daily NAV (cash + MV)
    _strat_snap_cash = {sr['date']: sr['cash'] for sr in strat_results}
    _strat_raw = {}
    for dt in dates_in_range:
        _strat_raw[dt.strftime('%Y-%m-%d')] = _read_source_holdings(
            _BASE_DIR, dt, 'portfolio_strategy', 'strategy')

    daily_nav_strat = []
    _s_h, _s_cash = {}, float(initial_capital)
    for day_str in trading_days:
        if day_str in _snap_dates_str and day_str in _strat_raw:
            _s_h = _strat_raw[day_str]
            if day_str in _strat_snap_cash:
                _s_cash = _strat_snap_cash[day_str]
        _dd = _dt_mod.datetime.strptime(day_str, '%Y-%m-%d').date()
        mv = sum(
            (get_close_price(stock_data, t, _dd) or h['avg_cost']) * h['shares']
            for t, h in _s_h.items()
        )
        daily_nav_strat.append({'date': day_str, 'nav': _s_cash + mv})

    # 5b. 策略外 daily NAV (V4.22: sell_proceeds - buy_costs + MV)
    _other_snap_nav = {sr['date']: sr['current_total'] for sr in other_results}
    _other_raw = {}
    for dt in dates_in_range:
        _other_raw[dt.strftime('%Y-%m-%d')] = _read_source_holdings(
            _BASE_DIR, dt, 'portfolio_other', 'other')

    daily_nav_other = []
    _o_h = {}
    _o_net_cash = 0.0  # V4.22: 淨現金流 = sell_proceeds - buy_costs
    for day_str in trading_days:
        if day_str in _snap_dates_str and day_str in _other_raw:
            _o_h = _other_raw[day_str]
            if day_str in _other_snap_nav:
                # 從 analyze_other 取已扣除 buy_costs 的 current_total
                _dd_snap = _dt_mod.datetime.strptime(day_str, '%Y-%m-%d').date()
                _snap_mv = sum(
                    (get_close_price(stock_data, t, _dd_snap) or h['avg_cost']) * h['shares']
                    for t, h in _o_h.items()
                )
                _o_net_cash = _other_snap_nav[day_str] - _snap_mv
        _dd = _dt_mod.datetime.strptime(day_str, '%Y-%m-%d').date()
        mv = sum(
            (get_close_price(stock_data, t, _dd) or h['avg_cost']) * h['shares']
            for t, h in _o_h.items()
        )
        daily_nav_other.append({'date': day_str, 'nav': _o_net_cash + mv})

    # --- 6. 計算指標 ---
    def _nav_metrics(daily_nav, n_days):
        navs = [d['nav'] for d in daily_nav]
        if len(navs) < 2 or navs[0] <= 0:
            return {'return': 0, 'mdd': 0, 'start': 0, 'end': 0,
                    'sharpe': float('nan'), 'calmar': float('nan')}
        start_v, end_v = navs[0], navs[-1]
        total_ret = (end_v - start_v) / start_v * 100
        # MDD
        peak, max_dd = navs[0], 0
        for nv in navs:
            if nv > peak:
                peak = nv
            dd = (peak - nv) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        # Sharpe / Calmar (only if >= 30 trading days)
        sharpe, calmar = float('nan'), float('nan')
        if n_days >= 30:
            daily_rets = [navs[i] / navs[i-1] - 1 for i in range(1, len(navs)) if navs[i-1] > 0]
            if len(daily_rets) >= 2 and np.std(daily_rets) > 0:
                rf = 0.015 / 245
                excess = [r - rf for r in daily_rets]
                sharpe = (np.mean(excess) / np.std(daily_rets)) * np.sqrt(245)
            _years = n_days / 365.25
            if _years > 0 and start_v > 0:
                cagr = ((end_v / start_v) ** (1 / _years) - 1) * 100
                if max_dd >= 0.1:
                    calmar = cagr / max_dd
        return {'return': total_ret, 'mdd': max_dd, 'start': start_v, 'end': end_v,
                'sharpe': sharpe, 'calmar': calmar}

    _cal_days = (_ed - _sd).days
    s_m = _nav_metrics(daily_nav_strat, _cal_days)
    o_m = _nav_metrics(daily_nav_other, _cal_days)

    # 交易統計
    s_trade_count = len([t for t in strat_trades if t['action'] != 'INIT'])
    o_trade_count = len([t for t in other_trades if t['action'] != 'INIT'])

    # --- 7. 輸出比較 ---
    print(f"\n{'═'*70}")
    print(f"🆚 策略內 vs 策略外 績效比較")
    print(f"   區間: {start_date} ~ {end_date} ({len(trading_days)} 交易日)")
    print(f"   快照: {len(dates_in_range)} 個")
    print(f"{'═'*70}")

    _short = (_cal_days < 30)

    def _isnan(v):
        try:
            return math.isnan(v)
        except (TypeError, ValueError):
            return False

    def _fmt_v(v, is_pct=True, na_if_short=False):
        if _isnan(v) or (na_if_short and _short):
            return 'N/A*'
        return f"{v:+.2f}%" if is_pct else f"{v:.2f}"

    def _diff_str(a, b, is_pct=True, lower_better=False, na_if_short=False):
        if _isnan(a) or _isnan(b) or (na_if_short and _short):
            return ''
        d = a - b
        s = f"{d:+.2f}%" if is_pct else f"{d:+.2f}"
        if lower_better:
            tag = '👍' if d < 0 else ('👎' if d > 0.5 else '')
        else:
            tag = '👍' if d > 0 else ('👎' if d < -0.5 else '')
        return f"{s} {tag}"

    _rows = [
        ('起始總值', f"${s_m['start']:,.0f}", f"${o_m['start']:,.0f}", ''),
        ('結束總值', f"${s_m['end']:,.0f}",   f"${o_m['end']:,.0f}",   ''),
        ('總報酬率', _fmt_v(s_m['return']),    _fmt_v(o_m['return']),
                    _diff_str(s_m['return'], o_m['return'])),
        ('MDD',     f"{s_m['mdd']:.2f}%",     f"{o_m['mdd']:.2f}%",
                    _diff_str(s_m['mdd'], o_m['mdd'], lower_better=True)),
        ('Sharpe',  _fmt_v(s_m['sharpe'], is_pct=False, na_if_short=True),
                    _fmt_v(o_m['sharpe'], is_pct=False, na_if_short=True),
                    _diff_str(s_m['sharpe'], o_m['sharpe'], is_pct=False, na_if_short=True)),
        ('Calmar',  _fmt_v(s_m['calmar'], is_pct=False, na_if_short=True),
                    _fmt_v(o_m['calmar'], is_pct=False, na_if_short=True),
                    _diff_str(s_m['calmar'], o_m['calmar'], is_pct=False, na_if_short=True)),
        ('交易次數', f"{s_trade_count}",        f"{o_trade_count}",
                    f"{s_trade_count - o_trade_count:+d}"),
        ('持股檔數', f"{strat_results[-1]['n']}",
                    f"{other_results[-1]['n']}", ''),
    ]

    print(f"\n   {'指標':<10} {'策略內':>14} {'策略外':>14} {'差異(策略-外)':>18}")
    print(f"   {'─'*10} {'─'*14} {'─'*14} {'─'*18}")
    for label, sv, ov, dv in _rows:
        print(f"   {label:<10} {sv:>14} {ov:>14} {dv:>18}")

    if _short:
        print(f"\n   * 區間僅 {_cal_days} 天, Sharpe/Calmar 年化後失真, 改顯示 N/A")

    # 解讀
    _rd = s_m['return'] - o_m['return']
    print(f"\n   💡 解讀:")
    if _rd > 1:
        print(f"      策略內優於策略外 {_rd:.1f}% → 策略選股有效 ✅")
    elif _rd < -1:
        print(f"      策略外優於策略內 {-_rd:.1f}% → 考慮調整策略選股")
    else:
        print(f"      差距很小 ({_rd:+.1f}%) → 兩邊表現相近")
    if len(dates_in_range) < 10:
        print(f"      ⚠️ 快照僅 {len(dates_in_range)} 個, 建議累積更多數據再比較")

    # --- 8. 每日 NAV 曲線 (雙欄) ---
    print(f"\n{'═'*70}")
    print(f"📊 每日 NAV 曲線 (標準化為 100)")
    print(f"{'═'*70}")
    _s_base = daily_nav_strat[0]['nav'] if daily_nav_strat else 1
    _o_base = daily_nav_other[0]['nav'] if daily_nav_other else 1

    print(f"\n   {'日期':<12} {'策略內':>10} {'策略外':>10} {'策略-外':>10}")
    print(f"   {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
    for i in range(len(trading_days)):
        day_str = trading_days[i]
        s_idx = daily_nav_strat[i]['nav'] / _s_base * 100 if _s_base > 0 else 100
        o_idx = daily_nav_other[i]['nav'] / _o_base * 100 if _o_base > 0 else 100
        diff = s_idx - o_idx
        snap_mark = ' 📸' if day_str in _snap_dates_str else ''
        print(f"   {day_str:<12} {s_idx:>9.2f} {o_idx:>9.2f} {diff:>+9.2f}{snap_mark}")

    # --- 9. 交易明細 ---
    print(f"\n{'═'*70}")
    print(f"📜 交易明細")
    print(f"{'═'*70}")

    def _print_trades(trades, label):
        non_init = [t for t in trades if t['action'] != 'INIT']
        print(f"\n   {label} ({len(non_init)} 筆):")
        if not non_init:
            print(f"      (無交易)")
            return
        print(f"   {'日期':<12} {'類型':<8} {'代碼':<12} {'名稱':<8} {'股數':>6} {'價格':>8} {'損益':>8}")
        print(f"   {'─'*12} {'─'*8} {'─'*12} {'─'*8} {'─'*6} {'─'*8} {'─'*8}")
        for t in non_init:
            _pnl = f"{t['pnl_pct']:+.1f}%" if t.get('pnl_pct') and t['pnl_pct'] != 0 else ''
            print(f"   {t['date']:<12} {t['action']:<8} {t['ticker']:<12} "
                  f"{t['name'][:8]:<8} {t['shares']:>6} {t['price']:>8.1f} {_pnl:>8}")

    _print_trades(strat_trades, "📈 策略內")
    _print_trades(other_trades, "📦 策略外")

    # --- 10. 持股重疊分析 ---
    _s_end_h = _read_source_holdings(
        _BASE_DIR, dates_in_range[-1], 'portfolio_strategy', 'strategy')
    _o_end_h = _read_source_holdings(
        _BASE_DIR, dates_in_range[-1], 'portfolio_other', 'other')
    _overlap = set(_s_end_h.keys()) & set(_o_end_h.keys())
    if _overlap:
        print(f"\n   ⚠️ 重疊持股 ({len(_overlap)} 檔, 同時在策略內外):")
        for t in sorted(_overlap):
            _sn = _s_end_h[t]['name'] if t in _s_end_h else t
            _ss = _s_end_h[t]['shares'] if t in _s_end_h else 0
            _os = _o_end_h[t]['shares'] if t in _o_end_h else 0
            print(f"      {t} {_sn}  策略:{_ss}股 / 其他:{_os}股")

    print(f"\n{'═'*70}")
    return {
        'strat_nav': daily_nav_strat, 'other_nav': daily_nav_other,
        'strat_metrics': s_m, 'other_metrics': o_m,
        'strat_trades': strat_trades, 'other_trades': other_trades,
    }


# 🆚 Mode 8: 回測 vs 人工操作 績效比較 (V4.18)
# ==========================================

def run_backtest_vs_manual_comparison():
    """
    比較回測引擎 vs 人工操作績效
    - 回測側: 用起始日快照當初始持倉, 跑引擎到結束日
    - 人工側: 讀所有快照, 追蹤現金流+每日市值
    - 並排顯示 Sharpe / MDD / CAGR / 總報酬
    """
    import numpy as np
    from analyze_performance import (
        find_portfolio_dates, collect_all_tickers, download_price_data,
        analyze_strategy, _read_source_holdings, _calc_mv, get_close_price,
    )

    print("\n" + "=" * 70)
    print("🆚 回測 vs 人工操作 績效比較 (Mode 8)")
    print("=" * 70)

    # --- 0. 列出可用快照 ---
    all_dates = find_portfolio_dates(_BASE_DIR)
    if len(all_dates) < 2:
        print(f"\n   ❌ 快照數量不足 (目前 {len(all_dates)} 個, 至少需要 2 個)")
        if all_dates:
            print(f"      可用: {', '.join(d.strftime('%Y-%m-%d') for d in all_dates)}")
        print(f"   💡 每次跑完 Mode 5 後, 請儲存當日快照 (portfolio_strategy_YYYYMMDD.csv)")
        return None

    print(f"\n   📁 可用快照 ({len(all_dates)} 個):")
    for i, d in enumerate(all_dates):
        print(f"      {i+1}. {d.strftime('%Y-%m-%d')}")

    # --- 1. 選擇日期範圍 ---
    print(f"\n   建議: 起始日 = {all_dates[0].strftime('%Y-%m-%d')}, "
          f"結束日 = {all_dates[-1].strftime('%Y-%m-%d')}")
    _start_in = input(f"📅 起始日 (YYYY-MM-DD, Enter={all_dates[0].strftime('%Y-%m-%d')}): ").strip()
    start_date = _start_in if _start_in else all_dates[0].strftime('%Y-%m-%d')
    _end_in = input(f"📅 結束日 (YYYY-MM-DD, Enter={all_dates[-1].strftime('%Y-%m-%d')}): ").strip()
    end_date = _end_in if _end_in else all_dates[-1].strftime('%Y-%m-%d')

    # 篩選範圍內的快照
    import datetime as _dt_mod
    _sd = _dt_mod.datetime.strptime(start_date, '%Y-%m-%d').date()
    _ed = _dt_mod.datetime.strptime(end_date, '%Y-%m-%d').date()
    dates_in_range = [d for d in all_dates if _sd <= d <= _ed]
    if len(dates_in_range) < 2:
        print(f"\n   ❌ 範圍內快照不足 (需至少 2 個, 目前 {len(dates_in_range)} 個)")
        return None

    # --- 2. 選產業 ---
    industries = _select_daily_industries()
    if industries is None:
        return None

    # --- 3. 參數 ---
    budget = DAILY_BUDGET
    initial_capital = 900_000
    from strategy import DEFAULT_CONFIG as _dc8
    max_positions = _dc8.get('max_positions', 12)

    print(f"\n   📊 比較設定:")
    print(f"      區間: {start_date} ~ {end_date}")
    print(f"      快照: {len(dates_in_range)} 個")
    print(f"      產業: {' + '.join(industries)}")
    print(f"      每筆: ${budget:,} | 初始資金: ${initial_capital:,}")

    # 快照間隔警告
    for i in range(1, len(dates_in_range)):
        gap = (dates_in_range[i] - dates_in_range[i-1]).days
        if gap > 7:
            print(f"      ⚠️ {dates_in_range[i-1].strftime('%m/%d')} → "
                  f"{dates_in_range[i].strftime('%m/%d')} 間隔 {gap} 天, 中間交易無法精確追蹤")

    # ========================================
    # A. 回測側
    # ========================================
    print(f"\n{'─'*70}")
    print(f"📈 A. 回測引擎 (用起始日快照 + 預設參數跑引擎)")
    print(f"{'─'*70}")

    # A1: 載入起始日快照
    _start_snap = _find_best_snapshot(PORTFOLIO_STRATEGY_FILE, start_date)
    if _start_snap:
        _start_df = load_portfolio(_start_snap)
        print(f"   載入快照: {os.path.basename(_start_snap)}")
    else:
        _start_df = load_portfolio(PORTFOLIO_STRATEGY_FILE)
        print(f"   載入主檔: portfolio_strategy.csv (無匹配快照)")

    # A2: 過濾產業
    _industry_set = set(industries)
    if 'industry' in _start_df.columns:
        _start_df = _start_df[_start_df['industry'].isin(_industry_set)].copy()

    # A3: 轉引擎持倉
    bt_initial_pos = _strategy_df_to_engine_positions(_start_df, start_date)
    print(f"   初始持倉: {len(bt_initial_pos)} 檔")
    if bt_initial_pos:
        for t, p in list(bt_initial_pos.items())[:5]:
            print(f"      {t} {p['name']} {p['shares']}股 @ ${p['avg_cost']:.1f}")
        if len(bt_initial_pos) > 5:
            print(f"      ... 共 {len(bt_initial_pos)} 檔")

    # A4: 建股票池 + market_map
    stock_pool, pool_industry_map, _ = _build_daily_stock_pool(industries)
    market_map = reconstruct_market_history(start_date, end_date)

    # A5: per_industry_config
    per_industry_config = {}
    for ind in industries:
        if ind in INDUSTRY_CONFIGS and INDUSTRY_CONFIGS[ind].get('config'):
            per_industry_config[ind] = INDUSTRY_CONFIGS[ind]['config']

    # A6: 跑回測
    print(f"\n   ⏳ 回測中 ({start_date} ~ {end_date})...")
    bt_result = run_group_backtest(
        stock_pool, start_date, end_date, budget,
        market_map, exec_mode='next_open',
        initial_capital=initial_capital,
        initial_positions=bt_initial_pos,
        industry_map=pool_industry_map,
        per_industry_config=per_industry_config,
    )

    if bt_result is None:
        print(f"   ❌ 回測失敗")
        return None

    bt_trades = len(bt_result.get('trade_log', []))
    bt_return = bt_result.get('total_return_pct', 0)
    bt_cagr = bt_result.get('cagr', 0)
    bt_mdd = bt_result.get('mdd_pct', 0)
    bt_sharpe = bt_result.get('sharpe_ratio', 0)
    bt_calmar = bt_result.get('calmar_ratio', 0)
    bt_final = bt_result.get('final_total_value', initial_capital)
    bt_days = bt_result.get('backtest_days', 0)

    print(f"   ✅ 回測完成: {bt_days} 交易日, {bt_trades} 筆交易")

    # ========================================
    # B. 人工操作側
    # ========================================
    print(f"\n{'─'*70}")
    print(f"🧑 B. 人工操作 (根據快照推算)")
    print(f"{'─'*70}")

    # B1: 收集所有股票 + 下載股價
    all_tickers = collect_all_tickers(dates_in_range, _BASE_DIR)
    stock_data_m = download_price_data(all_tickers, dates_in_range)

    # B2: 用 analyze_strategy 算每個快照日的現金/市值
    strat_results, strat_trades = analyze_strategy(
        dates_in_range, _BASE_DIR, stock_data_m, initial_capital)

    # 過濾: 只算策略產業內的持倉
    # (analyze_strategy 讀全部 portfolio_strategy, 但我們只要選定產業)
    # 注意: analyze_strategy 本身不支援產業過濾, 所以我們用整體結果
    # 如果快照中有非策略產業的股票, 會被一起算入

    manual_trade_count = len([t for t in strat_trades if t['action'] != 'INIT'])

    # B3: 建每日 NAV 曲線 (快照之間用收盤價插值)
    # 取得交易日列表 (從 market_map)
    trading_days = sorted(market_map.keys())
    trading_days = [d for d in trading_days if start_date <= d <= end_date]

    # 讀取各快照日的持倉
    _snap_holdings = {}
    _snap_cash = {}
    for sr in strat_results:
        _snap_holdings[sr['date']] = sr
        _snap_cash[sr['date']] = sr['cash']

    # 也讀取 raw holdings 以便每日用收盤價重算
    _raw_snap_holdings = {}
    for dt in dates_in_range:
        ds = dt.strftime('%Y%m%d')
        h = _read_source_holdings(_BASE_DIR, dt, 'portfolio_strategy', 'strategy')
        _raw_snap_holdings[dt.strftime('%Y-%m-%d')] = h

    # 建每日 NAV
    daily_nav_manual = []
    _current_h = {}  # 目前持倉 (快照之間不變)
    _current_cash = float(initial_capital)
    _snap_dates_str = [dt.strftime('%Y-%m-%d') for dt in dates_in_range]

    for day_str in trading_days:
        # 如果這天有快照, 更新持倉和現金
        if day_str in _snap_dates_str and day_str in _raw_snap_holdings:
            _current_h = _raw_snap_holdings[day_str]
            if day_str in _snap_cash:
                _current_cash = _snap_cash[day_str]

        # 用今日收盤價算市值
        _day_date = _dt_mod.datetime.strptime(day_str, '%Y-%m-%d').date()
        mv = 0.0
        for t, h in _current_h.items():
            close = get_close_price(stock_data_m, t, _day_date)
            if close is None:
                close = h['avg_cost']
            mv += close * h['shares']

        nav = _current_cash + mv
        daily_nav_manual.append({'date': day_str, 'nav': nav})

    # B4: 從 daily NAV 算績效指標
    if len(daily_nav_manual) < 2:
        print(f"   ❌ 交易日不足, 無法計算績效")
        return None

    _nav_list = [d['nav'] for d in daily_nav_manual]
    _nav_start = _nav_list[0]
    _nav_end = _nav_list[-1]

    # 總報酬
    manual_return = (_nav_end - _nav_start) / _nav_start * 100 if _nav_start > 0 else 0

    # CAGR
    _cal_days = (_ed - _sd).days
    _years = _cal_days / 365.25 if _cal_days > 0 else 1
    if _nav_start > 0 and _nav_end > 0 and _years > 0:
        manual_cagr = ((_nav_end / _nav_start) ** (1 / _years) - 1) * 100
    else:
        manual_cagr = 0

    # MDD
    _peak = _nav_list[0]
    _max_dd_pct = 0
    for _nv in _nav_list:
        if _nv > _peak:
            _peak = _nv
        if _peak > 0:
            _dd = (_peak - _nv) / _peak * 100
            if _dd > _max_dd_pct:
                _max_dd_pct = _dd
    manual_mdd = _max_dd_pct

    # Sharpe
    _daily_rets = []
    for i in range(1, len(_nav_list)):
        if _nav_list[i-1] > 0:
            _daily_rets.append(_nav_list[i] / _nav_list[i-1] - 1)

    _rf_daily = 0.015 / 245  # 1.5% 年化無風險利率
    if len(_daily_rets) >= 2 and np.std(_daily_rets) > 0:
        _excess = [r - _rf_daily for r in _daily_rets]
        manual_sharpe = (np.mean(_excess) / np.std(_daily_rets)) * np.sqrt(245)
    else:
        manual_sharpe = float('nan')

    # Calmar
    if manual_mdd > 0:
        manual_calmar = manual_cagr / manual_mdd
    else:
        manual_calmar = float('nan')

    manual_final = _nav_end

    print(f"   ✅ 分析完成: {len(daily_nav_manual)} 交易日, {manual_trade_count} 筆交易")

    # ========================================
    # C. 比較輸出
    # ========================================
    print(f"\n{'═'*70}")
    print(f"🆚 回測 vs 人工操作 績效比較")
    print(f"   區間: {start_date} ~ {end_date} ({bt_days} 交易日)")
    print(f"   快照: {len(dates_in_range)} 個 | 產業: {' + '.join(industries)}")
    print(f"{'═'*70}")

    import math
    def _isnan(v):
        try:
            return math.isnan(v)
        except (TypeError, ValueError):
            return False

    def _fmt_pct(v, sign=True):
        if sign:
            return f"{v:+.2f}%"
        return f"{v:.2f}%"

    def _fmt_money(v):
        return f"${v:,.0f}"

    def _fmt_diff(a, b, is_pct=True, lower_better=False):
        d = a - b
        if is_pct:
            s = f"{d:+.2f}%"
        else:
            s = f"{d:+.2f}"
        # 顏色判斷: 正差 = 回測好 (通常是好事, MDD除外)
        if lower_better:
            tag = '👍' if d < 0 else ('👎' if d > 0.5 else '')
        else:
            tag = '👍' if d > 0 else ('👎' if d < -0.5 else '')
        return f"{s} {tag}"

    # 短期判斷: 交易日 < 30 天, 年化指標無意義
    _short_period = (bt_days < 30)
    _footnotes = []

    def _fmt_metric(val, is_nan=False, is_pct=True, short_na=False):
        """格式化指標, 短期或異常 → N/A"""
        if is_nan or (short_na and _short_period):
            return 'N/A*'
        if is_pct:
            return f"{val:+.2f}%"
        return f"{val:.2f}"

    def _fmt_diff_safe(a, b, is_pct=True, lower_better=False, short_na=False):
        """安全差異計算, 任一邊 N/A → 不顯示"""
        if short_na and _short_period:
            return ''
        if _isnan(a) or _isnan(b):
            return ''
        return _fmt_diff(a, b, is_pct=is_pct, lower_better=lower_better)

    # Calmar: MDD < 0.1% → N/A (除數接近 0 無意義)
    _bt_calmar_na = _isnan(bt_calmar) or bt_mdd < 0.1
    _man_calmar_na = _isnan(manual_calmar) or manual_mdd < 0.1

    _rows = [
        ('初始資金',  _fmt_money(initial_capital), _fmt_money(initial_capital), ''),
        ('結束總值',  _fmt_money(bt_final),         _fmt_money(manual_final),   ''),
        ('總報酬率',  _fmt_pct(bt_return),           _fmt_pct(manual_return),    _fmt_diff(bt_return, manual_return)),
        ('CAGR',     _fmt_metric(bt_cagr, short_na=True),
                     _fmt_metric(manual_cagr, short_na=True),
                     _fmt_diff_safe(bt_cagr, manual_cagr, short_na=True)),
        ('MDD',      _fmt_pct(bt_mdd, sign=False),  _fmt_pct(manual_mdd, sign=False), _fmt_diff(bt_mdd, manual_mdd, lower_better=True)),
        ('Sharpe',   _fmt_metric(bt_sharpe, is_pct=False, short_na=True),
                     _fmt_metric(manual_sharpe, is_nan=_isnan(manual_sharpe), is_pct=False, short_na=True),
                     _fmt_diff_safe(bt_sharpe, manual_sharpe, is_pct=False, short_na=True)),
        ('Calmar',   _fmt_metric(bt_calmar, is_nan=_bt_calmar_na, is_pct=False, short_na=True),
                     _fmt_metric(manual_calmar, is_nan=_man_calmar_na, is_pct=False, short_na=True),
                     _fmt_diff_safe(bt_calmar, manual_calmar, is_pct=False, short_na=True)),
        ('交易次數',  f"{bt_trades}",                 f"{manual_trade_count}",    f"{bt_trades - manual_trade_count:+d}"),
    ]

    if _short_period:
        _footnotes.append(f"* 區間僅 {bt_days} 交易日, CAGR/Sharpe/Calmar 年化後失真, 改顯示 N/A")
        _footnotes.append(f"  → 建議累積 ≥ 60 交易日 (約3個月) 再比較年化指標")

    print(f"\n   {'指標':<10} {'回測引擎':>14} {'人工操作':>14} {'差異(回測-人工)':>18}")
    print(f"   {'─'*10} {'─'*14} {'─'*14} {'─'*18}")
    for label, bt_val, man_val, diff_val in _rows:
        print(f"   {label:<10} {bt_val:>14} {man_val:>14} {diff_val:>18}")

    # 腳註
    if _footnotes:
        print()
        for fn in _footnotes:
            print(f"   {fn}")

    # 解讀
    _ret_diff = bt_return - manual_return
    print(f"\n   💡 解讀:")
    if _ret_diff > 1:
        print(f"      回測優於人工 {_ret_diff:.1f}% → 建議更嚴格跟隨引擎訊號")
    elif _ret_diff < -1:
        print(f"      人工優於回測 {-_ret_diff:.1f}% → 你的判斷力有加分")
    else:
        print(f"      差距很小 ({_ret_diff:+.1f}%) → 人工操作與引擎基本一致 ✅")

    if len(dates_in_range) < 10:
        print(f"      ⚠️ 快照僅 {len(dates_in_range)} 個, 建議累積更多數據再比較")

    # ========================================
    # D. 交易明細
    # ========================================
    print(f"\n{'═'*70}")
    print(f"📜 D. 交易明細")
    print(f"{'═'*70}")

    # D1: 回測交易
    _bt_log = bt_result.get('trade_log', [])
    print(f"\n   📈 回測引擎 ({len(_bt_log)} 筆):")
    if _bt_log:
        print(f"   {'日期':<12} {'類型':<6} {'代碼':<12} {'名稱':<8} {'股數':>6} {'價格':>8} {'損益':>10}")
        print(f"   {'─'*12} {'─'*6} {'─'*12} {'─'*8} {'─'*6} {'─'*8} {'─'*10}")
        for t in _bt_log:
            _pnl_str = f"{t['roi']:+.1f}%" if t.get('roi') is not None else ''
            print(f"   {t['date']:<12} {t['type']:<6} {t['ticker']:<12} {t['name']:<8} "
                  f"{t['shares']:>6} {t['price']:>8.1f} {_pnl_str:>10}")
    else:
        print(f"      (無交易)")

    # D2: 人工交易 (從快照推算)
    _man_trades = [t for t in strat_trades if t['action'] != 'INIT']
    print(f"\n   🧑 人工操作 ({len(_man_trades)} 筆, 從快照差異推算):")
    if _man_trades:
        print(f"   {'日期':<12} {'類型':<6} {'代碼':<12} {'名稱':<8} {'股數':>6} {'價格':>8} {'損益':>10}")
        print(f"   {'─'*12} {'─'*6} {'─'*12} {'─'*8} {'─'*6} {'─'*8} {'─'*10}")
        for t in _man_trades:
            _pnl_str = f"{t['pnl_pct']:+.1f}%" if t.get('pnl_pct') and t['pnl_pct'] != 0 else ''
            print(f"   {t['date']:<12} {t['action']:<6} {t['ticker']:<12} {t['name']:<8} "
                  f"{t['shares']:>6} {t['price']:>8.1f} {_pnl_str:>10}")
    else:
        print(f"      (無交易)")

    # D3: 每日 NAV 曲線 (debug 用, 簡要顯示)
    print(f"\n   📊 人工操作每日 NAV:")
    for d in daily_nav_manual:
        _chg = ''
        if daily_nav_manual.index(d) > 0:
            _prev = daily_nav_manual[daily_nav_manual.index(d)-1]['nav']
            if _prev > 0:
                _r = (d['nav'] - _prev) / _prev * 100
                _chg = f" ({_r:+.2f}%)"
        # 標記有快照的日期
        _snap_mark = ' 📸' if d['date'] in _snap_dates_str else ''
        print(f"      {d['date']}  ${d['nav']:,.0f}{_chg}{_snap_mark}")

    print(f"\n{'═'*70}")

    return {
        'bt_result': bt_result,
        'manual_nav': daily_nav_manual,
        'manual_trades': strat_trades,
        'bt_metrics': {
            'return': bt_return, 'cagr': bt_cagr, 'mdd': bt_mdd,
            'sharpe': bt_sharpe, 'calmar': bt_calmar, 'trades': bt_trades,
            'final': bt_final,
        },
        'manual_metrics': {
            'return': manual_return, 'cagr': manual_cagr, 'mdd': manual_mdd,
            'sharpe': manual_sharpe, 'calmar': manual_calmar, 'trades': manual_trade_count,
            'final': manual_final,
        },
    }


def main():
    # --- CLI 參數 ---
    parser = argparse.ArgumentParser(description='Group Backtest V2.5')
    parser.add_argument('--mode', type=str, default=None, help='直接選擇模式 (1-6)')
    parser.add_argument('--auto', action='store_true', help='Mode 5 自動模式')
    parser.add_argument('--date', type=str, default=None, help='Mode 5/6 分析日期')
    parser.add_argument('--verify', action='store_true', help='Mode 5 一致性驗證 (split)')
    parser.add_argument('--verify-daily', action='store_true', help='Mode 7 逐日 vs 回測一致性驗證')
    parser.add_argument('--scan', action='store_true', help='Mode 6 純產業掃描')
    parser.add_argument('--days', type=int, default=120, help='Mode 7 驗證天數 (預設120)')
    parser.add_argument('--start', type=str, default=None, help='Mode 7 起始日 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='Mode 7 結束日 (YYYY-MM-DD)')
    parser.add_argument('--capital', type=int, default=None, help='初始總資金')
    parser.add_argument('--budget', type=int, default=None, help='每筆金額')
    parser.add_argument('--industries', type=str, nargs='+', default=None, help='指定產業')
    args = parser.parse_args()

    # --- 快速路徑: CLI 指定 mode 5/6/7 或 --verify/--scan ---
    if args.verify:
        run_verify(industries=args.industries, budget=args.budget,
                   max_positions=DAILY_MAX_POSITIONS)
        return

    if args.verify_daily or args.mode == '7':
        run_daily_vs_backtest_verify(
            industries=args.industries, budget=args.budget,
            start_date_override=args.start,
            end_date_override=args.end,
        )
        return

    if args.scan or args.mode == '6':
        run_scan_mode(
            target_date=args.date,
            industries=args.industries,
            budget=args.budget,
            initial_capital=args.capital,
            auto_mode=args.scan or (args.mode == '6' and args.industries is None),
        )
        return

    if args.mode == '5' or args.auto:
        run_daily_signal_mode(
            target_date=args.date,
            industries=args.industries,
            budget=args.budget,
            initial_capital=args.capital,
            auto_mode=args.auto or (args.mode == '5' and args.industries is None),
        )
        return

    print("=" * 60)
    print("🚀 Group Backtest V2.5 (V4.2: 進階績效指標)")
    print("   模擬: Group Scanner 發現 → 買入")
    print("         Daily Scanner 追蹤 → 加碼/賣出")
    print("   ✅ 多產業掃描 + 跨股票績效彙總")
    print("   ✅ T+1 開盤 / 當日收盤 兩種模式")
    print("   ✅ 上車限制: 持倉上限 + 每日新建倉上限 + 排序")
    print("   ✅ 現金帳戶: 追蹤實際投入 → 帳戶總值報酬率")
    print("   ✅ 風險指標: CAGR / MDD% / Sharpe / Calmar")
    print("   ✅ 多檔換股: 每日最多換 N 檔 (max_swap_per_day)")
    print("   ✅ Ablation: 策略參數 + 上車限制 組合測試")
    print("   ✅ 每日實盤訊號: 讀取庫存 + 引擎分析 (Mode 5)")
    print("   ✅ 逐日 vs 回測一致性驗證 (Mode 7)")
    print("=" * 60)

    # --- 模式選擇 (移到共用設定之前) ---
    print("\n📋 選擇模式:")
    print("   1. 📈 標準回測 (使用預設參數)")
    print("   2. 🧪 Ablation 策略比較 (多組參數比拼)")
    print("   3. 🔀 多產業混合回測 (跨產業建倉, 各產業可用不同策略)")
    print("   4. 🔍 產業組合掃描 (自動找最佳產業組合)")
    print("   5. 📊 每日實盤訊號 (讀取庫存 CSV, 帶入部位分析)")
    print("   6. 🔎 產業掃描 (不帶部位, 純看買入候選)")
    print("   7. ✅ 驗證: 逐日訊號 vs 回測一致性")
    print("   8. 🆚 績效比較 (回測vs人工 / 策略vs策略外)")
    choice = args.mode or input("👉 選擇 (Enter=1): ").strip()

    # --- Mode 5: 專用流程 ---
    if choice == '5':
        print("\n📊 每日實盤訊號模式")
        print("─" * 40)

        # 日期
        date_in = input(f"📅 分析日期 (Enter=今天): ").strip()
        target_date = date_in or None

        # Budget
        try:
            bud_in = input(f"💰 每筆金額 (Enter={DAILY_BUDGET:,}): ").strip()
            m5_budget = int(bud_in) if bud_in else DAILY_BUDGET
        except ValueError:
            m5_budget = DAILY_BUDGET

        # Capital
        cap_default = 900_000
        try:
            cap_in = input(f"🏦 初始總資金 (Enter={cap_default:,}): ").strip()
            m5_capital = int(cap_in) if cap_in else cap_default
        except ValueError:
            m5_capital = cap_default

        run_daily_signal_mode(
            target_date=target_date,
            budget=m5_budget,
            initial_capital=m5_capital,
            auto_mode=False,  # 互動模式 (會顯示產業選單)
        )
        return

    # --- Mode 6: 純產業掃描 ---
    if choice == '6':
        print("\n🔎 產業掃描模式 (不帶部位)")
        print("─" * 40)

        date_in = input(f"📅 分析日期 (Enter=今天): ").strip()
        target_date = date_in or None

        run_scan_mode(
            target_date=target_date,
            auto_mode=False,  # 互動模式 (會顯示產業選單)
        )
        return

    # --- Mode 7: 逐日 vs 回測一致性驗證 ---
    if choice == '7':
        print("\n✅ 逐日訊號 vs 回測一致性驗證")
        print("─" * 40)

        # V4.17: 讓使用者選產業 (同 Mode 1/5)
        _v_industries = _select_daily_industries()
        if _v_industries is None:
            return

        # V4.15: 支援指定日期範圍 (預設與V5 ablation一致)
        _v_start = input("📅 起始日 (Enter=2022-01-01): ").strip() or '2022-01-01'
        _v_end = input("📅 結束日 (Enter=2026-02-11): ").strip() or '2026-02-11'

        run_daily_vs_backtest_verify(
            industries=_v_industries,
            start_date_override=_v_start,
            end_date_override=_v_end,
        )
        return

    # --- Mode 8: 績效比較 ---
    if choice == '8':
        print("\n🆚 績效比較 (Mode 8)")
        print("─" * 40)
        print("   1. 回測引擎 vs 人工操作")
        print("   2. 策略內 vs 策略外 (快照比較)")
        _m8_choice = input("👉 選擇 (1/2): ").strip()
        if _m8_choice == '2':
            run_strategy_vs_other_comparison()
        else:
            run_backtest_vs_manual_comparison()
        return

    # --- 共用設定 (modes 1-4) ---
    start_date, end_date = select_period()
    print(f"   📅 區間: {start_date} ~ {end_date}")

    exec_mode = select_exec_mode()

    try:
        budget_in = input(f"💰 每次買入/加碼金額 (Enter=25000): ").strip()
        budget = int(budget_in) if budget_in else 25_000
    except ValueError:
        budget = 25_000

    # V4.1: 初始總資金 (預設 90 萬)
    cap_default = 900_000
    try:
        cap_in = input(f"🏦 初始總資金 (Enter={cap_default:,}): ").strip()
        initial_capital = int(cap_in) if cap_in else cap_default
    except ValueError:
        initial_capital = cap_default

    if args.capital:
        initial_capital = args.capital
    if args.budget:
        budget = args.budget

    # V4.15: 倉位/每日新買/換股 可調整 (Enter=DEFAULT_CONFIG 預設值)
    from strategy import DEFAULT_CONFIG as _dc
    _def_pos = _dc.get('max_positions', 12)
    _def_buy = _dc.get('max_new_buy_per_day', 4)
    _def_swap = _dc.get('max_swap_per_day', 1)
    _def_tier_b = _dc.get('tier_b_net', 20)

    try:
        _pos_in = input(f"📊 最大持倉數 (Enter={_def_pos}): ").strip()
        max_positions = int(_pos_in) if _pos_in else _def_pos
    except ValueError:
        max_positions = _def_pos
    try:
        _buy_in = input(f"🛒 每日最多新買 (Enter={_def_buy}): ").strip()
        max_new_buy = int(_buy_in) if _buy_in else _def_buy
    except ValueError:
        max_new_buy = _def_buy
    try:
        _swap_in = input(f"🔄 每日最多換股 (Enter={_def_swap}): ").strip()
        max_swap = int(_swap_in) if _swap_in else _def_swap
    except ValueError:
        max_swap = _def_swap
    try:
        _tier_b_in = input(f"🛡️ Tier B 停利門檻% (Enter={_def_tier_b}): ").strip()
        tier_b_net = int(_tier_b_in) if _tier_b_in else _def_tier_b
    except ValueError:
        tier_b_net = _def_tier_b

    # 只有跟預設不同時才覆蓋
    _pos_override = {}
    if max_positions != _def_pos:
        _pos_override['max_positions'] = max_positions
    if max_new_buy != _def_buy:
        _pos_override['max_new_buy_per_day'] = max_new_buy
    if max_swap != _def_swap:
        _pos_override['max_swap_per_day'] = max_swap
    if tier_b_net != _def_tier_b:
        _pos_override['tier_b_net'] = tier_b_net

    # --- 建立大盤 ---
    market_map = reconstruct_market_history(start_date, end_date)
    if not market_map:
        print("❌ 大盤資料建立失敗")
        return

    # --- 選擇產業 (模式4自行處理, 其他模式用共用選擇) ---
    if choice != '4':
        selected_industries, all_stocks, all_industries, industry_map = _select_industries()
        if not all_stocks:
            return

    if choice == '2':
        # ===== Ablation 模式 =====
        print("\n🧪 選擇 Ablation 範圍:")
        print("   1. 🔍 倉位網格搜索 (找最佳 pos × buy × swap)")
        print("   2. 策略模組 (baseline + V3.8/V3.6對照 + 各模組關閉)")
        print("   3. 參數調整 (交叉ablation + 參數微調)")
        print("   4. 完整 (全部)")
        print("   5. 自選")
        print("   6. 🔬 V4.5 交叉驗證 (bias × swap 最佳組合 + 動態bias)")
        print("   7. 🔬 V4.6 精細化 (動態bias細調 + zombie搭配 + 高檔修正修補)")
        print("   8. 🔬 V4.7 持倉/換倉/加碼金額 網格搜索")
        print("   9. 🔬 V4.7 交叉驗證 (pos × swap × budget 最佳組合確認)")
        print("  10. 🔬 V4.8 減碼策略 ablation (分批停利 R1/R2 參數測試)")
        print("  11. 🔬 V4.9 大盤偵測門檻 ablation (crash/panic 敏感度調整)")
        print("  12. 🔬 V4.10 持倉組合恐慌 ablation (組合跌幅偵測+強制賣出)")
        print("  13. 🔬 V4.11 現金管理 ablation (禁止透支+加碼控制)")
        print("  14. 🔬 V4.11 交叉驗證 (保留資金% × 加碼次數 最佳組合)")
        print("  15. 🔬 V4.12 組合驗證 (停利S1 × 濾網F1 × 資金reserve 交叉)")
        print("  16. ✅ V4.12 新baseline確認 (新default vs 舊baseline vs 關鍵對照)")
        print("  17. 🔬 V4.13 滑價壓力測試 (0% ~ 1.0%)")
        print("  18. 🔬 V4.13 交易參數最佳化 (持倉/金額/換股/殭屍, 90萬固定)")
        print("  19. 🔬 V4.13 K8 組合驗證 (buy2×swap2×殭屍10 交叉)")
        print("  20. 🏭 跨產業參數搜索 (分層漏斗, 高效率找最佳參數)")
        print("  21. 🏭 其他電子業 分區間驗證 (L2冠軍 + 微調對照 + 防禦驗證)")
        print("  22. 🏭 電腦週邊業 分區間驗證 (L2冠軍 + brk5 + 微調 + 防禦驗證)")
        print("  23. 🏭 通信網路業 分區間驗證 (L2冠軍 + 6維度微調 + 防禦驗證)")
        print("  24. 🔬 半導體業 L2 交叉驗證 (bias×z×F1, 5+1組)")
        print("  25. 🔬 其他電子業 L2 交叉驗證 (bias×z + brk5加掛, 6+1組)")
        print("  26. 🔬 通信網路業 L2 交叉驗證 (bias×s1×z, 8+1組)")
        print("  27. 🔬 電子零組件業 L2 交叉驗證 (bias×z, 6+1組)")
        print("  28. 🔬 電機機械 L2 交叉驗證 (bias×z×F1 + brk5加掛, 8+1組)")
        print("  29. 🔬 電腦週邊業 L2 交叉驗證 (bias×s1×z + brk5加掛, 14+1組)")
        print("  30. 🔬 V4.20 漲停遞延 ablation (T+1→T+2 重試)")
        print("  31. 🔬 V4.20 S2 季線緩衝 ablation (有獲利時延遲賣出)")
        print("  32. 🔬 V4.20 TierB peak + 組合驗證")
        print("  33. 🔬 V4.20 OTC 大盤濾網模式 (or/and/off × entry_peak)")
        print("  34. 🔬 V4.21 動態曝險管理 (大盤弱勢時降低持倉上限, 8組)")
        print("  35. 🔬 V4.21 動態限買 (大盤弱勢降低每日新買上限, 4組)")
        print("  36. 🔬 V4.21 動態停損 (大盤弱勢收緊 hard_stop, 4組)")
        print("  37. 🔬 V4.21 波動率倉位控制 (高波動縮小 budget, 4組)")
        print("  38. 🔬 V4.21 參數精煉 (S1/swap/cash 微調, 6組)")
        print("  39. 🔬 V4.21 組合交叉驗證 (stop_B×pos12×保守換股×vol_C, 11組)")
        print("  40. 🔬 V4.21 冠軍微調 (pos×margin×swap 精細搜索, 15組)")
        abl_choice = input("👉 選擇 (Enter=1): ").strip()

        _dynamic_configs = {}  # V4.14: Layer 2/3 動態產生的 configs
        _skip_normal_ablation = False  # V4.14: 選項20自行處理執行流程

        if abl_choice == '2':
            keys = ['baseline', 'static_b4b7', 'v38_baseline', 'v36_baseline',
                    'no_filter', 'f1_strict', 'f1_moderate', 'f1_relaxed',
                    'no_tiered', 's1_tight', 's1_loose',
                    'no_zombie', 'no_swap', 'no_quality',
                    'no_breakout',
                    'no_fish_tail', 'no_shooting', 'no_bias', 'no_defense']
        elif abl_choice == '3':
            keys = ['baseline',
                    # V4.4 交叉 ablation (B7 × swap_margin)
                    'b7on_sw03', 'b7on_sw05', 'b7off_sw03', 'b7off_sw05',
                    # B4 乖離
                    'no_bias', 'bias_15', 'bias_25', 'bias_30',
                    # 其他參數
                    'breakout_5d', 'breakout_20d',
                    'fish_3d', 'fish_7d',
                    'sort_bias', 'zombie_10d', 'zombie_20d',
                    'swap_low', 'swap_high']
        elif abl_choice == '4':
            keys = list(GROUP_ABLATION_CONFIGS.keys())
        elif abl_choice == '5':
            print("\n   可選:")
            for k, (_, desc) in GROUP_ABLATION_CONFIGS.items():
                print(f"      {k:<18} {desc}")
            raw = input("👉 輸入 key (逗號分隔): ").strip()
            keys = [k.strip() for k in raw.split(',') if k.strip() in GROUP_ABLATION_CONFIGS]
            if 'baseline' not in keys:
                keys.insert(0, 'baseline')
        elif abl_choice == '6':
            keys = ['baseline',
                    # 單獨改動 (對照用)
                    'bias_25', 'swap_high',
                    # 交叉組合
                    'bi25_sw05', 'bi25_sw08', 'bi20_sw08', 'bi30_sw08',
                    # 動態 bias
                    'bi_dyn_25_20', 'bi_dyn_25_15', 'bi_dyn_30_20']
        elif abl_choice == '7':
            keys = ['baseline',
                    # V4.5 最佳候選 (對照基準)
                    'bi25_sw08', 'bi_dyn_30_20', 'bi30_sw08',
                    # A: 動態 bias 精細化 (中性/空頭最佳值)
                    'dyn_30_25_15', 'dyn_30_25_10',
                    'dyn_30_30_15', 'dyn_30_30_20',
                    # A2: zombie 搭配 swap 0.8
                    'dyn30_z10', 'dyn30_z20',
                    'bi25_z10', 'bi25_z20',
                    # B: 高檔修正弱點修補
                    'dyn30_z10_sw2', 'bi25_z10_sw2',
                    'dyn30_zr3', 'bi25_zr3']
        elif abl_choice == '8':
            keys = ['baseline',
                    # 持倉數量 (10=default)
                    'pos10', 'pos15', 'pos20', 'pos25', 'pos30',
                    # 每日換倉數量
                    'swap_off', 'swap_1d', 'swap_2d', 'swap_3d_v47', 'swap_5d',
                    # 每次加碼金額
                    'budget_1w', 'budget_2w', 'budget_5w', 'budget_10w', 'budget_20w']
        elif abl_choice == '9':
            keys = ['baseline',
                    # pos × swap 交叉
                    'p15_sw2', 'p15_sw1', 'p15_sw3',
                    'p10_sw2', 'p20_sw2',
                    # 最佳 pos×swap + budget 交叉
                    'p15s2_b1w', 'p15s2_b2w', 'p15s2_b3w', 'p15s2_b5w',
                    # p15_sw1 最佳組合 budget 精細化
                    'p15s1_b15k', 'p15s1_b20k']
        elif abl_choice == '10':
            keys = ['baseline',
                    # V4.8 減碼 ablation
                    'reduce_off', 'reduce_t15r50', 'reduce_t20r50',
                    'reduce_t25r50', 'reduce_t20r33', 'reduce_2tier']
        elif abl_choice == '11':
            keys = ['baseline',
                    # V4.9 大盤偵測門檻 ablation
                    'panic_default',
                    'crash_d20', 'crash_d15',
                    'panic_d25', 'panic_d20',
                    'panic_3d40', 'panic_3d35',
                    'sens_mid', 'sens_high']
        elif abl_choice == '12':
            keys = ['baseline',
                    # V4.10 持倉組合恐慌 ablation
                    'pp_off',
                    'pp_d3', 'pp_d4', 'pp_d5',
                    'pp_3d5', 'pp_3d7', 'pp_3d10',
                    'pp_sell_all', 'pp_loss_n5',
                    'pp_sens', 'pp_relaxed',
                    'pp_d3_3d5']
        elif abl_choice == '13':
            keys = ['baseline',
                    # V4.11 現金管理 ablation
                    'cash_unlimited', 'cash_strict',
                    'cash_reserve_10', 'cash_reserve_20',
                    'cash_no_add_unlimited', 'cash_no_add',
                    'cash_add_limit_3', 'cash_add_limit_5']
        elif abl_choice == '14':
            keys = ['baseline',
                    # V4.11 交叉驗證: reserve × add_limit
                    'cash_reserve_5', 'cash_reserve_10', 'cash_reserve_15',
                    'cash_add_limit_5', 'cash_add_limit_7', 'cash_add_limit_10',
                    'r5_a5', 'r5_a7',
                    'r10_a5', 'r10_a7', 'r10_a10',
                    'r15_a5', 'r15_a7']
        elif abl_choice == '15':
            keys = ['baseline',
                    # 單因子對照
                    's1_tight', 's1_loose', 's1_mid',
                    'f1_moderate', 'f1_relaxed',
                    'cash_reserve_5',
                    # S1 × F1 交叉
                    's1_loose_f1_mod', 's1_loose_f1_rel',
                    's1_mid_f1_mod', 's1_mid_f1_rel',
                    'base_f1_mod',
                    # S1 × reserve 交叉
                    's1_loose_r5', 's1_mid_r5',
                    # 三因子組合
                    's1_loose_f1_mod_r5', 's1_mid_f1_mod_r5']
        elif abl_choice == '16':
            keys = ['baseline',
                    # V4.13 新baseline確認: 舊版對照
                    'v411_baseline',       # V4.11 全舊 (S1嚴/F1strict/buy3/swap1)
                    'v412_baseline',       # V4.12 舊 (buy3/swap1, 只差K8改動)
                    # 核心因子單獨還原 (確認每個改動的貢獻)
                    'f1_strict',          # 只還原F1→strict (測S1改動的獨立效果)
                    's1_tight',           # S1收緊 (TierA=20/0.98)
                    # 關鍵替代方案
                    'f1_moderate',
                    'cash_reserve_5',
                    # 防禦模組確認 (新baseline下是否仍有效)
                    'no_filter', 'no_tiered',
                    'no_zombie', 'no_swap', 'no_breakout',
                    'no_fish_tail', 'no_bias', 'no_defense']
        elif abl_choice == '17':
            keys = ['baseline', 'slip_0', 'slip_01', 'slip_03', 'slip_05', 'slip_10']
        elif abl_choice == '18':
            print("\n   📋 交易參數子選單:")
            print("     a. K1 持倉數量 (8/10/12/15/20/25)")
            print("     b. K2 每次下單金額 (1萬~6萬, 含1.5萬/2.5萬)")
            print("     c. K3 每日換股上限 (0/1/2/3)")
            print("     d. K4 每日建倉上限 (1/2/3/5)")
            print("     e. K5 資金利用率組合 (budget×pos等水位)")
            print("     f. K6 殭屍清除參數 (10/15/20/30天/關)")
            print("     g. K7 換股門檻 (0.3/0.5/0.8/1.2/1.5)")
            print("     h. 全部跑 (K1~K7, 約30組)")
            sub = input("   👉 選擇 (Enter=h): ").strip().lower() or 'h'
            if sub == 'a':
                keys = ['baseline', 'k_pos8', 'k_pos10', 'k_pos12', 'k_pos20', 'k_pos25']
            elif sub == 'b':
                keys = ['baseline', 'k_bud10k', 'k_bud15k', 'k_bud20k', 'k_bud25k', 'k_bud30k', 'k_bud45k', 'k_bud60k']
            elif sub == 'c':
                keys = ['baseline', 'k_swap0', 'k_swap2', 'k_swap3']
            elif sub == 'd':
                keys = ['baseline', 'k_buy1', 'k_buy2', 'k_buy5']
            elif sub == 'e':
                keys = ['baseline', 'k_20k_30p', 'k_30k_20p', 'k_45k_15p', 'k_60k_10p', 'k_90k_8p']
            elif sub == 'f':
                keys = ['baseline', 'k_zomb_off', 'k_zomb10d', 'k_zomb20d', 'k_zomb30d']
            elif sub == 'g':
                keys = ['baseline', 'k_sm03', 'k_sm05', 'k_sm12', 'k_sm15']
            else:
                keys = ['baseline',
                        'k_pos8', 'k_pos10', 'k_pos12', 'k_pos20', 'k_pos25',
                        'k_bud10k', 'k_bud15k', 'k_bud20k', 'k_bud25k', 'k_bud30k', 'k_bud45k', 'k_bud60k',
                        'k_swap0', 'k_swap2', 'k_swap3',
                        'k_buy1', 'k_buy2', 'k_buy5',
                        'k_20k_30p', 'k_30k_20p', 'k_45k_15p', 'k_60k_10p', 'k_90k_8p',
                        'k_zomb_off', 'k_zomb10d', 'k_zomb20d', 'k_zomb30d',
                        'k_sm03', 'k_sm05', 'k_sm12', 'k_sm15']
        elif abl_choice == '19':
            keys = ['baseline',
                    'k8_buy2', 'k8_buy2_swap2', 'k8_buy2_z10',
                    'k8_buy2_s2_z10', 'k8_z10', 'k8_swap2']
        elif abl_choice == '21':
            # V4.15: 其他電子業分區間驗證
            keys = ['baseline',
                    'oe_champ', 'oe_champ_nz',
                    'oe_b15_mid_z30', 'oe_b10_tight_z30', 'oe_b10_loose_z30',
                    'oe_b10_mid_z20', 'oe_b10_mid_z15',
                    'oe_no_filter', 'oe_no_breakout', 'oe_no_bias',
                    'oe_no_fish', 'oe_no_s1']
        elif abl_choice == '22':
            # V4.15: 電腦及週邊設備業分區間驗證
            keys = ['baseline',
                    'cp_champ', 'cp_champ_brk5',
                    'cp_b15_mid_z25', 'cp_b10_tight_z25', 'cp_b10_loose_z25',
                    'cp_b10_mid_z20', 'cp_b10_mid_z30', 'cp_b10_mid_z15',
                    'cp_no_filter', 'cp_no_breakout', 'cp_no_bias',
                    'cp_no_fish', 'cp_no_s1', 'cp_no_zombie']
        elif abl_choice == '23':
            # V4.15: 通信網路業分區間驗證
            keys = ['baseline',
                    'cn_champ', 'cn_b20_tight_z20', 'cn_b30_tight_z20',
                    'cn_b25_mid_z20', 'cn_b25_loose_z20',
                    'cn_b25_tight_z15', 'cn_b25_tight_z30',
                    'cn_champ_bud15k', 'cn_champ_brk10', 'cn_champ_f1relax',
                    'cn_no_filter', 'cn_no_breakout', 'cn_no_bias',
                    'cn_no_fish', 'cn_no_s1', 'cn_no_zombie']
        elif abl_choice == '24':
            # V4.16: 半導體業 L2 交叉驗證
            keys = ['baseline',
                    'sc_b25_loose_z10', 'sc_b25_loose_z15',
                    'sc_b30_loose_z10', 'sc_b30_loose_z15',
                    'sc_b30_loose_z10_rel']
        elif abl_choice == '25':
            # V4.16: 其他電子業 L2 交叉驗證
            keys = ['baseline',
                    'oe2_b10_loose_z25', 'oe2_b10_loose_z30',
                    'oe2_b15_loose_z25', 'oe2_b15_loose_z30',
                    'oe2_b10_loose_z30_brk5', 'oe2_b10_loose_z25_brk5']
        elif abl_choice == '26':
            # V4.16: 通信網路業 L2 交叉驗證
            keys = ['baseline',
                    'cn2_b20_tight_z15', 'cn2_b20_tight_z20',
                    'cn2_b20_loose_z15', 'cn2_b20_loose_z20',
                    'cn2_b25_tight_z15', 'cn2_b25_tight_z20',
                    'cn2_b25_loose_z15', 'cn2_b25_loose_z20']
        elif abl_choice == '27':
            # V4.16: 電子零組件業 L2 交叉驗證
            keys = ['baseline',
                    'ec_b10_loose_z15', 'ec_b10_loose_z20', 'ec_b10_loose_z25',
                    'ec_b15_loose_z15', 'ec_b15_loose_z20', 'ec_b15_loose_z25']
        elif abl_choice == '28':
            # V4.16: 電機機械 L2 交叉驗證
            keys = ['baseline',
                    'em_b10_loose_z10', 'em_b10_loose_z15',
                    'em_b15_loose_z10', 'em_b15_loose_z15',
                    'em_b20_loose_z10', 'em_b20_loose_z15',
                    'em_b15_loose_z15_brk5', 'em_b15_loose_z10_brk5']
        elif abl_choice == '29':
            # V4.16: 電腦及週邊設備業 L2 交叉驗證
            keys = ['baseline',
                    'cp2_b20_tight_z15', 'cp2_b20_tight_z20',
                    'cp2_b20_loose_z15', 'cp2_b20_loose_z20',
                    'cp2_b25_tight_z15', 'cp2_b25_tight_z20',
                    'cp2_b25_loose_z15', 'cp2_b25_loose_z20',
                    'cp2_b30_tight_z15', 'cp2_b30_tight_z20',
                    'cp2_b30_loose_z15', 'cp2_b30_loose_z20',
                    'cp2_b25_tight_z20_brk5', 'cp2_b25_loose_z20_brk5']
        elif abl_choice == '30':
            # V4.20: 漲停遞延 ablation
            keys = ['baseline', 'lu_skip', 'lu_retry1', 'lu_retry2', 'lu_no_skip']
        elif abl_choice == '31':
            # V4.20: S2 季線緩衝 ablation
            keys = ['baseline', 's2_no_buf', 's2_buf_5_2d', 's2_buf_10_2d',
                    's2_buf_10_3d', 's2_buf_15_2d', 's2_buf_20_2d']
        elif abl_choice == '32':
            # V4.20: TierB peak + 組合驗證
            keys = ['baseline', 'tb_20d_peak', 'tb_entry_peak',
                    'v420_retry_s2buf', 'v420_retry_peak', 'v420_full']
        elif abl_choice == '33':
            # V4.20: OTC 大盤濾網模式 ablation
            keys = ['baseline', 'otc_or', 'otc_and', 'otc_off',
                    'otc_and_ep_off', 'otc_off_ep_off']
        elif abl_choice == '34':
            # V4.21: 動態曝險管理 ablation
            keys = ['baseline', 'dyn_A', 'dyn_B', 'dyn_C', 'dyn_D',
                    'dyn_E', 'dyn_F', 'dyn_G', 'dyn_H']
        elif abl_choice == '35':
            # V4.21: 動態限買 ablation
            keys = ['baseline', 'buy_A', 'buy_B', 'buy_C', 'buy_D']
        elif abl_choice == '36':
            # V4.21: 動態停損 ablation
            keys = ['baseline', 'stop_A', 'stop_B', 'stop_C', 'stop_D']
        elif abl_choice == '37':
            # V4.21: 波動率倉位控制 ablation
            keys = ['baseline', 'vol_A', 'vol_B', 'vol_C', 'vol_D']
        elif abl_choice == '38':
            # V4.21: 參數精煉 ablation
            keys = ['baseline', 'ref_A', 'ref_B', 'ref_C', 'ref_D', 'ref_E', 'ref_F']
        elif abl_choice == '39':
            # V4.21: 組合交叉驗證
            keys = ['baseline',
                    'combo_stopB', 'combo_pos12',
                    'combo_sB_p12', 'combo_sB_rD',
                    'combo_sB_p12_rD', 'combo_p12_rD',
                    'combo_sB_p12_vC', 'combo_full',
                    'combo_sB_p12_b9', 'combo_sB_p12_b11',
                    'combo_sB_p12_b35k']
        elif abl_choice == '40':
            # V4.21: 冠軍微調 (pos12+保守換股 精細搜索)
            keys = ['baseline',
                    # X1: 倉位數
                    'tune_p10_s10', 'tune_p11_s10', 'tune_p12_s10',
                    'tune_p13_s10', 'tune_p14_s10',
                    # X2: 換股門檻
                    'tune_p12_s08', 'tune_p12_s09',
                    'tune_p12_s11', 'tune_p12_s12',
                    # X3: 不換股
                    'tune_p12_noswap', 'tune_p11_noswap',
                    # X4: cash/buy
                    'tune_p12_s10_c5', 'tune_p12_s10_b3', 'tune_p12_s10_b5']
        elif abl_choice == '20':
            # === V4.15 跨產業參數搜索 (分層漏斗, 全自動 a→b→c 串接) ===
            _skip_normal_ablation = True  # 跳過後面的標準 ablation 流程

            print("\n🏭 跨產業參數搜索 — 分層漏斗法 (逐產業自動執行):")
            print("   a. 🔍 Layer 1: 單因子掃描 (~22組/產業, 找各維度冠軍)")
            print("   b. 🔬 Layer 2: 交叉驗證 (手動輸入L1冠軍 → 自動3×3交叉)")
            print("   c. ✅ Layer 3: 最終確認 (手動輸入最佳參數 → 驗證防禦模組)")
            print("   d. 🚀 全自動 (Layer 1→2→3 一鍵串接, 每層輸出報告)")
            print()
            print("   💡 推薦 d: 全自動跑完三層, 每層都有完整報告可驗證")
            sub = input("   👉 選擇 (Enter=d): ").strip().lower() or 'd'

            # --- 共用: S1 停利預設值 ---
            _s1_presets = {
                'tight': {'tier_a_net': 20, 'tier_a_ma_buf': 0.98, 'tier_b_net': 10, 'tier_b_drawdown': 0.5},
                'mid':   {'tier_a_net': 30, 'tier_a_ma_buf': 0.97, 'tier_b_net': 15, 'tier_b_drawdown': 0.6},
                'loose': {'tier_a_net': 40, 'tier_a_ma_buf': 0.95, 'tier_b_net': 20, 'tier_b_drawdown': 0.7},
            }

            # --- 共用: 從 config key 反推參數值 ---
            _key_to_bias = {'ind_bias10': 10, 'ind_bias15': 15, 'ind_bias20': 20, 'ind_bias25': 25, 'ind_bias30': 30}
            _key_to_s1 = {'ind_s1_tight': 'tight', 'ind_s1_mid': 'mid', 'ind_s1_loose': 'loose'}
            _key_to_zombie = {'ind_z10': 10, 'ind_z15': 15, 'ind_z20': 20, 'ind_z30': 30}
            _key_to_f1 = {'ind_f1_strict': 'strict', 'ind_f1_mod': 'moderate', 'ind_f1_relaxed': 'relaxed'}
            _key_to_bud = {'ind_bud10k': 10000, 'ind_bud15k': 15000, 'ind_bud20k': 20000, 'ind_bud30k': 30000}
            _key_to_brk = {'ind_brk5': 5, 'ind_brk10': 10, 'ind_brk20': 20}

            # --- 共用函式: 跑一批 configs 並回傳結果 ---
            def _run_layer(layer_name, per_ind_configs, ind_stocks, ind_data,
                           ind_name, start_date, end_date, budget, market_map,
                           exec_mode, initial_capital):
                """執行一層 ablation, 回傳 {config_name: result}"""
                print(f"\n   {'─'*90}")
                print(f"   {layer_name}: {len(per_ind_configs)} 組參數")
                print(f"   {'─'*90}")
                layer_results = {}
                layer_start = time.time()
                for config_name, (config_dict, desc) in per_ind_configs.items():
                    t_start = time.time()
                    run_budget = budget
                    run_config = config_dict
                    if config_dict and '_budget' in config_dict:
                        run_budget = config_dict['_budget']
                        run_config = {k: v for k, v in config_dict.items() if k != '_budget'}
                    result = run_group_backtest(
                        ind_stocks, start_date, end_date, run_budget,
                        market_map, exec_mode=exec_mode,
                        config_override=run_config if run_config else None,
                        initial_capital=initial_capital,
                        preloaded_data=ind_data,
                    )
                    elapsed = time.time() - t_start
                    layer_results[config_name] = result
                    if result:
                        sharpe = result.get('sharpe_ratio', 0)
                        ret_pct = result.get('total_return_pct', result['roi'])
                        print(f"   🧪 {config_name:<26} → "
                              f"Sharpe {sharpe:>5.2f} | "
                              f"報酬 {ret_pct:>+6.1f}% | "
                              f"勝率 {result['win_rate']:>3.0f}% | "
                              f"{result['trades']}次 "
                              f"⏱{elapsed:.1f}s")
                layer_elapsed = time.time() - layer_start
                print(f"   ⏱ {layer_name} 耗時: {layer_elapsed:.1f}秒")
                return layer_results

            # --- 共用函式: 從 L1 結果提取各維度冠軍 ---
            def _extract_l1_champions(ind_results, per_ind_configs):
                """回傳 {dim_name: (best_key, best_sharpe, param_value)}"""
                _dimensions = {
                    'B4 乖離': ([k for k in per_ind_configs if k.startswith('ind_bias')], _key_to_bias, 25),
                    'S1 停利': ([k for k in per_ind_configs if k.startswith('ind_s1_')], _key_to_s1, 'loose'),
                    '殭屍天數': ([k for k in per_ind_configs if k.startswith('ind_z')], _key_to_zombie, 15),
                    'F1 濾網': ([k for k in per_ind_configs if k.startswith('ind_f1_')], _key_to_f1, 'relaxed'),
                    '每次金額': ([k for k in per_ind_configs if k.startswith('ind_bud')], _key_to_bud, 15000),
                    'B6 突破': ([k for k in per_ind_configs if k.startswith('ind_brk')], _key_to_brk, 10),
                }
                champions = {}
                baseline_sharpe = 0
                if ind_results.get('baseline') and ind_results['baseline'].get('sharpe_ratio'):
                    baseline_sharpe = ind_results['baseline']['sharpe_ratio']

                for dim_name, (dim_keys, key_map, default_val) in _dimensions.items():
                    best_key = 'baseline'
                    best_sharpe = baseline_sharpe
                    for dk in dim_keys:
                        dr = ind_results.get(dk)
                        if dr and dr.get('sharpe_ratio', 0) > best_sharpe:
                            best_sharpe = dr['sharpe_ratio']
                            best_key = dk
                    # 反推參數值
                    param_val = key_map.get(best_key, default_val) if best_key != 'baseline' else default_val
                    champions[dim_name] = (best_key, best_sharpe, param_val)
                return champions

            # --- 共用函式: 從 L1 冠軍建立 L2 交叉 configs ---
            def _build_l2_configs(champions):
                """用 bias × s1 × zombie 前三維度交叉, 固定 F1/budget/brk 為各自冠軍"""
                bias_val = champions['B4 乖離'][2]
                s1_val = champions['S1 停利'][2]
                zombie_val = champions['殭屍天數'][2]
                f1_val = champions['F1 濾網'][2]
                bud_val = champions['每次金額'][2]
                brk_val = champions['B6 突破'][2]

                # 交叉: 冠軍值 ± 鄰近值 (共 3×3×3=27 組最多, 實際去重後更少)
                bias_candidates = sorted(set([bias_val,
                    max(10, bias_val - 5), min(30, bias_val + 5)]))
                s1_candidates = []
                s1_order = ['tight', 'mid', 'loose']
                s1_idx = s1_order.index(s1_val) if s1_val in s1_order else 2
                for offset in [-1, 0, 1]:
                    idx = s1_idx + offset
                    if 0 <= idx < len(s1_order):
                        if s1_order[idx] not in s1_candidates:
                            s1_candidates.append(s1_order[idx])
                zombie_candidates = sorted(set([zombie_val,
                    max(10, zombie_val - 5), min(30, zombie_val + 5)]))

                configs = {'baseline': ({}, '✅ baseline (現有default)')}
                cross_idx = 0
                for b4 in bias_candidates:
                    for s1_name in s1_candidates:
                        for z in zombie_candidates:
                            cross_idx += 1
                            cfg = {
                                'bias_limit_bull': b4, 'bias_limit_neutral': b4, 'bias_limit_bear': b4,
                                'zombie_hold_days': z,
                                'market_filter_mode': f1_val,
                                'breakout_lookback': brk_val,
                                **_s1_presets[s1_name],
                            }
                            if bud_val != 15000:
                                cfg['_budget'] = bud_val
                            key = f'L2_{cross_idx}_b{b4}_s{s1_name}_z{z}'
                            desc = f'🔬L2 乖離{b4}%+S1_{s1_name}+殭屍{z}天+F1_{f1_val}+brk{brk_val}'
                            configs[key] = (cfg, desc)
                return configs

            # --- 共用函式: 從 L2 結果建立 L3 防禦驗證 configs ---
            def _build_l3_configs(l2_results, l2_configs):
                """找 L2 冠軍 → 建立防禦驗證 configs"""
                best_key = 'baseline'
                best_sharpe = 0
                for cn, r in l2_results.items():
                    if r and r.get('sharpe_ratio', 0) > best_sharpe:
                        best_sharpe = r['sharpe_ratio']
                        best_key = cn
                # 取冠軍的 config_dict
                best_cfg = {}
                best_desc_str = ''
                if best_key in l2_configs:
                    best_cfg = dict(l2_configs[best_key][0])  # copy
                    best_desc_str = l2_configs[best_key][1]
                # 移除 _budget (防禦驗證不改金額)
                l3_budget_override = best_cfg.pop('_budget', None)
                best_cfg_clean = dict(best_cfg)

                configs = {
                    'baseline':         ({}, '✅ baseline (現有default)'),
                    'L3_best':          (best_cfg_clean, f'✅L3 冠軍: {best_desc_str}'),
                    'L3_no_filter':     ({**best_cfg_clean, 'enable_market_filter': False},
                                         f'❌L3 冠軍+關F1F2F3'),
                    'L3_no_tiered':     ({**best_cfg_clean, 'enable_tiered_stops': False},
                                         f'❌L3 冠軍+關S1停利'),
                    'L3_no_zombie':     ({**best_cfg_clean, 'enable_zombie_cleanup': False},
                                         f'❌L3 冠軍+關殭屍S4'),
                    'L3_no_breakout':   ({**best_cfg_clean, 'enable_breakout': False},
                                         f'❌L3 冠軍+關突破B6'),
                    'L3_no_fish_tail':  ({**best_cfg_clean, 'enable_fish_tail': False},
                                         f'❌L3 冠軍+關魚尾B5'),
                    'L3_no_bias':       ({**best_cfg_clean, 'enable_bias_limit': False},
                                         f'❌L3 冠軍+關乖離B4'),
                }
                if l3_budget_override:
                    for k in configs:
                        if k != 'baseline':
                            cfg_copy = dict(configs[k][0])
                            cfg_copy['_budget'] = l3_budget_override
                            configs[k] = (cfg_copy, configs[k][1])
                return configs, best_key, best_sharpe

            # ======================================================
            # 依 sub 執行 (a/b/c 手動 | d 全自動)
            # ======================================================

            if sub == 'a':
                # Layer 1 only: 單因子掃描
                _l1_keys = ['baseline',
                            'ind_bias10', 'ind_bias15', 'ind_bias20', 'ind_bias25', 'ind_bias30',
                            'ind_s1_tight', 'ind_s1_mid', 'ind_s1_loose',
                            'ind_z10', 'ind_z15', 'ind_z20', 'ind_z30',
                            'ind_f1_strict', 'ind_f1_mod', 'ind_f1_relaxed',
                            'ind_bud10k', 'ind_bud15k', 'ind_bud20k', 'ind_bud30k',
                            'ind_brk5', 'ind_brk10', 'ind_brk20']
                _per_ind_configs = {k: GROUP_ABLATION_CONFIGS[k] for k in _l1_keys if k in GROUP_ABLATION_CONFIGS}

            elif sub == 'b':
                # Layer 2 only: 手動輸入 L1 冠軍 → 交叉驗證
                print("\n   📋 請輸入 Layer 1 各維度冠軍 (看上一次報告的 Sharpe 最高者):")
                print("   ────────────────────────────────────────────")
                _b4_in = input("     B4 乖離上限 (10/15/20/25/30, 可逗號多選, Enter=25): ").strip() or '25'
                _b4_vals = [int(x.strip()) for x in _b4_in.split(',') if x.strip().isdigit()]
                if not _b4_vals: _b4_vals = [25]
                print("     S1 停利模式:")
                print("       tight = A20/B10  |  mid = A30/B15  |  loose = A40/B20")
                _s1_in = input("     S1 停利 (可逗號多選, Enter=loose): ").strip().lower() or 'loose'
                _s1_vals = [s.strip() for s in _s1_in.split(',') if s.strip() in _s1_presets]
                if not _s1_vals: _s1_vals = ['loose']
                _z_in = input("     殭屍天數 (10/15/20/30, 可逗號多選, Enter=15): ").strip() or '15'
                _z_vals = [int(x.strip()) for x in _z_in.split(',') if x.strip().isdigit()]
                if not _z_vals: _z_vals = [15]
                print("\n   (選填) 是否固定 F1/金額?")
                _f1_in = input("     F1 濾網 (strict/moderate/relaxed, Enter=跳過): ").strip().lower()
                _bud_in = input("     每次金額 (10000/15000/20000/30000, Enter=跳過): ").strip()
                _per_ind_configs = {'baseline': ({}, '✅ baseline (現有default)')}
                cross_idx = 0
                for b4 in _b4_vals:
                    for s1_name in _s1_vals:
                        for z in _z_vals:
                            cross_idx += 1
                            cfg = {
                                'bias_limit_bull': b4, 'bias_limit_neutral': b4, 'bias_limit_bear': b4,
                                'zombie_hold_days': z,
                                **_s1_presets[s1_name],
                            }
                            if _f1_in and _f1_in in ('strict', 'moderate', 'relaxed'):
                                cfg['market_filter_mode'] = _f1_in
                            if _bud_in and _bud_in.isdigit():
                                cfg['_budget'] = int(_bud_in)
                            key = f'cross_{cross_idx}_b{b4}_s{s1_name}_z{z}'
                            desc = f'🔬L2 乖離{b4}%+S1_{s1_name}+殭屍{z}天'
                            if _f1_in and _f1_in in ('strict', 'moderate', 'relaxed'):
                                desc += f'+F1_{_f1_in}'
                            if _bud_in and _bud_in.isdigit():
                                desc += f'+{int(_bud_in)//1000}K'
                            _per_ind_configs[key] = (cfg, desc)
                print(f"\n   🔬 交叉組合: {len(_per_ind_configs)-1} 組 + baseline")

            elif sub == 'c':
                # Layer 3 only: 手動輸入最佳參數 → 驗證防禦模組
                print("\n   ✅ 請輸入 Layer 2 冠軍的參數:")
                print("   ────────────────────────────────────────────")
                _b4_final = int(input("     B4 乖離上限 (Enter=25): ").strip() or '25')
                print("     S1 停利: tight=A20/B10 | mid=A30/B15 | loose=A40/B20")
                _s1_final = input("     S1 停利 (Enter=loose): ").strip().lower() or 'loose'
                _z_final = int(input("     殭屍天數 (Enter=15): ").strip() or '15')
                _f1_final = input("     F1 濾網 (strict/moderate/relaxed, Enter=relaxed): ").strip().lower() or 'relaxed'
                _brk_final = int(input("     B6 突破回看天數 (Enter=10): ").strip() or '10')
                _bud_final_in = input("     每次金額 (Enter=25000): ").strip()
                _bud_final = int(_bud_final_in) if _bud_final_in.isdigit() else 25000
                _s1_cfg = _s1_presets.get(_s1_final, _s1_presets['loose'])
                _best_cfg = {
                    'bias_limit_bull': _b4_final, 'bias_limit_neutral': _b4_final, 'bias_limit_bear': _b4_final,
                    'zombie_hold_days': _z_final,
                    'market_filter_mode': _f1_final,
                    'breakout_lookback': _brk_final,
                    **_s1_cfg,
                }
                if _bud_final != 15000:
                    _best_cfg['_budget'] = _bud_final
                _best_desc = (f'bias{_b4_final}/S1_{_s1_final}/z{_z_final}/'
                              f'F1_{_f1_final}/brk{_brk_final}/bud{_bud_final//1000}k')
                _per_ind_configs = {
                    'baseline':         ({}, '✅ baseline (現有default)'),
                    'L3_best':          (_best_cfg, f'✅L3 冠軍: {_best_desc}'),
                    'L3_no_filter':     ({**_best_cfg, 'enable_market_filter': False},
                                         f'❌L3 冠軍+關F1F2F3'),
                    'L3_no_tiered':     ({**_best_cfg, 'enable_tiered_stops': False},
                                         f'❌L3 冠軍+關S1停利'),
                    'L3_no_zombie':     ({**_best_cfg, 'enable_zombie_cleanup': False},
                                         f'❌L3 冠軍+關殭屍S4'),
                    'L3_no_breakout':   ({**_best_cfg, 'enable_breakout': False},
                                         f'❌L3 冠軍+關突破B6'),
                    'L3_no_fish_tail':  ({**_best_cfg, 'enable_fish_tail': False},
                                         f'❌L3 冠軍+關魚尾B5'),
                    'L3_no_bias':       ({**_best_cfg, 'enable_bias_limit': False},
                                         f'❌L3 冠軍+關乖離B4'),
                }
                print(f"\n   ✅ 共 {len(_per_ind_configs)-1} 組驗證 + baseline")

            elif sub == 'd':
                # d 模式: 全自動 → 在下面的逐產業迴圈中特殊處理
                _per_ind_configs = None  # placeholder, d 模式不用這個
            else:
                # 未知 → 預設全自動
                sub = 'd'
                _per_ind_configs = None

            # ======================================================
            # 🏭 逐產業執行 ablation + 彙總比較
            # ======================================================
            _sim_start = pd.Timestamp(start_date)
            _download_start = (_sim_start - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
            _download_end = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')

            # d 模式: Layer 1 configs
            if sub == 'd':
                _l1_keys = ['baseline',
                            'ind_bias10', 'ind_bias15', 'ind_bias20', 'ind_bias25', 'ind_bias30',
                            'ind_s1_tight', 'ind_s1_mid', 'ind_s1_loose',
                            'ind_z10', 'ind_z15', 'ind_z20', 'ind_z30',
                            'ind_f1_strict', 'ind_f1_mod', 'ind_f1_relaxed',
                            'ind_bud10k', 'ind_bud15k', 'ind_bud20k', 'ind_bud30k',
                            'ind_brk5', 'ind_brk10', 'ind_brk20']
                _l1_configs = {k: GROUP_ABLATION_CONFIGS[k] for k in _l1_keys if k in GROUP_ABLATION_CONFIGS}
                print(f"\n{'='*100}")
                print(f"🚀 全自動分層漏斗: {len(selected_industries)} 個產業")
                print(f"   Layer 1: {len(_l1_configs)} 組單因子掃描")
                print(f"   Layer 2: 自動交叉驗證 (bias×s1×zombie ±鄰近值)")
                print(f"   Layer 3: 自動防禦模組驗證 (7 組 on/off)")
                print(f"   區間: {start_date} ~ {end_date}")
                print(f"{'='*100}")
            else:
                print(f"\n{'='*100}")
                print(f"🏭 逐產業參數搜索: {len(selected_industries)} 個產業 × {len(_per_ind_configs)} 組參數")
                print(f"   區間: {start_date} ~ {end_date}")
                print(f"{'='*100}")

            # 收集跨產業結果
            _all_industry_results = {}   # {ind: {layer: {config: result}}}
            _all_industry_summary = {}   # {ind: {l1_champs, l2_best, l3_best, ...}}

            grand_total_start = time.time()

            for ind_idx, ind_name in enumerate(selected_industries):
                ind_stocks = get_stocks_by_industry(ind_name)
                if not ind_stocks:
                    print(f"\n⚠️ {ind_name}: 無股票，跳過")
                    continue

                print(f"\n{'='*100}")
                print(f"🏭 [{ind_idx+1}/{len(selected_industries)}] {ind_name} ({len(ind_stocks)} 檔)")
                print(f"{'='*100}")

                # 下載該產業的股票資料
                print(f"   ⏳ 下載 {len(ind_stocks)} 檔股票資料...")
                ind_data, ind_skipped = batch_download_stocks(
                    ind_stocks, _download_start, _download_end,
                    min_data_days=MIN_DATA_DAYS,
                )
                print(f"   ✅ 有效標的: {len(ind_data)} 檔", end="")
                if ind_skipped['download_fail'] > 0:
                    print(f" | 下載失敗: {ind_skipped['download_fail']}", end="")
                if ind_skipped['low_data'] > 0:
                    print(f" | 資料不足: {ind_skipped['low_data']}", end="")
                print()

                if not ind_data:
                    print(f"   ❌ 無有效標的，跳過")
                    continue

                if sub == 'd':
                    # ========================================
                    # 🚀 全自動模式: L1 → L2 → L3 串接
                    # ========================================
                    _all_industry_results[ind_name] = {}

                    # ── Layer 1: 單因子掃描 ──
                    print(f"\n   {'━'*90}")
                    print(f"   🔍 Layer 1: 單因子掃描 ({len(_l1_configs)} 組)")
                    print(f"   {'━'*90}")
                    l1_results = _run_layer(
                        '🔍 Layer 1', _l1_configs, ind_stocks, ind_data,
                        ind_name, start_date, end_date, budget, market_map,
                        exec_mode, initial_capital)
                    _all_industry_results[ind_name]['L1'] = l1_results

                    # L1 報告
                    print_ablation_report(l1_results, _l1_configs,
                                         [ind_name], start_date, end_date, budget, exec_mode)

                    # L1 冠軍提取
                    l1_champs = _extract_l1_champions(l1_results, _l1_configs)

                    print(f"\n   {'─'*90}")
                    print(f"   📊 Layer 1 各維度冠軍 — {ind_name}:")
                    print(f"   {'維度':<10} {'冠軍key':<20} {'Sharpe':>7} {'參數值'}")
                    print(f"   {'─'*70}")
                    for dim_name, (best_key, best_sharpe, param_val) in l1_champs.items():
                        marker = ' ← =default' if best_key == 'baseline' else ''
                        print(f"   {dim_name:<10} {best_key:<20} {best_sharpe:>7.2f} {param_val}{marker}")

                    # L1 CSV
                    for cn, r in l1_results.items():
                        if r and r.get('daily_snapshots'):
                            _safe_ind = ind_name.replace(' ', '_')
                            export_daily_csv(r, f'{_safe_ind}_L1_{cn}', start_date, end_date)

                    # ── Layer 2: 交叉驗證 ──
                    l2_configs = _build_l2_configs(l1_champs)
                    print(f"\n   {'━'*90}")
                    print(f"   🔬 Layer 2: 交叉驗證 ({len(l2_configs)} 組)")
                    print(f"   ━━ 基於 L1 冠軍: bias={l1_champs['B4 乖離'][2]}, "
                          f"s1={l1_champs['S1 停利'][2]}, "
                          f"zombie={l1_champs['殭屍天數'][2]}, "
                          f"f1={l1_champs['F1 濾網'][2]}, "
                          f"budget={l1_champs['每次金額'][2]}, "
                          f"brk={l1_champs['B6 突破'][2]}")
                    print(f"   {'━'*90}")

                    l2_results = _run_layer(
                        '🔬 Layer 2', l2_configs, ind_stocks, ind_data,
                        ind_name, start_date, end_date, budget, market_map,
                        exec_mode, initial_capital)
                    _all_industry_results[ind_name]['L2'] = l2_results

                    # L2 報告
                    print_ablation_report(l2_results, l2_configs,
                                         [ind_name], start_date, end_date, budget, exec_mode)

                    # 找 L2 冠軍
                    l2_best_key, l2_best_sharpe = 'baseline', 0
                    for cn, r in l2_results.items():
                        if r and r.get('sharpe_ratio', 0) > l2_best_sharpe:
                            l2_best_sharpe = r['sharpe_ratio']
                            l2_best_key = cn
                    l2_best_desc = l2_configs.get(l2_best_key, ({}, ''))[1]
                    print(f"\n   🏆 Layer 2 冠軍: {l2_best_key} (Sharpe {l2_best_sharpe:.2f})")
                    print(f"      {l2_best_desc}")

                    # L2 CSV
                    for cn, r in l2_results.items():
                        if r and r.get('daily_snapshots'):
                            _safe_ind = ind_name.replace(' ', '_')
                            export_daily_csv(r, f'{_safe_ind}_L2_{cn}', start_date, end_date)

                    # ── Layer 3: 防禦模組驗證 ──
                    l3_configs, l3_source_key, l3_source_sharpe = _build_l3_configs(l2_results, l2_configs)
                    print(f"\n   {'━'*90}")
                    print(f"   ✅ Layer 3: 防禦模組驗證 ({len(l3_configs)} 組)")
                    print(f"   ━━ 基於 L2 冠軍: {l3_source_key} (Sharpe {l3_source_sharpe:.2f})")
                    print(f"   {'━'*90}")

                    l3_results = _run_layer(
                        '✅ Layer 3', l3_configs, ind_stocks, ind_data,
                        ind_name, start_date, end_date, budget, market_map,
                        exec_mode, initial_capital)
                    _all_industry_results[ind_name]['L3'] = l3_results

                    # L3 報告
                    print_ablation_report(l3_results, l3_configs,
                                         [ind_name], start_date, end_date, budget, exec_mode)

                    # L3 CSV
                    for cn, r in l3_results.items():
                        if r and r.get('daily_snapshots'):
                            _safe_ind = ind_name.replace(' ', '_')
                            export_daily_csv(r, f'{_safe_ind}_L3_{cn}', start_date, end_date)

                    # 儲存該產業彙總
                    l3_best_key, l3_best_sharpe = 'baseline', 0
                    for cn, r in l3_results.items():
                        if r and r.get('sharpe_ratio', 0) > l3_best_sharpe:
                            l3_best_sharpe = r['sharpe_ratio']
                            l3_best_key = cn
                    _all_industry_summary[ind_name] = {
                        'l1_champs': l1_champs,
                        'l2_best_key': l2_best_key, 'l2_best_sharpe': l2_best_sharpe,
                        'l2_best_desc': l2_best_desc,
                        'l3_best_key': l3_best_key, 'l3_best_sharpe': l3_best_sharpe,
                        'l3_configs': l3_configs,
                        'l3_results': l3_results,
                    }

                    print(f"\n   {'━'*90}")
                    print(f"   🏆 {ind_name} 最終結果:")
                    print(f"      L2 冠軍: {l2_best_key} (Sharpe {l2_best_sharpe:.2f})")
                    print(f"      L3 最佳: {l3_best_key} (Sharpe {l3_best_sharpe:.2f})")
                    r_final = l3_results.get(l3_best_key)
                    if r_final:
                        print(f"      報酬率: {r_final.get('total_return_pct', 0):+.1f}% | "
                              f"CAGR: {r_final.get('cagr', 0):.1f}% | "
                              f"MDD: {r_final.get('mdd_pct', 0):.1f}% | "
                              f"勝率: {r_final.get('win_rate', 0):.0f}% | "
                              f"交易: {r_final.get('trades', 0)}次")
                    print(f"   {'━'*90}")

                else:
                    # ========================================
                    # a/b/c 手動模式: 單層執行
                    # ========================================
                    ind_results = _run_layer(
                        f'Layer {"1" if sub == "a" else "2" if sub == "b" else "3"}',
                        _per_ind_configs, ind_stocks, ind_data,
                        ind_name, start_date, end_date, budget, market_map,
                        exec_mode, initial_capital)

                    _all_industry_results[ind_name] = {'single': ind_results}

                    # 找冠軍
                    best_name, best_sharpe = 'baseline', 0
                    for cn, r in ind_results.items():
                        if r and r.get('sharpe_ratio', 0) > best_sharpe:
                            best_sharpe = r['sharpe_ratio']
                            best_name = cn
                    best_desc = _per_ind_configs[best_name][1] if best_name in _per_ind_configs else ''
                    print(f"\n   🏆 {ind_name} 冠軍: {best_name} (Sharpe {best_sharpe:.2f})")
                    print(f"      {best_desc}")

                    # 報告
                    print_ablation_report(ind_results, _per_ind_configs,
                                         [ind_name], start_date, end_date, budget, exec_mode)

                    # CSV
                    for cn, r in ind_results.items():
                        if r and r.get('daily_snapshots'):
                            _safe_ind = ind_name.replace(' ', '_')
                            export_daily_csv(r, f'{_safe_ind}_{cn}', start_date, end_date)

                    # Layer a 時輸出各維度冠軍明細
                    if sub == 'a':
                        l1_champs = _extract_l1_champions(ind_results, _per_ind_configs)
                        print(f"\n   📊 各維度冠軍 — {ind_name}:")
                        print(f"   {'維度':<10} {'冠軍key':<20} {'Sharpe':>7} {'參數值'}")
                        print(f"   {'─'*70}")
                        for dim_name, (bk, bs, pv) in l1_champs.items():
                            marker = ' ← =default' if bk == 'baseline' else ''
                            print(f"   {dim_name:<10} {bk:<20} {bs:>7.2f} {pv}{marker}")

                    _all_industry_summary[ind_name] = {
                        'best_key': best_name, 'best_sharpe': best_sharpe,
                        'best_desc': best_desc, 'results': ind_results,
                    }

            grand_elapsed = time.time() - grand_total_start

            # ======================================================
            # 🏆 跨產業最終彙總
            # ======================================================
            if _all_industry_summary:
                print(f"\n{'='*100}")
                if sub == 'd':
                    print(f"🏆 跨產業全自動搜索彙總 (Layer 1→2→3)")
                else:
                    print(f"🏆 跨產業參數搜索彙總 (Layer {'1' if sub == 'a' else '2' if sub == 'b' else '3'})")
                print(f"   區間: {start_date} ~ {end_date} | 每次 ${budget:,} | 初始 ${initial_capital:,}")
                print(f"{'='*100}")

                if sub == 'd':
                    # 全自動彙總: 每個產業顯示 L1→L2→L3 結果
                    print(f"\n{'產業':<14} {'L2冠軍':<28} {'L2 Sharpe':>9} "
                          f"{'L3最佳':<16} {'L3 Sharpe':>9} {'報酬%':>7} {'CAGR%':>7} "
                          f"{'MDD%':>7} {'勝率':>6} {'交易':>5}")
                    print("─" * 130)

                    for ind_name in selected_industries:
                        s = _all_industry_summary.get(ind_name)
                        if not s:
                            continue
                        l3r = s.get('l3_results', {}).get(s['l3_best_key'])
                        if l3r:
                            print(f"{ind_name:<14} {s['l2_best_key']:<28} "
                                  f"{s['l2_best_sharpe']:>9.2f} "
                                  f"{s['l3_best_key']:<16} "
                                  f"{s['l3_best_sharpe']:>9.2f} "
                                  f"{l3r.get('total_return_pct', 0):>7.1f} "
                                  f"{l3r.get('cagr', 0):>7.1f} "
                                  f"{l3r.get('mdd_pct', 0):>7.1f} "
                                  f"{l3r.get('win_rate', 0):>5.0f}% "
                                  f"{l3r.get('trades', 0):>5}")

                    # 各產業 L1 冠軍明細
                    print(f"\n{'─'*100}")
                    print(f"📊 各產業 Layer 1 維度冠軍:")
                    for ind_name in selected_industries:
                        s = _all_industry_summary.get(ind_name)
                        if not s or 'l1_champs' not in s:
                            continue
                        print(f"\n   🏭 {ind_name}:")
                        print(f"   {'維度':<10} {'冠軍key':<20} {'Sharpe':>7} {'最佳參數值'}")
                        print(f"   {'─'*70}")
                        for dim_name, (bk, bs, pv) in s['l1_champs'].items():
                            marker = ' ← =default' if bk == 'baseline' else ''
                            print(f"   {dim_name:<10} {bk:<20} {bs:>7.2f} {pv}{marker}")

                    # 各產業 L3 防禦模組效果
                    print(f"\n{'─'*100}")
                    print(f"🛡️ 各產業防禦模組效果 (L3 冠軍 vs 關閉各模組):")
                    for ind_name in selected_industries:
                        s = _all_industry_summary.get(ind_name)
                        if not s or 'l3_results' not in s:
                            continue
                        l3r = s['l3_results']
                        best_r = l3r.get('L3_best')
                        if not best_r:
                            continue
                        best_sharpe = best_r.get('sharpe_ratio', 0)
                        print(f"\n   🏭 {ind_name} (L3冠軍 Sharpe={best_sharpe:.2f}):")
                        print(f"   {'模組':<22} {'Sharpe':>7} {'差異':>8} {'效果'}")
                        print(f"   {'─'*60}")
                        for cn in ['L3_no_filter', 'L3_no_tiered', 'L3_no_zombie',
                                   'L3_no_breakout', 'L3_no_fish_tail', 'L3_no_bias']:
                            r = l3r.get(cn)
                            if not r:
                                continue
                            sh = r.get('sharpe_ratio', 0)
                            diff = sh - best_sharpe
                            if diff < -0.1:
                                effect = '🟢 模組有效 (關掉變差)'
                            elif diff > 0.1:
                                effect = '🔴 模組有害 (關掉反而好)'
                            else:
                                effect = '⚪ 影響不大'
                            cfg_desc = s['l3_configs'].get(cn, ({}, ''))[1]
                            print(f"   {cfg_desc:<22} {sh:>7.2f} {diff:>+8.2f} {effect}")

                else:
                    # a/b/c 手動模式彙總
                    print(f"\n{'產業':<16} {'冠軍配置':<24} {'Sharpe':>7} {'CAGR%':>7} {'MDD%':>7} "
                          f"{'報酬%':>7} {'勝率':>6} {'交易':>5}")
                    print("─" * 100)
                    for ind_name in selected_industries:
                        s = _all_industry_summary.get(ind_name)
                        if not s:
                            continue
                        r = s.get('results', {}).get(s['best_key'])
                        if r:
                            print(f"{ind_name:<16} {s['best_key']:<24} "
                                  f"{r.get('sharpe_ratio', 0):>7.2f} "
                                  f"{r.get('cagr', 0):>7.1f} "
                                  f"{r.get('mdd_pct', 0):>7.1f} "
                                  f"{r.get('total_return_pct', 0):>7.1f} "
                                  f"{r.get('win_rate', 0):>5.0f}% "
                                  f"{r.get('trades', 0):>5}")

                print(f"\n   ⏱ 總耗時: {grand_elapsed:.1f}秒")
                print(f"{'='*100}")

            # 設 keys=[] 讓後面的標準流程不會重跑
            keys = []

        else:
            keys = ['baseline',
                    'p8_b3_s1', 'p8_b3_s2',
                    'p10_b3_s1', 'p10_b3_s2', 'p10_b5_s2',
                    'p12_b3_s2', 'p12_b5_s2', 'p12_b5_s3',
                    'p15_b3_s2', 'p15_b5_s3',
                    'p20_b5_s3', 'p20_b8_s3',
                    'no_swap']

        # V4.14: 選項20自行處理，跳過標準流程
        if not _skip_normal_ablation:
            configs_to_run = {k: GROUP_ABLATION_CONFIGS[k] for k in keys if k in GROUP_ABLATION_CONFIGS}

            # V4.14: 合併動態產生的 configs (Layer 2/3 交叉驗證)
            if _dynamic_configs:
                configs_to_run.update(_dynamic_configs)

            print(f"\n🔄 開始: {len(configs_to_run)} 組策略 × {len(all_stocks)} 檔標的")

            # ✅ 加速: 資料只下載一次，所有 ablation 配置共享
            _sim_start = pd.Timestamp(start_date)
            _download_start = (_sim_start - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
            _download_end = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')

            # ✅ 快取檢查: 回測前先比對快取 vs 最新數據
            print(f"\n🔍 檢查資料快取...")
            cache_info = check_cache_vs_latest(all_stocks, _download_start, _download_end)
            print_cache_check(cache_info)

            force_refresh = False
            if cache_info is not None:
                if not cache_info['data_match']:
                    print(f"\n   ⚠️ 快取與最新數據有差異！(可能有除權息調整)")
                    refresh_choice = input("   👉 要更新資料嗎？(Y=重新下載 / Enter=繼續用快取): ").strip().upper()
                    force_refresh = (refresh_choice == 'Y')
                else:
                    print(f"   ✅ 快取數據與最新一致，直接使用")

            print(f"\n⏳ 載入 {len(all_stocks)} 檔股票資料 (所有 ablation 共用)...")
            shared_data, shared_skipped = batch_download_stocks(
                all_stocks, _download_start, _download_end,
                min_data_days=MIN_DATA_DAYS, force_refresh=force_refresh,
            )
            print(f"   ✅ 有效標的: {len(shared_data)} 檔")
            if shared_skipped['download_fail'] > 0:
                print(f"   ⚠️ 下載失敗: {shared_skipped['download_fail']} 檔")
            if shared_skipped['low_data'] > 0:
                print(f"   ⚠️ 資料不足: {shared_skipped['low_data']} 檔")

            # V4.12: 除權息驗證 (資料品質檢查)
            div_check = check_dividend_adjustment(shared_data, sample_n=10)
            print_dividend_check(div_check)

            # V4.20: Ablation 也帶入產業 L2 參數 (與 Mode 5 實盤一致)
            _abl_per_ind_config = {}
            if industry_map:
                for _ind in selected_industries:
                    if _ind in INDUSTRY_CONFIGS:
                        _abl_per_ind_config[_ind] = INDUSTRY_CONFIGS[_ind]['config']
                if _abl_per_ind_config:
                    _ind_names = ', '.join(_abl_per_ind_config.keys())
                    print(f"\n   📋 Ablation 套用產業 L2 參數: {_ind_names}")

            # V4.20 fix: 建立 ablation 全域基底 config
            # 問題: per_industry_config 只影響 per-ticker 訊號評估,
            # 但殭屍清除等全域邏輯用的是 config_override → global cfg。
            # 必須把 L2 參數 + _pos_override 合進 config_override,
            # 才能讓 ablation baseline 與 Mode 1 結果一致。
            _abl_global_base = {}
            if _abl_per_ind_config and len(selected_industries) == 1:
                # 單產業: L2 參數直接作為全域基底
                _single_ind = selected_industries[0]
                if _single_ind in _abl_per_ind_config and _abl_per_ind_config[_single_ind]:
                    _abl_global_base.update(_abl_per_ind_config[_single_ind])
            elif _abl_per_ind_config and len(selected_industries) > 1:
                # 多產業: 取各產業 L2 的「交集」(所有產業都相同的參數值)
                _all_cfgs = [c for c in _abl_per_ind_config.values() if c]
                if _all_cfgs:
                    _common = dict(_all_cfgs[0])
                    for _c in _all_cfgs[1:]:
                        _common = {k: v for k, v in _common.items()
                                   if k in _c and _c[k] == v}
                    _abl_global_base.update(_common)
            # 用戶輸入的倉位參數 (max_positions 等) 也要帶入
            if _pos_override:
                _abl_global_base.update(_pos_override)
            if _abl_global_base:
                _base_desc = ', '.join(f'{k}={v}' for k, v in _abl_global_base.items())
                print(f"   📋 Ablation 全域基底: {_base_desc}")

            # V4.20: 預建不同 OTC 模式的 market_map (ablation 33 用)
            _has_otc_configs = any(
                cd.get('_otc_mode') for cd, _ in configs_to_run.values() if cd
            )
            _otc_market_maps = {}
            if _has_otc_configs:
                for _om in ['or', 'and', 'off']:
                    _need = any(
                        cd.get('_otc_mode') == _om
                        for cd, _ in configs_to_run.values() if cd
                    )
                    if _need and _om != 'or':  # 'or' = 現有 market_map
                        print(f"\n   🌐 預建 OTC={_om} 大盤地圖...")
                        _otc_market_maps[_om] = reconstruct_market_history(
                            start_date, end_date, otc_unsafe_mode=_om)
                _otc_market_maps['or'] = market_map  # 現有的就是 'or' 模式

            all_results = {}
            ablation_total_start = time.time()
            for config_name, (config_dict, desc) in configs_to_run.items():
                print(f"\n   🧪 {config_name}: {desc}")
                t_start = time.time()
                # _budget 特殊 key: 覆蓋 budget_per_trade (非策略參數)
                run_budget = budget
                # 合併: 全域基底 → ablation 設定疊加
                run_config = dict(_abl_global_base)  # L2 + pos_override 為底
                # V4.20: _otc_mode 特殊 key — 切換 market_map
                run_market_map = market_map
                if config_dict:
                    _abl_clean = dict(config_dict)
                    if '_budget' in _abl_clean:
                        run_budget = _abl_clean.pop('_budget')
                    if '_otc_mode' in _abl_clean:
                        _otc_m = _abl_clean.pop('_otc_mode')
                        if _otc_m in _otc_market_maps:
                            run_market_map = _otc_market_maps[_otc_m]
                    run_config.update(_abl_clean)  # ablation 參數覆蓋
                result = run_group_backtest(
                    all_stocks, start_date, end_date, run_budget,
                    run_market_map, exec_mode=exec_mode,
                    config_override=run_config if run_config else None,
                    initial_capital=initial_capital,
                    preloaded_data=shared_data,
                    industry_map=industry_map if _abl_per_ind_config else None,
                    per_industry_config=_abl_per_ind_config if _abl_per_ind_config else None,
                )
                elapsed = time.time() - t_start
                all_results[config_name] = result
                if result:
                    ret_pct = result.get('total_return_pct', result['roi'])
                    print(f"      → 損益 ${int(result['total_pnl']):+,} "
                          f"(報酬率 {ret_pct:.1f}%, "
                          f"{result['trades']}次, 勝率{result['win_rate']:.0f}%) "
                          f"⏱ {elapsed:.1f}秒")
                else:
                    print(f"      ⏱ {elapsed:.1f}秒")
            ablation_total_elapsed = time.time() - ablation_total_start
            print(f"\n   ⏱ Ablation 總耗時: {ablation_total_elapsed:.1f}秒 "
                  f"({len(configs_to_run)} 組, 平均 {ablation_total_elapsed/len(configs_to_run):.1f}秒/組)")

            print_ablation_report(all_results, configs_to_run,
                                  selected_industries, start_date, end_date, budget, exec_mode)

            # V4.7: 自動輸出每日快照 CSV
            csv_count = 0
            for config_name, result in all_results.items():
                if result and result.get('daily_snapshots'):
                    fp = export_daily_csv(result, config_name, start_date, end_date)
                    if fp:
                        csv_count += 1
            if csv_count > 0:
                print(f"\n📄 已輸出 {csv_count} 個每日快照 CSV → {REPORT_DIR}/")

    elif choice == '3':
        # ===== 多產業混合回測 =====
        print("\n🔀 多產業混合回測 — 跨產業建倉")
        print("=" * 60)

        # 顯示已選產業及其策略
        print("\n📋 已選產業及策略:")
        for ind in selected_industries:
            if ind in INDUSTRY_CONFIGS:
                cfg_info = INDUSTRY_CONFIGS[ind]
                if cfg_info['config']:
                    print(f"   {ind}: {cfg_info['desc']}")
                else:
                    print(f"   {ind}: DEFAULT_CONFIG (原生適配)")
            else:
                print(f"   {ind}: DEFAULT_CONFIG (無專屬參數)")

        # 檢查通信網路業是否已選
        has_comm = '通信網路業' in selected_industries

        print("\n📋 選擇混合策略模式:")
        print("   1. 🎯 各產業各自策略 — 統一池排名 (實驗A)")
        print("   2. 🔬 全用半導體業策略 (DEFAULT_CONFIG)")
        print("   3. 🔬 全用其他電子業策略")
        if not has_comm:
            print("   4. 🌐 加入通信網路業 (各產業各自策略)")
        else:
            print("   4. 🌐 (通信網路業已在列表中, 等同模式1)")
        _quota_sum = sum(INDUSTRY_QUOTA.values())
        _quota_desc = '/'.join(f"{v}" for v in INDUSTRY_QUOTA.values())
        print(f"   5. 📊 各產業配額制 — 產業名額限制 (實驗B, 配額 {_quota_desc}={_quota_sum})")

        mix_choice = input("👉 選擇 (Enter=1): ").strip() or '1'

        # 模式 4: 需要加入通信網路業的股票
        mix_all_stocks = list(all_stocks)  # copy
        mix_industry_map = dict(industry_map)  # copy
        mix_industries = list(selected_industries)  # copy

        if mix_choice == '4' and not has_comm:
            comm_stocks = get_stocks_by_industry('通信網路業')
            new_count = 0
            seen = set(t for t, _ in mix_all_stocks)
            for ticker, name in comm_stocks:
                if ticker not in seen:
                    mix_all_stocks.append((ticker, name))
                    mix_industry_map[ticker] = '通信網路業'
                    seen.add(ticker)
                    new_count += 1
            mix_industries.append('通信網路業')
            print(f"\n   ✅ 新增通信網路業: {new_count} 檔")
            print(f"   合計: {len(mix_all_stocks)} 檔 (去重後)")

        # 構建 per_industry_config
        per_industry_config = None
        mix_config_override = None
        mix_mode_label = ''
        mix_industry_quota = None  # V2.11: 配額制

        if mix_choice == '1' or mix_choice == '4' or mix_choice == '5':
            # 各產業各自策略 (1/4=統一池, 5=配額制)
            per_industry_config = {}
            for ind in mix_industries:
                if ind in INDUSTRY_CONFIGS and INDUSTRY_CONFIGS[ind]['config']:
                    per_industry_config[ind] = INDUSTRY_CONFIGS[ind]['config']
                # else: 該產業用 DEFAULT_CONFIG → config_override=None → fallback

            if mix_choice == '5':
                # V2.11: 配額制模式 — 確保所選產業都有配額
                mix_industry_quota = {}
                for ind in mix_industries:
                    q = INDUSTRY_QUOTA.get(ind, 0)
                    if q > 0:
                        mix_industry_quota[ind] = q
                    else:
                        # 不在配額表中的產業，給 1 個保底名額
                        mix_industry_quota[ind] = 1
                mix_mode_label = f'各產業配額制 ({"/".join(f"{ind[:2]}{q}" for ind, q in mix_industry_quota.items())})'
                print(f"\n📊 模式: 各產業配額制")
                for ind in mix_industries:
                    cfg_tag = '專屬參數' if ind in per_industry_config else 'DEFAULT'
                    print(f"   {ind}: 配額 {mix_industry_quota.get(ind, 0)} 倉 ({cfg_tag})")
                print(f"   合計配額: {sum(mix_industry_quota.values())} 倉")
            else:
                mix_mode_label = '各產業各自策略 (統一池)'
                print(f"\n🎯 模式: 各產業各自策略 (統一池排名)")
                for ind in mix_industries:
                    if ind in per_industry_config:
                        print(f"   {ind}: 專屬參數")
                    else:
                        print(f"   {ind}: DEFAULT_CONFIG")

        elif mix_choice == '2':
            # 全用半導體策略 (DEFAULT_CONFIG)
            mix_config_override = None
            mix_mode_label = '全用半導體策略'
            print(f"\n🔬 模式: 全用半導體策略 (DEFAULT_CONFIG)")

        elif mix_choice == '3':
            # 全用其他電子業策略
            if '其他電子業' in INDUSTRY_CONFIGS and INDUSTRY_CONFIGS['其他電子業']['config']:
                mix_config_override = INDUSTRY_CONFIGS['其他電子業']['config']
            mix_mode_label = '全用其他電子業策略'
            print(f"\n🔬 模式: 全用其他電子業策略")
            if mix_config_override:
                print(f"   覆蓋項目: {', '.join(mix_config_override.keys())}")

        else:
            mix_mode_label = '各產業各自策略'

        print(f"\n   產業: {' + '.join(mix_industries)}")
        print(f"   標的: {len(mix_all_stocks)} 檔")
        print(f"   策略: {mix_mode_label}")

        # V2.11: 倉位數量選擇
        print(f"\n📦 倉位設定 (目前 DEFAULT = 15):")
        print(f"   Enter=15 / 輸入數字 (10/15/20/25/30)")
        _pos_input = input("👉 最大持倉數: ").strip()
        _mix_pos_override = None
        if _pos_input and _pos_input.isdigit():
            _pos_val = int(_pos_input)
            if _pos_val != 15 and 5 <= _pos_val <= 50:
                _mix_pos_override = {'max_positions': _pos_val}
                # 自動調整每日買入數: 大約 pos/5, 最少2最多6
                _buy_per_day = max(2, min(6, _pos_val // 5))
                _mix_pos_override['max_new_buy_per_day'] = _buy_per_day
                print(f"   ✅ 設定: {_pos_val} 倉, 每日最多買 {_buy_per_day} 檔")
        if _mix_pos_override:
            # max_positions 是全局參數, 透過 config_override 傳入 run_group_backtest
            # (per_industry_config 只管策略參數, 不管倉位)
            if mix_config_override is None:
                mix_config_override = _mix_pos_override
            else:
                mix_config_override.update(_mix_pos_override)

        # V5.0: Budget 比較模式
        print(f"\n💰 Budget 比較:")
        print(f"   Enter = 只跑 ${budget:,} (單次)")
        print(f"   B     = 跑多個 budget 比較 (10K/15K/20K/25K/30K/45K/60K)")
        _bud_choice = input("👉 選擇: ").strip().upper()

        _budget_list = None
        if _bud_choice == 'B':
            _budget_list = [10000, 15000, 20000, 25000, 30000, 45000, 60000]
            print(f"   ✅ 將跑 {len(_budget_list)} 個 budget 級距: {', '.join(f'${b:,}' for b in _budget_list)}")

        # 下載資料
        _sim_start = pd.Timestamp(start_date)
        _download_start = (_sim_start - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
        _download_end = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')

        print(f"\n🔍 檢查資料快取...")
        cache_info = check_cache_vs_latest(mix_all_stocks, _download_start, _download_end)
        print_cache_check(cache_info)

        force_refresh = False
        if cache_info is not None and not cache_info['data_match']:
            print(f"\n   ⚠️ 快取與最新數據有差異！")
            refresh_choice = input("   👉 要更新資料嗎？(Y=重新下載 / Enter=繼續用快取): ").strip().upper()
            force_refresh = (refresh_choice == 'Y')
        elif cache_info is not None:
            print(f"   ✅ 快取數據與最新一致，直接使用")

        if _budget_list:
            # === Budget 比較模式 ===
            print(f"\n{'='*100}")
            print(f"💰 Budget 比較模式 — 各產業各自 L2 參數")
            print(f"   產業: {' + '.join(mix_industries)}")
            print(f"   策略: {mix_mode_label}")
            print(f"   倉位: {_mix_pos_override.get('max_positions', 12) if _mix_pos_override else 12}")
            print(f"{'='*100}")

            _bud_results = {}
            _bud_configs = {}
            for _bud in _budget_list:
                _bud_label = f'budget_{_bud//1000}k'
                _bud_desc = f'💰 每次 ${_bud:,}'
                _bud_configs[_bud_label] = (None, _bud_desc)
                _bud_capital = int(_bud * (_mix_pos_override.get('max_positions', 12) if _mix_pos_override else 12) * 1.5)
                _bud_capital = max(_bud_capital, initial_capital)

                print(f"\n   🧪 {_bud_label}: {_bud_desc} (初始資金 ${_bud_capital:,})")
                t_start = time.time()
                _bud_result = run_group_backtest(
                    mix_all_stocks, start_date, end_date, _bud,
                    market_map, exec_mode=exec_mode,
                    config_override=mix_config_override,
                    initial_capital=_bud_capital,
                    force_refresh=force_refresh,
                    industry_map=mix_industry_map,
                    per_industry_config=per_industry_config,
                    industry_quota=mix_industry_quota,
                    preloaded_data=None,
                )
                elapsed = time.time() - t_start
                _bud_results[_bud_label] = _bud_result
                if _bud_result:
                    _ret = _bud_result.get('total_return_pct', 0)
                    _sh = _bud_result.get('sharpe_ratio', 0)
                    _mdd = _bud_result.get('mdd_pct', 0)
                    print(f"      → 報酬 {_ret:>+7.1f}% | Sharpe {_sh:>5.2f} | MDD {_mdd:>5.1f}% | "
                          f"勝率 {_bud_result['win_rate']:.0f}% | {_bud_result['trades']}次 ⏱{elapsed:.1f}s")

            # 使用 baseline key = 當前 budget
            _baseline_label = f'budget_{budget//1000}k'
            if _baseline_label in _bud_results:
                _bud_configs[_baseline_label] = (None, f'💰 每次 ${budget:,} (目前設定)')

            print_ablation_report(_bud_results, _bud_configs,
                                  mix_industries, start_date, end_date, budget, exec_mode)

            # CSV 輸出
            csv_count = 0
            for _bud_label, _bud_result in _bud_results.items():
                if _bud_result and _bud_result.get('daily_snapshots'):
                    fp = export_daily_csv(_bud_result, f'mix_{_bud_label}', start_date, end_date)
                    if fp:
                        csv_count += 1
            if csv_count > 0:
                print(f"\n📄 已輸出 {csv_count} 個每日快照 CSV → {REPORT_DIR}/")

        else:
            # === 單次回測 ===
            result = run_group_backtest(
                mix_all_stocks, start_date, end_date, budget,
                market_map, exec_mode=exec_mode,
                config_override=mix_config_override,
                initial_capital=initial_capital,
                force_refresh=force_refresh,
                industry_map=mix_industry_map,
                per_industry_config=per_industry_config,
                industry_quota=mix_industry_quota,
            )
            print_group_report(result, mix_industries, start_date, end_date, budget)

            # CSV 輸出
            if result and result.get('daily_snapshots'):
                _safe_mode = f'mixed_mode{mix_choice}'
                fp = export_daily_csv(result, _safe_mode, start_date, end_date)
                if fp:
                    print(f"\n📄 每日快照 CSV → {fp}")

    elif choice == '4':
        # ===== 產業組合掃描 =====
        import itertools

        print("\n🔍 產業組合掃描 — 自動找最佳產業組合")
        print("=" * 80)
        print("   策略: 全用半導體策略 (DEFAULT_CONFIG)")
        print("   方法: 以半導體為基底, 逐步加入其他產業, 比較績效")

        # --- 載入產業資料庫 ---
        print("\n⏳ 載入產業資料庫...")
        df_all = get_all_companies()
        scan_all_industries = list_industries(df_all)

        if not scan_all_industries:
            print("❌ 無法載入產業資料")
            return

        # --- 選擇基底產業 ---
        print(f"\n🏭 基底產業: 半導體業 (固定)")

        # --- 選擇候選產業 ---
        print(f"\n🏭 候選產業列表 ({len(scan_all_industries)} 個):")
        for i, ind in enumerate(scan_all_industries):
            marker = " ★" if ind == '半導體業' else ""
            print(f"  [{i+1:02d}] {ind:<12}{marker}", end="")
            if (i + 1) % 4 == 0:
                print()
        print()

        print("\n📋 選擇候選產業 (半導體業自動包含):")
        print("   a. 🔬 電子相關 (其他電子/通信網路/電腦週邊/光電/電子通路/電子零組件/電機機械)")
        print("   b. 📋 自選 (輸入編號)")
        print("   c. 🌐 全部產業 (很慢!)")
        scan_sel = input("👉 選擇 (Enter=a): ").strip().lower() or 'a'

        _base_industry = '半導體業'
        _candidate_industries = []

        if scan_sel == 'a':
            _elec_names = ['其他電子業', '通信網路業', '電腦及週邊設備業', '光電業',
                           '電子通路業', '電子零組件業', '電機機械']
            _candidate_industries = [ind for ind in _elec_names if ind in scan_all_industries]
        elif scan_sel == 'c':
            _candidate_industries = [ind for ind in scan_all_industries if ind != _base_industry]
        else:
            # 自選
            raw = input("👉 輸入候選產業編號 (逗號分隔, 半導體自動包含): ").strip()
            for part in raw.split(','):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(scan_all_industries):
                        ind = scan_all_industries[idx]
                        if ind != _base_industry and ind not in _candidate_industries:
                            _candidate_industries.append(ind)

        if not _candidate_industries:
            print("❌ 未選擇任何候選產業")
            return

        print(f"\n✅ 基底: {_base_industry}")
        print(f"✅ 候選: {', '.join(_candidate_industries)} ({len(_candidate_industries)} 個)")

        # --- 選擇最大組合大小 ---
        print(f"\n📋 組合掃描範圍:")
        print(f"   單產業加入: {len(_candidate_industries)} 組")
        if len(_candidate_industries) >= 2:
            n2 = len(list(itertools.combinations(_candidate_industries, 2)))
            print(f"   雙產業加入: {n2} 組")
        if len(_candidate_industries) >= 3:
            n3 = len(list(itertools.combinations(_candidate_industries, 3)))
            print(f"   三產業加入: {n3} 組")
        total_max = sum(len(list(itertools.combinations(_candidate_industries, r)))
                        for r in range(1, len(_candidate_industries) + 1))
        print(f"   全部組合:   {total_max + 1} 組 (含基底 only)")

        try:
            max_combo_in = input(f"👉 最多同時加入幾個產業? (Enter=3, max={len(_candidate_industries)}): ").strip()
            max_combo_size = int(max_combo_in) if max_combo_in else min(3, len(_candidate_industries))
            max_combo_size = min(max_combo_size, len(_candidate_industries))
        except ValueError:
            max_combo_size = min(3, len(_candidate_industries))

        # 產生所有組合
        combos = [()]  # 基底 only
        for r in range(1, max_combo_size + 1):
            combos.extend(itertools.combinations(_candidate_industries, r))

        print(f"\n🔄 共 {len(combos)} 組組合待測試")

        # --- 下載資料 ---
        # 收集所有可能用到的股票 (一次下載)
        print(f"\n⏳ 收集所有產業股票...")
        _all_ind_stocks = {}  # {industry: [(ticker, name), ...]}
        _all_scan_stocks = []
        _seen = set()

        # 基底
        base_stocks = get_stocks_by_industry(_base_industry)
        _all_ind_stocks[_base_industry] = base_stocks
        for t, n in base_stocks:
            if t not in _seen:
                _all_scan_stocks.append((t, n))
                _seen.add(t)

        # 候選
        for ind in _candidate_industries:
            ind_stocks = get_stocks_by_industry(ind)
            _all_ind_stocks[ind] = ind_stocks
            for t, n in ind_stocks:
                if t not in _seen:
                    _all_scan_stocks.append((t, n))
                    _seen.add(t)

        print(f"   {_base_industry}: {len(base_stocks)} 檔")
        for ind in _candidate_industries:
            print(f"   {ind}: {len(_all_ind_stocks[ind])} 檔")
        print(f"   合計: {len(_all_scan_stocks)} 檔 (去重後)")

        _sim_start = pd.Timestamp(start_date)
        _download_start = (_sim_start - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
        _download_end = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')

        print(f"\n⏳ 一次性下載 {len(_all_scan_stocks)} 檔股票資料...")
        shared_data, shared_skipped = batch_download_stocks(
            _all_scan_stocks, _download_start, _download_end,
            min_data_days=MIN_DATA_DAYS,
        )
        print(f"   ✅ 有效標的: {len(shared_data)} 檔")
        if shared_skipped['download_fail'] > 0:
            print(f"   ⚠️ 下載失敗: {shared_skipped['download_fail']} 檔")
        if shared_skipped['low_data'] > 0:
            print(f"   ⚠️ 資料不足: {shared_skipped['low_data']} 檔")

        # V4.17: 建立 per-industry 參數 (各產業用各自 L2)
        _scan_per_industry_config = {}
        _scan_ind_with_l2 = []
        for _ind in [_base_industry] + _candidate_industries:
            if _ind in INDUSTRY_CONFIGS and INDUSTRY_CONFIGS[_ind]['config']:
                _scan_per_industry_config[_ind] = INDUSTRY_CONFIGS[_ind]['config']
                _scan_ind_with_l2.append(_ind)

        # 建立全域 industry_map (ticker → industry)
        _scan_industry_map = {}
        for _ind, _stocks in _all_ind_stocks.items():
            for _t, _n in _stocks:
                _scan_industry_map[_t] = _ind

        # --- 逐組合回測 ---
        print(f"\n{'='*100}")
        print(f"🔄 開始產業組合掃描: {len(combos)} 組")
        if _scan_per_industry_config:
            print(f"   策略: 各產業 L2 參數 ({len(_scan_ind_with_l2)} 個有L2, 其餘用DEFAULT)")
        else:
            print(f"   策略: DEFAULT_CONFIG")
        print(f"   區間: {start_date} ~ {end_date}")
        print(f"   每次: ${budget:,} | 初始: ${initial_capital:,}")
        print(f"{'='*100}")

        scan_results = []  # [(combo_label, industries_list, result), ...]
        scan_total_start = time.time()

        for combo_idx, extra_inds in enumerate(combos):
            # 組合標的
            combo_industries = [_base_industry] + list(extra_inds)
            combo_stocks = []
            combo_seen = set()
            for ind in combo_industries:
                for t, n in _all_ind_stocks.get(ind, []):
                    if t not in combo_seen:
                        combo_stocks.append((t, n))
                        combo_seen.add(t)

            # 標籤
            if not extra_inds:
                combo_label = _base_industry
            else:
                combo_label = _base_industry + ' + ' + ' + '.join(extra_inds)

            # 篩選 preloaded_data 中屬於此組合的股票
            combo_data = {t: shared_data[t] for t in combo_seen if t in shared_data}

            print(f"\n   [{combo_idx+1}/{len(combos)}] {combo_label}")
            print(f"      標的: {len(combo_stocks)} 檔 (有效: {len(combo_data)})")

            t_start = time.time()
            # V4.17: 組合掃描也用各產業 L2
            _combo_industry_map = {t: _scan_industry_map[t] for t in combo_seen if t in _scan_industry_map}
            result = run_group_backtest(
                combo_stocks, start_date, end_date, budget,
                market_map, exec_mode=exec_mode,
                config_override=None,
                initial_capital=initial_capital,
                preloaded_data=combo_data,
                industry_map=_combo_industry_map,
                per_industry_config=_scan_per_industry_config,
            )
            elapsed = time.time() - t_start

            if result:
                ret_pct = result.get('total_return_pct', result['roi'])
                sharpe = result.get('sharpe_ratio', 0)
                mdd_pct = result.get('mdd_pct', 0)
                calmar = result.get('calmar_ratio', 0)
                print(f"      → 報酬 {ret_pct:>+7.1f}% | Sharpe {sharpe:>5.2f} | "
                      f"MDD {mdd_pct:>5.1f}% | Calmar {calmar:>5.2f} | "
                      f"勝率 {result['win_rate']:.0f}% | {result['trades']}次 "
                      f"⏱{elapsed:.1f}s")
                scan_results.append((combo_label, combo_industries, result))

                # CSV
                if result.get('daily_snapshots'):
                    _safe_label = combo_label.replace(' ', '_').replace('+', '_')
                    export_daily_csv(result, f'scan_{_safe_label}', start_date, end_date)
            else:
                print(f"      → ❌ 回測失敗 ⏱{elapsed:.1f}s")

        scan_total_elapsed = time.time() - scan_total_start

        # --- 比較報告 ---
        print(f"\n{'='*120}")
        print(f"🏆 產業組合掃描結果 (全用半導體策略)")
        print(f"   區間: {start_date} ~ {end_date} | 每次 ${budget:,} | 初始 ${initial_capital:,}")
        print(f"   掃描: {len(scan_results)} 組 | 耗時: {scan_total_elapsed:.1f}秒")
        print(f"{'='*120}")

        # 排序: Sharpe 降序
        scan_results.sort(key=lambda x: x[2].get('sharpe_ratio', 0), reverse=True)

        # 找基底結果 (對照)
        base_result = None
        for label, inds, r in scan_results:
            if inds == [_base_industry]:
                base_result = r
                break
        base_sharpe = base_result.get('sharpe_ratio', 0) if base_result else 0
        base_mdd = base_result.get('mdd_pct', 0) if base_result else 0
        base_calmar = base_result.get('calmar_ratio', 0) if base_result else 0
        base_ret = base_result.get('total_return_pct', 0) if base_result else 0

        print(f"\n{'排名':>4} {'產業組合':<50} {'報酬%':>7} {'CAGR%':>7} "
              f"{'MDD%':>7} {'Sharpe':>7} {'Calmar':>7} {'勝率':>6} {'交易':>5} {'標的':>5}")
        print("─" * 130)

        for rank, (label, inds, r) in enumerate(scan_results, 1):
            ret_pct = r.get('total_return_pct', 0)
            cagr = r.get('cagr', 0)
            mdd_pct = r.get('mdd_pct', 0)
            sharpe = r.get('sharpe_ratio', 0)
            calmar = r.get('calmar_ratio', 0)
            win_rate = r.get('win_rate', 0)
            trades = r.get('trades', 0)
            n_stocks = len([t for ind in inds for t, _ in _all_ind_stocks.get(ind, [])])
            marker = " ← 基底" if inds == [_base_industry] else ""
            print(f"{rank:>4} {label:<50} {ret_pct:>7.1f} {cagr:>7.1f} "
                  f"{mdd_pct:>7.1f} {sharpe:>7.2f} {calmar:>7.2f} "
                  f"{win_rate:>5.0f}% {trades:>5} {n_stocks:>5}{marker}")

        # --- 差異分析 (vs 基底) ---
        if base_result:
            print(f"\n📊 相對基底 ({_base_industry}) 的差異:")
            print(f"{'排名':>4} {'產業組合':<50} {'Δ報酬%':>8} {'ΔMDD%':>8} "
                  f"{'ΔSharpe':>8} {'ΔCalmar':>8} {'解讀'}")
            print("─" * 130)

            for rank, (label, inds, r) in enumerate(scan_results, 1):
                if inds == [_base_industry]:
                    continue
                ret_diff = r.get('total_return_pct', 0) - base_ret
                mdd_diff = r.get('mdd_pct', 0) - base_mdd
                sharpe_diff = r.get('sharpe_ratio', 0) - base_sharpe
                calmar_diff = r.get('calmar_ratio', 0) - base_calmar

                # 解讀
                if sharpe_diff > 0.05 and mdd_diff <= 0:
                    interp = "🟢 更好 (Sharpe↑ MDD↓)"
                elif sharpe_diff > 0.05 and mdd_diff > 0:
                    interp = "🟡 報酬↑但風險也↑"
                elif sharpe_diff < -0.05 and mdd_diff <= 0:
                    interp = "🟡 報酬↓但風險也↓"
                elif sharpe_diff < -0.05:
                    interp = "🔴 更差 (Sharpe↓)"
                else:
                    interp = "⚪ 差異不大"

                # 特殊標記
                n_extra = len(inds) - 1
                prefix = f"{'  '*n_extra}+{''.join([i[:2] for i in inds[1:]])}"

                print(f"{rank:>4} {label:<50} {ret_diff:>+8.1f} {mdd_diff:>+8.1f} "
                      f"{sharpe_diff:>+8.2f} {calmar_diff:>+8.2f} {interp}")

        # --- 最佳組合推薦 ---
        print(f"\n{'─'*80}")
        print(f"🏆 推薦排名 (依 Sharpe 排序):")
        for rank, (label, inds, r) in enumerate(scan_results[:5], 1):
            sharpe = r.get('sharpe_ratio', 0)
            calmar = r.get('calmar_ratio', 0)
            mdd = r.get('mdd_pct', 0)
            ret = r.get('total_return_pct', 0)
            print(f"   #{rank}: {label}")
            print(f"       Sharpe={sharpe:.2f} | Calmar={calmar:.2f} | "
                  f"報酬={ret:+.1f}% | MDD={mdd:.1f}%")

        # 最佳 Calmar (風險調整最佳)
        best_calmar = max(scan_results, key=lambda x: x[2].get('calmar_ratio', 0))
        print(f"\n   🛡️ 最佳風險調整 (Calmar): {best_calmar[0]}")
        print(f"       Calmar={best_calmar[2].get('calmar_ratio', 0):.2f} | "
              f"Sharpe={best_calmar[2].get('sharpe_ratio', 0):.2f} | "
              f"MDD={best_calmar[2].get('mdd_pct', 0):.1f}%")

        # 最低 MDD
        best_mdd = min(scan_results, key=lambda x: abs(x[2].get('mdd_pct', 999)))
        print(f"\n   📉 最低回撤 (MDD): {best_mdd[0]}")
        print(f"       MDD={best_mdd[2].get('mdd_pct', 0):.1f}% | "
              f"Sharpe={best_mdd[2].get('sharpe_ratio', 0):.2f} | "
              f"報酬={best_mdd[2].get('total_return_pct', 0):+.1f}%")

        print(f"\n   ⏱ 掃描總耗時: {scan_total_elapsed:.1f}秒 "
              f"({len(scan_results)} 組, 平均 {scan_total_elapsed/max(1,len(scan_results)):.1f}秒/組)")
        print(f"{'='*120}")

    else:
        # ===== 標準回測 =====

        # V6: 子模式選擇 — 回測模擬 vs 真實部位分析
        print(f"\n🔧 回測模式:")
        print(f"   A. 回測模擬 (標準回測, 輸出 sim_portfolio CSV)")
        print(f"   B. 真實部位分析 (讀取 portfolio CSV, 分析績效 + T+1 訊號)")
        _mode1_sub = input("   👉 選擇 (Enter=A): ").strip().upper()

        if _mode1_sub == 'B':
            # --- 真實部位分析 ---
            _ind_label = '_'.join(selected_industries)

            # V6: 產業參數選擇 — 與 Mode 1A 相同邏輯, 確保 T+1 訊號一致
            _rp_config_override = None
            _rp_per_industry = {}
            _rp_use_per_industry = False

            if len(selected_industries) == 1:
                _ind_name = selected_industries[0]
                if _ind_name in INDUSTRY_CONFIGS:
                    _ind_cfg = INDUSTRY_CONFIGS[_ind_name]
                    _rp_config_override = _ind_cfg['config'] if _ind_cfg['config'] else None
                    _has_v5 = _ind_name in INDUSTRY_CONFIGS_V5 and INDUSTRY_CONFIGS_V5[_ind_name].get('config')
                    if _rp_config_override:
                        print(f"\n🏭 產業參數選擇 [{_ind_name}] (真實部位):")
                        print(f"   ★ L3: {_ind_cfg['desc']}")
                        if _has_v5:
                            _v5_cfg = INDUSTRY_CONFIGS_V5[_ind_name]
                            print(f"   🧪 V5: {_v5_cfg['desc']}")
                            print(f"\n   A. 使用 L3 最佳參數 (推薦)")
                            print(f"   B. 🧪 使用 V5 冠軍參數 (實驗性)")
                            print(f"   C. 使用 DEFAULT_CONFIG")
                            _rp_param = input("   👉 選擇 (Enter=A): ").strip().upper()
                            if _rp_param == 'B':
                                _rp_config_override = _v5_cfg['config']
                                print(f"   → 使用 V5 冠軍參數 🧪")
                            elif _rp_param == 'C':
                                _rp_config_override = None
                                print(f"   → 使用 DEFAULT_CONFIG")
                            else:
                                print(f"   → 使用 L3 參數 ✅")
                        else:
                            print(f"   → 使用 L3 參數 ✅")
            else:
                for _ind in selected_industries:
                    if _ind in INDUSTRY_CONFIGS and INDUSTRY_CONFIGS[_ind]['config']:
                        _rp_per_industry[_ind] = INDUSTRY_CONFIGS[_ind]['config']
                        _rp_use_per_industry = True

            rp_result = run_real_portfolio_analysis(
                selected_industries=selected_industries,
                industry_map=industry_map,
                all_stocks=list(all_stocks),  # copy to avoid mutation
                start_date=start_date,
                end_date=end_date,
                budget=budget,
                initial_capital=initial_capital,
                market_map=market_map,
                config_override=_rp_config_override,
                per_industry_config=_rp_per_industry,
                use_per_industry=_rp_use_per_industry,
            )
            return  # 真實部位分析完成, 不繼續跑回測

        # V4.15+V4.17: 產業專屬參數自動偵測
        _industry_override = None
        _industry_cfg_desc = None
        _use_per_industry = False
        _per_industry_config = {}

        if len(selected_industries) == 1:
            _ind_name = selected_industries[0]
            if _ind_name in INDUSTRY_CONFIGS:
                _ind_cfg = INDUSTRY_CONFIGS[_ind_name]
                _industry_override = _ind_cfg['config'] if _ind_cfg['config'] else None
                _industry_cfg_desc = _ind_cfg['desc']
                # 檢查是否有 V5 升級版
                _has_v5 = _ind_name in INDUSTRY_CONFIGS_V5 and INDUSTRY_CONFIGS_V5[_ind_name].get('config')
                if _industry_override:
                    print(f"\n🏭 產業參數選擇 [{_ind_name}]:")
                    print(f"   ★ L3: {_industry_cfg_desc}")
                    print(f"   覆蓋項目: {', '.join(_industry_override.keys())}")
                    if _has_v5:
                        _v5_cfg = INDUSTRY_CONFIGS_V5[_ind_name]
                        print(f"   🧪 V5: {_v5_cfg['desc']}")
                        print(f"   V5 覆蓋項目: {', '.join(_v5_cfg['config'].keys())}")
                        print(f"\n   A. 使用 L3 最佳參數 (推薦, 已驗證)")
                        print(f"   B. 🧪 使用 V5 冠軍參數 (實驗性)")
                        print(f"   C. 使用 DEFAULT_CONFIG (無產業優化)")
                        _param_choice = input("   👉 選擇 (Enter=A): ").strip().upper()
                        if _param_choice == 'B':
                            _industry_override = _v5_cfg['config']
                            print(f"   → 使用 V5 冠軍參數 🧪")
                        elif _param_choice == 'C':
                            _industry_override = None
                            print(f"   → 使用 DEFAULT_CONFIG")
                        else:
                            print(f"   → 使用 L3 參數 ✅")
                    else:
                        print(f"\n   A. 使用 L3 最佳參數 (推薦)")
                        print(f"   B. 使用 DEFAULT_CONFIG")
                        _param_choice = input("   👉 選擇 (Enter=A): ").strip().upper()
                        if _param_choice == 'B':
                            _industry_override = None
                            print(f"   → 使用 DEFAULT_CONFIG")
                        else:
                            print(f"   → 使用 L3 參數 ✅")
                else:
                    print(f"\n🏭 [{_ind_name}]: 使用 DEFAULT_CONFIG (無專屬參數)")

        elif len(selected_industries) > 1:
            # V4.17: 多產業 — 讓使用者選擇參數模式
            _configs_found = []
            for _ind in selected_industries:
                if _ind in INDUSTRY_CONFIGS and INDUSTRY_CONFIGS[_ind]['config']:
                    _configs_found.append((_ind, INDUSTRY_CONFIGS[_ind]['config']))
                    _per_industry_config[_ind] = INDUSTRY_CONFIGS[_ind]['config']
                else:
                    _configs_found.append((_ind, None))

            _n_with_l2 = sum(1 for _, c in _configs_found if c is not None)

            if _n_with_l2 > 0:
                # 收集有 L2 的產業供選項 C 使用
                _l2_industries = [(i, c) for i, c in _configs_found if c is not None]

                print(f"\n🏭 多產業參數選擇 ({_n_with_l2}/{len(selected_industries)} 個產業有 L2/L3 參數):")
                for _ind, _cfg in _configs_found:
                    if _cfg:
                        print(f"   ★ {_ind}: {INDUSTRY_CONFIGS[_ind]['desc']}")
                    else:
                        print(f"   · {_ind}: DEFAULT_CONFIG")
                print(f"\n   A. 各產業用各自最佳參數 (無L2的用DEFAULT)")
                print(f"   B. 全部統一用 DEFAULT_CONFIG")
                print(f"   C. 全部統一用某一產業的參數 (擴大選股池, 參數不變)")
                _param_choice = input("   👉 選擇 (Enter=A): ").strip().upper()
                if _param_choice == 'C':
                    # 讓使用者選哪個產業的 L2
                    if len(_l2_industries) == 1:
                        _chosen_ind, _chosen_cfg = _l2_industries[0]
                    else:
                        print(f"\n   選擇要統一使用哪個產業的 L2:")
                        for _ci, (_cind, _ccfg) in enumerate(_l2_industries, 1):
                            print(f"      {_ci}. {_cind}")
                        _c_choice = input(f"      👉 選擇 (Enter=1): ").strip()
                        _c_idx = int(_c_choice) - 1 if _c_choice.isdigit() else 0
                        _c_idx = max(0, min(_c_idx, len(_l2_industries) - 1))
                        _chosen_ind, _chosen_cfg = _l2_industries[_c_idx]
                    _industry_override = _chosen_cfg
                    _per_industry_config = {}
                    _use_per_industry = False
                    print(f"   → 全部股票統一用 {_chosen_ind} L2 參數 ✅")
                    print(f"     ({', '.join(f'{k}={v}' for k, v in _chosen_cfg.items())})")
                elif _param_choice == 'B':
                    _per_industry_config = {}
                    print(f"   → 統一使用 DEFAULT_CONFIG")
                else:
                    _use_per_industry = True
                    _industry_override = None  # 不用單一 override
                    print(f"   → 各產業套用各自最佳參數 ✅")
            else:
                print(f"\n🏭 多產業: 全部使用 DEFAULT_CONFIG (無產業有 L2 參數)")

        # =============================================
        # V5: 載入初始庫存 (可選 — 從 CSV 帶入持倉)
        # =============================================
        _init_pos = None
        _init_pos_source = None
        _ind_label = '_'.join(selected_industries)  # e.g. '半導體業'

        print(f"\n📦 初始庫存:")
        print(f"   A. 空倉開始 (標準回測)")
        print(f"   B. 從 CSV 載入初始部位")
        _pos_choice = input("   👉 選擇 (Enter=A): ").strip().upper()

        if _pos_choice == 'B':
            # 自動尋找可用的庫存 CSV
            _prev_date_ts = pd.Timestamp(start_date) - pd.Timedelta(days=1)
            _prev_date = _prev_date_ts.strftime('%Y%m%d')

            _candidates = []

            # 搜尋 portfolio_{產業}_{日期}.csv 格式
            _base_ind_csv = os.path.join(_BASE_DIR, f"portfolio_{_ind_label}.csv")
            _snap_ind = _find_best_snapshot(_base_ind_csv, start_date)
            if _snap_ind:
                _candidates.append(('industry', _snap_ind))

            # 搜尋 sim_portfolio_{產業}_{日期}.csv 格式 (回測輸出)
            _base_sim_csv = os.path.join(_BASE_DIR, f"sim_portfolio_{_ind_label}.csv")
            _snap_sim = _find_best_snapshot(_base_sim_csv, start_date)
            if _snap_sim:
                _candidates.append(('sim', _snap_sim))

            # 搜尋 portfolio_strategy_{日期}.csv (舊格式)
            _snap_strat = _find_best_snapshot(PORTFOLIO_STRATEGY_FILE, start_date)
            if _snap_strat:
                _candidates.append(('strategy', _snap_strat))

            # 搜尋 reports/ 目錄下的快照 (Q6 輸出, 含 sim_portfolio 和 portfolio)
            import glob as _glob_init
            _report_pattern = os.path.join(REPORT_DIR, f"positions_*", f"*portfolio_*_{_prev_date}.csv")
            _report_snaps = _glob_init.glob(_report_pattern)
            for _rs in sorted(_report_snaps):
                _candidates.append(('report', _rs))

            # 去重 (同一檔案可能被多種方式找到)
            _seen_paths = set()
            _unique_candidates = []
            for src, fp in _candidates:
                _real = os.path.realpath(fp)
                if _real not in _seen_paths:
                    _seen_paths.add(_real)
                    _unique_candidates.append((src, fp))
            _candidates = _unique_candidates

            if _candidates:
                print(f"   找到可用庫存:")
                for i, (src, fp) in enumerate(_candidates, 1):
                    _df_preview = pd.read_csv(fp, dtype={'ticker': str})
                    print(f"      {i}. {os.path.basename(fp)} ({len(_df_preview)} 檔)")
                print(f"      {len(_candidates)+1}. 手動輸入路徑")
                _csv_choice = input(f"      👉 選擇 (Enter=1): ").strip()
                _chosen = None
                if _csv_choice.isdigit() and 1 <= int(_csv_choice) <= len(_candidates):
                    _, _chosen = _candidates[int(_csv_choice)-1]
                elif _csv_choice == str(len(_candidates)+1):
                    _chosen = input("      📂 CSV 路徑: ").strip()
                else:
                    _, _chosen = _candidates[0]
            else:
                print(f"   未找到自動匹配的庫存 CSV")
                _chosen = input("      📂 請輸入 CSV 路徑 (或 Enter=空倉): ").strip()
                if not _chosen:
                    _chosen = None

            if _chosen and os.path.exists(_chosen):
                _pos_df = load_portfolio(_chosen)
                _init_pos = _strategy_df_to_engine_positions(_pos_df, start_date)
                _init_pos_source = os.path.basename(_chosen)
                print(f"   ✅ 載入: {_init_pos_source} ({len(_init_pos)} 檔)")
                for _t, _p in sorted(_init_pos.items(), key=lambda x: x[1].get('name', x[0])):
                    print(f"      {_t} {_p['name']} {_p['shares']}股 @ ${_p['avg_cost']:.1f}")

                # 確保初始部位的 ticker 在下載清單中
                _existing_tickers = set(t for t, _ in all_stocks)
                for _t, _p in _init_pos.items():
                    if _t not in _existing_tickers:
                        all_stocks.append((_t, _p.get('name', _t)))
                        _existing_tickers.add(_t)
                        print(f"   📌 庫存補充: {_t} {_p.get('name', _t)} (不在產業池)")
            else:
                print(f"   → 空倉開始")

        # ✅ 快取檢查
        _sim_start = pd.Timestamp(start_date)
        _download_start = (_sim_start - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
        _download_end = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')

        print(f"\n🔍 檢查資料快取...")
        cache_info = check_cache_vs_latest(all_stocks, _download_start, _download_end)
        print_cache_check(cache_info)

        force_refresh = False
        if cache_info is not None and not cache_info['data_match']:
            print(f"\n   ⚠️ 快取與最新數據有差異！")
            refresh_choice = input("   👉 要更新資料嗎？(Y=重新下載 / Enter=繼續用快取): ").strip().upper()
            force_refresh = (refresh_choice == 'Y')
        elif cache_info is not None:
            print(f"   ✅ 快取數據與最新一致，直接使用")

        # 合併: 產業專屬參數 + 用戶輸入的倉位參數
        _final_override = {}
        if _industry_override:
            _final_override.update(_industry_override)
        if _pos_override:
            _final_override.update(_pos_override)  # 用戶輸入優先

        if _use_per_industry:
            # V4.17: 多產業各自 L2
            result = run_group_backtest(
                all_stocks, start_date, end_date, budget,
                market_map, exec_mode=exec_mode,
                config_override=_pos_override or None,  # 只傳倉位參數, 策略參數由 per_industry 負責
                initial_capital=initial_capital,
                force_refresh=force_refresh,
                industry_map=industry_map,
                per_industry_config=_per_industry_config,
                initial_positions=_init_pos,   # V5: CSV 初始部位
                _capture_positions=True,       # V5: 每日部位快照
            )
        else:
            result = run_group_backtest(
                all_stocks, start_date, end_date, budget,
                market_map, exec_mode=exec_mode,
                config_override=_final_override or None,
                initial_capital=initial_capital,
                force_refresh=force_refresh,
                initial_positions=_init_pos,   # V5: CSV 初始部位
                _capture_positions=True,       # V5: 每日部位快照
            )
        print_group_report(result, selected_industries, start_date, end_date, budget)

        # V4.7: 標準回測也輸出 CSV
        if result and result.get('daily_snapshots'):
            fp = export_daily_csv(result, 'standard', start_date, end_date)
            if fp:
                print(f"\n📄 每日快照 CSV → {fp}")

        # V5: 每日部位快照 (portfolio CSV 格式, 可直接載入)
        if result and result.get('_positions_history'):
            _ind_map = industry_map if 'industry_map' in dir() else None
            snap_files = export_portfolio_snapshots(
                result, 'standard', start_date, end_date,
                industry_map=_ind_map,
                industry_label=_ind_label,
            )
            if snap_files:
                snap_dir = os.path.dirname(snap_files[0])
                print(f"📁 每日部位快照 ({len(snap_files)} 天) → {snap_dir}/")

        # V5: 最終部位 CSV (可直接做為 portfolio_strategy.csv 使用)
        if result and result.get('_raw_positions'):
            os.makedirs(REPORT_DIR, exist_ok=True)
            _final_rows = {}
            _ind_map = industry_map if 'industry_map' in dir() else None
            for _ft, _fp in result['_raw_positions'].items():
                _f_ind = _ind_map.get(_ft, '') if _ind_map else ''
                _final_rows[_ft] = {
                    'ticker': _ft,
                    'name': _fp.get('name', _ft),
                    'industry': _f_ind,
                    'shares': _fp['shares'],
                    'avg_cost': round(_fp['avg_cost'], 2),
                    'buy_price': round(_fp.get('buy_price', _fp['avg_cost']), 2),
                    'peak_since_entry': round(_fp.get('peak_since_entry', _fp['avg_cost']), 2),
                    'note': 'Mode1回測最終部位',
                }
            _final_path = os.path.join(REPORT_DIR, f"sim_portfolio_{_ind_label}_final_{end_date.replace('-', '')}.csv")
            _save_portfolio_csv(_final_rows, _final_path, end_date, 'Mode1回測最終部位')


if __name__ == "__main__":
    main()
