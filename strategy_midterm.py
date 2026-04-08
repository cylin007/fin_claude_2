"""
策略 C: 中長期趨勢回檔 (Midterm Trend-Following Strategy)

核心邏輯:
  - 趨勢確認: MA20 > MA60，MA60 斜率向上，週線多頭
  - 進場時機: 回檔到 MA20 支撐區 (-5% ~ +3%)
  - RSI 過濾: < 70 (不超買)
  - 量縮確認: 回檔量 < 5日均量 70%
  - 出場: 連續3天跌破 MA60 / 追蹤停利 (40%觸發, 回撤15%) / 硬停損 -12%
  - 殭屍清除: 90天獲利 < 5%
  - 倉位: 4 檔集中持股，每檔 15% NAV

vs 策略 A (短線動能):
  - A: 追突破, 12檔分散, 15天zombie, 快進快出
  - C: 買回檔, 4檔集中, 90天zombie, 中長期持有
"""

import pandas as pd
import numpy as np
from strategy import calculate_net_pnl, calculate_fee, calculate_tax, DEFAULT_CONFIG


# ==========================================
# 中長期策略參數
# ==========================================
MIDTERM_CONFIG = {
    # === 進場條件 ===
    'mt_ma20_above_ma60':    True,    # MA20 > MA60 (趨勢向上)
    'mt_pullback_lo':       -5.0,     # 價格離 MA20 最低距離%
    'mt_pullback_hi':        3.0,     # 價格離 MA20 最高距離%
    'mt_rsi_max':            70,      # RSI 上限 (不超買)
    'mt_vol_shrink_ratio':   0.7,     # 量縮門檻 (vol < vol_ma5 * ratio)
    'mt_min_ma60_slope':     0.0,     # MA60 最低斜率% (20日) — 0=不檢查
    'mt_ma60_slope_days':    20,      # MA60 斜率計算天數
    'mt_require_weekly':     True,    # 要求週線 MA20 > MA60

    # === 出場條件 ===
    'mt_hard_stop_pct':     -12.0,    # 硬停損 (淨利%)
    'mt_exit_below_ma60':    True,    # 跌破 MA60 出場
    'mt_ma60_break_days':    3,       # 連續 N 天收盤 < MA60 才出場
    'mt_trail_trigger':      40.0,    # 開始追蹤停利門檻 (帳面%)
    'mt_trail_drop':         15.0,    # 從高點回撤多少%出場
    'mt_zombie_days':        90,      # 持有超過 N 天
    'mt_zombie_min_profit':   5.0,    # zombie 時至少要有此淨利%才留

    # === 倉位 ===
    'mt_max_positions':       4,      # 最大持股數
    'mt_budget_pct':          15.0,   # 每筆 = NAV × 此%

    # === 大盤過濾 ===
    'mt_block_bear':          True,   # bear 市不買
    'mt_block_panic':         True,   # panic 不買
}


def _compute_rsi(close_series, period=14):
    """計算 RSI-14"""
    if len(close_series) < period + 1:
        return 50.0  # 資料不足 → 中性
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if not pd.isna(val) else 50.0


def _compute_weekly_trend(history_df):
    """計算週線 MA20 > MA60 (用日線模擬週線: 每5日取收盤)"""
    if history_df is None or len(history_df) < 300:
        return True  # 資料不足 → 不過濾
    try:
        weekly = history_df['Close'].resample('W-FRI').last().dropna()
        if len(weekly) < 60:
            return True
        w_ma20 = weekly.tail(20).mean()
        w_ma60 = weekly.tail(60).mean()
        return w_ma20 > w_ma60
    except Exception:
        return True


def _count_below_ma60_days(history_df, n_days=3):
    """計算最近 N 天收盤低於 MA60 的天數"""
    if history_df is None or len(history_df) < 61 + n_days:
        return 0
    try:
        close_s = history_df['Close']
        ma60_s = close_s.rolling(60).mean()
        recent_close = close_s.iloc[-n_days:]
        recent_ma60 = ma60_s.iloc[-n_days:]
        below_count = sum(1 for c, m in zip(recent_close, recent_ma60)
                         if not pd.isna(m) and c < m)
        return below_count
    except Exception:
        return 0


def _compute_ma60_slope(history_df, ma60_now, slope_days=20):
    """計算 MA60 的 N 日斜率%"""
    if history_df is None or len(history_df) < 60 + slope_days:
        return 0.0
    try:
        ma60_series = history_df['Close'].rolling(60).mean()
        ma60_past = ma60_series.iloc[-1 - slope_days]
        if pd.isna(ma60_past) or ma60_past <= 0:
            return 0.0
        return (ma60_now - ma60_past) / ma60_past * 100
    except (IndexError, KeyError):
        return 0.0


def check_midterm_signal(stock_id, info, held_cost=0, held_shares=0,
                          market_status=None, history_df=None, config=None,
                          reduce_stage=0, last_reduce_day_idx=-99,
                          current_day_idx=0, peak_price_since_entry=None):
    """
    策略 C 主函數: 中長期趨勢回檔

    與 check_strategy_signal 相同介面，可直接替換到 run_group_backtest(signal_func=...)
    """
    cfg = {**DEFAULT_CONFIG, **MIDTERM_CONFIG}
    if config:
        cfg.update(config)

    default_result = {
        'action': 'hold', 'reason': '',
        'pnl': None, 'consecutive': 0, 'is_fish_tail': False,
        'reduce_ratio': 0, 'reduce_stage': reduce_stage,
    }

    # --- 數據解包 ---
    required_keys = ['close', 'ma20', 'ma60', 'volume', 'vol_ma5', 'prev_close']
    for key in required_keys:
        if key not in info or info[key] is None:
            return {**default_result, 'reason': f'⚠️ 缺少: {key}'}

    try:
        price      = float(info['close'])
        ma20       = float(info['ma20'])
        ma60       = float(info['ma60'])
        vol        = float(info['volume'])
        vol_ma5    = float(info['vol_ma5'])
        prev_close = float(info['prev_close'])
    except (ValueError, TypeError):
        return {**default_result, 'reason': '⚠️ 格式錯誤'}

    if ma20 <= 0 or ma60 <= 0 or price <= 0:
        return default_result

    bias_ma20 = (price - ma20) / ma20 * 100
    is_unsafe = market_status.get('is_unsafe', False) if market_status else False
    is_panic  = market_status.get('is_panic', False) if market_status else False
    twii_trend = market_status.get('twii', {}).get('trend', 'neutral') if market_status else 'neutral'

    # =========================================
    # 🛑 賣出邏輯 (有庫存時)
    # =========================================
    if held_shares > 0 and held_cost > 0:
        pnl_detail = calculate_net_pnl(stock_id, held_cost, price, held_shares)
        net_pct = pnl_detail['net_pnl_pct']
        profit_pct = (price - held_cost) / held_cost * 100

        def make_sell(reason):
            return {
                'action': 'sell', 'reason': reason,
                'pnl': pnl_detail, 'consecutive': 0, 'is_fish_tail': False,
                'reduce_ratio': 0, 'reduce_stage': reduce_stage,
            }

        # S1: 硬停損 — 安全網
        if net_pct <= cfg['mt_hard_stop_pct']:
            return make_sell(
                f'🔴 MT硬停損 (淨利{net_pct:+.1f}% ≤ {cfg["mt_hard_stop_pct"]}%)')

        # S2: 連續 N 天跌破 MA60 → 趨勢破壞
        if cfg['mt_exit_below_ma60']:
            break_days = cfg['mt_ma60_break_days']
            below_count = _count_below_ma60_days(history_df, break_days)
            if below_count >= break_days:
                return make_sell(
                    f'📉 MT趨勢破壞 (連續{below_count}天 < MA60 {ma60:.0f}, '
                    f'淨利{net_pct:+.1f}%)')

        # S3: 追蹤停利 — 從最高點回撤
        if peak_price_since_entry and peak_price_since_entry > 0:
            peak_profit = (peak_price_since_entry - held_cost) / held_cost * 100
            if peak_profit >= cfg['mt_trail_trigger']:
                drop_from_peak = (peak_price_since_entry - price) / peak_price_since_entry * 100
                if drop_from_peak >= cfg['mt_trail_drop']:
                    return make_sell(
                        f'📈 MT追蹤停利 (高點+{peak_profit:.0f}%, '
                        f'回撤{drop_from_peak:.1f}% ≥ {cfg["mt_trail_drop"]}%, '
                        f'淨利{net_pct:+.1f}%)')

        # S4: Zombie — 持有太久獲利不足 (由 group_backtest 統一控制)
        # 這裡不處理，讓 group_backtest 用 config 中的 zombie 參數處理

        return {**default_result, 'pnl': pnl_detail}

    # =========================================
    # 🟢 買進邏輯 (無庫存時)
    # =========================================

    # F1: 大盤過濾 — bear 市不開新倉
    if cfg['mt_block_bear'] and twii_trend in ('bear', 'weak'):
        return default_result

    # F2: 恐慌不買
    if cfg['mt_block_panic'] and is_panic:
        return default_result

    # B1: 日線趨勢 — MA20 > MA60
    if cfg['mt_ma20_above_ma60'] and ma20 <= ma60:
        return default_result

    # B2: MA60 斜率 > 0 (長期趨勢上升)
    if cfg['mt_min_ma60_slope'] > 0:
        slope = _compute_ma60_slope(history_df, ma60, cfg['mt_ma60_slope_days'])
        if slope < cfg['mt_min_ma60_slope']:
            return default_result

    # B3: 週線趨勢 — 週 MA20 > 週 MA60
    if cfg['mt_require_weekly']:
        weekly_ok = _compute_weekly_trend(history_df)
        if not weekly_ok:
            return default_result

    # B4: 回檔位置 — 價格在 MA20 附近 (-5% ~ +3%)
    if not (cfg['mt_pullback_lo'] <= bias_ma20 <= cfg['mt_pullback_hi']):
        return default_result

    # B5: RSI < 70 (不超買)
    if history_df is not None and len(history_df) >= 15:
        rsi = _compute_rsi(history_df['Close'])
        if rsi > cfg['mt_rsi_max']:
            return default_result
        rsi_str = f'RSI={rsi:.0f}'
    else:
        rsi_str = 'RSI=N/A'

    # B6: 量縮確認 (回檔時量應該縮)
    vol_ratio = vol / vol_ma5 if vol_ma5 > 0 else 1.0
    if vol_ratio > cfg['mt_vol_shrink_ratio']:
        return default_result

    # ✅ 所有條件通過 → 買進
    return {
        'action': 'buy',
        'reason': (f'🟢 MT回檔買 (MA20偏離{bias_ma20:+.1f}%, {rsi_str}, '
                   f'量比={vol_ratio:.2f}x, 大盤={twii_trend})'),
        'pnl': None, 'consecutive': 1, 'is_fish_tail': False,
        'reduce_ratio': 0, 'reduce_stage': reduce_stage,
    }


# ==========================================
# 策略 D: 趨勢存續型持股 (Trend-Persistence Strategy)
# ==========================================
# 核心差異 vs C:
#   - 出場: 非對稱 (虧損快砍/小獲利中等/大獲利寬鬆) + 趨勢健康度判斷
#   - 進場: 支援 momentum (策略A) / pullback (策略C) / hybrid 三種模式
#   - 部位: 5檔集中, 每檔12% NAV, 每週限買2檔
# ==========================================

TREND_PERSISTENCE_CONFIG = {
    # === 進場模式 ===
    'tp_entry_mode':         'pullback',  # 'momentum' | 'pullback' | 'hybrid'

    # --- 非對稱出場: 虧損層 (net < 0%) — 嚴格快砍 ---
    'tp_loss_hard_stop':     -10.0,       # 硬停損%
    'tp_loss_ma60_break_days': 3,         # MA60 破線容忍天數
    'tp_loss_zombie_days':   20,          # 虧損持有超過N天 → 砍

    # --- 非對稱出場: 小獲利層 (0% ~ tp_big_profit_threshold) — 中等耐心 ---
    'tp_small_ma60_break_days': 5,        # MA60 破線容忍天數 (比虧損寬)
    'tp_small_trend_health': True,        # 趨勢健康度檢查 (MA20<MA60 + MA60斜率負)
    'tp_small_zombie_days':  60,          # 效率zombie: 持有>N天且獲利低 → 砍
    'tp_small_zombie_min_pct': 8.0,       # 效率zombie: 淨利至少要>此%才留

    # --- 非對稱出場: 大獲利層 (>= tp_big_profit_threshold) — 寬鬆讓利潤奔跑 ---
    'tp_big_profit_threshold': 15.0,      # 進入大獲利模式的門檻%
    'tp_big_use_trend_exit':  True,       # 用趨勢判斷出場 (不用回撤%)
    'tp_big_ma60_slope_days': 5,          # MA60斜率連續N天為負才觸發
    'tp_big_require_weekly_bear': True,   # 需要週線翻空才賣
    'tp_big_max_drawdown':    25.0,       # 保底: 從最高點回撤>此%強制賣
    'tp_big_ma60_break_days': 7,          # 大獲利時MA60破線容忍更久

    # --- 進場: pullback 模式 ---
    'tp_pullback_lo':        -8.0,        # 價格離MA20最低距離%
    'tp_pullback_hi':         5.0,        # 價格離MA20最高距離%
    'tp_rsi_max':             65,         # RSI上限
    'tp_vol_ratio_max':       1.2,        # 量比上限 (放寬, 不必量縮)
    'tp_require_weekly_bull': True,       # 要求週線多頭
    'tp_ma20_above_ma60':     True,       # 要求MA20 > MA60

    # --- 進場: momentum 模式 ---
    'tp_require_breakout':    False,      # 是否要求突破N日高 (False=只看趨勢+量增)

    # --- 部位管理 ---
    'tp_max_positions':       5,          # 最大持股數
    'tp_budget_pct':          12.0,       # 每筆 = NAV × 此%
    'tp_weekly_max_buy':      2,          # 每週最多新買N檔 (0=不限)
    'tp_block_bear':          True,       # bear/weak 市不買
    'tp_block_panic':         True,       # panic 不買
    'tp_cash_reserve_pct':    15.0,       # 現金保留%
}


def _check_trend_health(history_df):
    """
    趨勢健康度判斷: MA20 > MA60 且 MA60 斜率 > 0
    回傳 (is_healthy: bool, details: str)
    """
    if history_df is None or len(history_df) < 80:
        return True, 'data_insufficient'
    try:
        close_s = history_df['Close']
        ma20 = close_s.rolling(20).mean().iloc[-1]
        ma60 = close_s.rolling(60).mean().iloc[-1]
        if pd.isna(ma20) or pd.isna(ma60) or ma60 <= 0:
            return True, 'ma_na'
        ma20_above = ma20 > ma60
        slope = _compute_ma60_slope(history_df, ma60, slope_days=20)
        slope_positive = slope > 0
        healthy = ma20_above or slope_positive  # 任一成立就算健康
        detail = f'MA20{">" if ma20_above else "<"}MA60,slope={slope:+.2f}%'
        return healthy, detail
    except Exception:
        return True, 'error'


def _count_negative_slope_days(history_df, slope_days=20, check_days=5):
    """
    計算最近 check_days 天中, MA60 斜率連續為負的天數
    用於大獲利層判斷趨勢是否持續惡化
    """
    if history_df is None or len(history_df) < 60 + slope_days + check_days:
        return 0
    try:
        close_s = history_df['Close']
        ma60_s = close_s.rolling(60).mean()
        count = 0
        for i in range(check_days):
            idx = -1 - i
            past_idx = idx - slope_days
            ma60_now = ma60_s.iloc[idx]
            ma60_past = ma60_s.iloc[past_idx]
            if pd.isna(ma60_now) or pd.isna(ma60_past) or ma60_past <= 0:
                break
            slope = (ma60_now - ma60_past) / ma60_past * 100
            if slope < 0:
                count += 1
            else:
                break  # 一旦不連續就停止
        return count
    except Exception:
        return 0


def check_trend_persistence_signal(
        stock_id, info, held_cost=0, held_shares=0,
        market_status=None, history_df=None, config=None,
        reduce_stage=0, last_reduce_day_idx=-99,
        current_day_idx=0, peak_price_since_entry=None):
    """
    策略 D: 趨勢存續型持股

    與 check_strategy_signal / check_midterm_signal 相同介面。
    核心差異: 非對稱出場 (虧損快砍 / 小獲利中等 / 大獲利寬鬆)。
    """
    cfg = {**DEFAULT_CONFIG, **TREND_PERSISTENCE_CONFIG}
    if config:
        cfg.update(config)

    default_result = {
        'action': 'hold', 'reason': '',
        'pnl': None, 'consecutive': 0, 'is_fish_tail': False,
        'reduce_ratio': 0, 'reduce_stage': reduce_stage,
    }

    # --- 數據解包 ---
    required_keys = ['close', 'ma20', 'ma60', 'volume', 'vol_ma5', 'prev_close']
    for key in required_keys:
        if key not in info or info[key] is None:
            return {**default_result, 'reason': f'⚠️ 缺少: {key}'}
    try:
        price      = float(info['close'])
        ma20       = float(info['ma20'])
        ma60       = float(info['ma60'])
        vol        = float(info['volume'])
        vol_ma5    = float(info['vol_ma5'])
        prev_close = float(info['prev_close'])
    except (ValueError, TypeError):
        return {**default_result, 'reason': '⚠️ 格式錯誤'}

    if ma20 <= 0 or ma60 <= 0 or price <= 0:
        return default_result

    bias_ma20 = (price - ma20) / ma20 * 100
    is_unsafe = market_status.get('is_unsafe', False) if market_status else False
    is_panic  = market_status.get('is_panic', False) if market_status else False
    twii_trend = market_status.get('twii', {}).get('trend', 'neutral') if market_status else 'neutral'

    # =========================================
    # 🛑 賣出邏輯: 非對稱三層出場
    # =========================================
    if held_shares > 0 and held_cost > 0:
        pnl_detail = calculate_net_pnl(stock_id, held_cost, price, held_shares)
        net_pct = pnl_detail['net_pnl_pct']

        def make_sell(reason):
            return {
                'action': 'sell', 'reason': reason,
                'pnl': pnl_detail, 'consecutive': 0, 'is_fish_tail': False,
                'reduce_ratio': 0, 'reduce_stage': reduce_stage,
            }

        # === 判斷獲利層級 ===
        big_threshold = cfg['tp_big_profit_threshold']

        if net_pct < 0:
            # ────────────────────────────────────
            #  虧損層: 嚴格快砍
            # ────────────────────────────────────

            # S1: 硬停損
            if net_pct <= cfg['tp_loss_hard_stop']:
                return make_sell(
                    f'🔴 TP硬停損 (淨利{net_pct:+.1f}% ≤ {cfg["tp_loss_hard_stop"]}%)')

            # S2: MA60 破線 (3天)
            break_days = cfg['tp_loss_ma60_break_days']
            below = _count_below_ma60_days(history_df, break_days)
            if below >= break_days:
                return make_sell(
                    f'📉 TP虧損MA60破 (連{below}天<MA60, 淨利{net_pct:+.1f}%)')

            # S3: 虧損 zombie — 持有太久仍虧損
            # 注意: 實際 hold_days 由 group_backtest 的 zombie 機制處理
            # 這裡透過 config 設定 zombie_hold_days 和 zombie_net_range 讓引擎處理

        elif net_pct < big_threshold:
            # ────────────────────────────────────
            #  小獲利層: 中等耐心
            # ────────────────────────────────────

            # S4: MA60 破線 (5天, 比虧損層寬)
            break_days = cfg['tp_small_ma60_break_days']
            below = _count_below_ma60_days(history_df, break_days)
            if below >= break_days:
                return make_sell(
                    f'📉 TP小獲利MA60破 (連{below}天<MA60, 淨利{net_pct:+.1f}%)')

            # S5: 趨勢健康度失敗 (MA20<MA60 且 MA60斜率負)
            if cfg['tp_small_trend_health']:
                healthy, detail = _check_trend_health(history_df)
                if not healthy:
                    return make_sell(
                        f'📊 TP趨勢不健康 ({detail}, 淨利{net_pct:+.1f}%)')

            # S6: 效率 zombie — 由 group_backtest 引擎用
            #   tp_small_zombie_days / tp_small_zombie_min_pct 映射到
            #   zombie_hold_days / zombie_net_range

        else:
            # ────────────────────────────────────
            #  大獲利層: 寬鬆讓利潤奔跑
            # ────────────────────────────────────

            # S7: 保底回撤 — 從最高點回撤超過限制
            if peak_price_since_entry and peak_price_since_entry > 0:
                drop_from_peak = (peak_price_since_entry - price) / peak_price_since_entry * 100
                if drop_from_peak >= cfg['tp_big_max_drawdown']:
                    return make_sell(
                        f'🛡️ TP保底回撤 (高點回撤{drop_from_peak:.1f}% ≥ '
                        f'{cfg["tp_big_max_drawdown"]}%, 淨利{net_pct:+.1f}%)')

            # S8: MA60 破線 (7天, 非常寬鬆)
            break_days = cfg['tp_big_ma60_break_days']
            below = _count_below_ma60_days(history_df, break_days)
            if below >= break_days:
                # 若啟用趨勢出場, 額外檢查斜率+週線
                if cfg['tp_big_use_trend_exit']:
                    neg_slope_days = _count_negative_slope_days(
                        history_df, slope_days=20,
                        check_days=cfg['tp_big_ma60_slope_days'])
                    weekly_bear = not _compute_weekly_trend(history_df)

                    if neg_slope_days >= cfg['tp_big_ma60_slope_days']:
                        if not cfg['tp_big_require_weekly_bear'] or weekly_bear:
                            return make_sell(
                                f'📉 TP大獲利趨勢崩壞 (MA60破{below}天+'
                                f'斜率負{neg_slope_days}天'
                                f'{"+週線空" if weekly_bear else ""}, '
                                f'淨利{net_pct:+.1f}%)')
                else:
                    # 不用趨勢出場, 純看MA60破線天數
                    return make_sell(
                        f'📉 TP大獲利MA60破 (連{below}天<MA60, 淨利{net_pct:+.1f}%)')

        return {**default_result, 'pnl': pnl_detail}

    # =========================================
    # 🟢 買進邏輯: 依 entry_mode 分派
    # =========================================
    entry_mode = cfg.get('tp_entry_mode', 'pullback')

    # F1: 大盤過濾
    if cfg['tp_block_bear'] and twii_trend in ('bear', 'weak'):
        return default_result
    if cfg['tp_block_panic'] and is_panic:
        return default_result

    if entry_mode == 'momentum':
        return _tp_entry_momentum(
            stock_id, info, market_status, history_df, cfg,
            reduce_stage, twii_trend, ma20, ma60, price, prev_close, vol, vol_ma5)

    elif entry_mode == 'hybrid':
        # 先嘗試 momentum, 不成功再嘗試 pullback
        result = _tp_entry_momentum(
            stock_id, info, market_status, history_df, cfg,
            reduce_stage, twii_trend, ma20, ma60, price, prev_close, vol, vol_ma5)
        if result['action'] == 'buy':
            return result
        return _tp_entry_pullback(
            stock_id, info, history_df, cfg,
            reduce_stage, twii_trend, bias_ma20, vol, vol_ma5)

    else:  # 'pullback' (default)
        return _tp_entry_pullback(
            stock_id, info, history_df, cfg,
            reduce_stage, twii_trend, bias_ma20, vol, vol_ma5)


def _tp_entry_pullback(stock_id, info, history_df, cfg,
                       reduce_stage, twii_trend, bias_ma20, vol, vol_ma5):
    """策略D pullback進場 (沿用策略C邏輯, 參數可調)"""
    price = float(info['close'])
    ma20 = float(info['ma20'])
    ma60 = float(info['ma60'])
    default_result = {
        'action': 'hold', 'reason': '',
        'pnl': None, 'consecutive': 0, 'is_fish_tail': False,
        'reduce_ratio': 0, 'reduce_stage': reduce_stage,
    }

    # B1: MA20 > MA60
    if cfg['tp_ma20_above_ma60'] and ma20 <= ma60:
        return default_result

    # B2: 週線多頭
    if cfg['tp_require_weekly_bull']:
        if not _compute_weekly_trend(history_df):
            return default_result

    # B3: 回檔區間
    if not (cfg['tp_pullback_lo'] <= bias_ma20 <= cfg['tp_pullback_hi']):
        return default_result

    # B4: RSI
    rsi_str = 'N/A'
    if history_df is not None and len(history_df) >= 15:
        rsi = _compute_rsi(history_df['Close'])
        if rsi > cfg['tp_rsi_max']:
            return default_result
        rsi_str = f'{rsi:.0f}'

    # B5: 量比
    vol_ratio = vol / vol_ma5 if vol_ma5 > 0 else 1.0
    if vol_ratio > cfg['tp_vol_ratio_max']:
        return default_result

    return {
        'action': 'buy',
        'reason': (f'🟢 TP回檔買 (MA20偏離{bias_ma20:+.1f}%, RSI={rsi_str}, '
                   f'量比={vol_ratio:.2f}x, 大盤={twii_trend})'),
        'pnl': None, 'consecutive': 1, 'is_fish_tail': False,
        'reduce_ratio': 0, 'reduce_stage': reduce_stage,
    }


def _tp_entry_momentum(stock_id, info, market_status, history_df, cfg,
                       reduce_stage, twii_trend, ma20, ma60, price,
                       prev_close, vol, vol_ma5):
    """策略D momentum進場 (沿用策略A的 B1-B4+B6 突破邏輯)"""
    default_result = {
        'action': 'hold', 'reason': '',
        'pnl': None, 'consecutive': 0, 'is_fish_tail': False,
        'reduce_ratio': 0, 'reduce_stage': reduce_stage,
    }

    # B1: 趨勢 — MA20 > MA60
    if ma20 <= ma60:
        return default_result

    # B2: 量增 — 成交量 > 5日均量
    if vol_ma5 > 0 and vol <= vol_ma5:
        return default_result
    vol_ratio = vol / vol_ma5 if vol_ma5 > 0 else 1.0

    # B3: 價漲 — 收盤 > 前日收盤
    if price <= prev_close:
        return default_result

    # B4: 乖離率限制 (用 bull 級放寬, 因為我們只在趨勢好時進場)
    bias = (price - ma20) / ma20 * 100
    bias_limit = cfg.get('bias_limit_bull', 30)
    if bias > bias_limit:
        return default_result

    # B6: 突破 — 收盤創 N 日新高 (用前一天之前的高點, 排除今天自己)
    _tp_require_breakout = cfg.get('tp_require_breakout', False)
    if _tp_require_breakout and history_df is not None and len(history_df) >= 12:
        lookback = cfg.get('breakout_lookback', 10)
        if 'High' in history_df.columns:
            # 取前 lookback 天的高點 (不含今天)
            prev_highs = history_df['High'].iloc[-(lookback+1):-1]
            if len(prev_highs) >= lookback:
                if price <= prev_highs.max():
                    return default_result

    # RSI 過濾 (偏寬)
    rsi_str = 'N/A'
    if history_df is not None and len(history_df) >= 15:
        rsi = _compute_rsi(history_df['Close'])
        if rsi > cfg.get('tp_rsi_max', 65):
            return default_result
        rsi_str = f'{rsi:.0f}'

    return {
        'action': 'buy',
        'reason': (f'🟢 TP動能買 (MA20上+量增{vol_ratio:.1f}x, '
                   f'偏離{bias:+.1f}%, RSI={rsi_str}, 大盤={twii_trend})'),
        'pnl': None, 'consecutive': 1, 'is_fish_tail': False,
        'reduce_ratio': 0, 'reduce_stage': reduce_stage,
    }
