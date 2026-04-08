"""
策略 B: 中期回檔買進 (Pullback Strategy)

邏輯:
  - 趨勢確認後，等回檔到 MA20 支撐位再進場
  - 持有 1-3 個月，讓趨勢跑完
  - 只在趨勢破壞 (跌破 MA60) 或獲利回吐時出場

vs 策略 A (現有動能追漲):
  - 策略 A: 追突破 → 勝率低(25%) × 大贏家拉回
  - 策略 B: 買回檔 → 勝率高(50%+) × 穩定獲利
"""

from strategy import calculate_net_pnl, calculate_fee, calculate_tax, DEFAULT_CONFIG


PULLBACK_CONFIG = {
    # === 進場條件 ===
    'pb_ma20_above_ma60':    True,   # MA20 > MA60 (趨勢向上)
    'pb_pullback_lo':       -5.0,    # 價格離 MA20 的最低距離% (太低=趨勢可能破)
    'pb_pullback_hi':        3.0,    # 價格離 MA20 的最高距離% (太高=已經漲一段)
    'pb_rsi_lo':             35,     # RSI 下限 (太低=弱勢)
    'pb_rsi_hi':             65,     # RSI 上限 (太高=超買)
    'pb_vol_shrink':         True,   # 回檔時量縮 (vol < vol_ma5)
    'pb_min_ma60_slope':     0.0,    # MA60 斜率最低要求 (0=不檢查)

    # === 出場條件 ===
    'pb_stop_loss_pct':     -8.0,    # 硬停損 (淨利%)
    'pb_exit_below_ma60':    True,   # 跌破 MA60 出場
    'pb_trail_trigger':      20.0,   # 開始追蹤停利的門檻 (淨利%)
    'pb_trail_drop':         10.0,   # 從高點回撤多少%出場
    'pb_zombie_days':        45,     # 持有超過 N 天
    'pb_zombie_min_profit':   5.0,   # zombie 時至少要有此淨利%才留

    # === 倉位 ===
    'pb_max_positions':       5,     # 最大持股數
    'pb_budget_pct':          15.0,  # 每筆 = NAV × 此%

    # === 大盤過濾 ===
    'pb_block_unsafe':        False, # unsafe 時不買 (False=不管大盤)
}


def check_pullback_signal(stock_id, info, held_cost=0, held_shares=0,
                           market_status=None, history_df=None, config=None,
                           reduce_stage=0, last_reduce_day_idx=-99,
                           current_day_idx=0, peak_price_since_entry=None):
    """
    策略 B 主函數: 回檔買進 + 趨勢跟隨

    與 check_strategy_signal 相同介面，可直接替換
    """
    cfg = {**DEFAULT_CONFIG, **PULLBACK_CONFIG}
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
        price     = float(info['close'])
        ma20      = float(info['ma20'])
        ma60      = float(info['ma60'])
        vol       = float(info['volume'])
        vol_ma5   = float(info['vol_ma5'])
        prev_close = float(info['prev_close'])
        rsi       = float(info.get('rsi', 50))
    except (ValueError, TypeError):
        return {**default_result, 'reason': '⚠️ 格式錯誤'}

    if ma20 <= 0 or ma60 <= 0 or price <= 0:
        return default_result

    bias_ma20 = (price - ma20) / ma20 * 100  # 離 MA20 距離%
    is_unsafe = market_status.get('is_unsafe', False) if market_status else False

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

        # S1: 硬停損
        if net_pct <= cfg['pb_stop_loss_pct']:
            return make_sell(f'🔴 PB停損 (淨利{net_pct:+.1f}% ≤ {cfg["pb_stop_loss_pct"]}%)')

        # S2: 跌破 MA60 → 趨勢破壞
        if cfg['pb_exit_below_ma60'] and price < ma60:
            return make_sell(f'📉 PB趨勢破壞 (價格{price:.0f} < MA60 {ma60:.0f}, 淨利{net_pct:+.1f}%)')

        # S3: 追蹤停利 (從高點回撤)
        if peak_price_since_entry and peak_price_since_entry > 0:
            peak_profit = (peak_price_since_entry - held_cost) / held_cost * 100
            if peak_profit >= cfg['pb_trail_trigger']:
                drop_from_peak = (peak_price_since_entry - price) / peak_price_since_entry * 100
                if drop_from_peak >= cfg['pb_trail_drop']:
                    return make_sell(
                        f'📈 PB追蹤停利 (高點+{peak_profit:.0f}%, '
                        f'回撤{drop_from_peak:.1f}% ≥ {cfg["pb_trail_drop"]}%, '
                        f'淨利{net_pct:+.1f}%)')

        # S4: Zombie — 持有太久獲利不足
        days_held = current_day_idx - (reduce_stage if reduce_stage > 0 else 0)
        # 用 last_reduce_day_idx 當作 entry day 的 proxy (backtest 會傳)
        # 但更準確的 days_held 由 group_backtest 的 pos['last_buy_date_idx'] 控制
        # 這裡只做基本檢查，實際 zombie 由 group_backtest 的 S4 處理

        return {**default_result, 'pnl': pnl_detail}

    # =========================================
    # 🟢 買進邏輯 (無庫存時)
    # =========================================

    # F1: 大盤過濾
    if cfg['pb_block_unsafe'] and is_unsafe:
        return default_result

    # B1: 趨勢確認 — MA20 > MA60
    if cfg['pb_ma20_above_ma60'] and ma20 <= ma60:
        return default_result

    # B2: 回檔位置 — 價格在 MA20 附近
    if not (cfg['pb_pullback_lo'] <= bias_ma20 <= cfg['pb_pullback_hi']):
        return default_result

    # B3: RSI 區間
    if not (cfg['pb_rsi_lo'] <= rsi <= cfg['pb_rsi_hi']):
        return default_result

    # B4: 量縮確認 (回檔時量應該縮)
    if cfg['pb_vol_shrink'] and vol_ma5 > 0 and vol > vol_ma5 * 1.2:
        return default_result  # 量太大 = 可能是恐慌賣壓，不是健康回檔

    # B5: MA60 斜率 (趨勢強度)
    if cfg['pb_min_ma60_slope'] > 0 and history_df is not None and len(history_df) >= 10:
        try:
            ma60_5d_ago = float(history_df['ma60'].iloc[-6]) if 'ma60' in history_df.columns else ma60
            ma60_slope = (ma60 - ma60_5d_ago) / ma60_5d_ago * 100 if ma60_5d_ago > 0 else 0
            if ma60_slope < cfg['pb_min_ma60_slope']:
                return default_result
        except (IndexError, KeyError):
            pass

    # ✅ 所有條件通過 → 買進
    return {
        'action': 'buy',
        'reason': f'🟢 PB回檔買 (MA20偏離{bias_ma20:+.1f}%, RSI={rsi:.0f}, 量比={vol/vol_ma5:.1f}x)',
        'pnl': None, 'consecutive': 1, 'is_fish_tail': False,
        'reduce_ratio': 0, 'reduce_stage': reduce_stage,
    }
