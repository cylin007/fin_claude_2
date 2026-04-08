import pandas as pd
import numpy as np

# ==========================================
# 🛠️ 基礎計算工具 (費率與稅)
# ==========================================
def calculate_fee(price, shares, discount=0.6):
    """
    計算手續費 (依元大投資先生 APP 規則)
    - 整股交易 (shares >= 1000): 最低 20 元
    - 零股交易 (shares < 1000):  最低 1 元
    手續費率: 成交金額 × 0.1425% × 折扣
    """
    if shares <= 0:
        return 0
    fee = int(price * shares * 0.001425 * discount)
    min_fee = 1 if shares < 1000 else 20
    return max(min_fee, fee)


def calculate_tax(stock_id, price, shares):
    """計算交易稅 (ETF 千分之1, 個股 千分之3)"""
    if shares <= 0:
        return 0
    # 先去掉 .TW/.TWO 後綴再判斷 ETF (修正: 帶後綴時 len 不正確)
    normalized_id = str(stock_id).replace('.TWO', '').replace('.TW', '').strip()
    is_etf = normalized_id.startswith("00") and len(normalized_id) in [4, 5]
    rate = 0.001 if is_etf else 0.003
    return int(price * shares * rate)


def calculate_net_pnl(stock_id, buy_price, sell_price, shares, discount=0.6):
    """計算扣除手續費與交易稅後的「實際淨損益」"""
    gross_pnl = (sell_price - buy_price) * shares
    buy_fee = calculate_fee(buy_price, shares, discount)
    sell_fee = calculate_fee(sell_price, shares, discount)
    sell_tax = calculate_tax(stock_id, sell_price, shares)
    total_cost = buy_fee + sell_fee + sell_tax
    net_pnl = gross_pnl - total_cost
    cost_basis = buy_price * shares + buy_fee
    net_pnl_pct = (net_pnl / cost_basis) * 100 if cost_basis > 0 else 0

    return {
        'gross_pnl': gross_pnl,
        'buy_fee': buy_fee,
        'sell_fee': sell_fee,
        'sell_tax': sell_tax,
        'total_cost': total_cost,
        'net_pnl': net_pnl,
        'net_pnl_pct': net_pnl_pct,
    }


# ==========================================
# 🐟 魚尾偵測 (Fish Tail Detection)
# ==========================================
# TODO: 目前魚尾偵測只看個股 B1-B4 條件，不帶大盤狀態 (F1/F2)。
#       這表示即使某天大盤偏空實際不會觸發買進，魚尾仍會將該天計入連續天數。
#       這是設計上的取捨：魚尾旨在偵測「個股自身的連續強勢」，與大盤狀態無關。
#       若未來需要更精確的回測，可考慮傳入各日大盤狀態。
def _count_consecutive_buy_days(history_df, lookback=5):
    """
    回頭檢查過去 N 個交易日，有幾天「連續」符合買進條件。
    回傳: int (連續天數, 不含今天)
    """
    if history_df is None or len(history_df) < 65:
        return 0
    
    count = 0
    total_len = len(history_df)
    
    for offset in range(1, lookback + 1):
        end_idx = total_len - offset
        if end_idx < 61:
            break
        
        day_slice = history_df.iloc[:end_idx]
        
        try:
            close = float(day_slice['Close'].iloc[-1])
            prev_close = float(day_slice['Close'].iloc[-2])
            ma20 = float(day_slice['Close'].tail(20).mean())
            ma60 = float(day_slice['Close'].tail(60).mean())
            vol = float(day_slice['Volume'].iloc[-1])
            vol_ma5 = float(day_slice['Volume'].tail(5).mean())
        except (ValueError, TypeError, KeyError, IndexError):
            break
        
        pct_change = (close - prev_close) / prev_close * 100 if prev_close > 0 else 0
        is_strong = pct_change > 1.5
        
        cond_trend = (close > ma20) and (close > ma60) and (ma20 > ma60)
        cond_vol = (vol > vol_ma5) or (vol > vol_ma5 * 0.7 and is_strong)
        cond_price = (close > prev_close)
        
        max_bias = 15 if close > 100 else 18
        bias = (close - ma20) / ma20 * 100
        cond_bias = bias <= max_bias
        
        if cond_trend and cond_vol and cond_price and cond_bias:
            count += 1
        else:
            break
    
    return count


# ==========================================
# ⚙️ 預設策略設定 (可由 ablation 覆蓋)
# ==========================================
DEFAULT_CONFIG = {
    # 買進模組開關
    'enable_bias_limit':    True,   # B4: 乖離率上限
    'bias_limit_bull':      30,     # B4 V4.6: bull 時高價股上限 (V4.3:20→30, 多頭放手追動能)
    'bias_limit_neutral':   25,     # B4 V4.6: neutral 時高價股上限 (V4.3:20→25, 中性適度放寬)
    'bias_limit_bear':      20,     # B4 V4.6: bear/weak 時維持20 (空頭不放寬, ablation實證收緊無效但不放寬)
    'enable_fish_tail':     True,   # B5: 魚尾偵測
    'fish_tail_lookback':   5,      # B5: 回看天數 (連續幾天算魚尾, V3.6 參數化)
    'enable_breakout':      True,   # B6: 突破前N日高點 (V3.1 新增)
    'breakout_lookback':    10,     # B6: 回看天數 (V3.2: 5→10, 過濾2週盤整)
    'enable_shooting_star': False,  # B7: 避雷針濾網 (V4.3: 關閉, ablation實證全期Sharpe+0.10/MDD更低)
    'shooting_star_ratio':  3.0,    # B7: 上影線 / 實體 > 此倍數就擋
    'shooting_star_bull':   3.0,    # B7 V4.0: bull 時 3.0x (同 V3.9, 方案3不動態化)
    'shooting_star_neutral':3.0,    # B7 V4.0: neutral 時 3.0x (同 V3.9)
    'shooting_star_bear':   3.0,    # B7 V4.0: bear/weak 時 3.0x (同 V3.9, 方案3不動態化)

    # 賣出模組開關 + 參數 (V3.8 參數化)
    'enable_tiered_stops':  True,   # S1: 階梯停利 (關閉則只留 S2+S3)
    'tier_a_net':           40,     # S1 Tier A: 淨利 > 此值 → 跌破月線就賣 (V4.12: 30→40, 6區間ablation實證放寬停利)
    'tier_a_ma_buf':        0.95,   # S1 Tier A: 月線緩衝 (V4.12: 0.97→0.95, 多給5%空間讓獲利奔跑)
    'tier_b_net':           20,     # S1 Tier B: 淨利 > 此值 → 動態回撤 (V4.12: 15→20, 配合TierA放寬)
    'tier_b_drawdown':      0.7,    # S1 Tier B: 從峰值回撤此比例觸發 (V4.12: 0.6→0.7, 減少過早停利)
    'tier_b_use_entry_peak': True,  # V4.20: True=用建倉後最高價做peak (需positions追蹤), False=用近20日高 (ablation實證: Sharpe+0.02, MDD-0.2%)
    'hard_stop_net':        -15,    # S3: 硬停損淨利% (恐慌時另外收緊)

    # V6.9: 獲利追蹤停利 (Profit Trailing Stop)
    #   當淨利 > activation → 啟動追蹤, 從峰值淨利回撤 > trail_pct 就出場
    #   概念: 讓贏家跑, 但一旦反轉就鎖住利潤
    'enable_profit_trailing':   False,       # 總開關
    'profit_trail_activation':  15,          # 啟動門檻: 淨利 > X% 才啟動追蹤
    'profit_trail_pct':         -8,          # 從峰值淨利回撤 X% (絕對值) 就賣
    'profit_trail_min_lock':    5,           # 最低鎖利: 出場時淨利至少 > X%

    # V4.20 S2 跌破季線緩衝 (有獲利時不急賣)
    's2_buffer_enabled':    False,  # True=獲利時跌破季線給緩衝, False=原始無條件賣
    's2_buffer_net_pct':    10,     # 淨利 > 此值% 時，跌破季線改為觀察不立即賣
    's2_buffer_days':       2,      # 觀察天數: 連續 N 天都在季線下才賣

    # 濾網模組開關
    'enable_market_filter': True,   # F1+F2+F3: 大盤濾網全開
    'market_filter_mode':   'relaxed',  # V4.12: strict→relaxed (6區間ablation: 修正+8.3%/多頭+4.4%/V轉+73%, 熊市僅-1.9%)

    # V4.9 大盤偵測門檻 (參數化, 供 ablation 調整)
    'crash_day_threshold':  -0.025, # is_crash: 單日跌幅門檻 (預設 -2.5%)
    'crash_bias_threshold': -0.02,  # is_crash 輔助: 乖離 < 此值 且 price < MA20
    'panic_day_threshold':  -0.03,  # is_panic: 單日跌幅門檻 (預設 -3.0%)
    'panic_3d_threshold':   -0.035, # is_panic: 3日累跌門檻 (V4.9 ablation: -5%→-3.5%, Sharpe 1.10/Calmar 1.10)
    'overheat_bias_threshold': 0.08, # is_overheated: 乖離 > 此值 (預設 8%)

    # V6.9 信念持股 (Conviction Hold)
    #   加碼 ≥ N 次的股票 → 放寬出場條件 (zombie/stop-loss/tier)
    'enable_conviction_hold':   False,
    'conviction_min_buys':      3,      # 加碼次數 ≥ 此值觸發
    'conviction_zombie_extra':  5,      # zombie 額外天數
    'conviction_zombie_range':  3.0,    # zombie 淨利範圍額外放寬%
    'conviction_stop_extra':    3.0,    # stop-loss 額外容忍%
    'conviction_tier_a_extra':  15.0,   # tier_a 停利門檻額外放寬%

    # V6.10 市場狀態自適應 (Regime-Adaptive)
    #   unsafe (weak/bear) 時使用保守參數
    'enable_regime_adaptive':   False,
    'regime_unsafe_zombie_days':    7,      # unsafe 時 zombie 天數
    'regime_unsafe_tier_b_net':     10,     # unsafe 時 tier_b 停利門檻
    'regime_unsafe_max_new_buy':    1,      # unsafe 時每日最多買幾檔

    # V6.8 族群相對估值 (Peer Group Relative Valuation)
    #   用 N日報酬 Z-score 衡量個股相對同族群是否「太貴」
    #   z > threshold → 扣分 (漲太多)  z < -threshold → 加分 (相對便宜)
    'enable_peer_zscore':       False,  # 總開關
    'peer_zscore_lookback':     60,     # 報酬率回看天數
    'peer_zscore_expensive':    1.5,    # z > 此值 → 偏貴扣分
    'peer_zscore_cheap':       -1.0,    # z < 此值 → 偏宜加分
    'peer_zscore_penalty':     -0.3,    # 偏貴扣分值
    'peer_zscore_bonus':        0.15,   # 偏宜加分值
    'peer_zscore_block':        2.5,    # z > 此值 → 直接不買 (族群內極端偏貴)
    'peer_min_stocks':          3,      # 族群至少 N 檔有數據才計算

    # V6.7 週線趨勢過濾 (Weekly Trend Filter)
    #   週線 MA20 < MA60 → 週線空頭 → 停止新建倉 (不影響既有持倉)
    'enable_weekly_filter':     False,  # 總開關

    # V4.10 持倉組合級恐慌偵測 (方向2: 解決 TWII 偵測不到類股特異風險)
    'enable_portfolio_panic':       False,  # V4.10 ablation: 弊>利, 預設關閉 (動量策略不適合恐慌賣出)
    'portfolio_panic_day_pct':      -4.0,   # 組合單日跌幅 > 此值% → 觸發組合恐慌 (賣出虧損倉)
    'portfolio_panic_3d_pct':       -7.0,   # 組合3日累跌 > 此值% → 觸發組合恐慌
    'portfolio_panic_action':       'sell_losers',  # 恐慌動作: 'sell_losers'=賣虧損倉 / 'sell_all'=全賣
    'portfolio_panic_loss_threshold': 0.0,  # 組合恐慌時, 淨利 < 此值% 的持倉才賣 (0=虧損的才賣)
    'portfolio_panic_cooldown':     3,      # 組合恐慌觸發後冷卻天數 (避免連續觸發)

    # V7: NAV-based position sizing (scale-invariant)
    'budget_pct':           0.0,    # 每筆交易 = NAV * budget_pct% (0=使用固定金額 budget_per_trade)

    # V4.11 現金管理 (禁止透支 + 加碼控制)
    'enable_cash_check':    True,   # 買入/加碼前檢查現金是否足夠
    'cash_reserve_pct':     10.0,   # 保留初始資金的 10% 不動用 (V4.11 H2 ablation 冠軍)
    'enable_add':           True,   # 是否允許加碼 (False=只建倉不加碼)
    'max_add_per_stock':    99,     # 每檔最多加碼 N 次 (99=不限)

    # V12: 拉回買入 (B8) — 體質好的股票拉回到均線附近買入
    'enable_pullback_buy':      False,  # 開啟 B8 拉回買入
    'pullback_ma60_slope_days': 20,     # MA60 斜率計算天數
    'pullback_ma60_min_slope':  0.0,    # MA60 最低斜率 (>0 = 趨勢上升)
    'pullback_bias_max':        3.0,    # 乖離率上限 (接近 MA20 = 0~3%)
    'pullback_bias_min':        -5.0,   # 乖離率下限 (跌太深不接)
    'pullback_max_drawdown':    12.0,   # 從近期高點最大回檔% (防接刀)
    'pullback_drawdown_days':   20,     # 計算回檔的近期高點天數
    'pullback_vol_ratio_max':   0.8,    # 量比上限 (量縮才是健康拉回)
    'pullback_require_above_ma60': True, # 股價必須在 MA60 之上

    # V12: 大盤急跌掃貨 (B9) — 大盤急跌時買抗跌股
    'enable_dip_buy':           False,  # 開啟 B9 大盤急跌掃貨
    'dip_market_drop_threshold': -1.5,  # 大盤單日跌幅門檻 (%)
    'dip_stock_vs_market_max':  0.5,    # 個股跌幅 < 大盤跌幅 * 此比例 = 抗跌
    'dip_ma60_min_slope':       0.0,    # MA60 最低斜率
    'dip_require_above_ma60':   True,   # 股價必須在 MA60 之上
    'dip_max_drawdown':         12.0,   # 從近期高點最大回檔%

    # V4.13 滑價模擬 (回測真實性)
    'slippage_pct':         0.3,    # V4.13: 單邊滑價% (零股0.3%, 整股0.1%, 0=完美成交)

    # V4.19 漲停跳過 (零股真實性: 開盤 OR 收盤漲停都買不到)
    'skip_limit_up':        True,   # 開盤或收盤漲停 → 跳過不買 (零股撮合不連續, 鎖漲停搓不到)
    'limit_up_threshold':   1.095,  # 價格 >= 前收 * 此值 視為漲停 (9.5% 含檔位誤差)

    # V4.18 漲停遞補 (新買入漲停 → 從備選遞補; 加碼漲停 → 暫緩不遞補)
    'enable_backup_fill':   False,   # 開啟後, 新買入被漲停跳過時自動從備選遞補

    # V4.20 漲停遞延 (T+1漲停買不到 → T+2再試一次)
    'limit_up_retry':       False,   # 開啟後, 漲停跳過的買單保留到隔天重試 (最多重試 N 次)
    'limit_up_max_retry':   1,       # 最多重試次數 (1=只試T+2, 2=試到T+3)

    # Group Scanner 上車限制 (V4.3: 網格搜索最佳化)
    'max_positions':        12,     # 同時最多持有 N 檔 (V4.22: 15→12, ablation 39/40 實證 pos12+swap保守 Sharpe 1.29/Calmar 1.02/MDD 30.7%)
    'max_new_buy_per_day':  4,      # 每天最多新建倉 N 檔 (V4.15: 2→4, 15倉+買4 Sharpe 1.30/Calmar 0.87/MDD 31%)
    'entry_sort_by':        'score', # V3.9: 'score'(品質分數) / 'bias'(乖離低優先) / 'volume'(量比高優先)

    # V3.9 倉位品質管理
    'enable_zombie_cleanup': True,  # S4: 殭屍倉位清除
    'zombie_hold_days':      15,    # S4: 持有超過 N 天
    'zombie_net_range':      5.0,   # S4: 淨利在 ±此值% 內視為殭屍
    # V13: RS 自適應殭屍 — RS 強的慢爬股延長觀察天數
    'zombie_rs_adaptive':    False, # 開啟 RS 自適應殭屍
    'zombie_rs_lookback':    60,    # RS 計算回看天數
    'zombie_rs_top_pct':     30,    # RS 排名前 N% 放寬殭屍
    'zombie_rs_extra_days':  10,    # RS 強勢股額外等待天數
    'zombie_rs_extra_range': 3.0,   # RS 強勢股額外容忍淨利範圍%
    'enable_position_swap':  True,  # 滿倉時允許換股 (踢掉最差持倉)
    'swap_score_margin':     1.0,   # 新候選分數須 > 最差持倉分數 + 此值 才換 (V4.22: 0.8→1.0, ablation 40 margin≥1.0飽和, 更保守換股讓贏家倉位充分成長)
    'max_swap_per_day':      1,     # 每天最多換股 N 檔 (V4.22: 2→1, ablation 40 noswap更差但swap=1最穩, 配合pos12集中持倉減少頻繁換股)

    # V4.8 減碼模組 (分批停利)
    'enable_reduce':         False, # R1/R2 總開關 (V4.8 ablation實證: 關閉Sharpe 1.06最佳, 開啟會砍動能)
    'reduce_tier1_net':      20,    # R1: 淨利 > 此值% 觸發首次減碼
    'reduce_tier1_ratio':    0.5,   # R1: 賣出持股比例 (50%)
    'reduce_tier2_net':      40,    # R2: 淨利 > 此值% 觸發二次減碼 (需 R1 已觸發)
    'reduce_tier2_ratio':    0.5,   # R2: 賣出「剩餘」持股比例 (再50% = 原始25%)
    'reduce_require_trend':  True,  # 減碼時要求股價 > MA20 (趨勢健康才減碼，不健康走全賣)
    'reduce_cooldown_days':  3,     # 減碼後 N 天內不再減碼 (避免同一週連續觸發)

    # V4.21 動態曝險管理 (大盤弱勢時降低持倉上限, 強制賣出超額弱勢持倉)
    'enable_dynamic_exposure':  False,  # 總開關 (預設關閉, ablation 測試用)
    'dyn_max_bull':     15,     # 多頭: 最多持有 N 檔
    'dyn_max_neutral':  15,     # 中性: 最多持有 N 檔
    'dyn_max_weak':     12,     # 偏弱: 最多持有 N 檔
    'dyn_max_bear':      8,     # 空頭: 最多持有 N 檔
    'dyn_max_panic':     5,     # 恐慌: 最多持有 N 檔

    # V4.21 動態限買 (大盤弱勢時降低每日新買上限, 不強制賣出)
    'enable_dyn_buy_limit':     False,  # 總開關
    'dyn_buy_bull':     4,      # 多頭: 每日最多新建倉 N 檔
    'dyn_buy_neutral':  4,      # 中性
    'dyn_buy_weak':     2,      # 偏弱
    'dyn_buy_bear':     1,      # 空頭
    'dyn_buy_panic':    0,      # 恐慌: 完全停買

    # V4.21 動態停損 (大盤弱勢時收緊 hard_stop_net)
    'enable_dyn_stop':          False,  # 總開關
    'hard_stop_weak':   -15,    # 偏弱時停損% (預設同 hard_stop_net)
    'hard_stop_bear':   -10,    # 空頭時收緊

    # V4.21 波動率倉位控制 (高波動時縮小單筆金額)
    'enable_vol_sizing':        False,  # 總開關
    'vol_lookback':     20,     # 滾動波動率回看天數
    'vol_target_pct':   1.5,    # 目標日波動率% (超過就縮小)
    'vol_scale_floor_pct': 50,  # 最低縮放% (50=最少用一半 budget)

    # ==========================================
    # V6 新增: 品質/RS/動能 三因子 (參考 fin_claude v10→v11)
    # ==========================================

    # V6.1 品質預篩 (Quality Pre-filter)
    #   在 Group Scanner 階段額外過濾低品質標的
    'enable_quality_filter':     False,  # 總開關
    'qf_min_trade_ratio':        0.80,   # 過去60天有交易天數 / 60 >= 此值 (流動性門檻)
    'qf_vol_stability_min':      0.3,    # 20日量 std/mean < 此值視為穩定; > 2.0 過濾 (量能穩定性)
    'qf_vol_stability_max':      2.5,    # 量能波動上限 (超過=量能極不穩定, 過濾)
    'qf_min_avg_turnover_20d':   30_000_000,  # 20日均成交額門檻 (低於=流動性不足)

    # V6.2 相對強度 (Relative Strength)
    #   對候選股依近N日報酬排序, 過濾最弱, 加分最強
    'enable_rs_filter':          False,  # 總開關
    'rs_lookback':               60,     # RS 回看天數 (60日報酬率)
    'rs_cutoff_bottom_pct':      30,     # 過濾掉底部 N% (報酬率最差的30%不買)
    'rs_bonus_top_pct':          20,     # 頂部 N% 額外加分
    'rs_bonus_score':            0.3,    # 頂部加分值 (加到品質分數 score 上)

    # V6.3 產業動能 (Sector Momentum)
    #   計算產業均線動能, 順勢加分/逆勢扣分
    'enable_sector_momentum':    False,  # 總開關
    'sm_lookback':               20,     # 產業動能回看天數 (20日平均報酬)
    'sm_positive_bonus':         0.15,   # 產業動能 > 0 時, 候選加分
    'sm_negative_penalty':       -0.3,   # 產業動能 < 0 時, 候選扣分
    'sm_strong_threshold':       0.02,   # 產業20日報酬 > 此值 = 強勁 (額外加分)
    'sm_strong_bonus':           0.2,    # 強勁動能額外加分
    'sm_weak_threshold':        -0.03,   # 產業20日報酬 < 此值 = 疲弱 (大幅扣分)
    'sm_weak_penalty':          -0.5,    # 疲弱動能大幅扣分

    # ==========================================
    # V6.4 EWT 隔夜情緒濾網 (iShares MSCI Taiwan ETF)
    #   EWT 在 NYSE 交易, 收盤 16:00 ET ≈ 台灣次日 04:00-05:00
    #   台灣 T 日開盤前可參考 EWT US T-1 的收盤
    # ==========================================
    'enable_ewt_filter':        False,  # F4: EWT 隔夜跌 > 門檻 → 暫停買進
    'ewt_drop_threshold':       -0.02,  # F4: EWT 單日跌幅門檻 (預設 -2%)
    'ewt_3d_threshold':         -0.035, # F4: EWT 3日累跌門檻 (預設 -3.5%)
    'ewt_tighten_stop':         False,  # EWT 隔夜大跌 → 收緊 hard_stop
    'ewt_tighten_threshold':    -0.03,  # 收緊觸發門檻 (EWT 日跌 > 此值)
    'ewt_tighten_amount':       3,      # 收緊幅度 (e.g., -15 → -12)

    # V6.5 EWT 加速器 (EWT 偏多時放寬進場條件)
    'ewt_accelerator':          False,  # 總開關
    'ewt_accel_fish_threshold': 6,      # EWT 偏多時魚尾門檻放寬 (4→6)
    'ewt_accel_breakout_lookback': 5,   # EWT 偏多時突破回看縮短 (10→5)

    # V6.6 EWT Score Boost (連續分數加減, 非二元開/關)
    #   T日掛單 → T日夜間 EWT 收盤 → T+1開盤前用 EWT 調整排序
    #   實作: 在 T+1 pending 執行前, 根據 market_map[T+1].ewt 調整 can_buy
    'enable_ewt_score_boost':   False,  # 總開關
    'ewt_boost_strong_up':      0.02,   # EWT 漲幅門檻: 大漲 (>+2%)
    'ewt_boost_up':             0.005,  # EWT 漲幅門檻: 偏多 (>+0.5%)
    'ewt_boost_down':          -0.01,   # EWT 跌幅門檻: 偏空 (<-1%)
    'ewt_boost_strong_down':   -0.02,   # EWT 跌幅門檻: 大跌 (<-2%)
    'ewt_boost_score_strong_up':  0.3,  # 大漲加分
    'ewt_boost_score_up':        0.15,  # 偏多加分
    'ewt_boost_score_down':     -0.15,  # 偏空減分
    'ewt_boost_score_strong_down': -0.3, # 大跌減分
    'ewt_boost_can_buy_bonus':   1,     # EWT 大漲時 can_buy 額外 +1
    'ewt_boost_can_buy_penalty': -1,    # EWT 大跌時 can_buy -1 (最低=0)
    'ewt_boost_market_adaptive': False, # 市場自適應: 空頭/恐慌時自動關閉 boost
    #   bull   → boost × 1.0 (全開)
    #   neutral→ boost × 1.0 (全開)
    #   weak   → boost × 0.5 (減半)
    #   bear   → boost × 0.0 (關閉)
    #   panic  → boost × 0.0 (關閉)

    # ==========================================
    # V8: 子題材動量 Boost (Theme Momentum Boost)
    #   熱門題材內的買入訊號獲得 score 加分，排序時優先被選入
    # ==========================================
    'enable_theme_boost':       False,  # 總開關 (預設關閉, ablation 開啟測試)
    'theme_boost_max':          0.3,    # theme_score=1.0 時最大加分 — 0.3 為最佳甜蜜點
    'theme_lookback':           20,     # 題材動量回看天數
    'theme_min_stocks':         3,      # 群內至少 N 檔有數據才計算題材熱度
    # V8.1: 市場自適應 Boost (Market-Adaptive)
    'theme_market_adaptive':    False,  # 開啟後依市場狀態自動調節 boost 強度
    # V8.2: 題材動量方向過濾 (Direction Filter)
    'theme_direction_filter':   False,  # 開啟後 20 日報酬 < 0 的題材不給 boost
    # ==========================================
    # V11: 子題材輪動 (Theme Rotation)
    #   每日只允許最熱的 N 個題材買入，冷門題材直接封鎖
    # ==========================================
    'enable_theme_rotation':            False,  # 總開關 (需 enable_theme_boost=True)
    'theme_rotation_min_score':         0.4,    # 題材 score >= 此值才允許買入 (0~1)
    'theme_rotation_min_themes':        3,      # 最少允許 N 個題材 (即使都低於門檻)
    'theme_rotation_max_themes':        8,      # 最多允許 N 個題材
    'theme_rotation_unmapped':          'allow',  # 未分類股: 'allow'/'block'/'penalty'
    'theme_rotation_unmapped_penalty':  -0.3,   # penalty 模式下的分數扣減
}


# ==========================================
# 📋 買進條件評估器 (B1-B7 + F1-F3)
#
#    統一由此函數評估，供「新建倉」和「加碼」共用
#    回傳:
#      passed:       True/False
#      tag:          買進訊號文字 (passed=True 時)
#      fail_short:   簡短失敗原因 (用於 hold 顯示)
#      fail_detail:  完整失敗原因 (用於無庫存顯示)
#      consecutive:  連續買進天數
#      is_fish_tail: 是否為魚尾
# ==========================================
def _evaluate_buy_conditions(current_price, high_price, open_price,
                             ma20, ma60, vol, vol_ma5,
                             prev_close, pct_change, bias_pct,
                             is_unsafe, is_overheated,
                             market_status, cfg, history_df):
    
    result = {
        'passed': False,
        'tag': '',
        'fail_short': '',
        'fail_detail': '',
        'consecutive': 0,
        'is_fish_tail': False,
    }
    
    # --- F1+F2+F3: 大盤濾網 (可關閉) ---
    if cfg['enable_market_filter']:
        # ✅ V3.7: F1 嚴格度參數化
        #   strict   = is_unsafe (破月線 or crash) → 擋  (原始邏輯)
        #   moderate = 只有 bear (破月+季) 或 crash 才擋，weak (只破月線) 放行
        #   relaxed  = 只有 bear 才擋 (crash 也不擋)
        filter_mode = cfg.get('market_filter_mode', 'strict')
        twii_trend = market_status.get('twii', {}).get('trend', 'neutral') if market_status else 'neutral'
        twii_crash = market_status.get('twii', {}).get('is_crash', False) if market_status else False

        if filter_mode == 'strict':
            f1_block = is_unsafe  # 原邏輯: bear/weak/crash 全擋
        elif filter_mode == 'moderate':
            f1_block = (twii_trend == 'bear') or twii_crash  # 只擋 bear + crash
        elif filter_mode == 'relaxed':
            f1_block = (twii_trend == 'bear')  # 只擋 bear
        else:
            f1_block = is_unsafe

        if f1_block:
            mode_label = {'strict': '嚴格', 'moderate': '中等', 'relaxed': '寬鬆'}.get(filter_mode, '')
            result['fail_short'] = '大盤偏空'
            result['fail_detail'] = f'🚫 大盤趨勢偏空 ({twii_trend}, F1={mode_label})，暫停買進'
            return result
        if is_overheated:
            twii_bias = market_status.get('twii', {}).get('bias_pct', 0) if market_status else 0
            result['fail_short'] = '大盤過熱'
            result['fail_detail'] = f'🔥 大盤過熱 (加權乖離 {twii_bias*100:.1f}% > 8%)，暫停買進'
            return result
        # ✅ F3: 大盤恐慌時暫停買進/加碼 (避免恐慌日加碼的矛盾)
        is_panic = market_status.get('is_panic', False) if market_status else False
        if is_panic:
            result['fail_short'] = '大盤恐慌'
            result['fail_detail'] = '🔴 大盤恐慌 (單日跌幅劇烈或3日累計重挫)，暫停買進'
            return result

    # --- F5: 週線趨勢過濾 (V6.7) ---
    if cfg.get('enable_weekly_filter', False):
        weekly_bullish = market_status.get('weekly_bullish', True) if market_status else True
        if not weekly_bullish:
            result['fail_short'] = '週線空頭'
            result['fail_detail'] = '📉 週線趨勢偏空 (週MA20 < 週MA60)，暫停新建倉'
            return result

    # --- F4: EWT 隔夜情緒濾網 (V6.4) ---
    if cfg.get('enable_ewt_filter', False):
        ewt_data = market_status.get('ewt') if market_status else None
        if ewt_data is not None:
            ewt_daily = ewt_data.get('daily_chg', 0)
            ewt_3d = ewt_data.get('cum_3d_chg', 0)
            ewt_drop_th = cfg.get('ewt_drop_threshold', -0.02)
            ewt_3d_th = cfg.get('ewt_3d_threshold', -0.035)
            if ewt_daily < ewt_drop_th or ewt_3d < ewt_3d_th:
                result['fail_short'] = 'EWT隔夜弱'
                result['fail_detail'] = (
                    f'🌙 EWT 隔夜偏弱 (日跌 {ewt_daily*100:.1f}% / '
                    f'3日 {ewt_3d*100:.1f}%)，暫停買進'
                )
                return result

    # --- B1: 趨勢 (股價 > MA20 > MA60) ---
    cond_trend = (current_price > ma20) and (current_price > ma60) and (ma20 > ma60)
    if not cond_trend:
        if ma20 < ma60:
            result['fail_short'] = '均線死叉'
            result['fail_detail'] = '❌ 均線死叉 (MA20 < MA60, 大趨勢偏空)'
        else:
            result['fail_short'] = '趨勢偏空'
            result['fail_detail'] = '❌ 趨勢偏空 (股價在月/季線之下)'
        return result

    # --- B2: 量能 ---
    is_strong_price = pct_change > 1.5
    cond_vol_standard = (vol > vol_ma5)
    cond_vol_relaxed  = (vol > vol_ma5 * 0.7) and is_strong_price
    cond_vol = cond_vol_standard or cond_vol_relaxed

    if not cond_vol:
        if vol > vol_ma5 * 0.7:
            result['fail_short'] = '量能偏弱'
            result['fail_detail'] = '❌ 量能偏弱 (量略縮且漲幅不足 1.5%)'
        else:
            result['fail_short'] = '量能不足'
            result['fail_detail'] = '❌ 量能不足 (今日量 < 5日均量 70%)'
        return result

    # --- B3: 收紅K ---
    cond_price = (current_price > prev_close)
    if not cond_price:
        result['fail_short'] = '今日收跌'
        result['fail_detail'] = '❌ 價格疲弱 (今日收跌/平盤)'
        return result

    # --- B4: 乖離率上限 (可關閉, V4.0 動態化) ---
    if cfg['enable_bias_limit']:
        # V4.0: 根據大盤趨勢動態調整乖離上限
        twii_trend_b4 = market_status.get('twii', {}).get('trend', 'neutral') if market_status else 'neutral'
        if twii_trend_b4 == 'bull':
            base_bias = cfg.get('bias_limit_bull', 15)
            extra = 3   # 低價股額外放寬 (方案3: 同 neutral)
        elif twii_trend_b4 in ('bear', 'weak'):
            base_bias = cfg.get('bias_limit_bear', 15)
            extra = 3   # 低價股額外放寬 (同 V3.9)
        else:  # neutral
            base_bias = cfg.get('bias_limit_neutral', 15)
            extra = 3
        max_bias = base_bias if current_price > 100 else base_bias + extra
        if bias_pct > max_bias:
            result['fail_short'] = f'乖離{bias_pct:.0f}%'
            result['fail_detail'] = f'⚠️ 乖離過大 (月線乖離 {bias_pct:.1f}% > {max_bias}%, 大盤={twii_trend_b4})'
            return result

    # --- V6.5: EWT 加速器 — EWT 偏多時放寬 B5/B6 ---
    ewt_bullish = False
    if cfg.get('ewt_accelerator', False) and market_status:
        ewt_data = market_status.get('ewt')
        if ewt_data is not None:
            ewt_bullish = (ewt_data.get('daily_chg', 0) > 0 and
                           ewt_data.get('above_ma20', False))

    # --- B5: 魚尾偵測 (可關閉) ---
    consecutive = 0
    if cfg['enable_fish_tail']:
        fish_lookback = cfg.get('fish_tail_lookback', 5)
        consecutive = _count_consecutive_buy_days(history_df, lookback=fish_lookback)
        total_days = consecutive + 1

        fish_limit = 4
        if ewt_bullish:
            fish_limit = cfg.get('ewt_accel_fish_threshold', 6)
        if consecutive >= fish_limit:
            result['fail_short'] = f'魚尾{total_days}天'
            result['fail_detail'] = f'🐟 魚尾警告 (連續 {total_days} 天買進訊號，追高風險大)'
            result['consecutive'] = total_days
            result['is_fish_tail'] = True
            return result

    # --- B6: 突破前N日高點 (可關閉) ---
    if cfg['enable_breakout']:
        lookback_n = cfg.get('breakout_lookback', 5)
        if ewt_bullish:
            lookback_n = cfg.get('ewt_accel_breakout_lookback', 5)
        if history_df is not None and len(history_df) >= lookback_n + 1:
            recent_highs = history_df['High'].iloc[-(lookback_n + 1):-1]
            breakout_level = float(recent_highs.max())
            
            if current_price <= breakout_level:
                result['fail_short'] = '未突破高點'
                result['fail_detail'] = (
                    f'❌ 未突破近{lookback_n}日高點 '
                    f'(現價 {current_price:.1f} ≤ 高點 {breakout_level:.1f}，盤整中)'
                )
                return result

    # --- B7: 避雷針濾網 (可關閉, V4.0 動態化) ---
    # 修正: 使用 abs(close - open) 作為實體，避免跳空高開收假陰線時 body <= 0 導致失效
    if cfg.get('enable_shooting_star', True):
        # V4.0: 根據大盤趨勢動態調整避雷針敏感度
        twii_trend_b7 = market_status.get('twii', {}).get('trend', 'neutral') if market_status else 'neutral'
        if twii_trend_b7 == 'bull':
            ratio_limit = cfg.get('shooting_star_bull', 5.0)
        elif twii_trend_b7 in ('bear', 'weak'):
            ratio_limit = cfg.get('shooting_star_bear', 2.0)
        else:  # neutral — 先查 V4.0 key，再 fallback 到舊 key
            ratio_limit = cfg.get('shooting_star_neutral', cfg.get('shooting_star_ratio', 3.0))

        body = abs(current_price - open_price)
        upper_shadow = high_price - max(current_price, open_price)  # 上影線 = 最高價 - K棒上緣

        if body > 0 and upper_shadow > 0 and upper_shadow > body * ratio_limit:
            shadow_ratio = upper_shadow / body
            result['fail_short'] = f'避雷針{shadow_ratio:.1f}x'
            result['fail_detail'] = (
                f'⚡ 避雷針風險 (上影線 {upper_shadow:.1f} > 實體 {body:.1f} 的 {shadow_ratio:.1f} 倍，'
                f'門檻{ratio_limit:.1f}x, 大盤={twii_trend_b7})'
            )
            return result

    # --- V7: 動能確認門檻 (RSI / BB%B / 10日漲幅 / 量比) ---
    if history_df is not None and len(history_df) >= 20:
        close_s = history_df['Close'].astype(float)
        n = len(close_s)

        # RSI 14
        min_rsi = cfg.get('min_rsi', 0)
        if min_rsi > 0 and n >= 15:
            delta = close_s.diff()
            gain = delta.where(delta > 0, 0.0)
            loss_s = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.rolling(14, min_periods=14).mean()
            avg_loss = loss_s.rolling(14, min_periods=14).mean()
            rs = avg_gain / avg_loss.replace(0, float('nan'))
            rsi_val = (100 - (100 / (1 + rs))).iloc[-1]
            if not pd.isna(rsi_val) and rsi_val < min_rsi:
                result['fail_short'] = f'RSI低{rsi_val:.0f}'
                result['fail_detail'] = f'⚠️ 動能不足 RSI={rsi_val:.1f} < {min_rsi}'
                return result

        # 布林 %B
        min_bb = cfg.get('min_bb_pct_b', 0)
        if min_bb > 0 and n >= 20:
            ma_bb = close_s.rolling(20).mean()
            std_bb = close_s.rolling(20).std()
            upper_bb = ma_bb + 2 * std_bb
            lower_bb = ma_bb - 2 * std_bb
            width = upper_bb.iloc[-1] - lower_bb.iloc[-1]
            if width > 0:
                pct_b = (close_s.iloc[-1] - lower_bb.iloc[-1]) / width
                if pct_b < min_bb:
                    result['fail_short'] = f'BB低{pct_b:.1f}'
                    result['fail_detail'] = f'⚠️ 布林%B={pct_b:.2f} < {min_bb} (未突破上軌)'
                    return result

        # 10 日漲幅
        min_10d = cfg.get('min_10d_return', 0)
        if min_10d > 0 and n >= 11:
            ret_10d = (close_s.iloc[-1] / close_s.iloc[-11] - 1) * 100
            if ret_10d < min_10d:
                result['fail_short'] = f'10d弱{ret_10d:.0f}%'
                result['fail_detail'] = f'⚠️ 10日漲幅={ret_10d:.1f}% < {min_10d}%'
                return result

        # 量比
        min_vol = cfg.get('min_vol_ratio', 0)
        if min_vol > 0 and n >= 20 and 'Volume' in history_df.columns:
            vol_s = history_df['Volume'].astype(float)
            vol_ma20 = vol_s.rolling(20).mean().iloc[-1]
            if vol_ma20 > 0:
                vol_r = vol_s.iloc[-1] / vol_ma20
                if vol_r < min_vol:
                    result['fail_short'] = f'量弱{vol_r:.1f}x'
                    result['fail_detail'] = f'⚠️ 量比={vol_r:.1f}x < {min_vol}x'
                    return result

    # --- ✅ 全部通過 ---
    total_days = consecutive + 1
    result['passed'] = True
    result['consecutive'] = total_days

    if cond_vol_relaxed and not cond_vol_standard:
        result['tag'] = f'📈 價強量穩 (乖離 {bias_pct:.1f}%, 漲 {pct_change:.1f}%)'
    else:
        result['tag'] = f'🚀 標準多頭訊號 (乖離 {bias_pct:.1f}%)'
    
    if consecutive >= 2:
        result['tag'] += f' ⚠️ 連續{total_days}天(留意追高)'
    
    return result


# ==========================================
# 📋 拉回買入評估器 (B8 + B9)
# ==========================================
def _evaluate_pullback_buy(current_price, ma20, ma60, vol, vol_ma5,
                           prev_close, market_status, cfg, history_df):
    """
    B8: 拉回買入 — 趨勢向上的股票拉回到 MA20 附近時買入
    B9: 大盤急跌掃貨 — 大盤急跌但個股抗跌時買入

    三道防接刀護欄:
      1. MA60 斜率 > 0 (長期趨勢仍上升)
      2. 從近期高點回檔 < 12% (不是崩盤)
      3. 量縮 (量比 < 0.8, 健康拉回不是恐慌拋售)

    回傳: {'passed': bool, 'tag': str, 'signal_type': 'B8'/'B9'}
    """
    result = {'passed': False, 'tag': '', 'signal_type': None}

    if history_df is None or len(history_df) < 65:
        return result

    close_s = history_df['Close'].astype(float)

    # ── 共用護欄計算 ──
    # 護欄1: MA60 斜率
    slope_days = cfg.get('pullback_ma60_slope_days', 20)
    min_slope = cfg.get('pullback_ma60_min_slope', 0.0)
    if len(close_s) > slope_days + 60:
        # 用 history_df 算 MA60 的斜率
        ma60_series = close_s.rolling(60).mean()
        ma60_now = ma60_series.iloc[-1]
        ma60_past = ma60_series.iloc[-1 - slope_days]
        if pd.isna(ma60_now) or pd.isna(ma60_past) or ma60_past <= 0:
            return result
        ma60_slope = (ma60_now - ma60_past) / ma60_past
        if ma60_slope <= min_slope:
            return result  # MA60 沒在上升 → 趨勢已壞，不接
    else:
        return result

    # 護欄2: 回檔幅度
    dd_days = cfg.get('pullback_drawdown_days', 20)
    max_dd = cfg.get('pullback_max_drawdown', 12.0)
    recent_high = float(close_s.iloc[-dd_days:].max()) if len(close_s) >= dd_days else float(close_s.max())
    if recent_high > 0:
        drawdown_pct = (recent_high - current_price) / recent_high * 100
        if drawdown_pct > max_dd:
            return result  # 跌太深 → 可能是接刀

    # 護欄3: 量縮確認
    vol_ratio = vol / vol_ma5 if vol_ma5 > 0 else 99

    # 大盤狀態
    twii_data = market_status.get('twii', {}) if market_status else {}
    market_daily_chg = twii_data.get('daily_chg', 0)
    is_panic = market_status.get('is_panic', False) if market_status else False

    pct_change = (current_price - prev_close) / prev_close * 100 if prev_close > 0 else 0
    bias_pct = (current_price - ma20) / ma20 * 100 if ma20 > 0 else 0

    # ══════════════════════════════════════
    # B8: 拉回買入
    # ══════════════════════════════════════
    if cfg.get('enable_pullback_buy', False) and not is_panic:
        b8_bias_max = cfg.get('pullback_bias_max', 3.0)
        b8_bias_min = cfg.get('pullback_bias_min', -5.0)
        b8_vol_max = cfg.get('pullback_vol_ratio_max', 0.8)
        b8_above_ma60 = cfg.get('pullback_require_above_ma60', True)

        cond_bias = b8_bias_min <= bias_pct <= b8_bias_max
        cond_vol = vol_ratio <= b8_vol_max
        cond_above_ma60 = (not b8_above_ma60) or (current_price > ma60)
        cond_trend = ma20 > ma60  # 均線多頭排列

        if cond_bias and cond_vol and cond_above_ma60 and cond_trend:
            result['passed'] = True
            result['signal_type'] = 'B8'
            result['tag'] = (f'🔄 拉回買入 (乖離 {bias_pct:+.1f}%, '
                             f'量比 {vol_ratio:.1f}, 回檔 {drawdown_pct:.1f}%)')
            return result

    # ══════════════════════════════════════
    # B9: 大盤急跌掃貨
    # ══════════════════════════════════════
    if cfg.get('enable_dip_buy', False) and not is_panic:
        dip_threshold = cfg.get('dip_market_drop_threshold', -1.5) / 100
        dip_vs_market = cfg.get('dip_stock_vs_market_max', 0.5)
        dip_above_ma60 = cfg.get('dip_require_above_ma60', True)

        if market_daily_chg < dip_threshold:
            # 大盤在急跌
            stock_drop = pct_change / 100
            cond_resilient = stock_drop > (market_daily_chg * dip_vs_market)  # 個股跌幅 < 大盤跌幅*比例
            cond_above = (not dip_above_ma60) or (current_price > ma60)
            cond_trend = ma20 > ma60

            if cond_resilient and cond_above and cond_trend:
                result['passed'] = True
                result['signal_type'] = 'B9'
                resilience = (1 - stock_drop / market_daily_chg) * 100 if market_daily_chg != 0 else 0
                result['tag'] = (f'🛡️ 急跌掃貨 (大盤 {market_daily_chg*100:+.1f}%, '
                                 f'個股 {pct_change:+.1f}%, 抗跌 {resilience:.0f}%)')
                return result

    return result


# ==========================================
# 🧠 策略核心 V4.0 (V3.9 + 動態B4/B7 + 換股修正)
#
# V3.5 合併 (基於 V3.4):
#   - 所有賣出門檻 (S1/S2/S3) 改用 net_pnl_pct (扣費後淨利) 判斷
#   - Tier B 動態回撤: peak 也用淨利計算, 避免帳面虛胖
#   - F3 恐慌濾網: 大盤恐慌時暫停買進/加碼, 避免恐慌日仍建議加碼的矛盾
#   - calculate_fee: 區分整股(低消20元)/零股(低消1元), 依元大投資先生規則
#   - B6 off-by-one: lookback 邊界 > 改 >=
#
# V3.3 修正:
#   - S1 elif 結構漏洞修正: Tier A / Tier B 改為各自獨立 if
#   - S2 跌破季線獨立化: 不再受 S1 Tier 的 elif 影響
#   - Tier B 改為動態回撤: 用近 20 日最高價作為 peak, 回撤一半觸發
#   - B7 避雷針修正: 使用 abs(close-open) 避免假陰線失效
#   - ETF 判斷修正: 先 normalize 去掉 .TW/.TWO 後綴
#
# 流程:
#   有庫存 → 先賣出檢查 (S1→S2→S3) → 沒觸發 → 買進條件檢查
#            → 通過 = 加碼  → 沒過 = 持有(附原因)
#   沒庫存 → 買進條件檢查
#            → 通過 = 買進  → 沒過 = 觀望(附原因)
#
# ✅ group_scanner 說不買 → main/daily 也不會說買(加碼)
#    因為共用同一組 _evaluate_buy_conditions
# ✅ 唯一差異: main/daily 可能說 sell (group 不會)
# ==========================================

def check_strategy_signal(stock_id, info, held_cost=0, held_shares=0,
                          market_status=None, history_df=None, config=None,
                          reduce_stage=0, last_reduce_day_idx=-99,
                          current_day_idx=0,
                          peak_price_since_entry=None):
    """
    策略主函數 V4.8

    回傳:
      - action:       'buy' / 'sell' / 'reduce' / 'hold'
      - reason:       中文說明
      - pnl:          損益明細 (持股時才有)
      - consecutive:  連續買進天數
      - is_fish_tail: 是否魚尾
      - reduce_ratio: 減碼比例 (action='reduce' 時)
      - reduce_stage: 更新後的減碼階段 (action='reduce' 時)
    """

    # --- 合併設定 ---
    cfg = {**DEFAULT_CONFIG}
    if config:
        cfg.update(config)

    # ------------------------------------------
    # 0. 數據解包與防呆
    # ------------------------------------------
    default_result = {
        'action': 'hold', 'reason': '',
        'pnl': None, 'consecutive': 0, 'is_fish_tail': False,
        'reduce_ratio': 0, 'reduce_stage': reduce_stage,
    }
    
    required_keys = ['close', 'ma20', 'ma60', 'volume', 'vol_ma5', 'prev_close']
    for key in required_keys:
        if key not in info or info[key] is None:
            return {**default_result, 'reason': f'⚠️ 缺少數據: {key}'}

    try:
        current_price = float(info['close'])
        ma20          = float(info['ma20'])
        ma60          = float(info['ma60'])
        vol           = float(info['volume'])
        vol_ma5       = float(info['vol_ma5'])
        prev_close    = float(info['prev_close'])
        high_price    = float(info.get('high', current_price))
        open_price    = float(info.get('open', prev_close))
    except (ValueError, TypeError):
        return {**default_result, 'reason': '⚠️ 數據格式錯誤'}

    pct_change = (current_price - prev_close) / prev_close * 100
    bias_pct   = (current_price - ma20) / ma20 * 100

    # 解析大盤狀態
    is_unsafe     = market_status.get('is_unsafe', False) if market_status else False
    is_overheated = market_status.get('is_overheated', False) if market_status else False
    is_panic      = market_status.get('is_panic', False) if market_status else False

    # =========================================
    # 🛑 賣出邏輯 (有庫存時優先檢查)
    # =========================================
    if held_shares > 0 and held_cost > 0:
        profit_pct = (current_price - held_cost) / held_cost * 100  # 帳面（顯示用）
        pnl_detail = calculate_net_pnl(stock_id, held_cost, current_price, held_shares)
        net_pct = pnl_detail['net_pnl_pct']  # ✅ V3.5: 所有賣出門檻改用淨利判斷

        def make_sell_result(reason):
            return {
                'action': 'sell', 'reason': reason,
                'pnl': pnl_detail, 'consecutive': 0, 'is_fish_tail': False,
                'reduce_ratio': 0, 'reduce_stage': reduce_stage,
            }

        def make_reduce_result(reason, ratio, new_stage):
            return {
                'action': 'reduce', 'reason': reason,
                'pnl': pnl_detail, 'consecutive': 0, 'is_fish_tail': False,
                'reduce_ratio': ratio, 'reduce_stage': new_stage,
            }

        # --- 恐慌收緊 (F3) ---
        # V3.8: S1 門檻從 cfg 讀取，恐慌時收緊為正常值的一半
        if is_panic and cfg['enable_market_filter']:
            tier_a_net    = cfg.get('tier_a_net', 30) / 2      # 恐慌: 30→15
            tier_a_ma_buf = 1.00                                # 恐慌: 精確跌破月線就賣
            tier_b_net    = cfg.get('tier_b_net', 15) / 2      # 恐慌: 15→7.5
            tier_b_dd     = max(0.3, cfg.get('tier_b_drawdown', 0.6) - 0.2)  # 恐慌: 0.6→0.4
            hard_stop_net = cfg.get('hard_stop_net', -15) + 5  # 恐慌: -15→-10
            mode_tag      = " [恐慌收緊🔴]"
        else:
            tier_a_net    = cfg.get('tier_a_net', 30)
            tier_a_ma_buf = cfg.get('tier_a_ma_buf', 0.97)
            tier_b_net    = cfg.get('tier_b_net', 15)
            tier_b_dd     = cfg.get('tier_b_drawdown', 0.6)
            hard_stop_net = cfg.get('hard_stop_net', -15)
            mode_tag      = ""

        # --- V6.4: EWT 隔夜大跌 → 額外收緊 hard_stop (可與恐慌疊加) ---
        if cfg.get('ewt_tighten_stop', False) and market_status:
            ewt_data = market_status.get('ewt')
            if ewt_data is not None:
                ewt_daily = ewt_data.get('daily_chg', 0)
                ewt_tight_th = cfg.get('ewt_tighten_threshold', -0.03)
                if ewt_daily < ewt_tight_th:
                    ewt_tight_amt = cfg.get('ewt_tighten_amount', 3)
                    hard_stop_net = hard_stop_net + ewt_tight_amt  # e.g., -15 → -12
                    mode_tag += " [EWT收緊🌙]"

        # --- R1/R2: 減碼 (V4.8 分批停利) ---
        # 減碼優先於全賣：趨勢健康時先鎖部分獲利，趨勢轉壞才走 S1-S3 全賣
        if cfg.get('enable_reduce', False) and held_shares > 0:
            r_cooldown = cfg.get('reduce_cooldown_days', 3)
            days_since_reduce = current_day_idx - last_reduce_day_idx
            trend_ok = (not cfg.get('reduce_require_trend', True)) or (current_price > ma20)

            if trend_ok and days_since_reduce >= r_cooldown:
                # R1: 首次減碼
                if reduce_stage == 0 and net_pct > cfg.get('reduce_tier1_net', 20):
                    r1_ratio = cfg.get('reduce_tier1_ratio', 0.5)
                    return make_reduce_result(
                        f'✂️ R1 首次減碼 (帳面 {profit_pct:+.1f}%, '
                        f'淨利 {net_pct:+.1f}% > {cfg.get("reduce_tier1_net", 20)}%, '
                        f'賣出 {r1_ratio:.0%}){mode_tag}',
                        r1_ratio, 1
                    )
                # R2: 二次減碼 (需 R1 已觸發)
                elif reduce_stage == 1 and net_pct > cfg.get('reduce_tier2_net', 40):
                    r2_ratio = cfg.get('reduce_tier2_ratio', 0.5)
                    return make_reduce_result(
                        f'✂️ R2 二次減碼 (帳面 {profit_pct:+.1f}%, '
                        f'淨利 {net_pct:+.1f}% > {cfg.get("reduce_tier2_net", 40)}%, '
                        f'賣出剩餘 {r2_ratio:.0%}){mode_tag}',
                        r2_ratio, 2
                    )

        # --- S1: 階梯停利 (可關閉, V3.8 全參數化) ---
        # 修正: 拆開 if/elif 結構，Tier A 和 Tier B 各自獨立判斷
        # V3.5: 所有賣出門檻改用 net_pct（扣除手續費+交易稅後的淨利）
        # V3.8: 門檻從 cfg 讀取，不再 hardcode
        if cfg['enable_tiered_stops']:
            # S1 Tier A: 高獲利保護 (淨利 > tier_a → 跌破月線×buf 就賣)
            if net_pct > tier_a_net:
                if current_price < ma20 * tier_a_ma_buf:
                    return make_sell_result(
                        f'💰 獲利保護停利 (帳面 {profit_pct:+.1f}%, '
                        f'淨利 {net_pct:+.1f}%, 跌破月線×{tier_a_ma_buf}){mode_tag}'
                    )

            # S1 Tier B: 動態回撤停利 (tier_b < 淨利 <= tier_a)
            # V4.20: 可選用建倉後最高價 (tier_b_use_entry_peak) 或近20日高
            if tier_b_net < net_pct <= tier_a_net:
                _use_entry_peak = cfg.get('tier_b_use_entry_peak', False)
                if _use_entry_peak and peak_price_since_entry is not None:
                    peak_price = peak_price_since_entry
                elif history_df is not None and len(history_df) >= 20:
                    peak_price = float(history_df['Close'].tail(20).max())
                else:
                    peak_price = current_price
                # 峰值淨利 = 假設在 peak_price 賣出的淨利%
                peak_pnl = calculate_net_pnl(stock_id, held_cost, peak_price, held_shares)
                peak_net_pct = peak_pnl['net_pnl_pct']
                # floor 淨利 = 峰值淨利 × (1 - drawdown比例)
                floor_net_pct = peak_net_pct * (1 - tier_b_dd)
                if net_pct < floor_net_pct:
                    return make_sell_result(
                        f'🛡️ 動態回撤停利 (帳面 {profit_pct:+.1f}%, '
                        f'淨利 {net_pct:+.1f}%, '
                        f'峰值淨利{peak_net_pct:.1f}%→回撤至{floor_net_pct:.1f}%){mode_tag}'
                    )

        # --- S1.5: 獲利追蹤停利 (Profit Trailing Stop) ---
        # 淨利超過 activation 後，從峰值淨利回撤超過 trail_pct 就出場
        # 優先級在 S1 之後、S2 之前 (S1 沒觸發才看 trailing)
        _pt_enabled = cfg.get('enable_profit_trailing', False)
        if _pt_enabled and peak_price_since_entry is not None and peak_price_since_entry > 0:
            _pt_activation = cfg.get('profit_trail_activation', 15)
            _pt_trail = cfg.get('profit_trail_pct', -8)
            _pt_min_lock = cfg.get('profit_trail_min_lock', 5)
            # 計算峰值淨利
            _pt_peak_pnl = calculate_net_pnl(stock_id, held_cost, peak_price_since_entry, held_shares)
            _pt_peak_net = _pt_peak_pnl['net_pnl_pct']
            # 只在峰值淨利曾超過 activation 時才啟動
            if _pt_peak_net > _pt_activation:
                _pt_drawdown = net_pct - _pt_peak_net  # 負數 = 從峰值回撤
                if _pt_drawdown < _pt_trail and net_pct > _pt_min_lock:
                    return make_sell_result(
                        f'🔒 追蹤停利 (峰值淨利{_pt_peak_net:+.1f}%→目前{net_pct:+.1f}%, '
                        f'回撤{_pt_drawdown:.1f}%>{_pt_trail}%, 鎖利>{_pt_min_lock}%){mode_tag}'
                    )

        # --- S2: 跌破季線 (獨立判斷，不受 S1 Tier 影響) ---
        # V4.20: S2 緩衝 — 有獲利時跌破季線不急賣，等連續 N 天確認
        if current_price < ma60:
            _s2_buf = cfg.get('s2_buffer_enabled', False)
            _s2_buf_net = cfg.get('s2_buffer_net_pct', 10)
            _s2_buf_days = cfg.get('s2_buffer_days', 2)

            _s2_sell = True  # 預設賣出
            if _s2_buf and net_pct > _s2_buf_net:
                # 有獲利 → 計算連續低於季線天數
                _below_days = 0
                if history_df is not None and len(history_df) >= 62:
                    _recent_close = history_df['Close'].iloc[-_s2_buf_days:]
                    _recent_ma60 = history_df['Close'].rolling(60).mean().iloc[-_s2_buf_days:]
                    _below_days = sum(1 for c, m in zip(_recent_close, _recent_ma60)
                                      if not pd.isna(m) and c < m)
                if _below_days < _s2_buf_days:
                    _s2_sell = False  # 還在緩衝期，不賣

            if _s2_sell:
                return make_sell_result(
                    f'📉 跌破季線 (帳面 {profit_pct:+.1f}%, '
                    f'淨利 {net_pct:+.1f}%){mode_tag}'
                )

        # S3: 硬停損 (淨利) — 可選 trailing stop
        _trailing_enabled = cfg.get('enable_trailing_stop', False)
        _trailing_from_peak = cfg.get('trailing_stop_from_peak', -15)
        if _trailing_enabled and peak_price_since_entry is not None and peak_price_since_entry > 0:
            # trailing stop: 從建倉後最高價回撤 X% 就賣
            peak_pnl_info = calculate_net_pnl(stock_id, held_cost, peak_price_since_entry, held_shares)
            peak_net = peak_pnl_info['net_pnl_pct']
            drawdown_from_peak = net_pct - peak_net
            # 只在淨利為負時才觸發 (避免獲利中誤觸)
            if net_pct < 0 and drawdown_from_peak < _trailing_from_peak:
                return make_sell_result(
                    f'💀 追蹤停損 (從峰值{peak_net:+.1f}%回撤{drawdown_from_peak:.1f}%>{_trailing_from_peak}%, '
                    f'帳面 {profit_pct:+.1f}%, 淨利 {net_pct:+.1f}%){mode_tag}'
                )
        if net_pct < hard_stop_net:
            return make_sell_result(
                f'💀 硬停損 ({hard_stop_net}%) (帳面 {profit_pct:+.1f}%, '
                f'淨利 {net_pct:+.1f}%){mode_tag}'
            )

        # ==================================================
        # ✅ V3.2 新增: 沒觸發賣出 → 檢查是否適合加碼
        # ==================================================
        panic_note = " ⚡恐慌警戒" if (is_panic and cfg['enable_market_filter']) else ""
        pos_tag = f"庫存 帳面{profit_pct:+.1f}% 淨利{net_pct:+.1f}%"
        
        buy_eval = _evaluate_buy_conditions(
            current_price, high_price, open_price,
            ma20, ma60, vol, vol_ma5,
            prev_close, pct_change, bias_pct,
            is_unsafe, is_overheated,
            market_status, cfg, history_df
        )
        
        if buy_eval['passed']:
            # ✅ 買進條件全過 → 建議加碼 (股數由回測引擎依 budget_per_trade 計算)
            return {
                'action': 'buy',
                'reason': f'{buy_eval["tag"]} [加碼 {pos_tag}]{panic_note}',
                'pnl': pnl_detail,
                'consecutive': buy_eval['consecutive'],
                'is_fish_tail': False,
                'reduce_ratio': 0, 'reduce_stage': reduce_stage,
            }
        else:
            # ❌ 買進條件未達 → 持有 + 說明為何不加碼
            return {
                'action': 'hold',
                'reason': f'持有 ({pos_tag}) '
                          f'不加碼: {buy_eval["fail_short"]}{panic_note}',
                'pnl': pnl_detail,
                'consecutive': buy_eval['consecutive'],
                'is_fish_tail': buy_eval['is_fish_tail'],
                'reduce_ratio': 0, 'reduce_stage': reduce_stage,
            }

    # =========================================
    # 🚀 買進邏輯 (無庫存)
    # =========================================
    buy_eval = _evaluate_buy_conditions(
        current_price, high_price, open_price,
        ma20, ma60, vol, vol_ma5,
        prev_close, pct_change, bias_pct,
        is_unsafe, is_overheated,
        market_status, cfg, history_df
    )
    
    if buy_eval['passed']:
        # 股數由回測引擎依 budget_per_trade 計算
        return {
            'action': 'buy',
            'reason': buy_eval['tag'],
            'pnl': None,
            'consecutive': buy_eval['consecutive'],
            'is_fish_tail': False,
            'reduce_ratio': 0, 'reduce_stage': 0,
        }
    else:
        return {
            'action': 'hold',
            'reason': buy_eval['fail_detail'],
            'pnl': None,
            'consecutive': buy_eval['consecutive'],
            'is_fish_tail': buy_eval['is_fish_tail'],
            'reduce_ratio': 0, 'reduce_stage': 0,
        }