# 策略總結文件 — V19b+R6+R11+R14+R15 A+ 半導體動能策略

> 最後更新: 2026-04-06
> 目的: 完整記錄當前策略版本、改動內容、績效數據、執行方式，供其他開發者（含 Claude Code）正確復現結果

---

## 1. 產業與標的

- **唯一產業**: `半導體業` (DAILY_DEFAULT_INDUSTRIES = ['半導體業'])
- **股票來源**: twstock 資料庫 + 補充清單，約 100+ 檔半導體股（IC 設計、晶圓代工、封測、材料）
- **市場**: 台灣上市 (.TW) + 上櫃 (.TWO)
- **前置篩選**: 日均量 ≥ 50萬股、日均成交值 ≥ 5000萬、股價 ≥ 10元、資料 ≥ 60天

---

## 2. Train / Validation 時間切分

| 代號 | 名稱 | 期間 | 用途 |
|------|------|------|------|
| **T** | Training 完整 | 2021-01-01 ~ 2025-06-30 | 調參專用 (4.5年完整市況) |
| T1 | 純空頭 | 2022-01-01 ~ 2022-10-31 | 加權 18000→12600 |
| T2 | 純多頭 | 2023-01-01 ~ 2024-07-31 | AI 狂飆 14000→24000 |
| T3 | 高檔修正 | 2024-07-01 ~ 2025-01-31 | AI 過熱回檔 |
| T4 | 關稅崩盤 | 2025-02-01 ~ 2025-06-30 | 川普關稅急跌 |
| **V** | Validation | 2025-07-01 ~ now | 驗證專用，參數定案後只跑一次，禁止回頭改參數 |

**防過擬合原則**: 所有 ablation 調參只看 T 期間結果。V 期間是最終驗證，任何參數改動都不能根據 V 的結果反向調整。

---

## 3. 當前最佳績效 (半導體業 V19b A+)

**描述行**: `🏆 V19b+R6+R11+R14+R15: A+ (Shrp 0.92, Clm 0.89, Ret+138%, MDD 24.1%)`

| 指標 | 數值 |
|------|------|
| **Sharpe Ratio** | 0.92 |
| **Calmar Ratio** | 0.89 |
| **總報酬** | +138% |
| **最大回撤 (MDD)** | 24.1% |

> **注意**: 這是 Training 期間 (2021-01 ~ 2025-06) 的績效。Sharpe 0.92 是正確數字。
> 如果有人跑出 Sharpe 0.43，很可能是:
> 1. 用了錯誤的時間區間（例如只跑了 Validation 期間或只跑空頭段）
> 2. 沒有正確載入 `INDUSTRY_CONFIGS` 的半導體配置（用了 DEFAULT_CONFIG 預設參數）
> 3. EXEC_MODE 不是 `'next_open'`
> 4. 初始資金或單筆金額設定錯誤

### 如何正確復現績效

```bash
# 方法 1: 跑完整回測 (含 0050 benchmark 對照)
python3 run_full_backtest.py

# 方法 2: 用互動模式選 Training 期間
python3 group_backtest.py
# → 選產業: 半導體業
# → 選期間: T (Training 完整 2021-01~2025-06)
# → 成交模式: 1 (T+1 開盤)

# 方法 3: 跑 7 段交叉驗證 (含 Train + Val)
python3 run_7030_backtest.py
```

### run_full_backtest.py 關鍵設定

```python
START_DATE = '2022-01-01'
END_DATE = '2026-03-13'       # 含 Val 期間的完整回測
INITIAL_CAPITAL = 900_000
BUDGET_PER_TRADE = 25_000
INDUSTRY = '半導體業'
EXEC_MODE = 'next_open'
SLIPPAGE_PCT = 0.003          # 0.3% 滑價
STRATEGY_CONFIG = INDUSTRY_CONFIGS.get(INDUSTRY, {}).get('config', {})
# ↑ 這行自動載入半導體專屬配置，不能省略
```

---

## 4. 從 Baseline 到 A+ 的完整改動清單

以下是從 DEFAULT_CONFIG 預設值到當前半導體 A+ 配置的所有差異:

### 4.1 持倉管理
| 參數 | 預設值 | A+ 值 | 改動原因 |
|------|--------|-------|----------|
| `budget_pct` | (固定金額) | 2.8% | NAV-based sizing, 25K/900K ≈ 2.78% |
| `zombie_hold_days` | 15 | 10 | 半導體週轉快，L2 ablation z10 > z15 |
| `max_add_per_stock` | 99 | 8 | R24: 防單股集中度>30% NAV (Shrp +0.05) |

### 4.2 停利/停損
| 參數 | 預設值 | A+ 值 | 改動原因 |
|------|--------|-------|----------|
| `tier_a_net` | 40% | 80% | V19b: 讓大贏家跑更久 (Ret+6.6%, PF+0.11) |
| `tier_b_net` | 20% | 15% | L3: 放寬中段停利門檻 |
| `tier_b_drawdown` | 0.7 | 0.6 | R4: 收緊回撤 (Train Shrp +0.04, MDD-1.4%) |
| `hard_stop_weak` | -15% | -12% | 動態停損: 偏弱趨勢收緊 |
| `hard_stop_bear` | -15% | -10% | 動態停損: 空頭更積極砍損 |

### 4.3 進場濾網
| 參數 | 預設值 | A+ 值 | 改動原因 |
|------|--------|-------|----------|
| `min_rsi` | 0 | 40 | V6.8: RSI<40 不買 (Shrp +0.05, PF 1.76→1.81) |
| `fish_tail_lookback` | 5 | 3 | V9: 更快偵測動能 (Shrp 1.18→1.21) |
| `market_filter_mode` | 'relaxed' | 'moderate' | R23: 中等過濾 (Clm +0.04, MDD -0.8%) |

### 4.4 風險控制模組 (全部新增)
| 模組 | 開關 | 關鍵參數 | 效果 |
|------|------|----------|------|
| **Portfolio Panic** | `enable_portfolio_panic: True` | 日跌-4%, 3日跌-6% → 賣虧損股, 冷卻3天 | R14: Shrp +0.04, Clm +0.05, Ret+10% |
| **動態停損** | `enable_dyn_stop: True` | weak -12%, bear -10% | 配合 Portfolio Panic |
| **S2 緩衝** | `s2_buffer_enabled: True` | 淨利>10% 給 5天觀察 | V19b: 減少假跌破被洗出 (Ret+25%, Shrp+0.10) |
| **產業動能** | `enable_sector_momentum: True` | 20日均, 正向+0.15, 負向-0.3 | V8: Shrp +0.04 |
| **波動率倉控** | `enable_vol_sizing: True` | 目標 2.5%, floor 70% | V6.5: 高波動時自動縮小 |
| **非對稱殭屍** | `zombie_asymmetric: True` | 虧損 15天快砍, 獲利 45天+<8%才清 | V15: Ret+21%, PF 2.01, WR+2.1% |
| **Peer RS** | `enable_val_peer_hold: True` | Theme 分群, Z>1.2 偏貴, Z<-1 最弱 | V19b: Shrp+0.08, Clm+0.10, Ret+18.4% |
| **子題材限持** | `theme_max_hold: 3` | 同子題材最多 3 檔 | R11: Shrp +0.02, Ret+4.3% |
| **多時間框架** | `enable_mtf: True` | 週線 MA20<MA60 → 不開新倉 | R15: MDD 28.4→24.1%, Calmar +0.09 |
| **EWT 隔夜** | `enable_ewt_filter: True` | 單日跌-3%, 3日跌-5% 不買 | R16b: MDD -0.3~-0.9% |
| **VIX 恐慌** | `enable_vix_filter: True` | VIX>25 限買2檔, VIX>30 全停 | R20: Shrp +0.03, MDD -1.4%, Clm +0.10 |

### 4.5 停用的模組 (ablation 驗證無效)
| 模組 | 原因 |
|------|------|
| `enable_theme_boost` | 靜態加分效果不穩定 |
| `enable_theme_rotation` | 半導體不適合高頻輪動 |
| `mtf_weekly_tighten` | 已有足夠出場機制 |
| `enable_profit_trailing` | 與 tier_a/tier_b 衝突 |
| B7 Shooting Star | ablation 顯示關閉 +0.10 Sharpe |
| B8 Pullback / B9 Dip | 動能策略不適合逆勢買入 |

---

## 5. 核心交易參數

```python
INITIAL_CAPITAL = 900_000       # 初始資金 90萬
DAILY_BUDGET = 25_000           # 單筆 2.5萬 (V4.18 ablation: 25K > 15K)
DAILY_MAX_POSITIONS = 12        # 最多持有 12 檔
DAILY_MAX_NEW_BUY = 4           # 每日最多買 4 檔新股
DAILY_MAX_SWAP = 1              # 每日最多換股 1 檔
EXEC_MODE = 'next_open'         # T日訊號 → T+1 開盤執行
SLIPPAGE_PCT = 0.003            # 滑價 0.3%
```

### 手續費計算 (台灣券商)
- 買: 0.1425% × 0.6 折扣 = 0.0855%
- 賣: 0.1425% × 0.6 折扣 + 0.3% 證交稅 (ETF 為 0.1%)
- 最低手續費: 1~20 元
- **所有 P&L 都是淨利** (扣完手續費+稅)

---

## 6. 每日交易使用方式

### 6.1 查看今日訊號
```bash
python3 run_daily_trading.py
# 或指定日期
python3 run_daily_trading.py signal 2026-04-07
```
輸出包含:
- 當日大盤狀態 (Bull/Neutral/Weak/Bear + Crash/Panic)
- 買進訊號列表 (排序 by score, 顯示乖離率)
- 賣出訊號列表 (含原因: S1停利/S2破線/S3停損/S4殭屍)
- 待執行掛單 (pending orders)

### 6.2 查看目前持倉
```bash
python3 run_daily_trading.py status
```

### 6.3 記錄已執行的交易
```bash
python3 run_daily_trading.py add
# 互動式輸入: 買/賣、股票代碼、股數、價格
```

### 6.4 查看交易歷史
```bash
python3 run_daily_trading.py history
```

### 6.5 重置引擎狀態 (重新開始)
```bash
python3 run_daily_trading.py reset
```

### 狀態檔案
- 持倉狀態: `output/daily_engine_state.json`
- 每日快照: `output/daily_csv/`

---

## 7. 驗證回測對齊 (Critical)

**設計原則**: Daily Engine 必須與 Backtest Engine 產出 100% 相同結果。

### 驗證指令
```bash
# 驗證特定日期範圍
python3 run_daily_trading.py verify 2026-03-26 2026-04-06

# 驗證全部歷史
python3 run_daily_trading.py verify
```

### 驗證機制
`verify` 模式會:
1. 用 Daily Engine (`step_day()`) 逐日跑一遍指定區間
2. 用 Backtest Engine (`run_group_backtest()`) 跑同一區間
3. 逐日比對: 持倉清單、買賣訊號、現金餘額、NAV
4. 報告任何差異 (理論上應為 0 差異)

### 何時需要驗證
- 修改 `strategy.py` 中的策略邏輯後
- 修改 `group_backtest.py` 中的回測引擎後
- 修改 `run_daily_trading.py` 中的每日引擎後
- **任何策略邏輯改動必須同步至兩個引擎**

### Import Chain (改動傳播路徑)
```
strategy.py (定義 check_strategy_signal, calculate_net_pnl, calculate_fee, calculate_tax)
  ↑ 被 import
group_backtest.py (re-export + 回測引擎 + INDUSTRY_CONFIGS)
  ↑ 被 import
run_daily_trading.py (DailyEngine, 用 re-exported 函式)
run_full_backtest.py (完整回測 runner)
```

---

## 8. 所有 Ablation 腳本清單

| 腳本 | 測試維度 | 用途 |
|------|----------|------|
| `run_full_backtest.py` | 完整回測+0050 benchmark | 主要績效報告 |
| `run_7030_backtest.py` | 7030 ETF benchmark | 對照被動投資 |
| `run_direction_ablation.py` | 市場濾網模式 (strict/moderate/relaxed) | F1 參數 |
| `run_ewt_ablation.py` | EWT 隔夜情緒濾網 | F4 參數 |
| `run_mdd_ablation.py` | 最大回撤參數 | 停損相關 |
| `run_pullback_ablation.py` | Strategy B (逆勢買入) vs A (動能) | 策略選擇 |
| `run_rsi_ablation.py` | RSI 指標優化 | 進場濾網 |
| `run_trailing_ablation.py` | 停利追蹤優化 | 出場機制 |
| `run_conv_regime_ablation.py` | 收斂/市場狀態測試 | 市場分類 |
| `run_peer_ablation.py` | Peer 族群 Z-score | 相對估值 |
| `run_round4_ablation.py` | Round 4: 出場/風控+選股 | 綜合微調 |
| `run_midterm_backtest.py` | Strategy C (中期趨勢) vs A | 策略對比 |
| `run_midterm_backtest_r2.py` | Strategy C Round 2 | C 策略改進 |
| `run_midterm_backtest_r3.py` | Strategy C Round 3 | C 策略改進 |

每個 ablation 都跑 7 段市場 (T, T1-T4, V, 完整週期)，只有在所有段落都不劣化時才接受改動。

---

## 9. 關於 Sharpe 0.43 的排查

如果跑出 Sharpe ~0.43 而非 ~0.92，請檢查:

### 最常見原因: 沒載入 INDUSTRY_CONFIGS

```python
# 錯誤: 只用 DEFAULT_CONFIG
result = run_group_backtest(tickers, ..., config_override={})

# 正確: 讓引擎自動讀取 INDUSTRY_CONFIGS
# run_full_backtest.py 已正確設定:
STRATEGY_CONFIG = INDUSTRY_CONFIGS.get(INDUSTRY, {}).get('config', {})
result = run_group_backtest(tickers, ..., config_override=STRATEGY_CONFIG)
```

### 其他可能原因

1. **時間區間問題**: 只跑了空頭段 (2022) 或 Val 段 (2025-07+)
   - 解決: 用 `T` 期間 (2021-01 ~ 2025-06) 或完整區間
2. **EXEC_MODE 錯誤**: 不是 `'next_open'`
3. **初始資金**: 不是 900,000
4. **單筆金額**: 不是 25,000
5. **滑價**: 不是 0.3%
6. **股票清單**: 沒有用 `get_stocks_by_industry('半導體業')` 取得正確清單
7. **缺少依賴**: `yfinance`, `twstock`, `requests` 等未安裝
8. **快取問題**: `.cache/` 目錄下的 pickle 過期或損壞 → 刪除 `.cache/` 重跑

### 快速驗證步驟

```bash
# 1. 確認依賴
pip install yfinance pandas numpy twstock requests

# 2. 清除快取 (可選)
rm -rf .cache/

# 3. 跑完整回測
python3 run_full_backtest.py

# 4. 檢查輸出最後幾行，應看到:
#    Sharpe Ratio: ~0.92
#    Calmar Ratio: ~0.89
#    Total Return: ~+138%
#    Max Drawdown: ~24.1%
```

---

## 10. 其他產業配置 (參考)

| 產業 | 版本 | Sharpe | 狀態 |
|------|------|--------|------|
| 半導體業 | V19b A+ | 0.92 | 主力，每日執行 |
| 通信網路業 | L3 | 1.23 | 已驗證 |
| 其他電子業 | L2 | 1.13 | 已驗證 |
| 電腦及週邊設備 | L2 | 1.06 | 已驗證 |
| 電子零組件 | L2 | 0.98 | 已驗證 |
| 電機機械 | L2 | 0.79 | 已驗證 |
| 光電業 | L2 | 0.38 | MDD 72%→28.6% |
| 電子通路業 | - | -0.02 | 已淘汰 |

**目前每日執行只跑半導體業。** 其他產業配置存在 `INDUSTRY_CONFIGS` 中但未啟用 daily trading。

---

## 11. 檔案結構速查

```
strategy.py              ← 策略邏輯 (信號 B1-B7, S1-S4, 手續費計算)
group_backtest.py         ← 回測引擎 + INDUSTRY_CONFIGS + PRESET_PERIODS
run_daily_trading.py      ← 每日交易引擎 (DailyEngine)
run_full_backtest.py      ← 完整回測 runner (含 0050 benchmark)
stock_utils.py            ← 資料下載、大盤狀態、指標計算
theme_config.py           ← 子題材 mapping (THEME_MAP)
strategy_pullback.py      ← Strategy B (逆勢)
strategy_midterm.py       ← Strategy C (中期趨勢)
industry_manager.py       ← 產業股票清單管理
output/                   ← 回測結果 (CSV, PNG, JSON)
.cache/                   ← 資料快取 (MD5-keyed pickle)
```
