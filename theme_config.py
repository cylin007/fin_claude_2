#!/usr/bin/env python3
"""
🏷️ 子題材動量 Boost (Theme Momentum Boost)

半導體及相關供應鏈的子題材分群 + 題材熱度計算。
用於在買入排序時，對處於「熱門題材」的股票給予 score boost。

維護說明:
  - THEME_MAP: 每月檢視一次，有新股上市/分類調整時更新
  - EXTRA_THEME_STOCKS: 跨產業股票，需要拉進半導體掃描池的
  - 未分類股票照常掃描，只是沒有 theme boost
"""

import numpy as np

# ============================================================
# 子題材 → 股票代碼 (純代號, 不含 .TW/.TWO)
# ============================================================
THEME_MAP = {
    # --- IC 設計 ---
    'AI_HPC設計': [
        '2454',  # 聯發科 (MediaTek) - 手機/AI SoC
        '3661',  # 世芯-KY (Alchip) - AI ASIC
        '3443',  # 創意 (GUC) - ASIC design service
        '5269',  # 祥碩 (ASMedia) - PCIe/USB 高速 IC
        '5274',  # 信驊 (Aspeed) - BMC/伺服器管理 IC
        '4966',  # 譜瑞-KY (Parade) - 高速介面 IC
        '6533',  # 晶心科 (Andes) - RISC-V IP
        '6643',  # M31 - 高速 IP
        '3035',  # 智原 (Faraday) - ASIC/IP
    ],
    '驅動IC': [
        '3034',  # 聯詠 (Novatek) - DDIC
        '3592',  # 瑞鼎 (Raydium) - DDIC
        '4961',  # 天鈺 - DDIC
        '3545',  # 敦泰 (FocalTech) - 觸控/DDIC
        '6462',  # 神盾 (Egis) - 指紋/觸控
        '8016',  # 矽創 (Sitronix) - DDIC
        '2458',  # 義隆 (ELAN) - 觸控
        '6962',  # 奕力-KY - DDIC
    ],
    '網通RF': [
        '2379',  # 瑞昱 (Realtek) - 網通/音效 IC
        '6526',  # 達發 (Airoha) - BT/WiFi
        '4968',  # 立積 (RichWave) - RF IC
        '5272',  # 笙科 - RF IC
        '3094',  # 聯傑 - 網通 IC
        '3169',  # 亞信 - 網通 IC
    ],
    'MCU': [
        '4919',  # 新唐 (Nuvoton) - MCU
        '6202',  # 盛群 (Holtek) - MCU
        '5471',  # 松翰 (Sonix) - MCU
        '6716',  # 應廣 - MCU
        '6494',  # 九齊 - MCU
        '3122',  # 笙泉 - MCU
        '6457',  # 紘康 - MCU
        '6907',  # 雅特力-KY - MCU
        '6909',  # 創控 - MCU
    ],
    '電源類比': [
        '6415',  # 矽力*-KY (Silergy) - PMIC
        '3588',  # 通嘉 (Leadtrend) - 電源 IC
        '6138',  # 茂達 (Anpec) - 電源 IC
        '8081',  # 致新 (GMT) - 電源 IC
        '6719',  # 力智 (uPI) - 電源 IC
        '5299',  # 杰力 - 電源 IC
        '8261',  # 富鼎 - MOSFET
        '6799',  # 來頡 - 電源 IC
        '2481',  # 強茂 (Panjit) - 二極體
        '5425',  # 台半 (TSC) - 分離式元件
        '3675',  # 德微 - 分離式元件
    ],

    # --- 製造 ---
    '晶圓代工': [
        '2330',  # 台積電 (TSMC)
        '2303',  # 聯電 (UMC)
        '5347',  # 世界 (Vanguard)
        '6770',  # 力積電 (PSMC)
    ],
    '封測': [
        '3711',  # 日月光投控 (ASE)
        '6239',  # 力成 (PTI)
        '8150',  # 南茂 (ChipMOS)
        '6147',  # 頎邦 (Chipbond) - COF
        '2441',  # 超豐 (Greatek)
        '6257',  # 矽格 (Sigurd)
        '3264',  # 欣銓 (Ardentec) - IC測試
        '2449',  # 京元電子 (KYEC) - IC測試
        '6552',  # 易華電
        '5285',  # 界霖 - 導線架
        '2329',  # 華泰 (OSE)
    ],

    # --- 設備/材料 ---
    '測試介面設備': [
        '6515',  # 穎崴 (MPI) - 探針卡
        '6223',  # 旺矽 (MJC) - 探針卡
        '7751',  # 竑騰 - 測試治具
        '6510',  # 精測 - 探針卡
        '3680',  # 家登 (Gudeng) - 光罩盒
        '3413',  # 京鼎 (Keystone) - 設備
        '3583',  # 辛耘 (Scientech) - 濕製程設備
        '6937',  # 天虹 - PVD 設備
        '6640',  # 均華 - 黏晶機
        '6261',  # 久元 - 測試分選
    ],

    # --- 記憶體 ---
    '記憶體': [
        '2337',  # 旺宏 (Macronix) - NOR Flash
        '2344',  # 華邦電 (Winbond) - DRAM/Flash
        '2408',  # 南亞科 (Nanya) - DRAM
        '8299',  # 群聯 (Phison) - Flash 控制器
        '4967',  # 十銓 (TeamGroup) - 記憶體模組
        '2451',  # 創見 (Transcend) - 記憶體模組
        '3260',  # 威剛 (ADATA) - 記憶體模組
        '8271',  # 宇瞻 (Apacer) - 記憶體模組
        '4973',  # 廣穎 (Silicon Power) - 記憶體模組
    ],

    # --- 材料 ---
    '矽晶圓': [
        '5483',  # 中美晶 (SAS)
        '6488',  # 環球晶 (GlobalWafers)
        '6182',  # 合晶 (Wafer Works)
        '3532',  # 台勝科 (SUMCO Taiwan)
        '3016',  # 嘉晶 (Epitech) - 磊晶
    ],
    '化合物半導體': [
        '3105',  # 穩懋 (Win Semi) - GaAs 代工
        '8086',  # 宏捷科 (VPEC) - GaAs PA
        '3707',  # 漢磊 (Episil) - SiC
        '4991',  # 環宇-KY - GaAs 基板
    ],

    # --- 光通訊 / CPO ---
    '光通訊': [
        '4960',  # 奇致科技 (Centera) - 光收發模組 ★跨產業
        '3714',  # 富采 (Ennostar) - 光電
        '2393',  # 億光 (Everlight) - LED/光通訊
        '6285',  # 啟碁 (WNC) - 網通光模組 ★跨產業
        '3583',  # 辛耘 - 光通訊設備
        '4113',  # 聯上光電 - 光通訊元件
    ],
    'CPO': [
        '3081',  # 聯亞 (LandMark) - InP 雷射晶粒 (CPO 核心)
        '2455',  # 全新 (Tong Hsing) - 光電封裝
        '6669',  # 緯穎 (Wiwynn) - AI 伺服器 (CPO 應用端) ★跨產業
        '3105',  # 穩懋 (Win Semi) - GaAs/InP 代工 (CPO 光源)
        '4968',  # 立積 (RichWave) - 光通訊 IC
    ],

    # --- PCB ---
    'PCB': [
        '2313',  # 華通 (Compeq) - HDI/AI 伺服器 PCB ★跨產業
        '3037',  # 欣興 (Unimicron) - ABF 載板 + PCB ★跨產業
        '8046',  # 南電 (Nanya PCB) - IC 載板 ★跨產業
        '5264',  # 鑫永銓 (SYQ) - 軟板 ★跨產業
        '6153',  # 嘉聯益 (Career) - FPC ★跨產業
        '8213',  # 志超 (Tripod) - PCB ★跨產業
        '3044',  # 健鼎 (Tripod) - PCB ★跨產業
        '6451',  # 訊芯-KY (Coretronic Optronics) - PCB
    ],

    # --- 台積設備/供應鏈 ---
    '台積設備供應鏈': [
        '3413',  # 京鼎 (Keystone) - 蝕刻腔體/設備 (台積指定)
        '3583',  # 辛耘 (Scientech) - 濕製程設備 (台積供應商)
        '6937',  # 天虹 - PVD 設備 (台積供應商)
        '3680',  # 家登 (Gudeng) - EUV 光罩盒 (台積獨家)
        '6515',  # 穎崴 (MPI) - 探針卡 (台積測試)
        '6223',  # 旺矽 (MJC) - 探針卡
        '3438',  # 類比科 (Anadigics) - 測試設備
        '4510',  # 高力 - 真空設備零件 ★跨產業
        '6640',  # 均華 - 黏晶機 (先進封裝)
        '3017',  # 奇鋐 (AVC) - 台積散熱 ★跨產業
    ],

    # --- 跨產業題材 (需搭配 EXTRA_THEME_STOCKS 拉進掃描池) ---
    '載板': [
        '3037',  # 欣興 (Unimicron) - ABF 載板 ★跨產業
        '8046',  # 南電 (Nanya PCB) - IC 載板 ★跨產業
        '3189',  # 景碩 (Kinsus) - IC 載板 (已在半導體池)
        '5439',  # 高技 - 載板 ★跨產業
    ],
    '散熱': [
        '3017',  # 奇鋐 (AVC) - 散熱模組 ★跨產業
        '3324',  # 雙鴻 (Auras) - 散熱模組 ★跨產業
        '6230',  # 尼得科超眾 - 熱管 ★跨產業
        '3653',  # 健策 (Kenner) - 均熱板 ★跨產業
        '2421',  # 建準 (Sunonwealth) - 風扇 ★跨產業
    ],
}

# ============================================================
# 跨產業股票 — 需拉進半導體掃描池
# 格式: (Ticker, Name)
# ============================================================
EXTRA_THEME_STOCKS = [
    # 載板 (原屬電子零組件業)
    ('3037.TW',  '欣興'),
    ('8046.TW',  '南電'),
    ('5439.TWO', '高技'),
    # 散熱 (原屬電腦週邊/其他電子/電子零組件)
    ('3017.TW',  '奇鋐'),
    ('3324.TWO', '雙鴻'),
    ('6230.TW',  '尼得科超眾'),
    ('3653.TW',  '健策'),
    ('2421.TW',  '建準'),
    # 光通訊
    ('4960.TWO', '奇致科技'),
    ('6285.TW',  '啟碁'),
    # CPO
    ('6669.TW',  '緯穎'),
    # PCB
    ('2313.TW',  '華通'),
    ('5264.TWO', '鑫永銓'),
    ('6153.TW',  '嘉聯益'),
    ('8213.TW',  '志超'),
    ('3044.TW',  '健鼎'),
    # 台積設備供應鏈
    ('4510.TW',  '高力'),
]


# ============================================================
# 反查表: 股票代碼 → 所屬題材 (啟動時自動建立)
# ============================================================
_CODE_TO_THEME = {}
for _theme, _codes in THEME_MAP.items():
    for _c in _codes:
        _CODE_TO_THEME[_c] = _theme


def get_stock_theme(code):
    """查詢股票所屬子題材，無則回傳 None"""
    # 去掉 .TW / .TWO 後綴
    pure = code.split('.')[0] if '.' in code else code
    return _CODE_TO_THEME.get(pure)


def compute_all_theme_scores(all_price_data, date, pool_tickers,
                             lookback=20, min_stocks=3):
    """
    計算所有題材的熱度分數

    Parameters
    ----------
    all_price_data : dict
        {ticker: DataFrame (有 'Close', 'Volume' 等欄位, DatetimeIndex)}
    date : str or Timestamp
        計算日期
    pool_tickers : list
        全部掃描池的 ticker 列表 (用來算整體平均報酬做 relative strength)
    lookback : int
        回看天數 (預設 20)
    min_stocks : int
        群內至少 N 檔有足夠數據才計算 (預設 3)

    Returns
    -------
    dict : {theme_name: float (0~1)}
    """
    import pandas as pd
    date = pd.Timestamp(date)

    # --- 1. 算全池平均報酬 (做 relative strength 的基準) ---
    pool_returns = []
    for tk in pool_tickers:
        r = _calc_stock_return(all_price_data.get(tk), date, lookback)
        if r is not None:
            pool_returns.append(r)
    pool_avg_ret = np.mean(pool_returns) if pool_returns else 0.0

    # --- 2. 每個題材算兩個指標 ---
    raw_scores = {}
    raw_rs = {}  # 存 relative_strength 供後續 normalize

    for theme, codes in THEME_MAP.items():
        # 找到有數據的股票
        theme_tickers = _codes_to_tickers(codes, all_price_data)
        if len(theme_tickers) < min_stocks:
            continue

        above_ma20 = 0
        total_valid = 0
        theme_returns = []

        for tk in theme_tickers:
            df = all_price_data.get(tk)
            if df is None or df.empty:
                continue
            valid = df.index[df.index <= date]
            if len(valid) < lookback:
                continue

            close = float(df.loc[valid[-1], 'Close'])
            ma20 = float(df.loc[valid[-lookback:]]['Close'].mean())

            total_valid += 1
            if close > ma20:
                above_ma20 += 1

            ret = _calc_stock_return(df, date, lookback)
            if ret is not None:
                theme_returns.append(ret)

        if total_valid < min_stocks:
            continue

        # 指標 1: MA20 breadth (站上 MA20 比例)
        breadth = above_ma20 / total_valid

        # 指標 2: Relative strength (vs 全池)
        theme_avg_ret = np.mean(theme_returns) if theme_returns else 0.0
        rs = theme_avg_ret - pool_avg_ret

        raw_scores[theme] = breadth
        raw_rs[theme] = rs

    # --- 3. Normalize relative strength 到 0~1 ---
    if not raw_rs:
        return {}

    rs_values = list(raw_rs.values())
    rs_min = min(rs_values)
    rs_max = max(rs_values)
    rs_range = rs_max - rs_min

    theme_scores = {}
    for theme in raw_scores:
        breadth = raw_scores[theme]
        if rs_range > 1e-8:
            rs_norm = (raw_rs[theme] - rs_min) / rs_range
        else:
            rs_norm = 0.5  # 所有題材強度一樣 → 中性

        # 合成: breadth 50% + relative_strength 50%
        theme_scores[theme] = breadth * 0.5 + rs_norm * 0.5

    return theme_scores


def compute_theme_returns(all_price_data, date, lookback=20, min_stocks=3):
    """
    計算每個題材的 20 日絕對報酬率 (%)

    Returns
    -------
    dict : {theme_name: float (%)}  正值=題材上漲, 負值=題材下跌
    """
    import pandas as pd
    date = pd.Timestamp(date)

    theme_rets = {}
    for theme, codes in THEME_MAP.items():
        theme_tickers = _codes_to_tickers(codes, all_price_data)
        if len(theme_tickers) < min_stocks:
            continue
        returns = []
        for tk in theme_tickers:
            r = _calc_stock_return(all_price_data.get(tk), date, lookback)
            if r is not None:
                returns.append(r)
        if len(returns) >= min_stocks:
            theme_rets[theme] = float(np.mean(returns))
    return theme_rets


def _calc_stock_return(df, date, lookback):
    """計算單檔股票的 lookback 日報酬率 (%)"""
    if df is None or df.empty:
        return None
    valid = df.index[df.index <= date]
    if len(valid) < lookback:
        return None
    close_now = float(df.loc[valid[-1], 'Close'])
    close_prev = float(df.loc[valid[-lookback], 'Close'])
    if close_prev <= 0:
        return None
    return (close_now - close_prev) / close_prev * 100


def _codes_to_tickers(codes, all_price_data):
    """
    把純代碼 list 轉成 all_price_data 裡找得到的 ticker list。
    嘗試 .TW 和 .TWO 兩種後綴。
    """
    result = []
    for c in codes:
        for suffix in ['.TW', '.TWO']:
            tk = c + suffix
            if tk in all_price_data:
                result.append(tk)
                break
    return result
