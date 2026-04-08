import pandas as pd

# ============================================================
# 補充股票清單：twstock 資料庫缺漏的股票
# twstock v1.4.0 沒有收錄的上市/上櫃公司，在此手動補充
# 格式: { '代號': ('名稱', '產業', '市場') }
# ============================================================
SUPPLEMENTAL_STOCKS = {
    # ---- 手動補充 (持有中 / 歷史紀錄) ----
    '7750': ('新代', '其他電子業', '上市'),      # CNC控制器
    '7734': ('印能科技', '半導體業', '上櫃'),     # 半導體封裝測試設備
    '7769': ('鴻勁', '半導體業', '上市'),        # 半導體精密零組件

    # ---- 2026Q1 --check 掃描補充 (TWSE/TPEx 官方資料) ----
    # 半導體業
    '3135': ('凌航', '半導體業', '上市'),
    '3467': ('台灣精材', '半導體業', '上櫃'),
    '4749': ('新應材', '半導體業', '上櫃'),
    '6423': ('億而得', '半導體業', '上櫃'),
    '6720': ('久昌', '半導體業', '上櫃'),
    '6907': ('雅特力-KY', '半導體業', '上櫃'),
    '6909': ('創控', '半導體業', '上市'),
    '6962': ('奕力-KY', '半導體業', '上市'),
    '6996': ('力領科技', '半導體業', '上櫃'),
    '7704': ('明遠精密', '半導體業', '上櫃'),
    '7712': ('博盛半導體', '半導體業', '上櫃'),
    '7749': ('意騰-KY', '半導體業', '上市'),
    '7751': ('竑騰', '半導體業', '上櫃'),
    '7770': ('君曜', '半導體業', '上櫃'),
    '7810': ('捷創科技', '半導體業', '上櫃'),
    '8102': ('傑霖科技', '半導體業', '上櫃'),
    # 其他電子業
    '4585': ('達明', '其他電子業', '上市'),
    '6722': ('輝創', '其他電子業', '上市'),
    '6725': ('矽科宏晟', '其他電子業', '上櫃'),
    '6739': ('竹陞科技', '其他電子業', '上櫃'),
    '6903': ('巨漢', '其他電子業', '上櫃'),
    '7703': ('銳澤', '其他電子業', '上櫃'),
    '7728': ('光焱科技', '其他電子業', '上櫃'),
    '7792': ('安葆', '其他電子業', '上櫃'),
    # 電機機械
    '6982': ('大井泵浦', '電機機械', '上櫃'),
    '7642': ('昶瑞機電', '電機機械', '上櫃'),
    '7709': ('榮田', '電機機械', '上櫃'),

    # 未來發現 twstock 缺的股票，執行 python industry_lookup.py --check 即可
}


def get_all_companies():
    """
    獲取所有股票清單 (上市+上櫃)
    來源: twstock 資料庫 + SUPPLEMENTAL_STOCKS 補充清單
    回傳 DataFrame 欄位: Ticker, Code, Name, Industry, Market
    """
    try:
        import twstock

        data = []
        seen_codes = set()

        # ---- 主要來源: twstock.codes ----
        for code, info in twstock.codes.items():
            # 過濾條件：只抓「股票」且是「上市/上櫃」
            if info.type == '股票' and info.market in ['上市', '上櫃']:

                # 判斷後綴 (給 yfinance 用)
                suffix = ".TW" if info.market == "上市" else ".TWO"

                # 處理產業分類 (若無則填其他)
                industry = info.group if info.group else "其他"

                data.append({
                    'Ticker': f"{code}{suffix}",    # yfinance 用
                    'Code': str(code).strip(),      # 純代號
                    'Name': str(info.name).strip(), # 中文名稱
                    'Industry': industry,           # 產業
                    'Market': info.market
                })
                seen_codes.add(str(code).strip())

        # ---- 補充來源: SUPPLEMENTAL_STOCKS ----
        added_count = 0
        for code, (name, industry, market) in SUPPLEMENTAL_STOCKS.items():
            if code not in seen_codes:
                suffix = ".TW" if market == "上市" else ".TWO"
                data.append({
                    'Ticker': f"{code}{suffix}",
                    'Code': str(code).strip(),
                    'Name': name,
                    'Industry': industry,
                    'Market': market
                })
                seen_codes.add(code)
                added_count += 1

        if added_count > 0:
            pass  # 靜默補充，不印訊息（避免干擾 backtest 輸出）

        df = pd.DataFrame(data)
        return df

    except ImportError:
        print("⚠️ 錯誤: 請先安裝 twstock 套件 (pip install twstock)")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ 讀取股票清單失敗: {e}")
        return pd.DataFrame()


def list_industries(df_all):
    """列出所有不重複的產業名稱"""
    if df_all.empty: return []
    if 'Industry' in df_all.columns:
        return sorted(df_all['Industry'].unique().tolist())
    return []


def get_stocks_by_industry(industry_name):
    """
    根據產業名稱回傳股票列表
    回傳格式: [(Ticker, Name), ...]
    """
    df = get_all_companies()
    if df.empty: return []

    # 精確搜尋產業名稱
    target = df[df['Industry'] == industry_name]
    if target.empty:
        return []

    return list(zip(target['Ticker'], target['Name']))
