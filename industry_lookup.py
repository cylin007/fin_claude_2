#!/usr/bin/env python3
"""
🏭 產業 / 股票 查詢工具
用法:
  python industry_lookup.py              → 互動選單
  python industry_lookup.py -i 半導體業    → 查該產業所有股票
  python industry_lookup.py -i 半導體業 -p → 查該產業所有股票 + 即時現價
  python industry_lookup.py -s 2330       → 查該股票屬於哪個產業
  python industry_lookup.py -s 台積電      → 用名稱模糊搜尋
  python industry_lookup.py -l            → 列出所有產業 + 各產業股票數
  python industry_lookup.py -a            → 匯出全部產業×股票到 CSV
  python industry_lookup.py --check       → 檢查 twstock 缺漏的股票 (每季維護用)
"""

import sys
import os
import argparse
import pandas as pd
from industry_manager import get_all_companies, list_industries


def load_data():
    """載入全部股票資料"""
    print("⏳ 載入產業資料庫...")
    df = get_all_companies()
    if df.empty:
        print("❌ 無法載入資料，請確認已安裝 twstock (pip install twstock)")
        sys.exit(1)
    print(f"✅ 載入完成: {len(df)} 檔股票, {df['Industry'].nunique()} 個產業\n")
    return df


def show_all_industries(df):
    """列出所有產業 + 股票數量"""
    stats = df.groupby('Industry').agg(
        股票數=('Code', 'count'),
        上市=('Market', lambda x: (x == '上市').sum()),
        上櫃=('Market', lambda x: (x == '上櫃').sum()),
    ).sort_values('股票數', ascending=False)

    print(f"{'='*70}")
    print(f"🏭 全部產業列表 ({len(stats)} 個產業, 共 {len(df)} 檔股票)")
    print(f"{'='*70}")
    print(f"{'#':>3}  {'產業':<16} {'股票數':>6} {'上市':>6} {'上櫃':>6}")
    print(f"{'─'*50}")
    for i, (ind, row) in enumerate(stats.iterrows(), 1):
        print(f"{i:>3}  {ind:<16} {row['股票數']:>6} {row['上市']:>6} {row['上櫃']:>6}")
    print(f"{'─'*50}")
    print(f"{'':>3}  {'合計':<16} {stats['股票數'].sum():>6} {stats['上市'].sum():>6} {stats['上櫃'].sum():>6}")


def search_by_industry(df, keyword, with_price=False):
    """查詢某產業的所有股票"""
    # 精確匹配
    matches = df[df['Industry'] == keyword]

    # 若精確匹配無結果 → 模糊搜尋
    if matches.empty:
        matches = df[df['Industry'].str.contains(keyword, na=False)]

    if matches.empty:
        print(f"❌ 找不到包含「{keyword}」的產業")
        all_ind = list_industries(df)
        suggestions = [ind for ind in all_ind if keyword in ind or any(c in ind for c in keyword)]
        if suggestions:
            print(f"💡 你是不是要找: {', '.join(suggestions[:10])}")
        return

    industries_found = matches['Industry'].unique()

    # 如果需要現價，批次下載
    price_map = {}
    if with_price:
        all_tickers = matches['Ticker'].tolist()
        price_map = _batch_fetch_prices(all_tickers)

    for ind in sorted(industries_found):
        ind_df = matches[matches['Industry'] == ind].sort_values('Code')

        if with_price:
            # 統計
            ok = sum(1 for t in ind_df['Ticker'] if price_map.get(t, 0) > 0)
            fail = len(ind_df) - ok
            print(f"\n{'='*90}")
            print(f"🏭 {ind} ({len(ind_df)} 檔, ✅{ok} 有價 / ❌{fail} 無價)")
            print(f"{'='*90}")
            print(f"{'#':>3}  {'代號':<8} {'Ticker':<12} {'名稱':<12} {'市場':<6} {'現價':>10} {'狀態'}")
            print(f"{'─'*70}")
            for i, (_, row) in enumerate(ind_df.iterrows(), 1):
                price = price_map.get(row['Ticker'], 0)
                if price > 0:
                    status = ''
                    price_str = f"${price:>8.1f}"
                else:
                    status = '❌ 無資料'
                    price_str = f"{'---':>9}"
                print(f"{i:>3}  {row['Code']:<8} {row['Ticker']:<12} {row['Name']:<12} "
                      f"{row['Market']:<6} {price_str}  {status}")
        else:
            print(f"\n{'='*80}")
            print(f"🏭 {ind} ({len(ind_df)} 檔)")
            print(f"{'='*80}")
            print(f"{'#':>3}  {'代號':<10} {'Ticker':<12} {'名稱':<14} {'市場':<6}")
            print(f"{'─'*55}")
            for i, (_, row) in enumerate(ind_df.iterrows(), 1):
                print(f"{i:>3}  {row['Code']:<10} {row['Ticker']:<12} {row['Name']:<14} {row['Market']:<6}")


def _batch_fetch_prices(tickers):
    """批次抓取現價，回傳 {ticker: close_price}"""
    import yfinance as yf

    price_map = {}
    BATCH = 100

    total = len(tickers)
    print(f"\n⏳ 下載 {total} 檔現價...")

    for i in range(0, total, BATCH):
        batch = tickers[i:i + BATCH]
        print(f"   批次 {i // BATCH + 1}/{(total - 1) // BATCH + 1}: {len(batch)} 檔...",
              end='', flush=True)

        try:
            with open(os.devnull, 'w') as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    raw = yf.download(batch, period='5d', group_by='ticker',
                                      threads=True, progress=False)
                finally:
                    sys.stderr = old_stderr
        except Exception:
            raw = pd.DataFrame()

        ok = 0
        for ticker in batch:
            try:
                if len(batch) == 1:
                    df_t = raw
                else:
                    df_t = raw[ticker]

                if isinstance(df_t.columns, pd.MultiIndex):
                    df_t.columns = df_t.columns.get_level_values(-1)

                df_t = df_t.dropna(subset=['Close'])
                if len(df_t) > 0:
                    close = float(df_t['Close'].iloc[-1])
                    if close > 0:
                        price_map[ticker] = close
                        ok += 1
            except (KeyError, TypeError):
                pass

        print(f" ✅{ok} ❌{len(batch) - ok}")

    return price_map


def search_by_stock(df, keyword):
    """查詢某股票屬於哪個產業"""
    keyword = keyword.strip()

    # 先用代號精確匹配
    matches = df[df['Code'] == keyword]

    # 代號模糊 (例如只打 2330)
    if matches.empty:
        matches = df[df['Code'].str.contains(keyword, na=False)]

    # Ticker 匹配 (例如 2330.TW)
    if matches.empty:
        matches = df[df['Ticker'].str.contains(keyword, na=False, case=False)]

    # 名稱模糊搜尋
    if matches.empty:
        matches = df[df['Name'].str.contains(keyword, na=False)]

    if matches.empty:
        print(f"❌ 找不到「{keyword}」相關的股票")
        return

    print(f"\n🔍 搜尋「{keyword}」: 找到 {len(matches)} 筆")
    print(f"{'─'*75}")
    print(f"{'代號':<10} {'Ticker':<12} {'名稱':<14} {'產業':<16} {'市場':<6}")
    print(f"{'─'*75}")
    for _, row in matches.iterrows():
        print(f"{row['Code']:<10} {row['Ticker']:<12} {row['Name']:<14} "
              f"{row['Industry']:<16} {row['Market']:<6}")

    # 如果只有一筆，額外顯示同產業的其他股票
    if len(matches) == 1:
        ind = matches.iloc[0]['Industry']
        same_ind = df[df['Industry'] == ind]
        print(f"\n📋 同產業「{ind}」共 {len(same_ind)} 檔股票:")
        preview = same_ind.sort_values('Code').head(20)
        names = [f"{r['Code']} {r['Name']}" for _, r in preview.iterrows()]
        print(f"   {', '.join(names)}")
        if len(same_ind) > 20:
            print(f"   ... 還有 {len(same_ind) - 20} 檔 (用 -i {ind} 查看全部)")


def export_csv(df):
    """匯出全部資料到 CSV"""
    out_path = 'industry_all_stocks.csv'
    df_sorted = df.sort_values(['Industry', 'Code'])
    df_sorted.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已匯出 → {out_path} ({len(df_sorted)} 筆)")


# ==========================================
# 🔍 twstock 缺漏檢查 (每季維護用)
# ==========================================

# TWSE 產業代碼 → 產業名稱 對照表
_TWSE_INDUSTRY_MAP = {
    '01': '水泥工業', '02': '食品工業', '03': '塑膠工業', '04': '紡織纖維',
    '05': '電機機械', '06': '電器電纜', '08': '玻璃陶瓷', '09': '造紙工業',
    '10': '鋼鐵工業', '11': '橡膠工業', '12': '汽車工業', '14': '建材營造業',
    '15': '航運業', '16': '觀光餐旅', '17': '金融保險業', '18': '貿易百貨業',
    '20': '其他業', '21': '化學工業', '22': '生技醫療業', '23': '油電燃氣業',
    '24': '半導體業', '25': '電腦及週邊設備業', '26': '光電業', '27': '通信網路業',
    '28': '電子零組件業', '29': '電子通路業', '30': '資訊服務業', '31': '其他電子業',
    '35': '綠能環保', '36': '數位雲端', '37': '運動休閒', '38': '居家生活',
}

# TPEx 產業分類行情 se 代碼 → 產業名稱
_TPEX_SE_MAP = {
    '02': '食品工業', '03': '塑膠工業', '04': '紡織纖維',
    '05': '電機機械', '06': '電器電纜', '08': '玻璃陶瓷',
    '10': '鋼鐵工業', '11': '橡膠工業', '12': '汽車工業',
    '14': '建材營造業', '15': '航運業', '16': '觀光餐旅',
    '17': '金融保險業', '18': '貿易百貨業', '20': '其他',
    '21': '化學工業', '22': '生技醫療業', '23': '油電燃氣業',
    '24': '半導體業', '25': '電腦及週邊設備業', '26': '光電業',
    '27': '通信網路業', '28': '電子零組件業', '29': '電子通路業',
    '30': '資訊服務業', '31': '其他電子業',
    '35': '綠能環保', '36': '數位雲端', '37': '運動休閒', '38': '居家生活',
}


def _fetch_json(url, timeout=15, retries=3):
    """安全的 JSON 下載 (含 SSL 跳過 + 重試)"""
    import ssl
    import urllib.request
    import json as _json
    import time

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                return _json.loads(resp.read())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                raise e


def check_twstock_gaps():
    """
    檢查 twstock 是否有缺漏 (使用 TWSE / TPEx 官方 API)：
    1. 從 TWSE 上市公司基本資料 API 取得所有上市股票 + 產業分類
    2. 從 TPEx 上櫃分類行情 API 取得策略相關產業的上櫃股票
    3. 比對 twstock + SUPPLEMENTAL，找出遺漏
    4. 只報告策略用到的產業中的遺漏 (可選全部)
    """
    import twstock
    import datetime
    from industry_manager import SUPPLEMENTAL_STOCKS

    print("=" * 70)
    print("🔍 twstock 缺漏檢查工具 (每季維護用)")
    print("   資料來源: TWSE / TPEx 官方 API (非 yfinance 掃描)")
    print("=" * 70)

    print(f"\n📦 twstock 版本: {twstock.__version__}")
    # twstock 中的「股票」代號
    tw_stock_codes = set(
        c for c, info in twstock.codes.items()
        if info.type == '股票' and c.isdigit() and len(c) == 4
    )
    supp_codes = set(SUPPLEMENTAL_STOCKS.keys())
    known_codes = tw_stock_codes | supp_codes
    print(f"   twstock 股票代號: {len(tw_stock_codes)} 檔")
    print(f"   SUPPLEMENTAL_STOCKS: {len(supp_codes)} 檔")

    # ==========================================
    # 1. 上市 (TWSE) 缺漏檢查
    # ==========================================
    print(f"\n{'─'*70}")
    print("1️⃣  查詢 TWSE 上市公司基本資料...")
    print(f"{'─'*70}")

    twse_new = []  # 上市的 twstock 遺漏
    try:
        data = _fetch_json('https://openapi.twse.com.tw/v1/opendata/t187ap03_L')
        print(f"   TWSE API 回傳 {len(data)} 家上市公司")

        for d in data:
            code = d.get('公司代號', '').strip()
            if not code or not code.isdigit() or len(code) != 4:
                continue
            if code in known_codes:
                continue

            name = d.get('公司簡稱', '?').strip()
            industry_code = d.get('產業別', '?')
            industry = _TWSE_INDUSTRY_MAP.get(industry_code, f'代碼{industry_code}')
            is_innovation = name.endswith('-創') or name.endswith('KY創')

            twse_new.append({
                'code': code, 'name': name, 'industry': industry,
                'market': '上市', 'innovation': is_innovation,
            })

    except Exception as e:
        print(f"   ❌ TWSE API 失敗: {e}")

    # ==========================================
    # 2. 上櫃 (TPEx) 缺漏檢查 (用分類行情 API)
    # ==========================================
    print(f"\n{'─'*70}")
    print("2️⃣  查詢 TPEx 上櫃分類行情...")
    print(f"{'─'*70}")

    tpex_new = []  # 上櫃的 twstock 遺漏

    # 找最近有行情的交易日
    trade_date = None
    for offset in range(0, 14):
        dt = datetime.date.today() - datetime.timedelta(days=offset)
        tw_year = dt.year - 1911
        ds = f'{tw_year}/{dt.month:02d}/{dt.day:02d}'
        try:
            url = (f'https://www.tpex.org.tw/web/stock/aftertrading/'
                   f'otc_quotes_no1430/stk_wn1430_result.php?l=zh-tw&d={ds}&se=AL')
            resp = _fetch_json(url, timeout=10)
            tables = resp.get('tables', [])
            if tables and len(tables[0].get('data', [])) > 0:
                trade_date = ds
                break
        except Exception:
            continue

    if trade_date:
        print(f"   交易日: {trade_date}")

        # 掃所有產業的上櫃股票
        scanned = 0
        for se_code, ind_name in sorted(_TPEX_SE_MAP.items()):
            try:
                url = (f'https://www.tpex.org.tw/web/stock/aftertrading/'
                       f'otc_quotes_no1430/stk_wn1430_result.php?l=zh-tw&d={trade_date}&se={se_code}')
                resp = _fetch_json(url, timeout=10)
                tables = resp.get('tables', [])
                if not tables or not tables[0].get('data'):
                    continue

                for row in tables[0]['data']:
                    code = row[0].strip()
                    name = row[1].strip()
                    if not code.isdigit() or len(code) != 4:
                        continue
                    if code in known_codes:
                        continue

                    is_innovation = name.endswith('-創') or name.endswith('KY創')
                    tpex_new.append({
                        'code': code, 'name': name, 'industry': ind_name,
                        'market': '上櫃', 'innovation': is_innovation,
                    })
                    scanned += 1

            except Exception:
                pass

        print(f"   掃描完成, 發現 {scanned} 檔上櫃新股")
    else:
        print(f"   ⚠️ 找不到最近的交易日 (可能連假中)")

    # ==========================================
    # 3. 彙整結果 (聚焦策略產業)
    # ==========================================
    all_new = twse_new + tpex_new

    # 去重 (同代號)
    seen = set()
    unique_new = []
    for item in all_new:
        if item['code'] not in seen:
            seen.add(item['code'])
            unique_new.append(item)

    # 策略用到的產業 (從 daily_signal 的設定)
    STRATEGY_INDUSTRIES = {'半導體業', '其他電子業', '電機機械'}

    relevant = [x for x in unique_new if x['industry'] in STRATEGY_INDUSTRIES and not x['innovation']]
    relevant_innovation = [x for x in unique_new if x['industry'] in STRATEGY_INDUSTRIES and x['innovation']]
    other = [x for x in unique_new if x['industry'] not in STRATEGY_INDUSTRIES]

    print(f"\n{'─'*70}")
    print("3️⃣  策略產業缺漏 (需加入 SUPPLEMENTAL_STOCKS)")
    print(f"{'─'*70}")

    if relevant:
        print(f"\n   🔴 {len(relevant)} 檔一般股票 (影響策略選股池):")
        print(f"   {'代號':<8} {'名稱':<14} {'產業':<14} {'市場':<6}")
        print(f"   {'─'*50}")
        for item in sorted(relevant, key=lambda x: x['code']):
            print(f"   {item['code']:<8} {item['name']:<14} {item['industry']:<14} {item['market']:<6}")

        print(f"\n   💡 請在 industry_manager.py 的 SUPPLEMENTAL_STOCKS 新增:")
        for item in sorted(relevant, key=lambda x: x['code']):
            market_suffix = '上市' if item['market'] == '上市' else '上櫃'
            print(f"       '{item['code']}': ('{item['name']}', '{item['industry']}', '{market_suffix}'),")
    else:
        print(f"\n   ✅ 策略產業無遺漏 (一般股票)")

    if relevant_innovation:
        print(f"\n   🟡 {len(relevant_innovation)} 檔創新板 (不影響策略, 交易量通常太小):")
        for item in sorted(relevant_innovation, key=lambda x: x['code']):
            print(f"   {item['code']} {item['name']} ({item['industry']}, {item['market']})")

    # 其他產業的遺漏 (摘要)
    if other:
        other_industries = {}
        for item in other:
            ind = item['industry']
            if ind not in other_industries:
                other_industries[ind] = 0
            other_industries[ind] += 1
        print(f"\n   📋 其他產業遺漏合計 {len(other)} 檔:")
        for ind, cnt in sorted(other_industries.items(), key=lambda x: -x[1]):
            print(f"      {ind}: {cnt} 檔")

    # ==========================================
    # 4. 確認現有 SUPPLEMENTAL 狀態
    # ==========================================
    print(f"\n{'─'*70}")
    print("4️⃣  確認現有 SUPPLEMENTAL_STOCKS 狀態...")
    print(f"{'─'*70}")

    for code, (name, industry, market) in SUPPLEMENTAL_STOCKS.items():
        if code in tw_stock_codes:
            print(f"   ✅ {code} {name}: twstock 已收錄 → 可從 SUPPLEMENTAL 移除")
        else:
            # 看 TWSE/TPEx 是否有這隻
            found_in_api = False
            for item in all_new:
                if item['code'] == code:
                    found_in_api = True
                    break

            # 也可能 TWSE/TPEx 已有但 known_codes 排除了 → 不在 all_new 中
            # 直接查原始 API 結果
            print(f"   📝 {code} {name} ({industry}, {market}): twstock 仍缺 → 需保留")

    # ==========================================
    # 5. 總結
    # ==========================================
    print(f"\n{'='*70}")
    print("📋 總結")
    print(f"{'='*70}")
    print(f"   📊 全部遺漏: {len(unique_new)} 檔 (上市 {len(twse_new)}, 上櫃 {len(tpex_new)})")
    if relevant:
        print(f"   🔴 策略產業缺漏: {len(relevant)} 檔 ← 需處理!")
    else:
        print(f"   ✅ 策略產業無缺漏")
    if relevant_innovation:
        print(f"   🟡 策略產業創新板: {len(relevant_innovation)} 檔 (可忽略)")
    print(f"   📋 其他產業: {len(other)} 檔 (不影響當前策略)")
    print(f"   💡 建議每季執行: python industry_lookup.py --check")


def interactive_menu(df):
    """互動式選單"""
    while True:
        print(f"\n{'='*50}")
        print("🏭 產業 / 股票 查詢工具")
        print(f"{'='*50}")
        print("  1. 📋 列出所有產業 (含股票數)")
        print("  2. 🏭 查產業 → 看該產業所有股票")
        print("  3. 🏭 查產業 → 含即時現價 💰")
        print("  4. 🔍 查股票 → 看它屬於哪個產業")
        print("  5. 📄 匯出全部到 CSV")
        print("  6. 🔍 twstock 缺漏檢查 (每季維護)")
        print("  q. 離開")
        print()
        choice = input("👉 選擇 (1/2/3/4/5/6/q): ").strip()

        if choice == '1':
            show_all_industries(df)

        elif choice in ('2', '3'):
            with_price = (choice == '3')
            # 先列出產業清單方便選擇
            industries = list_industries(df)
            print(f"\n🏭 可選產業 ({len(industries)} 個):")
            for i, ind in enumerate(industries, 1):
                print(f"  [{i:>2}] {ind:<16}", end="")
                if i % 4 == 0:
                    print()
            if len(industries) % 4 != 0:
                print()
            print()
            inp = input("👉 輸入產業名稱或編號: ").strip()
            if inp.isdigit() and 1 <= int(inp) <= len(industries):
                search_by_industry(df, industries[int(inp) - 1], with_price=with_price)
            elif inp:
                search_by_industry(df, inp, with_price=with_price)

        elif choice == '4':
            inp = input("👉 輸入股票代號或名稱 (例: 2330, 台積電): ").strip()
            if inp:
                search_by_stock(df, inp)

        elif choice == '5':
            export_csv(df)

        elif choice == '6':
            check_twstock_gaps()

        elif choice.lower() == 'q':
            print("👋 掰掰")
            break


def main():
    parser = argparse.ArgumentParser(description='🏭 產業/股票查詢工具')
    parser.add_argument('-i', '--industry', type=str, help='查詢產業名稱 (列出該產業所有股票)')
    parser.add_argument('-p', '--price', action='store_true', help='附加即時現價 (搭配 -i 使用)')
    parser.add_argument('-s', '--stock', type=str, help='查詢股票代號或名稱 (顯示所屬產業)')
    parser.add_argument('-l', '--list', action='store_true', help='列出所有產業及股票數')
    parser.add_argument('-a', '--all', action='store_true', help='匯出全部產業×股票到 CSV')
    parser.add_argument('--check', action='store_true', help='twstock 缺漏檢查 (每季維護用)')
    args = parser.parse_args()

    # --check 不需要載入 df
    if args.check:
        check_twstock_gaps()
        return

    df = load_data()

    # 命令列模式
    if args.industry:
        search_by_industry(df, args.industry, with_price=args.price)
    elif args.stock:
        search_by_stock(df, args.stock)
    elif args.list:
        show_all_industries(df)
    elif args.all:
        export_csv(df)
    else:
        # 互動模式
        interactive_menu(df)


if __name__ == "__main__":
    main()
