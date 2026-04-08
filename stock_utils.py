import yfinance as yf
import pandas as pd
import sys
import os
import pickle
import hashlib
import time

def get_stock_data(stock_id, period="1y"):
    """
    [一般個股] 自動判斷上市(.TW)或上櫃(.TWO)，並強制靜音錯誤訊息
    修復: yfinance 多層索引問題
    """
    stock_id = str(stock_id).strip().upper()
    
    final_df = pd.DataFrame()
    
    candidates = []
    if ".TW" in stock_id or ".TWO" in stock_id:
        candidates.append(stock_id)
    else:
        candidates.append(f"{stock_id}.TW")
        candidates.append(f"{stock_id}.TWO")

    for ticker in candidates:
        try:
            with open(os.devnull, 'w') as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    df = yf.download(ticker, period=period, interval="1d", progress=False)
                finally:
                    sys.stderr = old_stderr
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if not df.empty and len(df) > 0:
                final_df = df
                break 
        except Exception:
            continue

    return final_df

def get_index_data(symbol, period="1y", max_retries=3):
    """
    [大盤專用] 下載指數資料 (含重試機制)
    V6.5: 加入重試，避免 Yahoo Finance 偶發失敗導致誤判
    """
    import time as _time
    for attempt in range(max_retries):
        try:
            with open(os.devnull, 'w') as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    df = yf.download(symbol, period=period, interval="1d", progress=False)
                finally:
                    sys.stderr = old_stderr

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if not df.empty:
                return df

            # 空的結果，重試
            if attempt < max_retries - 1:
                _time.sleep(2)
        except Exception:
            if attempt < max_retries - 1:
                _time.sleep(2)

    return pd.DataFrame()

def evaluate_single_index(df, name, target_date_str=None):
    """
    評估單一指數狀態
    ✅ 新增: cum_3d_change (3日累計漲跌幅, 用於恐慌偵測)
    """
    empty_result = {
        'name': name, 'price': 0, 'change_pct': 0, 'trend': 'unknown',
        'bias_pct': 0, 'is_crash': False, 'trend_msg': '無資料',
        'date': 'N/A', 'cum_3d_change': 0
    }
    
    if df.empty:
        return empty_result

    if target_date_str:
        try:
            target_ts = pd.Timestamp(target_date_str)
            # 使用 asof() 取得 <= target_date 的最近交易日 (比 searchsorted + while 更安全)
            actual_date = df.index.asof(target_ts)
            if pd.isna(actual_date):
                # target_date 早於所有資料，無法評估
                return empty_result
            idx = df.index.get_loc(actual_date)
            
            current_row = df.iloc[idx]
            current_date = df.index[idx]
            
            history = df.iloc[:idx+1]
            ma20 = history['Close'].tail(20).mean()
            ma60 = history['Close'].tail(60).mean()
            prev_close = float(df.iloc[idx-1]['Close']) if idx > 0 else float(current_row['Close'])
            
            # ✅ 3日前收盤價
            close_3d_ago = float(df.iloc[idx-3]['Close']) if idx >= 3 else float(current_row['Close'])
        except (ValueError, TypeError, KeyError, IndexError):
            current_row = df.iloc[-1]
            current_date = df.index[-1]
            ma20 = df['Close'].tail(20).mean()
            ma60 = df['Close'].tail(60).mean()
            prev_close = float(df.iloc[-2]['Close'])
            close_3d_ago = float(df.iloc[-4]['Close']) if len(df) >= 4 else float(df.iloc[0]['Close'])
    else:
        current_row = df.iloc[-1]
        current_date = df.index[-1]
        ma20 = df['Close'].tail(20).mean()
        ma60 = df['Close'].tail(60).mean()
        prev_close = float(df.iloc[-2]['Close'])
        close_3d_ago = float(df.iloc[-4]['Close']) if len(df) >= 4 else float(df.iloc[0]['Close'])

    price = float(current_row['Close'])
    change_pct = (price - prev_close) / prev_close
    bias_pct = (price - ma20) / ma20
    
    # ✅ 3日累計漲跌幅
    cum_3d_change = (price - close_3d_ago) / close_3d_ago if close_3d_ago > 0 else 0
    
    trend = "neutral"
    trend_msg = "盤整"
    
    if price < ma20 and price < ma60:
        trend = "bear"
        trend_msg = "空頭排列 (破月季線)"
    elif price < ma20:
        trend = "weak"
        trend_msg = "短線轉弱 (破月線)"
    elif price > ma20 and price > ma60:
        trend = "bull"
        trend_msg = "多頭排列"
    
    # 修正: 門檻從 -1.5% 調整為 -2.5%，避免正常回檔就觸發偏空
    is_crash = (change_pct < -0.025) or (price < ma20 and bias_pct < -0.02)

    return {
        'name': name,
        'price': price,
        'change_pct': change_pct,
        'trend': trend,
        'bias_pct': bias_pct,
        'is_crash': is_crash,
        'trend_msg': trend_msg,
        'date': current_date.strftime('%Y-%m-%d'),
        'cum_3d_change': cum_3d_change,
    }

def get_market_status(target_date_str=None):
    """
    [雙引擎] 同時看加權指數(TWII) & 櫃買指數(TPEX)
    ✅ 新增: is_overheated (大盤過熱), is_panic (大盤恐慌)
    ✅ 修正: 櫃買指數嘗試多個符號 (Yahoo Finance 對 ^TWO 支援不穩定)
    """
    twii_df = get_index_data("^TWII", period="2y")
    
    # 櫃買指數: 嘗試多個符號 (^TWOII 為 2026 實測可用)
    otc_df = pd.DataFrame()
    for otc_symbol in ["^TWOII", "^TWO", "^TWOTCI", "OTCI.TW"]:
        otc_df = get_index_data(otc_symbol, period="2y")
        if not otc_df.empty:
            break
    
    twii_status = evaluate_single_index(twii_df, "加權指數", target_date_str)
    otc_status = evaluate_single_index(otc_df, "櫃買指數", target_date_str)

    actual_date = twii_status['date']
    twii_available = twii_status.get('price', 0) > 0
    otc_available = otc_status.get('price', 0) > 0

    # --- 安全性修正 (V6.5): 資料缺失時預設為「不安全」---
    # 避免 Yahoo Finance 下載失敗導致誤判為安全而放出買進訊號
    if not twii_available:
        print("   ⛔ 加權指數無資料！預設為不安全 (禁止買進)")
        is_unsafe = True
    else:
        is_twii_bad = (twii_status['trend'] in ['bear', 'weak']) or twii_status['is_crash']
        is_otc_bad = (otc_status['trend'] in ['bear', 'weak']) or otc_status['is_crash']
        # 櫃買無資料時，只看加權 (但會印警告)
        is_unsafe = is_twii_bad or (is_otc_bad if otc_available else False)
        if not otc_available:
            print("   ⚠️ 櫃買無資料，僅依加權判斷 (風險較高)")
    
    # --- ✅ 新增: 大盤過熱 ---
    # 加權指數月線乖離率 > 8% → 市場過熱，不宜追買
    twii_bias = twii_status.get('bias_pct', 0)
    is_overheated = twii_bias > 0.08
    
    # --- ✅ 新增: 大盤恐慌 ---
    # 條件 1: 加權單日跌幅 > 3%
    # 條件 2: 加權 3 日累計跌幅 > 5%
    twii_day_change = twii_status.get('change_pct', 0)
    twii_3d_change = twii_status.get('cum_3d_change', 0)
    is_panic = (twii_day_change < -0.03) or (twii_3d_change < -0.035)
    
    return {
        'date': actual_date,
        'is_unsafe': is_unsafe,
        'is_overheated': is_overheated,
        'is_panic': is_panic,
        'otc_available': otc_available,
        'twii': twii_status,
        'otc': otc_status,
    }


# ==========================================
# 🖨️ 大盤狀態統一顯示 (消除三處重複)
# ==========================================
def print_market_status(market_status, target_date_str=None):
    """
    統一顯示大盤狀態，供 main.py / daily_scanner.py / group_scanner.py 共用。
    """
    twii = market_status.get('twii', {})
    otc = market_status.get('otc', {})
    otc_ok = market_status.get('otc_available', False)
    date_label = target_date_str or market_status.get('date', '')

    print(f"\n🌏 大盤狀態 ({date_label}):")
    if twii and twii.get('price', 0) > 0:
        twii_chg = twii.get('change_pct', 0) * 100
        twii_bias = twii.get('bias_pct', 0) * 100
        print(f"   加權: {twii['price']:.0f} ({twii_chg:+.2f}%) {twii.get('trend_msg', '')}"
              f" | 月線乖離 {twii_bias:+.1f}%")
    if otc_ok:
        otc_chg = otc.get('change_pct', 0) * 100
        print(f"   櫃買: {otc['price']:.0f} ({otc_chg:+.2f}%) {otc.get('trend_msg', '')}")
    else:
        print(f"   櫃買: ⚠️ 無資料 (Yahoo Finance 櫃買指數暫時無法取得)")

    status_tags = []
    if market_status.get('is_unsafe'):     status_tags.append("⚠️ 趨勢偏空")
    if market_status.get('is_overheated'): status_tags.append("🔥 過熱")
    if market_status.get('is_panic'):      status_tags.append("🔴 恐慌")
    if not status_tags:                    status_tags.append("✅ 安全")

    reasons = []
    if market_status.get('is_unsafe'):
        if twii.get('trend') in ['bear', 'weak']:
            reasons.append(f"加權{twii.get('trend_msg', '')}")
        if otc_ok and otc.get('trend') in ['bear', 'weak']:
            reasons.append(f"櫃買{otc.get('trend_msg', '')}")
    if market_status.get('is_overheated'):
        reasons.append(f"加權月線乖離 {twii.get('bias_pct', 0)*100:.1f}% > 8%")
    if market_status.get('is_panic'):
        reasons.append(f"加權跌幅劇烈")
    if not reasons:
        reasons.append("加權多頭排列，無異常訊號")

    print(f"   狀態: {' / '.join(status_tags)}")
    print(f"   原因: {'; '.join(reasons)}")


# ==========================================
# 📦 info 字典統一組裝 (消除三處重複)
# ==========================================
def build_info_dict(history_df, sim_price=None, sim_vol=None):
    """
    從 history_df 組裝策略所需的 info 字典。
    可選傳入 sim_price / sim_vol 覆蓋真實值（用於模擬模式）。

    回傳: dict with keys: close, volume, high, open, ma20, ma60, vol_ma5, prev_close
    若資料不足回傳 None。
    """
    if history_df is None or len(history_df) < 2:
        return None

    try:
        real_price = float(history_df['Close'].iloc[-1])
        real_vol = float(history_df['Volume'].iloc[-1])
        prev_close = float(history_df['Close'].iloc[-2])

        # 價格有效性檢查：close <= 0 視為無效資料
        if real_price <= 0 or prev_close <= 0:
            return None

        current_price = sim_price if sim_price is not None else real_price
        current_vol = sim_vol if sim_vol is not None else real_vol

        real_high = float(history_df['High'].iloc[-1])
        real_open = float(history_df['Open'].iloc[-1])

        # 模擬價格同步: 如果模擬價 > 真實最高價，high 也更新
        high_price = max(real_high, current_price) if sim_price is not None else real_high

        # MA 計算: 如果有模擬價格，替換最後一筆後重算
        if sim_price is not None:
            close_series = history_df['Close'].copy()
            close_series.iloc[-1] = sim_price
            ma20 = close_series.tail(20).mean()
            ma60 = close_series.tail(60).mean()
        else:
            ma20 = history_df['Close'].tail(20).mean()
            ma60 = history_df['Close'].tail(60).mean()

        # 量能 MA: 如果有模擬量能，替換最後一筆後重算
        if sim_vol is not None:
            vol_series = history_df['Volume'].copy()
            vol_series.iloc[-1] = sim_vol
            vol_ma5 = vol_series.tail(5).mean()
        else:
            vol_ma5 = history_df['Volume'].tail(5).mean()

        return {
            'close': current_price,
            'volume': current_vol,
            'high': high_price,
            'open': real_open,
            'ma20': ma20,
            'ma60': ma60,
            'vol_ma5': vol_ma5,
            'prev_close': prev_close,
        }
    except (ValueError, TypeError, KeyError, IndexError):
        return None


# ==========================================
# 🚀 批次下載 + 本地快取 (加速回測用)
# ==========================================
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


def _make_cache_key(tickers, start_date, end_date):
    """根據股票清單+日期產生唯一 cache key"""
    raw = f"{sorted(tickers)}|{start_date}|{end_date}"
    return hashlib.md5(raw.encode()).hexdigest()


def _get_cache_info(cache_file):
    """取得快取檔案的基本資訊，回傳 (stock_data, skipped, mtime) 或 None"""
    if not os.path.exists(cache_file):
        return None
    try:
        mtime = os.path.getmtime(cache_file)
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        return cached['stock_data'], cached['skipped'], mtime
    except Exception:
        return None


def check_cache_vs_latest(stock_list, start_date, end_date, sample_n=5):
    """
    比較快取資料 vs Yahoo Finance 最新資料，列出差異。
    只抽樣 sample_n 檔做比對（避免全量下載太慢）。

    回傳:
        cache_info: dict with cache status, or None if no cache
    """
    import datetime as _dt

    os.makedirs(CACHE_DIR, exist_ok=True)
    tickers_only = [t for t, _ in stock_list]
    cache_key = _make_cache_key(tickers_only, start_date, end_date)
    cache_file = os.path.join(CACHE_DIR, f"batch_{cache_key}.pkl")

    info = _get_cache_info(cache_file)
    if info is None:
        return None

    stock_data, skipped, mtime = info
    cache_time = _dt.datetime.fromtimestamp(mtime)
    age_hours = (time.time() - mtime) / 3600

    result = {
        'cache_file': cache_file,
        'cache_time': cache_time.strftime('%Y-%m-%d %H:%M'),
        'age_hours': age_hours,
        'n_stocks': len(stock_data),
        'skipped': skipped,
        'sample_diffs': [],
        'data_match': True,
    }

    # V4.22: 空快取直接標記為不一致，需要重新下載
    if len(stock_data) == 0:
        result['data_match'] = False
        result['sample_diffs'] = [{'error': '快取為空 (0 檔)，需要重新下載'}]
        return result

    # 抽樣比對
    cached_tickers = list(stock_data.keys())
    import random
    sample_tickers = random.sample(cached_tickers, min(sample_n, len(cached_tickers)))

    try:
        with open(os.devnull, 'w') as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                fresh_df = yf.download(
                    sample_tickers, start=start_date, end=end_date,
                    group_by='ticker', threads=True, progress=False,
                )
            finally:
                sys.stderr = old_stderr
    except Exception:
        result['sample_diffs'] = [{'error': '無法下載比對資料'}]
        return result

    for ticker in sample_tickers:
        cached_df = stock_data[ticker]['df']
        try:
            if len(sample_tickers) == 1:
                fresh_single = fresh_df.copy()
            else:
                fresh_single = fresh_df[ticker].copy()
            if isinstance(fresh_single.columns, pd.MultiIndex):
                fresh_single.columns = fresh_single.columns.get_level_values(-1)
            fresh_single = fresh_single.dropna(subset=['Close'])
        except (KeyError, TypeError):
            continue

        # 比較行數
        cached_rows = len(cached_df)
        fresh_rows = len(fresh_single)

        # V4.22 fix: yfinance 回傳 0 行 = API 暫時失敗，不應判為快取過期
        if fresh_rows == 0 and cached_rows > 0:
            result['sample_diffs'].append({
                'error': f'無法下載比對資料 ({ticker}，yfinance 暫時無回應)'
            })
            continue

        # 比較最後日期
        cached_last = cached_df.index[-1].strftime('%Y-%m-%d') if len(cached_df) > 0 else 'N/A'
        fresh_last = fresh_single.index[-1].strftime('%Y-%m-%d') if len(fresh_single) > 0 else 'N/A'

        # 比較重疊日期的 Close 差異
        common = cached_df.index.intersection(fresh_single.index)
        if len(common) > 0:
            close_diff = (cached_df.loc[common, 'Close'] - fresh_single.loc[common, 'Close']).abs()
            max_diff = float(close_diff.max())
            n_diff = int((close_diff > 0.01).sum())
        else:
            max_diff = -1
            n_diff = -1

        diff_info = {
            'ticker': ticker,
            'cached_rows': cached_rows,
            'fresh_rows': fresh_rows,
            'cached_last': cached_last,
            'fresh_last': fresh_last,
            'max_close_diff': max_diff,
            'n_rows_diff': fresh_rows - cached_rows,
        }
        result['sample_diffs'].append(diff_info)

        if fresh_rows != cached_rows or n_diff > 0:
            result['data_match'] = False

    return result


def print_cache_check(cache_info):
    """印出快取檢查結果"""
    if cache_info is None:
        print("   📦 無快取，將重新下載")
        return

    print(f"   📦 快取狀態:")
    print(f"      建立時間: {cache_info['cache_time']} ({cache_info['age_hours']:.1f} 小時前)")
    print(f"      有效標的: {cache_info['n_stocks']} 檔")

    if cache_info['sample_diffs']:
        has_diff = not cache_info['data_match']
        if has_diff:
            print(f"      🔴 抽樣比對發現差異:")
        else:
            print(f"      ✅ 抽樣比對一致 (抽 {len(cache_info['sample_diffs'])} 檔):")

        for d in cache_info['sample_diffs']:
            if 'error' in d:
                print(f"         ⚠️ {d['error']}")
                continue
            status = "✅" if d['n_rows_diff'] == 0 and d['max_close_diff'] < 0.01 else "🔴"
            extra = ""
            if d['n_rows_diff'] != 0:
                extra += f" 行數差{d['n_rows_diff']:+d}"
            if d['max_close_diff'] >= 0.01:
                extra += f" Close差{d['max_close_diff']:.2f}"
            print(f"         {status} {d['ticker']}: cache={d['cached_rows']}行/{d['cached_last']}"
                  f" vs 最新={d['fresh_rows']}行/{d['fresh_last']}{extra}")


def batch_download_stocks(stock_list, start_date, end_date, min_data_days=60,
                          force_refresh=False, cache_ttl_hours=None):
    """
    批次下載多檔股票，自動快取到本地 .cache/ 目錄。

    Args:
        stock_list:    [(ticker, name), ...] 股票清單 (ticker 不含 .TW/.TWO)
        start_date:    下載起始日 (含前置 MA 計算期)
        end_date:      下載結束日
        min_data_days: 最少要有幾天資料才算有效
        force_refresh: True → 強制重新下載 (忽略快取)
        cache_ttl_hours: 快取有效時間 (小時), 超過自動重新下載; None=不限 (V4.22)

    Returns:
        stock_data: {ticker: {'df': DataFrame, 'name': str}}
        skipped:    {'download_fail': int, 'low_data': int}
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    tickers_only = [t for t, _ in stock_list]
    cache_key = _make_cache_key(tickers_only, start_date, end_date)
    cache_file = os.path.join(CACHE_DIR, f"batch_{cache_key}.pkl")

    # --- 檢查快取 ---
    if not force_refresh and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            stock_data = cached['stock_data']
            skipped = cached['skipped']
            file_age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600

            # V4.22: 空快取視為無效 — 之前下載全失敗留下的殘留
            if len(stock_data) == 0:
                print(f"   ⚠️ 快取為空 (0 檔)，視為無效，重新下載...")
            # V4.22: TTL 檢查 — 超過有效時間自動重新下載
            elif cache_ttl_hours is not None and file_age_hours > cache_ttl_hours:
                print(f"   ⏰ 快取已過期 ({file_age_hours:.1f}小時 > TTL {cache_ttl_hours}小時)，重新下載...")
            else:
                print(f"   ⚡ 使用本地快取 ({len(stock_data)} 檔, {file_age_hours:.1f}小時前)")
                return stock_data, skipped
        except Exception:
            pass  # 快取損壞，重新下載

    # --- 分類: 依後綴分組，未帶後綴的先歸 .TW ---
    name_map = {t: n for t, n in stock_list}
    tw_group = []    # (原始ticker, yf_ticker) - 上市
    two_group = []   # (原始ticker, yf_ticker) - 上櫃
    bare_group = []  # (原始ticker, yf_ticker) - 未帶後綴，先試 .TW

    for ticker in tickers_only:
        t_upper = ticker.upper()
        if t_upper.endswith('.TWO'):
            two_group.append((ticker, ticker))
        elif t_upper.endswith('.TW'):
            tw_group.append((ticker, ticker))
        else:
            bare_group.append((ticker, f"{ticker}.TW"))

    stock_data = {}
    skipped = {'download_fail': 0, 'low_data': 0}

    def _batch_download_and_parse(ticker_pairs, label):
        """批次下載一組 ticker，回傳成功的 stock_data 和失敗的原始 ticker 清單"""
        if not ticker_pairs:
            return {}, []
        yf_tickers = [yf_t for _, yf_t in ticker_pairs]
        print(f"   📡 批次下載 {len(yf_tickers)} 檔 ({label})...")
        try:
            with open(os.devnull, 'w') as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    raw_df = yf.download(
                        yf_tickers,
                        start=start_date, end=end_date,
                        group_by='ticker', threads=True, progress=False,
                    )
                finally:
                    sys.stderr = old_stderr
        except Exception:
            raw_df = pd.DataFrame()

        result = {}
        failed = []
        for orig_ticker, yf_ticker in ticker_pairs:
            try:
                if len(yf_tickers) == 1:
                    df = raw_df.copy()
                else:
                    df = raw_df[yf_ticker].copy()

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(-1)

                df = df.dropna(subset=['Close'])
                if df.empty or len(df) < min_data_days + 1:
                    failed.append(orig_ticker)
                    continue

                result[orig_ticker] = {'df': df, 'name': name_map[orig_ticker]}
            except (KeyError, TypeError):
                failed.append(orig_ticker)
        return result, failed

    # --- 第一輪: 下載已確定後綴的 .TW 和 .TWO ---
    all_first_round = tw_group + two_group + bare_group
    if all_first_round:
        data1, failed1 = _batch_download_and_parse(all_first_round, ".TW + .TWO")
        stock_data.update(data1)
    else:
        failed1 = []

    # --- 第二輪: bare_group 中失敗的改試 .TWO ---
    bare_tickers_set = {t for t, _ in bare_group}
    retry_two = [(t, f"{t}.TWO") for t in failed1 if t in bare_tickers_set]
    truly_failed = [t for t in failed1 if t not in bare_tickers_set]

    if retry_two:
        data2, failed2 = _batch_download_and_parse(retry_two, ".TWO 重試")
        stock_data.update(data2)
        truly_failed.extend(failed2)

    for t in truly_failed:
        skipped['download_fail'] += 1

    # --- 寫入快取 (V4.22 fix: 下載全失敗時保留舊快取) ---
    if len(stock_data) > 0:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'stock_data': stock_data, 'skipped': skipped}, f)
        except Exception:
            pass
    else:
        # 全部下載失敗 → 不覆蓋，嘗試回退到舊快取
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                old_data = cached.get('stock_data', {})
                if len(old_data) > 0:
                    print(f"   ⚠️ 重新下載全部失敗，保留舊快取 ({len(old_data)} 檔)")
                    return old_data, cached.get('skipped', skipped)
            except Exception:
                pass

    return stock_data, skipped


# ==========================================
# 🔍 除權息驗證工具 (Dividend/Split Adjustment Check)
# ==========================================
def check_dividend_adjustment(stock_data, sample_n=10, threshold_pct=1.0):
    """
    驗證 yfinance 下載的數據是否有除權息調整問題。

    原理:
      yfinance 回傳 Close (原始收盤價) 和 Adj Close (還原權息後的調整價)。
      如果兩者差距大，代表有除權息事件。
      若策略使用未調整的 Close 計算 MA，除權息當天會出現假的跌破均線訊號。

    做法:
      1. 抽樣 N 檔股票
      2. 比較每檔的 Close vs Adj Close
      3. 找出差異 > threshold_pct% 的日期 (即除權息日)
      4. 報告影響程度

    Args:
        stock_data:    {ticker: {'df': DataFrame, 'name': str}}
        sample_n:      抽樣幾檔 (預設10)
        threshold_pct: Close vs Adj Close 差異超過此 % 才視為除權息日 (預設1%)

    Returns:
        dict with:
            'has_issue':      True/False (是否發現未調整的問題)
            'affected_stocks': [{ticker, name, n_events, max_diff_pct, dates}, ...]
            'clean_stocks':   無問題的股票清單
            'sample_size':    抽樣數量
    """
    import random

    tickers = list(stock_data.keys())
    sample = random.sample(tickers, min(sample_n, len(tickers)))

    affected = []
    clean = []

    for ticker in sample:
        df = stock_data[ticker]['df']
        name = stock_data[ticker].get('name', ticker)

        # 確認是否有 Adj Close 欄位
        if 'Adj Close' not in df.columns:
            adj_col = None
            for col in df.columns:
                if 'adj' in col.lower():
                    adj_col = col
                    break
            if adj_col is None:
                clean.append({'ticker': ticker, 'name': name, 'note': '無 Adj Close 欄位'})
                continue
        else:
            adj_col = 'Adj Close'

        # 計算每日 Close vs Adj Close 差異百分比
        close = df['Close'].astype(float)
        adj_close = df[adj_col].astype(float)

        # 避免除以零
        valid_mask = adj_close.abs() > 0.01
        diff_pct = ((close[valid_mask] - adj_close[valid_mask]) / adj_close[valid_mask] * 100).abs()
        diff_pct = diff_pct.dropna()

        # 找出差異超過門檻的日期
        event_mask = diff_pct > threshold_pct
        n_events = int(event_mask.sum())

        if n_events > 0:
            event_dates = diff_pct[event_mask].sort_values(ascending=False)
            max_diff = float(event_dates.iloc[0])
            # 取前 5 個最大差異日
            top_dates = [(d.strftime('%Y-%m-%d'), f'{v:.1f}%')
                         for d, v in event_dates.head(5).items()]
            affected.append({
                'ticker': ticker,
                'name': name,
                'n_events': n_events,
                'max_diff_pct': max_diff,
                'dates': top_dates,
            })
        else:
            clean.append({'ticker': ticker, 'name': name, 'note': 'Close ≈ Adj Close'})

    has_issue = len(affected) > 0

    return {
        'has_issue': has_issue,
        'affected_stocks': affected,
        'clean_stocks': clean,
        'sample_size': len(sample),
    }


def print_dividend_check(result):
    """印出除權息驗證結果"""
    print(f"\n🔍 除權息驗證 (抽樣 {result['sample_size']} 檔):")
    print(f"   {'─' * 60}")

    if not result['has_issue']:
        print(f"   ✅ 全部抽樣股票 Close ≈ Adj Close，無除權息差異問題")
        print(f"   → yfinance 數據已自動調整，或該區間無除權息事件")
        return

    affected = result['affected_stocks']
    clean = result['clean_stocks']

    print(f"   ⚠️ 發現 {len(affected)}/{result['sample_size']} 檔有除權息差異:")
    print()

    for a in sorted(affected, key=lambda x: -x['max_diff_pct']):
        print(f"   🔴 {a['ticker']} {a['name']}: {a['n_events']} 個除權息日, "
              f"最大差異 {a['max_diff_pct']:.1f}%")
        for date_str, diff_str in a['dates']:
            print(f"      {date_str}  Close vs Adj Close 差 {diff_str}")

    if clean:
        print(f"\n   ✅ {len(clean)} 檔無差異: "
              + ', '.join(f"{c['ticker']}" for c in clean))

    print(f"\n   {'─' * 60}")
    print(f"   📋 影響分析:")
    print(f"      如果 Close 已被 yfinance 調整 (Close = Adj Close)，則無影響。")
    print(f"      如果 Close 是原始價 (≠ Adj Close)，除權息日附近的")
    print(f"      MA20/MA60 會出現不連續跳動，可能產生假的賣出/買進訊號。")
    print(f"   💡 建議:")
    print(f"      若差異顯著，可在 build_info_dict() 中改用 Adj Close 計算 MA。")
