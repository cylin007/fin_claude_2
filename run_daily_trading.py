#!/usr/bin/env python3
"""
每日交易引擎 — 與回測引擎共用策略邏輯，確保 100% 一致

使用方式:
  python3 run_daily_trading.py                          # 今日訊號
  python3 run_daily_trading.py signal [date]            # 指定日期訊號
  python3 run_daily_trading.py status                   # 持倉狀態
  python3 run_daily_trading.py add                      # 互動式記錄交易
  python3 run_daily_trading.py history                  # 交易紀錄
  python3 run_daily_trading.py verify [start] [end]     # 驗證 vs 回測
  python3 run_daily_trading.py reset                    # 重置狀態

核心原理:
  - DailyEngine.step_day() 完全對齊 group_backtest.py 的每日迴圈
  - 共用 check_strategy_signal(), build_info_dict(), calculate_net_pnl() 等函數
  - verify 模式: 逐日模擬 vs run_group_backtest() 比對, 確保 100% 一致
"""

import sys
import os
import csv
import json
import warnings
import datetime
import time as _time
import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)

from industry_manager import get_stocks_by_industry
from stock_utils import batch_download_stocks, build_info_dict, get_market_status
from strategy import calculate_net_pnl, DEFAULT_CONFIG
from group_backtest import (
    INDUSTRY_CONFIGS, reconstruct_market_history, run_group_backtest,
    DAILY_BUDGET, DAILY_MAX_POSITIONS,
    check_strategy_signal, calculate_fee, calculate_tax,
)

# ==========================================
# Constants
# ==========================================
INDUSTRY = '半導體業'
INITIAL_CAPITAL = 900_000
BUDGET_PER_TRADE = DAILY_BUDGET  # 25_000
MAX_POSITIONS = DAILY_MAX_POSITIONS  # 12

_OUTPUT_DIR = os.path.join(_BASE_DIR, 'output')
os.makedirs(_OUTPUT_DIR, exist_ok=True)
STATE_FILE = os.path.join(_OUTPUT_DIR, 'daily_engine_state.json')
TRADES_FILE = os.path.join(_OUTPUT_DIR, 'my_trades.csv')

# 回測引擎的基本門檻 (group_backtest.py)
MIN_VOLUME_SHARES = 500_000
MIN_TURNOVER = 50_000_000
MIN_PRICE = 10
MIN_DATA_DAYS = 20

# 名稱 ↔ 代號對照
_NAME_TO_TICKER = {}
_TICKER_TO_NAME = {}


def _build_name_ticker_map():
    global _NAME_TO_TICKER, _TICKER_TO_NAME
    for ind in INDUSTRY_CONFIGS.keys():
        try:
            for ticker, name in get_stocks_by_industry(ind):
                _NAME_TO_TICKER[name] = ticker
                _TICKER_TO_NAME[ticker] = name
                short = name.replace('-KY', '').replace('*', '')
                if short != name:
                    _NAME_TO_TICKER[short] = ticker
        except Exception:
            pass


def _resolve_ticker(name_or_ticker):
    name_or_ticker = name_or_ticker.strip()
    if name_or_ticker in _TICKER_TO_NAME:
        return name_or_ticker, _TICKER_TO_NAME[name_or_ticker]
    for suffix in ['.TW', '.TWO']:
        t = name_or_ticker + suffix
        if t in _TICKER_TO_NAME:
            return t, _TICKER_TO_NAME[t]
    if name_or_ticker in _NAME_TO_TICKER:
        t = _NAME_TO_TICKER[name_or_ticker]
        return t, _TICKER_TO_NAME.get(t, name_or_ticker)
    matches = [(n, t) for n, t in _NAME_TO_TICKER.items()
               if name_or_ticker in n or n in name_or_ticker]
    if len(matches) == 1:
        return matches[0][1], matches[0][0]
    if len(matches) > 1:
        print(f"  ⚠️ '{name_or_ticker}' 有多個匹配:")
        for n, t in matches[:5]:
            print(f"     {t} {n}")
    else:
        print(f"  ❌ 找不到 '{name_or_ticker}'")
    return None, None


# ==========================================
# DailyEngine Class
# ==========================================
class DailyEngine:
    """
    每日交易引擎 — 完全對齊 group_backtest.py 的逐日迴圈

    state 包含回測引擎的所有 mutable state:
      positions, pending, cash, day_idx, backup_queue, vol_rets,
      pp_daily_returns, pp_last_trigger_idx, prev_nav, trade_log, etc.
    """

    def __init__(self, industry=INDUSTRY, initial_capital=INITIAL_CAPITAL,
                 budget_per_trade=BUDGET_PER_TRADE):
        self.industry = industry
        self.initial_capital = initial_capital
        self.budget_per_trade = budget_per_trade

        # 策略參數 (完全從 INDUSTRY_CONFIGS + DEFAULT_CONFIG 取)
        self.cfg = dict(DEFAULT_CONFIG)
        ind_cfg = INDUSTRY_CONFIGS.get(industry, {}).get('config', {})
        self.cfg.update(ind_cfg)

        # 核心狀態 (對齊回測)
        self.positions = {}     # {ticker: {shares, cost_total, avg_cost, buy_price, buy_count, ...}}
        self.pending = {}       # {ticker: {action, reason, ...}}
        self.cash = initial_capital
        self.day_idx = 0
        self.last_date = None
        self.start_date = None  # 引擎首次 add 的日期 (固定 dl_start 用)
        self.backup_queue = []  # 漲停遞補候選

        # 績效追蹤
        self.realized_profit = 0
        self.total_fees = 0
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.trade_log = []

        # Vol sizing
        self.vol_rets = []

        # Portfolio panic
        self.pp_last_trigger_idx = -99
        self.pp_daily_returns = []

        # NAV 追蹤
        self.prev_nav = initial_capital
        self.peak_nav = initial_capital

        # 預解析 config
        self._parse_config()

    def _parse_config(self):
        """從 cfg 預解析所有策略參數 (對齊 run_daily_vs_backtest.py L118-181)"""
        cfg = self.cfg
        self.max_new_buy_per_day = cfg.get('max_new_buy_per_day', 4)
        self.max_positions = cfg.get('max_positions', 12)
        self.min_hold_days = cfg.get('min_hold_days', 1)
        self.enable_add = cfg.get('enable_add', True)
        self.max_add = cfg.get('max_add_per_stock', 99)
        self.max_add_per_day = cfg.get('max_add_per_day', 99)  # V8: 每日加碼上限
        self.enable_zombie = cfg.get('enable_zombie_cleanup', True)
        self.zombie_days = cfg.get('zombie_hold_days', 15)
        self.zombie_range = cfg.get('zombie_net_range', 5.0)
        self.sm_enabled = cfg.get('enable_sector_momentum', False)
        self.sm_lookback = cfg.get('sm_lookback', 20)
        self.slippage = cfg.get('slippage_pct', 0.0) / 100
        self.cash_check = cfg.get('enable_cash_check', True)
        self.cash_reserve_pct = cfg.get('cash_reserve_pct', 0.0)
        self.cash_reserve = int(self.initial_capital * self.cash_reserve_pct / 100)
        self.enable_swap = cfg.get('enable_position_swap', True)
        self.swap_margin = cfg.get('swap_score_margin', 1.0)
        self.max_swap_per_day = cfg.get('max_swap_per_day', 1)
        self.skip_limit_up = cfg.get('skip_limit_up', True)
        self.limit_up_threshold = cfg.get('limit_up_threshold', 1.095)
        self.enable_backup_fill = cfg.get('enable_backup_fill', True)
        self.entry_sort_by = cfg.get('entry_sort_by', 'score')

        # 動態限買
        self.dyn_buy_enabled = cfg.get('enable_dyn_buy_limit', False)
        self.dyn_buy_map = {
            'bull':    cfg.get('dyn_buy_bull', self.max_new_buy_per_day),
            'neutral': cfg.get('dyn_buy_neutral', self.max_new_buy_per_day),
            'weak':    cfg.get('dyn_buy_weak', self.max_new_buy_per_day),
            'bear':    cfg.get('dyn_buy_bear', self.max_new_buy_per_day),
        }
        self.dyn_buy_panic = cfg.get('dyn_buy_panic', 0)

        # 動態停損
        self.dyn_stop_enabled = cfg.get('enable_dyn_stop', False)
        self.hard_stop_override = {}
        if self.dyn_stop_enabled:
            hs_w = cfg.get('hard_stop_weak')
            hs_b = cfg.get('hard_stop_bear')
            if hs_w is not None:
                self.hard_stop_override['weak'] = hs_w
            if hs_b is not None:
                self.hard_stop_override['bear'] = hs_b

        # Vol sizing
        self.vol_sizing_enabled = cfg.get('enable_vol_sizing', False)
        self.vol_target = cfg.get('vol_target_pct', 1.5) / 100
        self.vol_floor = cfg.get('vol_scale_floor_pct', 50) / 100
        self.vol_lookback = cfg.get('vol_lookback', 20)

        # Portfolio panic
        self.pp_enabled = cfg.get('enable_portfolio_panic', False)
        self.pp_day_th = cfg.get('portfolio_panic_day_pct', -4.0) / 100
        self.pp_3d_th = cfg.get('portfolio_panic_3d_pct', -7.0) / 100
        self.pp_action = cfg.get('portfolio_panic_action', 'sell_losers')
        self.pp_loss_th = cfg.get('portfolio_panic_loss_threshold', 0.0)
        self.pp_cooldown = cfg.get('portfolio_panic_cooldown', 3)

    def save_state(self, filepath=None):
        """儲存引擎狀態到 JSON"""
        filepath = filepath or STATE_FILE
        state = {
            'version': '2.0',
            'industry': self.industry,
            'initial_capital': self.initial_capital,
            'budget_per_trade': self.budget_per_trade,
            'day_idx': self.day_idx,
            'last_date': self.last_date,
            'start_date': self.start_date,
            'cash': self.cash,
            'positions': self.positions,
            'pending': self.pending,
            'backup_queue': self.backup_queue,
            'vol_rets': self.vol_rets[-50:],  # 只保留最近 50 筆
            'pp_daily_returns': self.pp_daily_returns,
            'pp_last_trigger_idx': self.pp_last_trigger_idx,
            'prev_nav': self.prev_nav,
            'peak_nav': self.peak_nav,
            'cash_reserve': self.cash_reserve,
            'realized_profit': self.realized_profit,
            'total_fees': self.total_fees,
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'trade_log': self.trade_log,  # 全部保留 (績效統計用)
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_state(cls, filepath=None):
        """從 JSON 載入引擎狀態"""
        filepath = filepath or STATE_FILE
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        engine = cls(
            industry=state.get('industry', INDUSTRY),
            initial_capital=state.get('initial_capital', INITIAL_CAPITAL),
            budget_per_trade=state.get('budget_per_trade', BUDGET_PER_TRADE),
        )
        engine.day_idx = state.get('day_idx', 0)
        engine.last_date = state.get('last_date')
        engine.start_date = state.get('start_date')
        engine.cash = state.get('cash', engine.initial_capital)
        engine.positions = state.get('positions', {})
        engine.pending = state.get('pending', {})
        engine.backup_queue = state.get('backup_queue', [])
        engine.vol_rets = state.get('vol_rets', [])
        engine.pp_daily_returns = state.get('pp_daily_returns', [])
        engine.pp_last_trigger_idx = state.get('pp_last_trigger_idx', -99)
        engine.prev_nav = state.get('prev_nav', engine.initial_capital)
        engine.peak_nav = state.get('peak_nav', engine.initial_capital)
        engine.cash_reserve = state.get('cash_reserve', 0)
        engine.realized_profit = state.get('realized_profit', 0)
        engine.total_fees = state.get('total_fees', 0)
        engine.trade_count = state.get('trade_count', 0)
        engine.win_count = state.get('win_count', 0)
        engine.loss_count = state.get('loss_count', 0)
        engine.trade_log = state.get('trade_log', [])
        return engine

    def step_day(self, date_str, stock_data, market_status):
        """
        處理一個交易日 — 直接呼叫 run_group_backtest() 確保 100% 一致

        Args:
            date_str: 'YYYY-MM-DD' 格式的日期
            stock_data: {ticker: {'df': DataFrame, 'name': str}, ...}
            market_status: {is_unsafe, is_overheated, is_panic, twii: {...}, ewt: {...}}

        Returns:
            dict: {
                'executed': [已執行的交易],
                'signals': {ticker: {action, reason}},
                'nav': 今日 NAV,
                'daily_ret': 日報酬率,
            }
        """
        import io
        import contextlib

        # 建立 run_group_backtest 的輸入
        stock_list = [(t, sd.get('name', t)) for t, sd in stock_data.items()]
        market_map = {date_str: market_status}
        initial_positions = {t: dict(p) for t, p in self.positions.items()}
        initial_pending = {t: dict(p) for t, p in self.pending.items()}

        ind_cfg = INDUSTRY_CONFIGS.get(self.industry, {}).get('config', {})

        initial_state = {
            'cash': self.cash,
            'prev_nav': self.prev_nav,
            'peak_nav': self.peak_nav,
            'vol_rets': list(self.vol_rets),
            'pp_daily_returns': list(self.pp_daily_returns),
            'pp_last_trigger_idx': self.pp_last_trigger_idx,
            'realized_profit': self.realized_profit,
            'total_fees': self.total_fees,
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
        }

        # 呼叫回測引擎 (單日, 靜音輸出)
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_group_backtest(
                stock_list=stock_list,
                start_date=date_str,
                end_date=date_str,
                budget_per_trade=self.budget_per_trade,
                market_map=market_map,
                exec_mode='next_open',
                config_override=ind_cfg,
                initial_capital=self.initial_capital,
                preloaded_data=stock_data,
                initial_positions=initial_positions,
                initial_pending=initial_pending,
                initial_state=initial_state,
                _cash_reserve_override=self.cash_reserve,
                initial_day_idx=self.day_idx,
            )

        # 從回測結果提取狀態
        new_state = result['_state']
        self.positions = result['_raw_positions']
        self.pending = result['_pending']
        self.cash = new_state['cash']
        self.prev_nav = new_state['prev_nav']
        self.peak_nav = new_state['peak_nav']
        self.vol_rets = new_state['vol_rets']
        self.pp_daily_returns = new_state['pp_daily_returns']
        self.pp_last_trigger_idx = new_state['pp_last_trigger_idx']
        self.realized_profit = new_state['realized_profit']
        self.total_fees = new_state['total_fees']
        self.trade_count = new_state['trade_count']
        self.win_count = new_state['win_count']
        self.loss_count = new_state['loss_count']
        self.backup_queue = result.get('_backup_queue', [])
        self.day_idx = result.get('_final_day_idx', self.day_idx + 1)
        self.last_date = date_str

        # 轉換 trade_log 格式 (backtest 用 'note', DailyEngine 用 'reason')
        bt_trades = result.get('trade_log', [])
        executed_trades = []
        for t in bt_trades:
            entry = {
                'date': t['date'],
                'type': t['type'],
                'ticker': t['ticker'],
                'name': t.get('name', ''),
                'shares': t['shares'],
                'price': round(t['price'], 2) if isinstance(t['price'], float) else t['price'],
                'fee': int(t.get('fee', 0)),
                'profit': int(t['profit']) if t.get('profit') is not None else None,
                'roi': round(t['roi'], 2) if t.get('roi') is not None else None,
                'reason': t.get('note', '') or t.get('reason', ''),
            }
            executed_trades.append(entry)
            self.trade_log.append(entry)

        # 從 daily_snapshots 取 NAV 和 daily_ret
        snapshots = result.get('daily_snapshots', [])
        if snapshots:
            nav_today = snapshots[-1]['nav']
            daily_ret = snapshots[-1]['daily_return_pct'] / 100
        else:
            nav_today = self.prev_nav
            daily_ret = 0

        return {
            'executed': executed_trades,
            'signals': dict(self.pending),
            'candidates': result.get('_last_candidates', []),
            'add_candidates': result.get('_last_add_candidates', []),
            'nav': nav_today,
            'daily_ret': daily_ret,
            'positions_count': len(self.positions),
            'theme_rotation_status': result.get('_theme_rotation_status', {}),
        }

    def record_trade(self, ticker, action, shares, price, date_str, note=''):
        """手動記錄實際成交 (更新引擎內部持倉狀態)"""
        if action == 'buy':
            buy_price_with_slip = price  # 實際成交價
            buy_fee = calculate_fee(buy_price_with_slip, shares)
            cost = shares * buy_price_with_slip + buy_fee

            if ticker in self.positions:
                pos = self.positions[ticker]
                old_val = pos['avg_cost'] * pos['shares']
                pos['shares'] += shares
                pos['avg_cost'] = (old_val + shares * buy_price_with_slip) / pos['shares']
                pos['cost_total'] += cost
                pos['buy_count'] += 1
                pos['buy_price'] = buy_price_with_slip
                trade_type = 'ADD'
            else:
                name = _TICKER_TO_NAME.get(ticker, ticker)
                self.positions[ticker] = {
                    'name': name,
                    'shares': shares,
                    'avg_cost': buy_price_with_slip,
                    'cost_total': cost,
                    'buy_price': buy_price_with_slip,
                    'buy_count': 1,
                    'last_buy_date_idx': self.day_idx,
                    'reduce_stage': 0,
                    'last_reduce_date_idx': -99,
                    'peak_since_entry': price,
                }
                trade_type = 'BUY'

            self.cash -= cost
            self.total_fees += buy_fee

            entry = {
                'date': date_str, 'type': trade_type, 'ticker': ticker,
                'name': _TICKER_TO_NAME.get(ticker, ticker),
                'shares': shares, 'price': round(price, 2),
                'fee': int(buy_fee), 'profit': None, 'roi': None,
                'reason': note or '手動記錄',
            }
            self.trade_log.append(entry)

            # 從 pending 移除
            self.pending.pop(ticker, None)

        elif action == 'sell':
            if ticker not in self.positions:
                print(f"  ❌ 未持有 {ticker}")
                return
            pos = self.positions[ticker]
            sell_fee = calculate_fee(price, shares)
            sell_tax = calculate_tax(ticker, price, shares)
            revenue = shares * price - sell_fee - sell_tax
            cost_ratio = shares / pos['shares'] if pos['shares'] > 0 else 1
            cost_of_sold = pos['cost_total'] * cost_ratio
            profit = revenue - cost_of_sold

            self.cash += revenue
            self.realized_profit += profit
            self.total_fees += sell_fee + sell_tax

            remaining = pos['shares'] - shares
            if remaining <= 0:
                self.trade_count += 1
                if profit > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
                del self.positions[ticker]
            else:
                pos['shares'] = remaining
                pos['cost_total'] -= cost_of_sold

            entry = {
                'date': date_str, 'type': 'SELL', 'ticker': ticker,
                'name': _TICKER_TO_NAME.get(ticker, ticker),
                'shares': shares, 'price': round(price, 2),
                'fee': int(sell_fee + sell_tax),
                'profit': int(profit),
                'roi': round(profit / cost_of_sold * 100, 2) if cost_of_sold > 0 else 0,
                'reason': note or '手動記錄',
            }
            self.trade_log.append(entry)
            self.pending.pop(ticker, None)

        # 也寫入 my_trades.csv (sync 模式下跳過，避免重複)
        if not getattr(self, '_sync_mode', False):
            _append_to_trades_csv(date_str, ticker, action, shares, price, note)

    def get_positions_summary(self, stock_data=None, date_str=None):
        """取得持倉摘要"""
        if not self.positions:
            return []
        rows = []
        total_cost = 0
        total_mv = 0
        for ticker, pos in sorted(self.positions.items(),
                                   key=lambda x: -(x[1]['shares'] * x[1]['avg_cost'])):
            close = 0
            if stock_data and ticker in stock_data:
                sdf = stock_data[ticker]['df']
                if date_str:
                    valid = sdf.index[sdf.index <= pd.Timestamp(date_str)]
                    if len(valid) > 0:
                        close = float(sdf.loc[valid[-1], 'Close'])
                elif len(sdf) > 0:
                    close = float(sdf.iloc[-1]['Close'])

            cost = pos['avg_cost'] * pos['shares']
            mv = close * pos['shares'] if close > 0 else cost
            pnl_pct = (close / pos['avg_cost'] - 1) * 100 if pos['avg_cost'] > 0 and close > 0 else 0

            total_cost += cost
            total_mv += mv
            rows.append({
                'ticker': ticker, 'name': pos['name'],
                'shares': pos['shares'], 'avg_cost': pos['avg_cost'],
                'close': close, 'pnl_pct': pnl_pct,
                'market_value': mv, 'buy_count': pos.get('buy_count', 1),
            })
        return rows


# ==========================================
# Helper Functions
# ==========================================
def _append_to_trades_csv(date_str, ticker, action, shares, price, note=''):
    """附加交易紀錄到 my_trades.csv"""
    file_exists = os.path.exists(TRADES_FILE)

    if file_exists:
        with open(TRADES_FILE, 'rb') as f:
            f.seek(0, 2)
            if f.tell() > 0:
                f.seek(-1, 2)
                if f.read(1) != b'\n':
                    with open(TRADES_FILE, 'a', encoding='utf-8-sig') as fa:
                        fa.write('\n')

    with open(TRADES_FILE, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['日期', '股票', '動作', '股數', '價格', '備註'])
        name = _TICKER_TO_NAME.get(ticker, ticker)
        writer.writerow([date_str, name, action, shares, price, note])


def _export_trade_log_csv(trade_log, filepath, label=None):
    """匯出 trade_log 為 CSV — 統一格式 (與 run_full_backtest.py 完全一致)

    欄位: Date, Ticker, Name, Type, Price, Shares, Fee, Profit, ROI%, Note
    支援回測的 'note' 欄位和 DailyEngine 的 'reason' 欄位
    """
    rows = []
    for t in trade_log:
        rows.append({
            'Date': t['date'],
            'Ticker': t['ticker'],
            'Name': t.get('name', ''),
            'Type': t['type'],
            'Price': round(t['price'], 2) if isinstance(t['price'], float) else t['price'],
            'Shares': t['shares'],
            'Fee': int(t.get('fee', 0)),
            'Profit': int(t['profit']) if t.get('profit') is not None else '',
            'ROI%': round(t['roi'], 2) if t.get('roi') is not None else '',
            'Note': t.get('note', '') or t.get('reason', ''),
        })
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    tag = f" ({label})" if label else ""
    print(f"  ✅ 匯出{tag}: {filepath} ({len(rows)} 筆)")


# ==========================================
# Verify Mode
# ==========================================
def run_verify(start_date='2026-01-01', end_date='2026-03-13'):
    """
    驗證模式: 逐日模擬 vs 回測引擎, 比對交易是否 100% 一致

    1. 用共用資料跑 run_group_backtest()
    2. 用 DailyEngine 逐日 step_day()
    3. 逐日比對 trade_log
    """
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  驗證模式: 回測引擎 vs DailyEngine 逐日模擬                       ║")
    print(f"║  區間: {start_date} ~ {end_date} (兩者都從空倉開始){'':>20}║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # ═══════════════════════════════════════════════════
    # 共用資料準備
    # ═══════════════════════════════════════════════════
    cfg_entry = INDUSTRY_CONFIGS.get(INDUSTRY, {})
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(cfg_entry.get('config', {}))

    stock_list = get_stocks_by_industry(INDUSTRY)
    print(f"\n  股票池: {len(stock_list)} 檔")

    print(f"  建立大盤歷史...")
    market_map = reconstruct_market_history(start_date, end_date)

    dl_start = (pd.Timestamp(start_date) - pd.Timedelta(days=250)).strftime('%Y-%m-%d')
    _today = pd.Timestamp.today().normalize()
    dl_end = (max(_today, pd.Timestamp(end_date)) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    print(f"  下載股票資料...")
    preloaded_data, _ = batch_download_stocks(
        stock_list, dl_start, dl_end, min_data_days=20)
    print(f"  預載: {len(preloaded_data)} 檔")

    # 取得交易日列表 (用 market_map 日期，跟回測引擎完全一致)
    market_dates = sorted(pd.Timestamp(d) for d in market_map.keys())
    trading_dates = [d for d in market_dates
                     if start_date <= d.strftime('%Y-%m-%d') <= end_date]
    print(f"  交易日: {len(trading_dates)} 天 "
          f"({trading_dates[0].strftime('%Y-%m-%d')} ~ "
          f"{trading_dates[-1].strftime('%Y-%m-%d')})")

    # ═══════════════════════════════════════════════════
    # Part A: 回測引擎
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  Part A: 回測引擎 ({start_date} ~ {end_date})")
    print(f"{'='*70}")

    result_bt = run_group_backtest(
        stock_list=stock_list,
        start_date=start_date,
        end_date=end_date,
        budget_per_trade=BUDGET_PER_TRADE,
        market_map=market_map,
        exec_mode='next_open',
        config_override=cfg_entry.get('config', {}),
        initial_capital=INITIAL_CAPITAL,
        preloaded_data=preloaded_data,
    )
    bt_trades = result_bt['trade_log']
    print(f"  回測交易: {len(bt_trades)} 筆")

    # ═══════════════════════════════════════════════════
    # Part B: DailyEngine 逐日模擬
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  Part B: DailyEngine 逐日模擬 (完全對齊回測)")
    print(f"{'='*70}")

    engine = DailyEngine(
        industry=INDUSTRY,
        initial_capital=INITIAL_CAPITAL,
        budget_per_trade=BUDGET_PER_TRADE,
    )

    for day_idx, curr_date in enumerate(trading_dates):
        date_str = curr_date.strftime('%Y-%m-%d')
        current_market = market_map.get(date_str, {})
        engine.step_day(date_str, preloaded_data, current_market)

    daily_trades = engine.trade_log
    print(f"  Daily 模擬交易: {len(daily_trades)} 筆")

    # ═══════════════════════════════════════════════════
    # 輸出 CSV
    # ═══════════════════════════════════════════════════
    _output_dir = os.path.join(_BASE_DIR, 'output')
    os.makedirs(_output_dir, exist_ok=True)
    bt_csv = os.path.join(_output_dir, f'verify_backtest_{start_date}_{end_date}.csv')
    daily_csv = os.path.join(_output_dir, f'verify_daily_{start_date}_{end_date}.csv')

    _export_trade_log_csv(bt_trades, bt_csv, '回測')
    _export_trade_log_csv(daily_trades, daily_csv, 'DailyEngine')

    # ═══════════════════════════════════════════════════
    # 逐日比對
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  逐日比對: 回測 vs DailyEngine")
    print(f"{'='*70}")

    bt_by_date = {}
    for t in bt_trades:
        bt_by_date.setdefault(t['date'], []).append(t)
    daily_by_date = {}
    for t in daily_trades:
        daily_by_date.setdefault(t['date'], []).append(t)

    all_trade_dates = sorted(set(list(bt_by_date.keys()) + list(daily_by_date.keys())))
    total_days = 0
    match_days = 0
    mismatch_days = 0

    for d in all_trade_dates:
        bt_d = bt_by_date.get(d, [])
        daily_d = daily_by_date.get(d, [])

        bt_set = set()
        for t in bt_d:
            _type = 'BUY' if t['type'] == 'ADD' else t['type']
            bt_set.add((t['ticker'], _type))
        daily_set = set()
        for t in daily_d:
            _type = 'BUY' if t['type'] == 'ADD' else t['type']
            daily_set.add((t['ticker'], _type))

        total_days += 1
        if bt_set == daily_set:
            match_days += 1
            status = '✅'
        else:
            mismatch_days += 1
            status = '❌'

        bt_desc = ', '.join(
            f"{t['type']} {t['ticker'].split('.')[0]}" for t in bt_d) or '(無)'
        daily_desc = ', '.join(
            f"{t['type']} {t['ticker'].split('.')[0]}" for t in daily_d) or '(無)'

        if bt_set != daily_set:
            print(f"  {status} {d}")
            print(f"       回測:       {bt_desc}")
            print(f"       DailyEngine: {daily_desc}")
            only_bt = bt_set - daily_set
            only_daily = daily_set - bt_set
            if only_bt:
                print(f"       ↑ 回測獨有: {only_bt}")
            if only_daily:
                print(f"       ↑ Daily 獨有: {only_daily}")
        else:
            print(f"  {status} {d} | {bt_desc}")

    # ═══════════════════════════════════════════════════
    # 總結
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  📊 驗證總結")
    print(f"{'='*70}")
    print(f"  回測交易: {len(bt_trades)} 筆")
    print(f"  Daily 模擬: {len(daily_trades)} 筆")
    print(f"  有交易的天數: {total_days} 天")
    if total_days > 0:
        print(f"  股票一致: {match_days}/{total_days} 天 ({match_days/total_days*100:.1f}%)")
        print(f"  不一致: {mismatch_days} 天")

    if mismatch_days == 0:
        print(f"\n  ✅✅✅ 每日交易 100% 一致! ✅✅✅")
    else:
        print(f"\n  ⚠️ 有 {mismatch_days} 天不一致，需要 debug")

    return mismatch_days == 0


# ==========================================
# Live Signal Mode
# ==========================================
def run_live_signal(target_date=None):
    """生成每日交易訊號 (live 模式)"""
    today_str = target_date or datetime.date.today().strftime('%Y-%m-%d')

    # 載入或建立引擎
    engine = DailyEngine.load_state()
    if engine is None:
        print("  📝 首次使用，建立新引擎...")
        engine = DailyEngine()
    else:
        print(f"  📂 載入引擎狀態 (上次: {engine.last_date}, day_idx: {engine.day_idx})")

    print(f"\n{'='*80}")
    print(f"🔮 策略訊號 — {today_str}")
    print(f"{'='*80}")

    # 下載資料
    stock_list = get_stocks_by_industry(INDUSTRY)
    print(f"\n⏳ 下載資料...")
    dl_start = (pd.Timestamp(today_str) - pd.Timedelta(days=250)).strftime('%Y-%m-%d')
    dl_end = (pd.Timestamp(today_str) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    # 判斷是否為今天 → 盤中用短 TTL 確保資料新鮮
    _is_today = (today_str == datetime.date.today().strftime('%Y-%m-%d'))
    _ttl = 1 if _is_today else 12  # 今天: 1小時 TTL, 歷史: 12小時
    _force = ('--fresh' in sys.argv or '-f' in sys.argv)
    stock_data, _ = batch_download_stocks(
        stock_list, dl_start, dl_end, min_data_days=20,
        cache_ttl_hours=_ttl, force_refresh=_force)
    print(f"   預載: {len(stock_data)} 檔")

    # 補下載持倉/pending 中缺少的股票
    _held_tickers = set(engine.positions.keys()) | set(engine.pending.keys())
    _missing = [(t, engine.positions.get(t, {}).get('name', t)) for t in _held_tickers if t not in stock_data]
    if _missing:
        print(f"   ⚠️ 持倉補下載: {[m[1] for m in _missing]}")
        _extra, _ = batch_download_stocks(
            _missing, dl_start, dl_end, min_data_days=5,
            cache_ttl_hours=_ttl, force_refresh=_force)
        stock_data.update(_extra)

    # 確認資料包含目標日期
    _target_ts = pd.Timestamp(today_str)
    _has_date = sum(1 for sd in stock_data.values() if _target_ts in sd['df'].index)
    if _has_date > 0:
        print(f"   ✅ {today_str} 資料確認 ({_has_date}/{len(stock_data)} 檔有當日資料)")
    else:
        print(f"\n  ❌ {today_str} 非交易日 (無任何股票有當日資料)")
        # 找出最近的交易日供提示
        _sample = next(iter(stock_data.values()))['df']
        _prev = _sample.index[_sample.index < _target_ts]
        _next = _sample.index[_sample.index > _target_ts]
        if len(_prev) > 0:
            print(f"     前一個交易日: {_prev[-1].strftime('%Y-%m-%d')}")
        if len(_next) > 0:
            print(f"     下一個交易日: {_next[0].strftime('%Y-%m-%d')}")
        return

    # 大盤狀態
    print(f"   取得大盤狀態...")
    market_status = get_market_status(today_str)
    twii_info = market_status.get('twii', {})
    market_trend = twii_info.get('trend', 'neutral')
    is_unsafe = market_status.get('is_unsafe', False)
    is_panic = market_status.get('is_panic', False)

    trend_emoji = {'bull': '🟢', 'neutral': '🟡', 'weak': '🟠', 'bear': '🔴'}.get(market_trend, '⚪')
    print(f"\n📈 大盤: {trend_emoji} {market_trend}", end='')
    if is_unsafe:
        print(f" | ⛔ 偏空", end='')
    if is_panic:
        print(f" | 🚨 恐慌", end='')
    print()

    # 執行 step_day
    result = engine.step_day(today_str, stock_data, market_status)

    # ── 顯示已執行的交易 ──
    if result['executed']:
        print(f"\n{'─'*70}")
        print(f"📋 今日已執行 ({len(result['executed'])} 筆):")
        for t in result['executed']:
            emoji = '🔴' if t['type'] in ('SELL', 'REDUCE') else '🟢'
            profit_str = f" P&L ${t['profit']:+,}" if t.get('profit') is not None else ''
            print(f"  {emoji} {t['type']:<6} {t['ticker']:<12} {t['name']:<10} "
                  f"{t['shares']}股 @${t['price']:.1f}{profit_str}")

    # ── 顯示持倉 ──
    pos_rows = engine.get_positions_summary(stock_data, today_str)
    if pos_rows:
        print(f"\n{'─'*70}")
        print(f"📦 持倉: {len(pos_rows)} 檔")
        print(f"{'代號':<12} {'名稱':<10} {'股數':>6} {'均價':>8} {'現價':>8} {'損益%':>8}")
        print("-" * 60)
        for r in pos_rows:
            close_str = f"${r['close']:.1f}" if r['close'] > 0 else "N/A"
            pnl_str = f"{r['pnl_pct']:+.1f}%" if r['close'] > 0 else "N/A"
            print(f"{r['ticker']:<12} {r['name']:<10} {r['shares']:>6} "
                  f"${r['avg_cost']:>7.1f} {close_str:>8} {pnl_str:>8}")
    else:
        print(f"\n📦 空倉")

    # ── 顯示明日訊號 ──
    signals = result['signals']
    sell_signals = {t: o for t, o in signals.items() if o['action'] == 'sell'}
    buy_signals = {t: o for t, o in signals.items() if o['action'] == 'buy'}
    reduce_signals = {t: o for t, o in signals.items() if o['action'] == 'reduce'}

    print(f"\n{'='*70}")
    print(f"🔮 明日訊號 (T+1 open 執行)")
    print(f"{'='*70}")

    if sell_signals:
        print(f"\n  🔴 賣出 ({len(sell_signals)} 檔):")
        for ticker, order in sell_signals.items():
            pos = engine.positions.get(ticker, {})
            name = pos.get('name', ticker)
            shares = pos.get('shares', '?')
            print(f"     {ticker} {name} ({shares}股) — {order['reason'][:60]}")

    if reduce_signals:
        print(f"\n  🟠 減碼 ({len(reduce_signals)} 檔):")
        for ticker, order in reduce_signals.items():
            pos = engine.positions.get(ticker, {})
            name = pos.get('name', ticker)
            shares = pos.get('shares', '?')
            ratio = order.get('reduce_ratio', 0.5)
            sell_n = int(shares * ratio) if isinstance(shares, int) else '?'
            print(f"     {ticker} {name} (賣{sell_n}/{shares}股) — {order['reason'][:60]}")

    if buy_signals:
        new_buys = {t: o for t, o in buy_signals.items() if t not in engine.positions}
        add_buys = {t: o for t, o in buy_signals.items() if t in engine.positions}
        if new_buys:
            print(f"\n  🟢 新買進 ({len(new_buys)} 檔):")
            for ticker, order in new_buys.items():
                cand = next((c for c in result.get('candidates', [])
                             if c['ticker'] == ticker), None)
                if cand:
                    shares = int(engine.budget_per_trade / cand['close_price']) if cand['close_price'] > 0 else 0
                    print(f"     {ticker} {cand['name']} @${cand['close_price']:.1f} | "
                          f"{shares}股 | 分數{cand['score']:.2f}")
                else:
                    print(f"     {ticker} — {order['reason'][:60]}")
        if add_buys:
            print(f"\n  🔵 加碼 ({len(add_buys)} 檔):")
            for ticker, order in add_buys.items():
                pos = engine.positions.get(ticker, {})
                print(f"     {ticker} {pos.get('name', ticker)} "
                      f"(庫存{pos.get('shares', '?')}股) — {order['reason'][:60]}")

    if not sell_signals and not buy_signals and not reduce_signals:
        print(f"\n  ✅ 無訊號，維持現狀")

    # V11: 題材輪動狀態
    _rot_status = result.get('theme_rotation_status', {})
    _rot_scores = _rot_status.get('scores', {})
    _rot_allowed = set(_rot_status.get('allowed', []))
    if _rot_scores:
        print(f"\n{'─'*70}")
        print(f"🏷️ 題材輪動 (允許 {len(_rot_allowed)}/{len(_rot_scores)} 個題材)")
        _sorted_themes = sorted(_rot_scores.items(), key=lambda x: -x[1])
        for _t_name, _t_sc in _sorted_themes:
            _tag = '🟢' if _t_name in _rot_allowed else '🔴'
            _bar_len = int(_t_sc * 20)
            _bar = '█' * _bar_len + '░' * (20 - _bar_len)
            print(f"  {_tag} {_t_name:<12} {_bar} {_t_sc:.2f}")

    # 候選排行 (透明化分數拆解)
    candidates = result.get('candidates', [])
    if candidates:
        print(f"\n{'─'*90}")
        print(f"📊 候選排行 ({len(candidates)} 檔通過策略)")
        print(f"  {'#':>2}  {'代號':<12} {'名稱':<8} {'收盤':>7} │ {'總分':>5} = {'量價':>5} + {'漲幅':>5} - {'乖離罰':>5} + {'題材':>5} + {'RS':>5} │ {'量比':>5} {'漲%':>5} {'乖離%':>6} {'題材':<6}")
        print(f"  {'─'*105}")
        for i, c in enumerate(candidates[:15]):
            tag = ' ★' if c['ticker'] in buy_signals else ''
            # 拆解各分項
            vol_r = min(c.get('vol_ratio', 0), 3.0)
            vol_score = vol_r * 0.4
            pct_ch = min(c.get('pct_change', 0), 5.0)
            pct_score = pct_ch / 5 * 0.3
            bias = c.get('bias_pct', 0)
            bias_penalty = max(0, bias - 5) / 15 * 0.3
            theme_boost = c.get('theme_boost', 0)
            # RS bonus 無法精確還原，用 total - 其他反推
            base_score = vol_score + pct_score - bias_penalty + theme_boost
            rs_bonus = c['score'] - base_score
            theme_name = c.get('theme', '') or ''
            if len(theme_name) > 6:
                theme_name = theme_name[:5] + '…'
            print(f"  {i+1:>2}. {c['ticker']:<12} {c['name']:<8} "
                  f"@${c['close_price']:>6.1f} │ "
                  f"{c['score']:>5.2f} = {vol_score:>5.2f} + {pct_score:>5.2f} - {bias_penalty:>5.2f} + {theme_boost:>5.2f} + {rs_bonus:>5.2f} │ "
                  f"{c.get('vol_ratio', 0):>5.1f} {c.get('pct_change', 0):>+5.1f} {bias:>+6.1f} {theme_name:<6}{tag}")

    # NAV
    print(f"\n{'─'*70}")
    print(f"💰 NAV: ${result['nav']:,.0f} | 現金: ${engine.cash:,.0f} | "
          f"日報酬: {result['daily_ret']*100:+.2f}%")
    print(f"{'='*70}")

    # ── 輸出明日操作 JSON 到 /log ──
    import json as _json
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')
    os.makedirs(log_dir, exist_ok=True)

    # 明日執行日 = 下一個交易日
    exec_date_str = today_str  # signal date 當天看，T+1 執行
    actions = []

    # 賣出
    for ticker, order in sell_signals.items():
        pos = engine.positions.get(ticker, {})
        actions.append({
            'action': 'SELL',
            'ticker': ticker,
            'name': pos.get('name', ticker),
            'shares': pos.get('shares', 0),
            'reason': order.get('reason', ''),
        })

    # 減碼
    for ticker, order in reduce_signals.items():
        pos = engine.positions.get(ticker, {})
        shares = pos.get('shares', 0)
        ratio = order.get('reduce_ratio', 0.5)
        sell_n = int(shares * ratio) if isinstance(shares, int) else 0
        actions.append({
            'action': 'REDUCE',
            'ticker': ticker,
            'name': pos.get('name', ticker),
            'shares_sell': sell_n,
            'shares_hold': shares - sell_n,
            'reason': order.get('reason', ''),
        })

    # 新買進
    for ticker, order in buy_signals.items():
        if ticker not in engine.positions:
            cand = next((c for c in result.get('candidates', [])
                         if c['ticker'] == ticker), None)
            close_price = cand['close_price'] if cand else 0
            est_shares = int(engine.budget_per_trade / close_price) if close_price > 0 else 0
            actions.append({
                'action': 'BUY',
                'ticker': ticker,
                'name': cand['name'] if cand else ticker,
                'est_price': close_price,
                'est_shares': est_shares,
                'budget': engine.budget_per_trade,
                'score': cand['score'] if cand else 0,
                'reason': order.get('reason', ''),
            })

    # 加碼
    for ticker, order in buy_signals.items():
        if ticker in engine.positions:
            pos = engine.positions[ticker]
            cand = next((c for c in result.get('candidates', [])
                         if c['ticker'] == ticker), None)
            close_price = cand['close_price'] if cand else 0
            est_shares = int(engine.budget_per_trade / close_price) if close_price > 0 else 0
            actions.append({
                'action': 'ADD',
                'ticker': ticker,
                'name': pos.get('name', ticker),
                'current_shares': pos.get('shares', 0),
                'est_add_shares': est_shares,
                'est_price': close_price,
                'budget': engine.budget_per_trade,
                'reason': order.get('reason', ''),
            })

    log_data = {
        'signal_date': today_str,
        'execute_next_open': True,
        'market_trend': market_trend,
        'is_unsafe': bool(is_unsafe),
        'is_panic': bool(is_panic),
        'nav': float(result['nav']),
        'cash': float(engine.cash),
        'daily_ret': round(float(result['daily_ret']) * 100, 2),
        'actions': actions,
    }

    log_path = os.path.join(log_dir, f"signal_{today_str}.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        _json.dump(log_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  📁 訊號已輸出: {log_path}")

    # signal 模式為唯讀，不自動儲存狀態
    # 狀態只在使用者透過 add 記錄交易時才更新
    print(f"\n  ℹ️  signal 為唯讀模式，不會修改引擎狀態")


# ==========================================
# CLI: Status / Add / History
# ==========================================
def show_status():
    """顯示持倉狀態"""
    engine = DailyEngine.load_state()
    if engine is None:
        print("  📦 尚無引擎狀態 (空倉)")
        return

    print(f"\n{'='*70}")
    print(f"📦 引擎狀態")
    print(f"{'='*70}")
    print(f"  上次處理: {engine.last_date}")
    print(f"  Day index: {engine.day_idx}")
    print(f"  現金: ${engine.cash:,.0f}")
    print(f"  已實現損益: ${engine.realized_profit:+,.0f}")
    print(f"  總手續費: ${engine.total_fees:,.0f}")
    print(f"  交易次數: {engine.trade_count} (Win: {engine.win_count}, Loss: {engine.loss_count})")

    if engine.positions:
        # 下載最新價格
        held_tickers = list(engine.positions.keys())
        print(f"\n  ⏳ 取得最新價格...")
        import yfinance as yf
        latest_prices = {}
        for ticker in held_tickers:
            try:
                old_stderr = sys.stderr
                sys.stderr = open(os.devnull, 'w')
                try:
                    df = yf.download(ticker, period='5d', progress=False)
                finally:
                    sys.stderr = old_stderr
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if not df.empty:
                    latest_prices[ticker] = float(df['Close'].iloc[-1])
            except Exception:
                pass

        # 找每檔持倉的最近一筆「有內容」的備註 (空備註不覆蓋舊備註)
        last_notes = {}
        for t in engine.trade_log:
            if t['ticker'] in engine.positions:
                note = t.get('reason', '') or t.get('note', '')
                if note and note != '手動記錄':
                    last_notes[t['ticker']] = note

        # 計算總市值
        total_cost = 0
        total_market_val = 0
        rows = []
        for ticker, pos in sorted(engine.positions.items()):
            cost_val = pos['shares'] * pos['avg_cost']
            cur_price = latest_prices.get(ticker, 0)
            market_val = pos['shares'] * cur_price if cur_price > 0 else cost_val
            pnl_pct = ((cur_price - pos['avg_cost']) / pos['avg_cost'] * 100) if cur_price > 0 else 0
            total_cost += cost_val
            total_market_val += market_val
            rows.append((ticker, pos, cost_val, cur_price, market_val, pnl_pct))

        print(f"\n  持倉 ({len(rows)} 檔):")
        print(f"  {'代號':<12} {'名稱':<8} {'股數':>5} {'均價':>7} {'成本':>9} {'佔比':>6} {'現價':>7} {'市值':>9} {'報酬%':>7}  備註")
        print(f"  {'-'*100}")
        for ticker, pos, cost_val, cur_price, market_val, pnl_pct in rows:
            weight = (market_val / total_market_val * 100) if total_market_val > 0 else 0
            price_str = f"${cur_price:.1f}" if cur_price > 0 else "N/A"
            mval_str = f"${market_val:,.0f}" if cur_price > 0 else "N/A"
            pnl_str = f"{pnl_pct:+.1f}%" if cur_price > 0 else "N/A"
            note = last_notes.get(ticker, '')
            print(f"  {ticker:<12} {pos['name']:<8} {pos['shares']:>5} "
                  f"${pos['avg_cost']:>6.1f} ${cost_val:>8,.0f} {weight:>5.1f}% "
                  f"{price_str:>7} {mval_str:>9} {pnl_str:>7}  {note}")

        total_pnl = total_market_val - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        print(f"  {'-'*100}")
        print(f"  {'合計':<12} {'':8} {'':>5} {'':>7} ${total_cost:>8,.0f} {'100%':>6} "
              f"{'':>7} ${total_market_val:>8,.0f} {total_pnl_pct:>+6.1f}%")
        print(f"\n  💰 現金: ${engine.cash:,.0f} | 持股市值: ${total_market_val:,.0f} | "
              f"總資產: ${engine.cash + total_market_val:,.0f} | 未實現損益: ${total_pnl:+,.0f}")
    else:
        print(f"\n  (空倉)")

    if engine.pending:
        print(f"\n  待執行掛單 ({len(engine.pending)} 筆):")
        for ticker, order in engine.pending.items():
            print(f"  → {order['action'].upper()} {ticker}: {order['reason'][:50]}")


def _save_signal_log(date_str, engine, signals, all_new_candidates, all_add_candidates, stock_data):
    """抓 yfinance 基本面，存成 signal_log_{date}.txt 純文字供 analyze 使用"""

    # 收集所有要抓的 tickers: pending + 所有候選
    tickers_pending = {}   # ticker -> (name, action, reason)
    tickers_candidate = []  # [(ticker, name, reason, score, bias, is_add)]

    if signals:
        for ticker, order in signals.items():
            name = (engine.positions.get(ticker, {}).get('name')
                    or stock_data.get(ticker, {}).get('name')
                    or _TICKER_TO_NAME.get(ticker, ticker))
            tickers_pending[ticker] = (name, order['action'], order.get('reason', ''))

    for c in (all_new_candidates or []):
        t = c['ticker']
        if t not in tickers_pending:
            name = (stock_data.get(t, {}).get('name') or _TICKER_TO_NAME.get(t, t))
            tickers_candidate.append((t, name, c.get('reason', ''), c.get('score', 0), c.get('bias_pct', 0), False))

    for c in (all_add_candidates or []):
        t = c['ticker']
        if t not in tickers_pending and t not in [x[0] for x in tickers_candidate]:
            name = (engine.positions.get(t, {}).get('name')
                    or stock_data.get(t, {}).get('name')
                    or _TICKER_TO_NAME.get(t, t))
            tickers_candidate.append((t, name, c.get('reason', ''), c.get('score', 0), c.get('bias_pct', 0), True))

    all_tickers = list(tickers_pending.keys()) + [x[0] for x in tickers_candidate]
    if not all_tickers:
        print(f"  ℹ️ 無訊號/候選，跳過 signal_log")
        return

    # ── 抓 yfinance 基本面 ──
    print(f"\n⏳ 抓取 {len(all_tickers)} 檔基本面 (yfinance)...")
    import yfinance as yf

    fund_data = {}  # ticker -> formatted string
    for ticker in all_tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            hist = t.history(period='5d')

            pe = info.get('trailingPE') or info.get('forwardPE')
            pb = info.get('priceToBook')
            cap = info.get('marketCap', 0)
            cap_str = ''
            if cap and cap > 0:
                if cap > 1e12:
                    cap_str = f"${cap/1e12:.1f}兆"
                elif cap > 1e8:
                    cap_str = f"${cap/1e8:.0f}億"

            rg = info.get('revenueGrowth')
            rg_str = f"{rg*100:.1f}%" if isinstance(rg, (int, float)) else 'N/A'
            pm = info.get('profitMargins')
            pm_str = f"{pm*100:.1f}%" if isinstance(pm, (int, float)) else 'N/A'

            price_strs = []
            if not hist.empty:
                for idx, row in hist.tail(3).iterrows():
                    price_strs.append(f"{idx.strftime('%m/%d')}: ${row['Close']:.1f}")

            lines = []
            lines.append(f"     PE: {pe if pe else 'N/A'} | PB: {pb if pb else 'N/A'} | 市值: {cap_str or 'N/A'}")
            lines.append(f"     營收成長: {rg_str} | 利潤率: {pm_str}")
            if price_strs:
                lines.append(f"     近期股價: {', '.join(price_strs)}")
            fund_data[ticker] = '\n'.join(lines)
        except Exception as e:
            fund_data[ticker] = f"     (yfinance 錯誤: {e})"
        sys.stdout.write('.')
        sys.stdout.flush()
    print(' done')

    # ── 組文字 log ──
    lines = []
    lines.append(f"📅 日期: {date_str}")
    lines.append(f"💰 現金: ${engine.cash:,.0f} | 持倉: {len(engine.positions)} 檔 | 已實現損益: ${engine.realized_profit:+,.0f}")

    # 持倉
    if engine.positions:
        lines.append(f"\n📦 現有持倉 ({len(engine.positions)} 檔):")
        for ticker, pos in engine.positions.items():
            lines.append(f"  {ticker:<12} {pos.get('name', ticker):<10} "
                         f"{pos['shares']:>5}股 均${pos['avg_cost']:.1f} 成本${pos['cost_total']:,.0f}")

    # pending
    if tickers_pending:
        lines.append(f"\n🔮 明日 pending ({len(tickers_pending)} 筆):")
        for ticker, (name, action, reason) in tickers_pending.items():
            emoji = '🔴' if action.upper() in ('SELL', 'REDUCE') else '🟢'
            lines.append(f"  {emoji} {action.upper():<6} {ticker:<12} {name:<10} — {reason}")
            if ticker in fund_data:
                lines.append(fund_data[ticker])

    # 候選
    if tickers_candidate:
        lines.append(f"\n📊 其他候選 ({len(tickers_candidate)} 筆):")
        for (ticker, name, reason, score, bias, is_add) in tickers_candidate:
            tag = '[加碼]' if is_add else ''
            lines.append(f"  ⬜ {ticker:<12} {name:<10} 乖離 {bias:.1f}%, 分數 {score:.2f} {tag}")
            if reason:
                lines.append(f"     理由: {reason}")
            if ticker in fund_data:
                lines.append(fund_data[ticker])

    log_text = '\n'.join(lines)
    log_path = os.path.join(_OUTPUT_DIR, f'signal_log_{date_str}.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(log_text)
    print(f"📄 signal_log 已存: {log_path}")


def add_from_signal(date_str):
    """用 step_day 推進引擎一天 — 跟回測完全一致

    流程：
    1. 載入引擎狀態
    2. 防重複：檢查 last_date 是否已 >= date_str
    3. 下載全股票池資料 + 大盤狀態
    4. 呼叫 step_day(date_str) → 執行 pending + 檢查賣出 + 產生新訊號
    5. 儲存引擎狀態 (day_idx, prev_nav, pending 全部更新)
    6. 把已執行的交易寫入 my_trades.csv
    """
    # 載入引擎
    engine = DailyEngine.load_state()
    if engine is None:
        print("  📝 首次使用，建立新引擎 (空倉)...")
        engine = DailyEngine()

    # 防重複執行
    if engine.last_date and engine.last_date >= date_str:
        print(f"  ⚠️ 引擎已處理到 {engine.last_date}，不能重複處理 {date_str}")
        print(f"  💡 如需重跑，請先 reset")
        return

    _build_name_ticker_map()

    # 首次 add 時記錄 start_date
    if engine.start_date is None:
        engine.start_date = date_str

    print(f"\n{'='*70}")
    print(f"📋 推進引擎: {date_str} (day_idx={engine.day_idx})")
    print(f"{'='*70}")

    # 下載股票池資料
    # dl_start: 固定用 engine.start_date - 250 天 (所有 add 呼叫共用同一 cache)
    # dl_end:   固定用 max(date_str, today) + 5 天 (跟 run_full_backtest.py 對齊)
    stock_list = get_stocks_by_industry(INDUSTRY)
    print(f"\n⏳ 下載資料...")
    dl_start = (pd.Timestamp(engine.start_date) - pd.Timedelta(days=250)).strftime('%Y-%m-%d')
    _today = pd.Timestamp.today().normalize()
    _date_ts = pd.Timestamp(date_str)
    dl_end = (max(_today, _date_ts) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    _force = ('--fresh' in sys.argv or '-f' in sys.argv)
    stock_data, _ = batch_download_stocks(
        stock_list, dl_start, dl_end, min_data_days=20,
        cache_ttl_hours=12, force_refresh=_force)
    print(f"   預載: {len(stock_data)} 檔")

    # 補下載持倉/pending 中缺少的股票 (可能被 min_data_days 過濾掉)
    _held_tickers = set(engine.positions.keys()) | set(engine.pending.keys())
    _missing = [(t, engine.positions.get(t, {}).get('name', t)) for t in _held_tickers if t not in stock_data]
    if _missing:
        print(f"   ⚠️ 持倉補下載: {[m[1] for m in _missing]}")
        _extra, _ = batch_download_stocks(
            _missing, dl_start, dl_end, min_data_days=5,
            cache_ttl_hours=12, force_refresh=_force)
        stock_data.update(_extra)

    # 確認資料包含目標日期
    _target_ts = pd.Timestamp(date_str)
    _has_date = sum(1 for sd in stock_data.values() if _target_ts in sd['df'].index)
    if _has_date == 0:
        print(f"\n  ❌ {date_str} 非交易日 (無任何股票有當日資料)")
        _sample = next(iter(stock_data.values()))['df']
        _prev = _sample.index[_sample.index < _target_ts]
        _next = _sample.index[_sample.index > _target_ts]
        if len(_prev) > 0:
            print(f"     前一個交易日: {_prev[-1].strftime('%Y-%m-%d')}")
        if len(_next) > 0:
            print(f"     下一個交易日: {_next[0].strftime('%Y-%m-%d')}")
        return

    # 大盤狀態
    market_status = get_market_status(date_str)
    market_trend = market_status.get('twii', {}).get('trend', 'neutral')
    is_unsafe = market_status.get('is_unsafe', False)
    is_panic = market_status.get('is_panic', False)
    trend_emoji = {'bull': '🟢', 'neutral': '🟡', 'weak': '🟠', 'bear': '🔴'}.get(market_trend, '⚪')
    print(f"\n📈 大盤: {trend_emoji} {market_trend}", end='')
    if is_unsafe:
        print(f" | ⛔ 偏空", end='')
    if is_panic:
        print(f" | 🚨 恐慌", end='')
    print()

    # ══ 核心: 呼叫 step_day (跟回測一模一樣) ══
    result = engine.step_day(date_str, stock_data, market_status)

    # ── 顯示已執行的交易 (PHASE 1 的 pending 執行) ──
    if result['executed']:
        print(f"\n{'─'*70}")
        print(f"📋 今日已執行 ({len(result['executed'])} 筆):")
        for t in result['executed']:
            emoji = '🔴' if t['type'] in ('SELL', 'REDUCE') else '🟢'
            profit_str = f" P&L ${t['profit']:+,}" if t.get('profit') is not None else ''
            print(f"  {emoji} {t['type']:<6} {t['ticker']:<12} {t['name']:<10} "
                  f"{t['shares']}股 @${t['price']:.1f}{profit_str}")

    # ── 顯示持倉 ──
    pos_rows = engine.get_positions_summary(stock_data, date_str)
    if pos_rows:
        print(f"\n📦 持倉: {len(pos_rows)} 檔")
        for r in pos_rows:
            close_str = f"${r['close']:.1f}" if r['close'] > 0 else "N/A"
            pnl_str = f"{r['pnl_pct']:+.1f}%" if r['close'] > 0 else "N/A"
            print(f"  {r['ticker']:<12} {r['name']:<10} {r['shares']:>5}股 "
                  f"均${r['avg_cost']:.1f} 現{close_str} {pnl_str}")
    else:
        print(f"\n📦 空倉")

    # ── 顯示明日待執行 (pending) ──
    signals = result['signals']
    if signals:
        print(f"\n🔮 明日 pending ({len(signals)} 筆):")
        for ticker, order in signals.items():
            action = order['action'].upper()
            emoji = '🔴' if action == 'SELL' else '🟢' if action == 'BUY' else '🟠'
            name = (engine.positions.get(ticker, {}).get('name')
                    or stock_data.get(ticker, {}).get('name')
                    or _TICKER_TO_NAME.get(ticker, ticker))
            print(f"  {emoji} {action:<6} {ticker:<12} {name:<10} — {order['reason'][:60]}")

    # V11: 題材輪動狀態
    _rot_status = result.get('theme_rotation_status', {})
    _rot_scores = _rot_status.get('scores', {})
    _rot_allowed = set(_rot_status.get('allowed', []))
    if _rot_scores:
        print(f"\n{'─'*70}")
        print(f"🏷️ 題材輪動 (允許 {len(_rot_allowed)}/{len(_rot_scores)} 個題材)")
        _sorted_themes = sorted(_rot_scores.items(), key=lambda x: -x[1])
        for _t_name, _t_sc in _sorted_themes:
            _tag = '🟢' if _t_name in _rot_allowed else '🔴'
            _bar_len = int(_t_sc * 20)
            _bar = '█' * _bar_len + '░' * (20 - _bar_len)
            print(f"  {_tag} {_t_name:<12} {_bar} {_t_sc:.2f}")

    # ── 候選排行 (分數拆解) ──
    _all_new = result.get('candidates', [])
    _all_add = result.get('add_candidates', [])
    _pending_tickers = set(signals.keys()) if signals else set()
    _all_cands = sorted(_all_new + _all_add, key=lambda x: -x.get('score', 0))
    if _all_cands:
        print(f"\n{'─'*90}")
        print(f"📊 候選排行 ({len(_all_cands)} 檔通過策略)")
        print(f"  {'#':>2}  {'代號':<12} {'名稱':<8} {'收盤':>7} │ {'總分':>5} = {'量價':>5} + {'漲幅':>5} - {'乖離罰':>5} + {'題材':>5} + {'RS':>5} │ {'量比':>5} {'漲%':>5} {'乖離%':>6} {'題材':<6}")
        print(f"  {'─'*105}")
        for i, c in enumerate(_all_cands[:15]):
            vol_r = min(c.get('vol_ratio', 0), 3.0)
            vol_score = vol_r * 0.4
            pct_ch = min(c.get('pct_change', 0), 5.0)
            pct_score = pct_ch / 5 * 0.3
            bias = c.get('bias_pct', 0)
            bias_penalty = max(0, bias - 5) / 15 * 0.3
            theme_boost = c.get('theme_boost', 0)
            base_score = vol_score + pct_score - bias_penalty + theme_boost
            rs_bonus = c['score'] - base_score
            theme_name = c.get('theme', '') or ''
            if len(theme_name) > 6:
                theme_name = theme_name[:5] + '…'
            is_pending = '★' if c['ticker'] in _pending_tickers else ''
            is_add = '加' if c['ticker'] in engine.positions else ''
            tag = is_pending + is_add
            print(f"  {i+1:>2}. {c['ticker']:<12} {c['name']:<8} "
                  f"@${c['close_price']:>6.1f} │ "
                  f"{c['score']:>5.2f} = {vol_score:>5.2f} + {pct_score:>5.2f} - {bias_penalty:>5.2f} + {theme_boost:>5.2f} + {rs_bonus:>5.2f} │ "
                  f"{c.get('vol_ratio', 0):>5.1f} {c.get('pct_change', 0):>+5.1f} {bias:>+6.1f} {theme_name:<6} {tag}")

    # ── 寫入 my_trades.csv (只寫今天 executed 的，含漲停跳過) ──
    for t in result['executed']:
        _append_to_trades_csv(
            t['date'], t['ticker'], t['type'], t['shares'], t['price'],
            t.get('reason', ''))

    # 儲存引擎狀態
    engine.save_state()

    print(f"\n{'─'*70}")
    print(f"💰 NAV: ${result['nav']:,.0f} | 現金: ${engine.cash:,.0f} | "
          f"日報酬: {result['daily_ret']*100:+.2f}%")
    print(f"💾 已儲存 (day_idx={engine.day_idx}, last_date={engine.last_date})")

    # ── 抓 yfinance 基本面 → 存 signal_log ──
    _save_signal_log(date_str, engine, signals, _all_new, _all_add, stock_data)

    print(f"{'='*70}")


def add_trade_interactive():
    """互動式新增交易"""
    engine = DailyEngine.load_state()
    if engine is None:
        print("  📝 首次使用，建立新引擎 (空倉)...")
        engine = DailyEngine()

    _build_name_ticker_map()

    print("\n📝 新增交易紀錄")
    print("-" * 40)

    today = datetime.date.today().strftime('%Y-%m-%d')
    date_str = input(f"  日期 (Enter={today}): ").strip()
    if not date_str:
        date_str = today
    else:
        try:
            date_str = pd.Timestamp(date_str).strftime('%Y-%m-%d')
        except Exception:
            print(f"  ❌ 無效日期: {date_str}")
            return

    stock = input("  股票名稱或代號: ").strip()
    if not stock:
        print("  ❌ 取消")
        return
    ticker, name = _resolve_ticker(stock)
    if ticker is None:
        return
    print(f"  → {ticker} {name}")

    action = input("  動作 (buy/sell): ").strip().lower()
    if action not in ('buy', 'sell'):
        print(f"  ❌ 無效動作: {action}")
        return

    try:
        shares = int(float(input("  股數: ").strip()))
    except ValueError:
        print("  ❌ 無效股數")
        return

    try:
        price = float(input("  價格: ").strip())
    except ValueError:
        print("  ❌ 無效價格")
        return

    note = input("  備註 (可空): ").strip()

    action_ch = '買進' if action == 'buy' else '賣出'
    print(f"\n  確認: {date_str} {action_ch} {name}({ticker}) {shares}股 @${price:.1f}")
    confirm = input("  確認? (y/N): ").strip().lower()
    if confirm != 'y':
        print("  ❌ 取消")
        return

    engine.record_trade(ticker, action, shares, price, date_str, note)
    engine.save_state()
    print(f"  ✅ 已記錄並更新引擎狀態")


def show_history():
    """顯示交易紀錄"""
    engine = DailyEngine.load_state()
    if engine is None or not engine.trade_log:
        print("\n📜 無交易紀錄")
        return

    print(f"\n{'='*80}")
    print(f"📜 交易紀錄: {len(engine.trade_log)} 筆")
    print(f"{'='*80}")
    print(f"{'日期':<12} {'類型':<8} {'代號':<12} {'名稱':<10} {'股數':>6} {'價格':>8} {'損益':>10}")
    print("-" * 75)

    for t in engine.trade_log[-50:]:  # 最近 50 筆
        profit_str = f"${t['profit']:+,}" if t.get('profit') is not None else ''
        print(f"{t['date']:<12} {t['type']:<8} {t['ticker']:<12} "
              f"{t.get('name', ''):<10} {t['shares']:>6} "
              f"${t['price']:>7.1f} {profit_str:>10}")


def reset_engine():
    """重置引擎狀態 (含 my_trades.csv)"""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print(f"  ✅ 已刪除 {STATE_FILE}")
    else:
        print(f"  ℹ️ 無狀態檔案需要刪除")
    if os.path.exists(TRADES_FILE):
        os.remove(TRADES_FILE)
        print(f"  ✅ 已刪除 {TRADES_FILE}")
    print(f"  📝 下次執行 signal 時將從空倉開始")


def sync_from_csv():
    """從 my_trades.csv 重建 daily_engine_state.json 的 positions 和 cash"""
    _build_name_ticker_map()

    if not os.path.exists(TRADES_FILE):
        print(f"  ❌ 找不到 {TRADES_FILE}")
        return

    engine = DailyEngine.load_state()
    if not engine:
        print(f"  ❌ 找不到引擎狀態 ({STATE_FILE})，請先 add 至少一天")
        return

    # 讀取 CSV
    trades = []
    with open(TRADES_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(row)

    if not trades:
        print(f"  ❌ my_trades.csv 是空的")
        return

    # 保存需要保留的引擎狀態
    saved_pending = engine.pending
    saved_day_idx = engine.day_idx
    saved_last_date = engine.last_date
    saved_start_date = engine.start_date
    saved_backup_queue = engine.backup_queue

    # 重建 positions 和 cash
    engine._sync_mode = True  # 避免 record_trade 重複寫入 CSV
    engine.positions = {}
    engine.cash = engine.initial_capital
    engine.trade_log = []
    engine.realized_profit = 0
    engine.total_fees = 0
    engine.trade_count = 0
    engine.win_count = 0
    engine.loss_count = 0

    buy_count = 0
    sell_count = 0
    skip_count = 0
    bad_count = 0

    for i, row in enumerate(trades, start=2):  # CSV 行號從 2 開始 (1 = header)
        action = row.get('動作', '').strip().upper()
        name = row.get('股票', '').strip()
        date_str = row.get('日期', '').strip()

        # 驗證日期格式 (YYYY-MM-DD)
        try:
            pd.Timestamp(date_str)
        except Exception:
            bad_count += 1
            if bad_count <= 3:
                print(f"  ⚠️ 第{i}行格式異常，跳過: {dict(row)}")
            continue

        # 跳過 SKIP_LIMIT_UP
        if action == 'SKIP_LIMIT_UP':
            skip_count += 1
            continue

        # 驗證動作
        if action not in ('BUY', 'ADD', 'SELL'):
            bad_count += 1
            if bad_count <= 3:
                print(f"  ⚠️ 第{i}行未知動作 '{action}'，跳過")
            continue

        try:
            shares = int(float(row['股數']))
            price = float(row['價格'])
        except (ValueError, KeyError):
            bad_count += 1
            if bad_count <= 3:
                print(f"  ⚠️ 第{i}行數值異常，跳過: {dict(row)}")
            continue

        note = row.get('備註', '')

        # 名稱 → ticker
        result = _resolve_ticker(name)
        if not result or not result[0]:
            print(f"  ⚠️ 找不到 '{name}' 的代號，跳過 (第{i}行)")
            continue
        ticker, resolved_name = result

        if action in ('BUY', 'ADD'):
            engine.record_trade(ticker, 'buy', shares, price, date_str, note)
            buy_count += 1
        elif action == 'SELL':
            engine.record_trade(ticker, 'sell', shares, price, date_str, note)
            sell_count += 1

    if bad_count > 3:
        print(f"  ⚠️ 共 {bad_count} 行格式異常被跳過")

    # 恢復保留的狀態 (pending 清空，因為使用者已手動處理交易)
    engine.pending = {}  # 清空：sync 後的 add 不該重複執行舊 pending
    engine.backup_queue = []
    engine.start_date = saved_start_date

    # 更新 last_date 為 CSV 中最新日期 (避免 add 重複處理已 sync 的日期)
    csv_dates = []
    for t in engine.trade_log:
        try:
            csv_dates.append(pd.Timestamp(t['date']).strftime('%Y-%m-%d'))
        except Exception:
            pass
    if csv_dates:
        engine.last_date = max(csv_dates)
        # day_idx 也要同步 (根據 CSV 中不同日期的數量)
        unique_dates = sorted(set(csv_dates))
        engine.day_idx = len(unique_dates) + 1  # +1 因為 day_idx 是下一天的索引
    else:
        engine.day_idx = saved_day_idx
        engine.last_date = saved_last_date

    engine._sync_mode = False  # 恢復正常模式
    engine.save_state()

    print(f"\n{'='*70}")
    print(f"🔄 Sync 完成")
    print(f"{'='*70}")
    print(f"  📄 讀取 {len(trades)} 筆 (BUY/ADD: {buy_count}, SELL: {sell_count}, SKIP: {skip_count})")
    print(f"  💰 現金: ${engine.cash:,.0f}")
    print(f"  📦 持倉: {len(engine.positions)} 檔")

    if engine.positions:
        for ticker, pos in sorted(engine.positions.items(),
                                   key=lambda x: x[1]['shares'] * x[1]['avg_cost'],
                                   reverse=True):
            name = pos.get('name', ticker)
            print(f"     {ticker:<12} {name:<10} {pos['shares']:>6}股 "
                  f"均${pos['avg_cost']:.1f}")

    print(f"  📊 已實現損益: ${engine.realized_profit:+,.0f}")
    print(f"  💾 已儲存到 {STATE_FILE}")
    print(f"{'='*70}")


def analyze_signals(date_str=None):
    """讀取 signal_log_{date}.txt → 加上分析指令 → 送 Claude"""
    import subprocess, glob as _glob

    # 找 log 檔
    if date_str:
        log_path = os.path.join(_OUTPUT_DIR, f'signal_log_{date_str}.txt')
    else:
        # 自動找最新的
        pattern = os.path.join(_OUTPUT_DIR, 'signal_log_*.txt')
        files = sorted(_glob.glob(pattern))
        if not files:
            print("  ❌ 找不到任何 signal_log_*.txt，請先跑 add")
            return
        log_path = files[-1]
        date_str = os.path.basename(log_path).replace('signal_log_', '').replace('.txt', '')

    if not os.path.exists(log_path):
        print(f"  ❌ 找不到 {log_path}")
        print(f"  💡 請先跑: python3 run_daily_trading.py add {date_str}")
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        log_text = f.read()

    print(f"\n📖 讀取: {log_path}")

    # 組 prompt = 角色指令 + log 內容 + 分析要求
    prompt_header = "你是台股半導體產業分析師。以下是我的交易系統產生的訊號與基本面資料，請分析每檔股票的投資價值。\n"

    prompt_footer = """
## 請回答
1. 對每檔 pending 訊號給出「✅ 建議執行」「⚠️ 觀望」「❌ 不建議」的評級
2. 簡述理由（2-3句）
3. 候選股中有沒有值得手動加入的？
4. 如果有持倉股票應該減碼/賣出的，也請標注
5. 最後給一個整體建議（是否繼續加碼、減碼、或觀望）

用繁體中文回答，格式簡潔。"""

    full_prompt = prompt_header + "\n" + log_text + "\n" + prompt_footer

    # 存完整 prompt 備用
    prompt_path = os.path.join(_OUTPUT_DIR, f'analyze_prompt_{date_str}.txt')
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(full_prompt)

    print(f"📄 Prompt 已存: {prompt_path} ({len(full_prompt)} 字)")
    print(f"\n📤 呼叫 claude -p 分析中...\n")
    print("─" * 70)

    # 用 claude CLI (-p 非互動模式，吃 Max 訂閱不額外計費)
    import subprocess
    _env = os.environ.copy()
    _env['PATH'] = '/opt/homebrew/bin:' + _env.get('PATH', '')
    try:
        proc = subprocess.run(
            ['/opt/homebrew/bin/claude', '-p'],
            input=full_prompt,
            capture_output=True, text=True, timeout=180,
            env=_env,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            reply = proc.stdout.strip()
            print(reply)

            # 存分析結果
            result_path = os.path.join(_OUTPUT_DIR, f'analyze_result_{date_str}.txt')
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(reply)
            print(f"\n📄 分析結果已存: {result_path}")
        else:
            err = proc.stderr.strip() if proc.stderr else ''
            out = proc.stdout.strip() if proc.stdout else ''
            print(f"  ❌ claude -p 失敗 (exit={proc.returncode})")
            if err:
                print(f"  stderr: {err[:500]}")
            if out:
                print(f"  stdout: {out[:500]}")
            print(f"  💡 Prompt 已存到 {prompt_path}，可手動貼到 Claude")
    except FileNotFoundError:
        print("  ❌ 找不到 /opt/homebrew/bin/claude")
        print("  💡 請確認: npm install -g @anthropic-ai/claude-code")
        print(f"  💡 Prompt 已存到 {prompt_path}，可手動貼到 Claude")
    except subprocess.TimeoutExpired:
        print("  ❌ 逾時 (180秒)，prompt 可能太長")
        print(f"  💡 Prompt 已存到 {prompt_path}，可手動貼到 Claude")
    except Exception as e:
        print(f"  ❌ 執行失敗: {e}")
        print(f"  💡 Prompt 已存到 {prompt_path}，可手動貼到 Claude")

    print("─" * 70)


# ==========================================
# Main
# ==========================================
def main():
    args = sys.argv[1:]

    if not args or (len(args) == 1 and args[0] not in (
            'status', 'add', 'history', 'verify', 'reset', 'export', 'signal', 'sync', 'analyze')):
        # 預設: 指定日期或今日訊號
        _build_name_ticker_map()
        target_date = args[0] if args and '-' in args[0] else None
        run_live_signal(target_date)
        return

    cmd = args[0]

    if cmd == 'signal':
        _build_name_ticker_map()
        target_date = None
        if len(args) > 1 and '-' in args[1]:
            try:
                pd.Timestamp(args[1])
                target_date = args[1]
            except Exception:
                print(f"  ❌ 無效日期: {args[1]}")
                return
        elif len(args) > 1:
            print(f"  ❌ 無效日期: {args[1]}")
            print("  用法: python3 run_daily_trading.py signal [YYYY-MM-DD]")
            return
        run_live_signal(target_date)

    elif cmd == 'status':
        _build_name_ticker_map()
        show_status()

    elif cmd == 'add':
        if len(args) > 1 and '-' in args[1]:
            # add {date} → 推進引擎一天
            try:
                pd.Timestamp(args[1])
            except Exception:
                print(f"  ❌ 無效日期: {args[1]}")
                return
            add_from_signal(args[1])
        else:
            add_trade_interactive()

    elif cmd == 'history':
        show_history()

    elif cmd == 'verify':
        start = args[1] if len(args) > 1 else '2026-01-01'
        end = args[2] if len(args) > 2 else '2026-03-13'
        run_verify(start, end)

    elif cmd == 'sync':
        sync_from_csv()

    elif cmd == 'analyze':
        a_date = None
        if len(args) > 1 and '-' in args[1]:
            a_date = args[1]
        analyze_signals(a_date)

    elif cmd == 'reset':
        reset_engine()

    elif cmd == 'export':
        engine = DailyEngine.load_state()
        if engine and engine.trade_log:
            path = os.path.join(_BASE_DIR, 'daily_trade_log.csv')
            _export_trade_log_csv(engine.trade_log, path)
        else:
            print("  ❌ 無交易紀錄可匯出")

    else:
        print(f"  ❌ 未知指令: {cmd}")
        print("  用法:")
        print("    python3 run_daily_trading.py                    # 今日訊號")
        print("    python3 run_daily_trading.py signal [date]      # 指定日期")
        print("    python3 run_daily_trading.py status             # 持倉狀態")
        print("    python3 run_daily_trading.py add                # 記錄交易")
        print("    python3 run_daily_trading.py history            # 交易紀錄")
        print("    python3 run_daily_trading.py verify [start end] # 驗證模式")
        print("    python3 run_daily_trading.py sync               # 從CSV同步部位")
        print("    python3 run_daily_trading.py reset              # 重置")
        print("    python3 run_daily_trading.py analyze [date]     # Claude 分析訊號")
        print("    python3 run_daily_trading.py export             # 匯出CSV")


if __name__ == '__main__':
    main()
