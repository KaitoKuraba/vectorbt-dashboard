"""
VectorBT Backtesting Dashboard - Backend API
=============================================
Flask API server that handles data fetching, backtesting, and parameter optimization
using the vectorbt library.
"""

import json
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Allow all origins for Railway + Vercel setup


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _normalize_columns(data):
    """Flatten MultiIndex columns and normalize column names."""
    if data is None or data.empty:
        return data

    # Flatten MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        # Try to get the Price level (first level in yfinance MultiIndex)
        data.columns = data.columns.get_level_values(0)

    # Remove duplicate columns
    data = data.loc[:, ~data.columns.duplicated()]

    # Normalize column names: capitalize first letter
    col_map = {}
    for c in data.columns:
        c_str = str(c).strip()
        c_lower = c_str.lower()
        if c_lower == "open":
            col_map[c] = "Open"
        elif c_lower == "high":
            col_map[c] = "High"
        elif c_lower == "low":
            col_map[c] = "Low"
        elif c_lower in ("close", "adj close", "adjclose"):
            col_map[c] = "Close"
        elif c_lower == "volume":
            col_map[c] = "Volume"
    if col_map:
        data = data.rename(columns=col_map)

    return data


def _has_required_cols(data):
    """Check if DataFrame has OHLCV columns."""
    if data is None or data.empty:
        return False
    required = {"Open", "High", "Low", "Close", "Volume"}
    return required.issubset(set(data.columns))


INTERVAL_TO_FREQ = {
    "1m": "1T", "2m": "2T", "5m": "5T", "15m": "15T", "30m": "30T",
    "60m": "1H", "90m": "90T", "1h": "1H",
    "1d": "1D", "5d": "5D", "1wk": "1W", "1mo": "1M", "3mo": "3M",
}

# yfinance intraday intervals have max lookback limits
INTRADAY_MAX_DAYS = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
    "60m": 730, "90m": 60, "1h": 730,
}


def fetch_ohlcv(ticker, start, end, interval="1d"):
    """Robustly fetch OHLCV data from yfinance, handling all API versions and quirks."""
    error_msgs = []
    yf_version = getattr(yf, "__version__", "unknown")
    print(f"[fetch_ohlcv] ticker={ticker}, start={start}, end={end}, interval={interval}, yfinance={yf_version}")

    # For intraday intervals, clamp the date range to yfinance's max lookback
    is_intraday = interval in INTRADAY_MAX_DAYS
    if is_intraday:
        max_days = INTRADAY_MAX_DAYS[interval]
        earliest_allowed = (datetime.now() - timedelta(days=max_days)).strftime("%Y-%m-%d")
        if start < earliest_allowed:
            start = earliest_allowed
            print(f"[fetch_ohlcv] Clamped start to {start} (max {max_days} days for {interval})")

    # ── Attempt 1: Ticker.history() — most reliable for single tickers ──
    for auto_adj in [True, False]:
        try:
            t = yf.Ticker(ticker)
            kwargs = dict(start=start, end=end, interval=interval)
            # auto_adjust param was removed/changed in some versions
            try:
                data = t.history(**kwargs, auto_adjust=auto_adj)
            except TypeError:
                data = t.history(**kwargs)
            data = _normalize_columns(data)
            if _has_required_cols(data):
                print(f"[fetch_ohlcv] SUCCESS via Ticker.history(auto_adjust={auto_adj}): {len(data)} rows")
                return data
            msg = f"Ticker.history(auto_adjust={auto_adj}): empty or missing cols"
            if data is not None:
                msg += f" (shape={data.shape}, cols={list(data.columns)})"
            error_msgs.append(msg)
        except Exception as e:
            error_msgs.append(f"Ticker.history(auto_adjust={auto_adj}): {e}")

    # ── Attempt 2: Ticker.history() with period instead of date range ──
    try:
        t = yf.Ticker(ticker)
        period = "7d" if is_intraday else "max"
        data = t.history(period=period, interval=interval)
        data = _normalize_columns(data)
        if _has_required_cols(data):
            # Filter to date range
            data.index = pd.to_datetime(data.index)
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            # Make timezone-aware comparison if data index is tz-aware
            if data.index.tz is not None:
                start_ts = start_ts.tz_localize(data.index.tz)
                end_ts = end_ts.tz_localize(data.index.tz)
            mask = (data.index >= start_ts) & (data.index <= end_ts)
            data = data.loc[mask]
            if not data.empty:
                print(f"[fetch_ohlcv] SUCCESS via Ticker.history(period='{period}'): {len(data)} rows")
                return data
        error_msgs.append(f"Ticker.history(period='{period}'): empty after filtering")
    except Exception as e:
        error_msgs.append(f"Ticker.history(period='{period}'): {e}")

    # ── Attempt 3: yf.download with multi_level_index=False ──
    try:
        data = yf.download(
            ticker, start=start, end=end, interval=interval,
            auto_adjust=True, multi_level_index=False, progress=False,
        )
        data = _normalize_columns(data)
        if _has_required_cols(data):
            print(f"[fetch_ohlcv] SUCCESS via yf.download(multi_level_index=False): {len(data)} rows")
            return data
        error_msgs.append(f"yf.download(multi_level_index=False): empty (shape={getattr(data, 'shape', None)})")
    except TypeError:
        error_msgs.append("yf.download: multi_level_index param not supported")
    except Exception as e:
        error_msgs.append(f"yf.download(multi_level_index=False): {e}")

    # ── Attempt 4: yf.download standard ──
    try:
        data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        data = _normalize_columns(data)
        if _has_required_cols(data):
            print(f"[fetch_ohlcv] SUCCESS via yf.download(standard): {len(data)} rows")
            return data
        error_msgs.append(f"yf.download(standard): empty (shape={getattr(data, 'shape', None)})")
    except Exception as e:
        error_msgs.append(f"yf.download(standard): {e}")

    # ── All attempts failed ──
    err_detail = "; ".join(error_msgs)
    print(f"[fetch_ohlcv] ALL ATTEMPTS FAILED: {err_detail}")
    raise ValueError(
        f"Could not fetch data for '{ticker}'. "
        f"yfinance version: {yf_version}. "
        f"Errors: {err_detail}. "
        f"\n\nTroubleshooting:\n"
        f"  1. Upgrade yfinance: pip install --upgrade yfinance\n"
        f"  2. Check internet connection\n"
        f"  3. Try a different ticker (SPY, AAPL, MSFT, BTC-USD)\n"
        f"  4. Yahoo Finance may be temporarily blocking requests — wait a minute and retry"
    )


def safe_val(v):
    """Convert numpy/pandas types to JSON-safe Python types."""
    if v is None:
        return None
    # Handle float NaN and Infinity (JSON does not support Infinity)
    if isinstance(v, (float, np.floating)):
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    # Catch any remaining non-JSON-safe types
    if isinstance(v, (int, str, bool)):
        return v
    try:
        # Last resort: check for Python float inf/nan
        if isinstance(v, (float,)) and (v != v or v == float('inf') or v == float('-inf')):
            return None
    except (TypeError, ValueError):
        pass
    return v


# ─── Candlestick Pattern Definitions ────────────────────────────────────────────

BULLISH_PATTERNS = {
    "cdl_hammer", "cdl_inverted_hammer", "cdl_bullish_engulfing",
    "cdl_bullish_harami", "cdl_three_white_soldiers", "cdl_morning_star",
    "cdl_marubozu_bull", "cdl_piercing_line",
}

BEARISH_PATTERNS = {
    "cdl_shooting_star", "cdl_bearish_engulfing", "cdl_bearish_harami",
    "cdl_three_black_crows", "cdl_evening_star", "cdl_marubozu_bear",
    "cdl_dark_cloud_cover",
}

CDL_PATTERN_META = {
    "cdl_doji":                  {"label": "Doji",                  "type": "neutral"},
    "cdl_hammer":                {"label": "Hammer",                 "type": "bullish"},
    "cdl_inverted_hammer":       {"label": "Inverted Hammer",        "type": "bullish"},
    "cdl_shooting_star":         {"label": "Shooting Star",          "type": "bearish"},
    "cdl_bullish_engulfing":     {"label": "Bullish Engulfing",      "type": "bullish"},
    "cdl_bearish_engulfing":     {"label": "Bearish Engulfing",      "type": "bearish"},
    "cdl_morning_star":          {"label": "Morning Star",           "type": "bullish"},
    "cdl_evening_star":          {"label": "Evening Star",           "type": "bearish"},
    "cdl_bullish_harami":        {"label": "Bullish Harami",         "type": "bullish"},
    "cdl_bearish_harami":        {"label": "Bearish Harami",         "type": "bearish"},
    "cdl_three_white_soldiers":  {"label": "Three White Soldiers",   "type": "bullish"},
    "cdl_three_black_crows":     {"label": "Three Black Crows",      "type": "bearish"},
    "cdl_marubozu_bull":         {"label": "Bullish Marubozu",       "type": "bullish"},
    "cdl_marubozu_bear":         {"label": "Bearish Marubozu",       "type": "bearish"},
    "cdl_piercing_line":         {"label": "Piercing Line",          "type": "bullish"},
    "cdl_dark_cloud_cover":      {"label": "Dark Cloud Cover",       "type": "bearish"},
}


def detect_candlestick_patterns(open_, high, low, close):
    """Detect 16 candlestick patterns using pure NumPy/pandas math (no TA-Lib)."""
    patterns = {}

    body = close - open_
    body_abs = body.abs()
    total_range = high - low
    candle_top = pd.concat([close, open_], axis=1).max(axis=1)
    candle_bot = pd.concat([close, open_], axis=1).min(axis=1)
    upper_wick = high - candle_top
    lower_wick = candle_bot - low
    range_safe = total_range.where(total_range > 0, 0.0001)
    body_ratio = body_abs / range_safe

    # ── 1. Doji ──
    patterns["cdl_doji"] = body_ratio < 0.1

    # ── 2. Hammer (bullish reversal at bottom) ──
    patterns["cdl_hammer"] = (
        (body_ratio < 0.35) &
        (lower_wick >= 1.5 * body_abs) &
        (upper_wick <= 0.5 * body_abs.where(body_abs > 0, 0.0001))
    )

    # ── 3. Inverted Hammer (bullish reversal) ──
    patterns["cdl_inverted_hammer"] = (
        (body_ratio < 0.35) &
        (upper_wick >= 1.5 * body_abs) &
        (lower_wick <= 0.5 * body_abs.where(body_abs > 0, 0.0001)) &
        (body >= 0)
    )

    # ── 4. Shooting Star (bearish reversal at top) ──
    patterns["cdl_shooting_star"] = (
        (body_ratio < 0.35) &
        (upper_wick >= 1.5 * body_abs) &
        (lower_wick <= 0.5 * body_abs.where(body_abs > 0, 0.0001)) &
        (body <= 0)
    )

    # ── 5. Bullish Engulfing ──
    patterns["cdl_bullish_engulfing"] = (
        (body.shift(1) < 0) &
        (body > 0) &
        (open_ <= close.shift(1)) &
        (close >= open_.shift(1))
    )

    # ── 6. Bearish Engulfing ──
    patterns["cdl_bearish_engulfing"] = (
        (body.shift(1) > 0) &
        (body < 0) &
        (open_ >= close.shift(1)) &
        (close <= open_.shift(1))
    )

    # ── 7. Bullish Harami ──
    patterns["cdl_bullish_harami"] = (
        (body.shift(1) < 0) &
        (body_abs.shift(1) > range_safe.shift(1) * 0.5) &
        (body > 0) &
        (open_ > close.shift(1)) &
        (close < open_.shift(1))
    )

    # ── 8. Bearish Harami ──
    patterns["cdl_bearish_harami"] = (
        (body.shift(1) > 0) &
        (body_abs.shift(1) > range_safe.shift(1) * 0.5) &
        (body < 0) &
        (open_ < close.shift(1)) &
        (close > open_.shift(1))
    )

    # ── 9. Three White Soldiers ──
    patterns["cdl_three_white_soldiers"] = (
        (body > 0) & (body.shift(1) > 0) & (body.shift(2) > 0) &
        (close > close.shift(1)) & (close.shift(1) > close.shift(2)) &
        (open_ > open_.shift(1)) & (open_.shift(1) > open_.shift(2))
    )

    # ── 10. Three Black Crows ──
    patterns["cdl_three_black_crows"] = (
        (body < 0) & (body.shift(1) < 0) & (body.shift(2) < 0) &
        (close < close.shift(1)) & (close.shift(1) < close.shift(2)) &
        (open_ < open_.shift(1)) & (open_.shift(1) < open_.shift(2))
    )

    # ── 11. Morning Star (3-candle bullish) ──
    day1_mid = (open_.shift(2) + close.shift(2)) / 2
    patterns["cdl_morning_star"] = (
        (body.shift(2) < 0) &
        (body_abs.shift(1) < body_abs.shift(2) * 0.35) &
        (body > 0) &
        (close > day1_mid)
    )

    # ── 12. Evening Star (3-candle bearish) ──
    day1_mid_e = (open_.shift(2) + close.shift(2)) / 2
    patterns["cdl_evening_star"] = (
        (body.shift(2) > 0) &
        (body_abs.shift(1) < body_abs.shift(2) * 0.35) &
        (body < 0) &
        (close < day1_mid_e)
    )

    # ── 13. Bullish Marubozu ──
    patterns["cdl_marubozu_bull"] = (
        (body > 0) &
        (upper_wick < body_abs * 0.08) &
        (lower_wick < body_abs * 0.08)
    )

    # ── 14. Bearish Marubozu ──
    patterns["cdl_marubozu_bear"] = (
        (body < 0) &
        (upper_wick < body_abs * 0.08) &
        (lower_wick < body_abs * 0.08)
    )

    # ── 15. Piercing Line ──
    patterns["cdl_piercing_line"] = (
        (body.shift(1) < 0) &
        (open_ < close.shift(1)) &
        (body > 0) &
        (close > (open_.shift(1) + close.shift(1)) / 2) &
        (close < open_.shift(1))
    )

    # ── 16. Dark Cloud Cover ──
    patterns["cdl_dark_cloud_cover"] = (
        (body.shift(1) > 0) &
        (open_ > close.shift(1)) &
        (body < 0) &
        (close < (open_.shift(1) + close.shift(1)) / 2) &
        (close > open_.shift(1))
    )

    for k in patterns:
        patterns[k] = patterns[k].fillna(False)

    return patterns


def compute_cdc_actionzone(close, params):
    """
    CDC ActionZone V3 — translated from Pine Script by piriya33.
    Based on EMA crossover (default EMA12 / EMA26).
    Zones define 6 market states; buy/sell on first bar entering Green/Red.
    """
    fast_period = int(params.get("cdc_fast", 12))
    slow_period = int(params.get("cdc_slow", 26))
    smooth      = int(params.get("cdc_smooth", 1))

    # xPrice = optional pre-smoothing of close (smooth=1 → no change)
    x_price = close.ewm(span=smooth, adjust=False).mean() if smooth > 1 else close.copy()

    # EMA lines
    fast_ma = x_price.ewm(span=fast_period, adjust=False).mean()
    slow_ma = x_price.ewm(span=slow_period, adjust=False).mean()

    # Trend direction
    bull = fast_ma > slow_ma
    bear = fast_ma < slow_ma

    # ── 6 Color Zones (matching Pine Script) ──
    green  = bull & (x_price > fast_ma)                            # Buy Zone
    blue   = bear & (x_price > fast_ma) & (x_price > slow_ma)     # Pre-Buy 2
    lblue  = bear & (x_price > fast_ma) & (x_price < slow_ma)     # Pre-Buy 1
    red    = bear & (x_price < fast_ma)                            # Sell Zone
    orange = bull & (x_price < fast_ma) & (x_price < slow_ma)     # Pre-Sell 2
    yellow = bull & (x_price < fast_ma) & (x_price > slow_ma)     # Pre-Sell 1

    # First bar entering each zone (zone transition)
    buycond  = green & ~green.shift(1).fillna(False)
    sellcond = red   & ~red.shift(1).fillna(False)

    # ── State machine: track bullish/bearish trend ──
    # (equivalent to Pine's barssince(buycond) < barssince(sellcond))
    current_state = 0   # 0 = unknown, 1 = bullish, -1 = bearish
    states = []
    for i in range(len(close)):
        if buycond.iloc[i]:
            current_state = 1
        elif sellcond.iloc[i]:
            current_state = -1
        states.append(current_state)
    state = pd.Series(states, index=close.index, dtype=int)

    bullish_trend = state == 1
    bearish_trend = state == -1

    # ── Final buy/sell signal ──
    # buy  = was bearish[1] AND now first green bar
    # sell = was bullish[1] AND now first red bar
    buy_signal  = (state.shift(1).fillna(0) == -1) & buycond
    sell_signal = (state.shift(1).fillna(0) ==  1) & sellcond

    # ── Zone label per bar (for chart coloring) ──
    zone = pd.Series("none", index=close.index)
    zone[green]  = "green"
    zone[blue]   = "blue"
    zone[lblue]  = "lblue"
    zone[red]    = "red"
    zone[orange] = "orange"
    zone[yellow] = "yellow"

    return {
        "cdc_fast_ma":  fast_ma,
        "cdc_slow_ma":  slow_ma,
        "cdc_green":    green,
        "cdc_blue":     blue,
        "cdc_lblue":    lblue,
        "cdc_red":      red,
        "cdc_orange":   orange,
        "cdc_yellow":   yellow,
        "cdc_bullish":  bullish_trend,
        "cdc_bearish":  bearish_trend,
        "cdc_buy":      buy_signal,
        "cdc_sell":     sell_signal,
        "cdc_buycond":  buycond,
        "cdc_sellcond": sellcond,
        "cdc_zone":     zone,          # string Series — handled separately in extract_results
    }


def compute_rsi_divergence(close, params):
    """
    RSI Divergence / Convergence detector.

    Divergence:  Price and RSI move in OPPOSITE directions → potential reversal.
    Convergence: Price and RSI move in the SAME direction  → trend confirmation.

    Types returned
    ──────────────
    rdiv_bull        : Bullish Divergence  — price ↓ but RSI ↑ (oversold area)
    rdiv_bear        : Bearish Divergence  — price ↑ but RSI ↓ (overbought area)
    rdiv_hidden_bull : Hidden Bullish Div  — price ↑ but RSI ↓ (RSI < 50, uptrend pullback)
    rdiv_hidden_bear : Hidden Bearish Div  — price ↓ but RSI ↑ (RSI > 50, downtrend bounce)
    rdiv_conv_bull   : Bullish Convergence — price ↑ AND RSI ↑ (RSI > 50, momentum confirm)
    rdiv_conv_bear   : Bearish Convergence — price ↓ AND RSI ↓ (RSI < 50, downtrend confirm)
    rdiv_buy         : Bullish div + RSI crosses back above rsi_lower (entry signal)
    rdiv_sell        : Bearish div + RSI crosses back below rsi_upper (exit signal)
    rdiv_buy_mom     : First bar of fresh bullish divergence (momentum signal)
    rdiv_sell_mom    : First bar of fresh bearish divergence (momentum signal)
    """
    rsi_period = int(params.get("rsi_period", 14))
    rsi_upper  = float(params.get("rsi_upper", 70))
    rsi_lower  = float(params.get("rsi_lower", 30))
    div_window = int(params.get("div_window", 14))   # look-back window for slope comparison

    rsi = vbt.RSI.run(close, window=rsi_period).rsi

    # Slope over div_window bars (positive = rising, negative = falling)
    price_delta = close - close.shift(div_window)
    rsi_delta   = rsi   - rsi.shift(div_window)

    # ── Divergences (price vs RSI moving opposite directions) ──
    # Bullish: price made lower low but RSI made higher low → reversal signal near oversold
    bull_div = (price_delta < 0) & (rsi_delta > 0) & (rsi < rsi_lower + 20)

    # Bearish: price made higher high but RSI made lower high → reversal signal near overbought
    bear_div = (price_delta > 0) & (rsi_delta < 0) & (rsi > rsi_upper - 20)

    # Hidden Bullish: price ↑ (higher low) but RSI ↓ → pullback in uptrend, continuation signal
    hidden_bull = (price_delta > 0) & (rsi_delta < 0) & (rsi < 50)

    # Hidden Bearish: price ↓ (lower high) but RSI ↑ → bounce in downtrend, continuation signal
    hidden_bear = (price_delta < 0) & (rsi_delta > 0) & (rsi > 50)

    # ── Convergences (price and RSI moving same direction) ──
    # Bullish Convergence: both rising above midline → strong bull momentum
    bull_conv = (price_delta > 0) & (rsi_delta > 0) & (rsi > 50)

    # Bearish Convergence: both falling below midline → strong bear momentum
    bear_conv = (price_delta < 0) & (rsi_delta < 0) & (rsi < 50)

    # ── Trade signals ──
    # RSI crossing levels (momentum confirmation after divergence)
    rsi_cross_up   = (rsi > rsi_lower) & (rsi.shift(1) <= rsi_lower)
    rsi_cross_down = (rsi < rsi_upper) & (rsi.shift(1) >= rsi_upper)

    # Entry: bullish divergence present AND RSI just crossed back above oversold
    buy_signal  = bull_div & rsi_cross_up
    # Exit: bearish divergence present AND RSI just crossed back below overbought
    sell_signal = bear_div & rsi_cross_down

    # Helper: shift a bool Series by 1 without triggering pandas FutureWarning
    # (fillna on object-dtype Series triggers a deprecation warning in pandas ≥2.0)
    def _shift_bool(s):
        arr = np.concatenate([[False], s.values[:-1].astype(bool)])
        return pd.Series(arr, index=s.index, dtype=bool)

    def _fill_bool(s):
        """Ensure boolean Series has no NaN (replace NaN→False) without FutureWarning."""
        return pd.Series(np.where(np.isnan(s.astype(float)), False, s.astype(float)).astype(bool), index=s.index)

    # Momentum: first bar of divergence (transition from non-divergence to divergence)
    buy_momentum  = bull_div & ~_shift_bool(bull_div)
    sell_momentum = bear_div & ~_shift_bool(bear_div)

    bull_div    = _fill_bool(bull_div)
    bear_div    = _fill_bool(bear_div)
    hidden_bull = _fill_bool(hidden_bull)
    hidden_bear = _fill_bool(hidden_bear)
    bull_conv   = _fill_bool(bull_conv)
    bear_conv   = _fill_bool(bear_conv)
    buy_signal  = _fill_bool(buy_signal)
    sell_signal = _fill_bool(sell_signal)
    buy_momentum  = _fill_bool(buy_momentum)
    sell_momentum = _fill_bool(sell_momentum)

    return {
        "rdiv_rsi":        rsi,
        "rdiv_bull":       bull_div,
        "rdiv_bear":       bear_div,
        "rdiv_hidden_bull": hidden_bull,
        "rdiv_hidden_bear": hidden_bear,
        "rdiv_conv_bull":  bull_conv,
        "rdiv_conv_bear":  bear_conv,
        "rdiv_buy":        buy_signal,
        "rdiv_sell":       sell_signal,
        "rdiv_buy_mom":    buy_momentum,
        "rdiv_sell_mom":   sell_momentum,
    }


def compute_indicators(open_, close, high, low, params):
    """Compute all technical indicators based on user params."""
    indicators = {}

    # Moving Averages
    fast_ma_period = params.get("fast_ma", 10)
    slow_ma_period = params.get("slow_ma", 30)
    ma_type = params.get("ma_type", "SMA")

    if ma_type == "EMA":
        indicators["fast_ma"] = vbt.MA.run(close, window=fast_ma_period, ewm=True).ma
        indicators["slow_ma"] = vbt.MA.run(close, window=slow_ma_period, ewm=True).ma
    else:
        indicators["fast_ma"] = vbt.MA.run(close, window=fast_ma_period).ma
        indicators["slow_ma"] = vbt.MA.run(close, window=slow_ma_period).ma

    # RSI
    rsi_period = params.get("rsi_period", 14)
    rsi_upper = params.get("rsi_upper", 70)
    rsi_lower = params.get("rsi_lower", 30)
    indicators["rsi"] = vbt.RSI.run(close, window=rsi_period).rsi

    # Bollinger Bands
    bb_period = params.get("bb_period", 20)
    bb_std = params.get("bb_std", 2.0)
    bb = vbt.BBANDS.run(close, window=bb_period, alpha=bb_std)
    indicators["bb_upper"] = bb.upper
    indicators["bb_middle"] = bb.middle
    indicators["bb_lower"] = bb.lower

    # MACD
    macd_fast = params.get("macd_fast", 12)
    macd_slow = params.get("macd_slow", 26)
    macd_signal = params.get("macd_signal", 9)
    macd = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    indicators["macd"] = macd.macd
    indicators["macd_signal"] = macd.signal
    indicators["macd_hist"] = macd.hist

    # Candlestick Patterns
    cdl_patterns = detect_candlestick_patterns(open_, high, low, close)
    indicators.update(cdl_patterns)

    # CDC ActionZone
    cdc = compute_cdc_actionzone(close, params)
    indicators.update(cdc)

    # RSI Divergence / Convergence
    rdiv = compute_rsi_divergence(close, params)
    indicators.update(rdiv)

    return indicators


def _eval_single_condition(cond, close, high, low, indicators, params):
    """Evaluate a single condition rule and return a boolean Series.

    Each condition is a dict like:
        { "indicator": "ma_crossover", "direction": "bullish" }
        { "indicator": "rsi", "compare": "below", "value": 30 }
        { "indicator": "bollinger", "compare": "below_lower" }
        { "indicator": "macd", "direction": "bullish" }
        { "indicator": "price_ma", "compare": "above_fast" }
    """
    ind = cond.get("indicator", "")
    compare = cond.get("compare", "")
    direction = cond.get("direction", "bullish")
    value = float(cond.get("value", 0))
    false_series = pd.Series(False, index=close.index)

    if ind == "ma_crossover":
        fast = indicators["fast_ma"]
        slow = indicators["slow_ma"]
        if direction == "bullish":
            return (fast > slow) & (fast.shift(1) <= slow.shift(1))
        else:
            return (fast < slow) & (fast.shift(1) >= slow.shift(1))

    elif ind == "ma_trend":
        fast = indicators["fast_ma"]
        slow = indicators["slow_ma"]
        if direction == "bullish":
            return fast > slow
        else:
            return fast < slow

    elif ind == "rsi":
        rsi = indicators["rsi"]
        threshold = value if value > 0 else (params.get("rsi_lower", 30) if compare == "below" else params.get("rsi_upper", 70))
        if compare == "below":
            return (rsi < threshold) & (rsi.shift(1) >= threshold)
        elif compare == "above":
            return (rsi > threshold) & (rsi.shift(1) <= threshold)
        elif compare == "is_below":
            return rsi < threshold
        elif compare == "is_above":
            return rsi > threshold
        return false_series

    elif ind == "bollinger":
        if compare == "below_lower":
            return (close < indicators["bb_lower"]) & (close.shift(1) >= indicators["bb_lower"].shift(1))
        elif compare == "above_upper":
            return (close > indicators["bb_upper"]) & (close.shift(1) <= indicators["bb_upper"].shift(1))
        elif compare == "is_below_lower":
            return close < indicators["bb_lower"]
        elif compare == "is_above_upper":
            return close > indicators["bb_upper"]
        elif compare == "crosses_middle_up":
            return (close > indicators["bb_middle"]) & (close.shift(1) <= indicators["bb_middle"].shift(1))
        elif compare == "crosses_middle_down":
            return (close < indicators["bb_middle"]) & (close.shift(1) >= indicators["bb_middle"].shift(1))
        return false_series

    elif ind == "macd":
        macd_line = indicators["macd"]
        signal_line = indicators["macd_signal"]
        if direction == "bullish":
            return (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        else:
            return (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

    elif ind == "macd_hist":
        hist = indicators["macd_hist"]
        if direction == "bullish":
            return (hist > 0) & (hist.shift(1) <= 0)
        else:
            return (hist < 0) & (hist.shift(1) >= 0)

    elif ind == "price_ma":
        fast = indicators["fast_ma"]
        slow = indicators["slow_ma"]
        if compare == "above_fast":
            return (close > fast) & (close.shift(1) <= fast.shift(1))
        elif compare == "below_fast":
            return (close < fast) & (close.shift(1) >= fast.shift(1))
        elif compare == "above_slow":
            return (close > slow) & (close.shift(1) <= slow.shift(1))
        elif compare == "below_slow":
            return (close < slow) & (close.shift(1) >= slow.shift(1))
        return false_series

    elif ind == "candlestick":
        pattern_key = cond.get("pattern", "")
        if pattern_key and pattern_key in indicators:
            sig = indicators[pattern_key]
            return sig.fillna(False).astype(bool)
        return false_series

    elif ind == "cdc_actionzone":
        key_map = {
            "buy":      "cdc_buy",
            "sell":     "cdc_sell",
            "bullish":  "cdc_bullish",
            "bearish":  "cdc_bearish",
            "green":    "cdc_green",
            "red":      "cdc_red",
            "blue":     "cdc_blue",
            "lblue":    "cdc_lblue",
            "orange":   "cdc_orange",
            "yellow":   "cdc_yellow",
            "buycond":  "cdc_buycond",
            "sellcond": "cdc_sellcond",
        }
        ikey = key_map.get(compare, "")
        if ikey and ikey in indicators:
            return indicators[ikey].fillna(False).astype(bool)
        return false_series

    elif ind == "rsi_divergence":
        # compare values map directly to rdiv_* keys
        key_map = {
            "buy":         "rdiv_buy",        # bullish div + RSI cross above oversold
            "sell":        "rdiv_sell",        # bearish div + RSI cross below overbought
            "bull":        "rdiv_bull",        # bullish divergence active
            "bear":        "rdiv_bear",        # bearish divergence active
            "hidden_bull": "rdiv_hidden_bull", # hidden bullish divergence (uptrend continuation)
            "hidden_bear": "rdiv_hidden_bear", # hidden bearish divergence (downtrend continuation)
            "conv_bull":   "rdiv_conv_bull",   # bullish convergence (strong bull momentum)
            "conv_bear":   "rdiv_conv_bear",   # bearish convergence (strong bear momentum)
            "buy_mom":     "rdiv_buy_mom",     # first bar of bullish divergence
            "sell_mom":    "rdiv_sell_mom",    # first bar of bearish divergence
        }
        ikey = key_map.get(compare, "")
        if ikey and ikey in indicators:
            return indicators[ikey].fillna(False).astype(bool)
        return false_series

    return false_series


def generate_signals(close, high, low, indicators, params):
    """Generate entry/exit signals based on separate condition lists.

    params should contain:
        entry_conditions: [{ indicator, direction/compare, value }, ...]
        exit_conditions:  [{ indicator, direction/compare, value }, ...]
        entry_logic: "AND" | "OR"
        exit_logic: "AND" | "OR"
    Falls back to legacy single-strategy mode if entry_conditions is absent.
    """
    entry_conds = params.get("entry_conditions", [])
    exit_conds = params.get("exit_conditions", [])
    entry_logic = params.get("entry_logic", "AND")
    exit_logic = params.get("exit_logic", "AND")

    # ── Fallback: legacy single-strategy mode ──
    if not entry_conds and not exit_conds:
        strategy = params.get("strategy", "ma_crossover")
        legacy_map = {
            "ma_crossover": (
                [{"indicator": "ma_crossover", "direction": "bullish"}],
                [{"indicator": "ma_crossover", "direction": "bearish"}],
            ),
            "rsi": (
                [{"indicator": "rsi", "compare": "below", "value": params.get("rsi_lower", 30)}],
                [{"indicator": "rsi", "compare": "above", "value": params.get("rsi_upper", 70)}],
            ),
            "bollinger": (
                [{"indicator": "bollinger", "compare": "below_lower"}],
                [{"indicator": "bollinger", "compare": "above_upper"}],
            ),
            "macd": (
                [{"indicator": "macd", "direction": "bullish"}],
                [{"indicator": "macd", "direction": "bearish"}],
            ),
            "combined": (
                [{"indicator": "ma_trend", "direction": "bullish"},
                 {"indicator": "rsi", "compare": "below", "value": params.get("rsi_lower", 30)}],
                [{"indicator": "ma_trend", "direction": "bearish"},
                 {"indicator": "rsi", "compare": "above", "value": params.get("rsi_upper", 70)}],
            ),
        }
        entry_conds, exit_conds = legacy_map.get(strategy, legacy_map["ma_crossover"])
        if strategy == "combined":
            entry_logic = "AND"
            exit_logic = "OR"

    # ── Evaluate entry conditions ──
    if entry_conds:
        entry_signals = [_eval_single_condition(c, close, high, low, indicators, params) for c in entry_conds]
        if entry_logic == "OR":
            entries = entry_signals[0]
            for s in entry_signals[1:]:
                entries = entries | s
        else:
            entries = entry_signals[0]
            for s in entry_signals[1:]:
                entries = entries & s
    else:
        entries = pd.Series(False, index=close.index)

    # ── Evaluate exit conditions ──
    if exit_conds:
        exit_signals = [_eval_single_condition(c, close, high, low, indicators, params) for c in exit_conds]
        if exit_logic == "OR":
            exits = exit_signals[0]
            for s in exit_signals[1:]:
                exits = exits | s
        else:
            exits = exit_signals[0]
            for s in exit_signals[1:]:
                exits = exits & s
    else:
        exits = pd.Series(False, index=close.index)

    entries = entries.fillna(False)
    exits = exits.fillna(False)

    return entries, exits


def run_backtest(open_, close, entries, exits, params):
    """
    Run vectorbt portfolio simulation.

    Execution model (next_bar_open=True, recommended):
      Signal fires at close of bar N  →  trade executes at OPEN of bar N+1.
      This avoids look-ahead bias and matches real-world market order flow.

    Execution model (next_bar_open=False):
      Trade executes at CLOSE of the signal bar itself — unrealistic for
      end-of-day strategies but useful for intraday limit-order simulation.
    """
    init_cash    = params.get("init_cash", 10000)
    fees         = params.get("fees", 0.001)
    slippage     = params.get("slippage", 0.001)
    sl_stop      = params.get("sl_stop", None)
    tp_stop      = params.get("tp_stop", None)
    interval     = params.get("_interval", "1d")
    freq         = INTERVAL_TO_FREQ.get(interval, "1D")
    next_bar_open = params.get("next_bar_open", True)   # default: realistic execution

    if next_bar_open:
        # ── Shift signals forward by 1 bar ──────────────────────────────────
        # Signal detected at close of bar N → trigger fires on bar N+1.
        # We also pass open_ as the execution price so VectorBT fills the
        # order at the open of bar N+1 rather than its close.
        exec_entries = entries.shift(1).fillna(False)
        exec_exits   = exits.shift(1).fillna(False)
    else:
        exec_entries = entries
        exec_exits   = exits

    pf_kwargs = dict(
        close=close,
        entries=exec_entries,
        exits=exec_exits,
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        freq=freq,
    )

    # Pass open prices for execution when next_bar_open is on
    # VectorBT ≥ 0.26 supports the `open` kwarg in from_signals for intraday fills
    if next_bar_open and open_ is not None:
        try:
            pf_kwargs["open"] = open_
        except Exception:
            pass   # silently skip if this VectorBT build doesn't support it

    if sl_stop and sl_stop > 0:
        pf_kwargs["sl_stop"] = sl_stop / 100.0
    if tp_stop and tp_stop > 0:
        pf_kwargs["tp_stop"] = tp_stop / 100.0

    pf = vbt.Portfolio.from_signals(**pf_kwargs)
    return pf


def extract_results(pf, open_, close, high, low, entries, exits, indicators, params):
    """Extract all metrics and chart data from the portfolio."""

    # ── Core Metrics ──
    stats = pf.stats()
    total_return = safe_val(pf.total_return())
    sharpe = safe_val(stats.get("Sharpe Ratio", None))
    sortino = safe_val(stats.get("Sortino Ratio", None))
    max_dd = safe_val(pf.max_drawdown())
    total_trades = safe_val(pf.trades.count())
    win_rate = safe_val(pf.trades.win_rate()) if total_trades > 0 else None
    profit_factor = safe_val(stats.get("Profit Factor", None))
    avg_trade = safe_val(pf.trades.pnl.mean()) if total_trades > 0 else None
    best_trade_pnl = safe_val(pf.trades.pnl.max()) if total_trades > 0 else None
    worst_trade_pnl = safe_val(pf.trades.pnl.min()) if total_trades > 0 else None
    calmar = safe_val(stats.get("Calmar Ratio", None))
    expectancy = safe_val(stats.get("Expectancy", None))
    omega = safe_val(stats.get("Omega Ratio", None))

    # ── Extended Metrics ──
    init_cash = params.get("init_cash", 10000)
    start_date = close.index[0].isoformat() if len(close) > 0 else None
    end_date = close.index[-1].isoformat() if len(close) > 0 else None

    # Period duration
    if len(close) > 1:
        period_td = close.index[-1] - close.index[0]
        period_days = period_td.days
        period_str = f"{period_days} days"
    else:
        period_days = 0
        period_str = "0 days"

    # Max Gross Exposure
    try:
        max_gross_exposure = safe_val(stats.get("Max Gross Exposure [%]", None))
        if max_gross_exposure is None:
            max_gross_exposure = safe_val(pf.gross_exposure().max() * 100)
    except Exception:
        max_gross_exposure = None

    # Total Fees Paid
    try:
        total_fees = safe_val(stats.get("Total Fees Paid", None))
        if total_fees is None:
            total_fees = safe_val(pf.trades.records_readable["Entry Fees"].sum() + pf.trades.records_readable["Exit Fees"].sum()) if total_trades > 0 else 0
    except Exception:
        total_fees = safe_val(stats.get("Total Fees Paid", 0))

    # Max Drawdown Duration
    try:
        max_dd_duration = stats.get("Max Drawdown Duration", None)
        if max_dd_duration is not None:
            max_dd_duration_str = str(max_dd_duration)
        else:
            max_dd_duration_str = None
    except Exception:
        max_dd_duration_str = None

    # Closed vs Open trades
    try:
        total_closed = safe_val(stats.get("Total Closed Trades", None))
        total_open = safe_val(stats.get("Total Open Trades", None))
        if total_closed is None:
            total_closed = total_trades
        if total_open is None:
            total_open = 0
    except Exception:
        total_closed = total_trades
        total_open = 0

    # Open Trade PnL
    try:
        open_pnl = safe_val(stats.get("Open Trade PnL", None))
        if open_pnl is None:
            open_pnl = 0
    except Exception:
        open_pnl = 0

    # Best/Worst Trade in percentage
    best_trade_pct = None
    worst_trade_pct = None
    avg_winning_pct = None
    avg_losing_pct = None
    avg_winning_duration = None
    avg_losing_duration = None

    if total_trades > 0:
        try:
            trade_returns = pf.trades.records_readable["Return"]
            best_trade_pct = safe_val(trade_returns.max() * 100)
            worst_trade_pct = safe_val(trade_returns.min() * 100)

            # Winning vs losing trade averages
            winning = trade_returns[trade_returns > 0]
            losing = trade_returns[trade_returns <= 0]
            avg_winning_pct = safe_val(winning.mean() * 100) if len(winning) > 0 else None
            avg_losing_pct = safe_val(losing.mean() * 100) if len(losing) > 0 else None
        except Exception:
            pass

        # Avg duration for winning/losing trades
        try:
            records = pf.trades.records_readable
            if "Entry Timestamp" in records.columns and "Exit Timestamp" in records.columns:
                records["Duration"] = pd.to_datetime(records["Exit Timestamp"]) - pd.to_datetime(records["Entry Timestamp"])
                records["IsWin"] = records["Return"] > 0
                winning_dur = records.loc[records["IsWin"], "Duration"]
                losing_dur = records.loc[~records["IsWin"], "Duration"]
                avg_winning_duration = str(winning_dur.mean()).split(".")[0] if len(winning_dur) > 0 else None
                avg_losing_duration = str(losing_dur.mean()).split(".")[0] if len(losing_dur) > 0 else None
        except Exception:
            pass

    # Benchmark return
    interval = params.get("_interval", "1d")
    freq = INTERVAL_TO_FREQ.get(interval, "1D")
    pf_bh = vbt.Portfolio.from_holding(close, init_cash=init_cash, freq=freq)
    benchmark_return = safe_val(pf_bh.total_return())

    metrics = {
        "start": start_date,
        "end": end_date,
        "period": period_str,
        "start_value": init_cash,
        "end_value": safe_val(pf.final_value()),
        "total_return": total_return,
        "total_return_pct": round(total_return * 100, 2) if total_return else None,
        "benchmark_return_pct": round(benchmark_return * 100, 2) if benchmark_return else None,
        "max_gross_exposure": round(max_gross_exposure, 2) if max_gross_exposure else None,
        "total_fees_paid": round(total_fees, 2) if total_fees else 0,
        "max_drawdown": round(max_dd * 100, 2) if max_dd else None,
        "max_drawdown_duration": max_dd_duration_str,
        "total_trades": total_trades,
        "total_closed_trades": total_closed,
        "total_open_trades": total_open,
        "open_trade_pnl": round(open_pnl, 2) if open_pnl else 0,
        "win_rate": round(win_rate * 100, 2) if win_rate else None,
        "best_trade_pct": round(best_trade_pct, 2) if best_trade_pct else None,
        "worst_trade_pct": round(worst_trade_pct, 2) if worst_trade_pct else None,
        "avg_winning_trade_pct": round(avg_winning_pct, 2) if avg_winning_pct else None,
        "avg_losing_trade_pct": round(avg_losing_pct, 2) if avg_losing_pct else None,
        "avg_winning_duration": avg_winning_duration,
        "avg_losing_duration": avg_losing_duration,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "omega_ratio": omega,
        "sortino_ratio": sortino,
        "avg_trade_pnl": avg_trade,
        "best_trade": best_trade_pnl,
        "worst_trade": worst_trade_pnl,
        "final_value": safe_val(pf.final_value()),
        "init_cash": init_cash,
    }

    # ── Equity Curve ──
    equity = pf.value()
    dates = [d.isoformat() for d in equity.index]
    equity_data = {
        "dates": dates,
        "equity": [safe_val(v) for v in equity.values],
        "close": [safe_val(v) for v in close.values],
    }

    # ── Drawdown ──
    dd = pf.drawdown()
    dd_data = {
        "dates": dates,
        "drawdown": [safe_val(v * 100) for v in dd.values],
    }

    # ── Monthly Returns Heatmap ──
    try:
        returns = pf.returns()
        monthly = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
        monthly_data = []
        for dt, ret in monthly.items():
            monthly_data.append({
                "year": dt.year,
                "month": dt.month,
                "return": round(safe_val(ret) * 100, 2) if safe_val(ret) is not None else 0,
            })
    except Exception:
        monthly_data = []

    # ── Trade Log ──
    trade_log = []
    if total_trades > 0:
        trades = pf.trades.records_readable
        for _, t in trades.iterrows():
            trade_log.append({
                "entry_date": str(t.get("Entry Timestamp", t.get("Entry Index", ""))),
                "exit_date": str(t.get("Exit Timestamp", t.get("Exit Index", ""))),
                "direction": str(t.get("Direction", "Long")),
                "size": safe_val(t.get("Size", 0)),
                "entry_price": safe_val(t.get("Avg Entry Price", t.get("Entry Price", 0))),
                "exit_price": safe_val(t.get("Avg Exit Price", t.get("Exit Price", 0))),
                "pnl": safe_val(t.get("PnL", 0)),
                "return_pct": round(safe_val(t.get("Return", 0)) * 100, 2),
            })

    # ── Indicator chart data ──
    indicator_data = {}
    for key, val in indicators.items():
        if key.startswith("cdl_"):
            continue                         # candlestick patterns handled in patterns{}
        if key == "cdc_zone":
            indicator_data[key] = list(val.values)  # string array — keep as-is
            continue
        if hasattr(val, "values"):
            try:
                indicator_data[key] = [safe_val(v) for v in val.values]
            except Exception:
                pass

    # CDC buy/sell signal dates (for chart markers)
    for sig_key, date_key in [("cdc_buy", "cdc_buy_dates"), ("cdc_sell", "cdc_sell_dates")]:
        if sig_key in indicators:
            mask = indicators[sig_key].fillna(False).astype(bool).values
            indicator_data[date_key] = [d.isoformat() for d, m in zip(close.index, mask) if m]

    # RSI divergence signal dates (for chart markers on the RSI subplot)
    rdiv_signal_keys = [
        ("rdiv_buy",        "rdiv_buy_dates"),
        ("rdiv_sell",       "rdiv_sell_dates"),
        ("rdiv_bull",       "rdiv_bull_dates"),
        ("rdiv_bear",       "rdiv_bear_dates"),
        ("rdiv_hidden_bull","rdiv_hidden_bull_dates"),
        ("rdiv_hidden_bear","rdiv_hidden_bear_dates"),
        ("rdiv_conv_bull",  "rdiv_conv_bull_dates"),
        ("rdiv_conv_bear",  "rdiv_conv_bear_dates"),
        ("rdiv_buy_mom",    "rdiv_buy_mom_dates"),
        ("rdiv_sell_mom",   "rdiv_sell_mom_dates"),
    ]
    for sig_key, date_key in rdiv_signal_keys:
        if sig_key in indicators:
            mask = indicators[sig_key].fillna(False).astype(bool).values
            indicator_data[date_key] = [d.isoformat() for d, m in zip(close.index, mask) if m]

    # ── Entry/Exit markers ──
    # When next_bar_open is on, signals were shifted by 1 in run_backtest,
    # so the actual trade executes one bar later than the raw signal.
    # We expose the ORIGINAL signal dates (for overlay on signal bar) and
    # the EXECUTION dates (shifted, where the order was actually filled).
    next_bar_open = params.get("next_bar_open", True)
    raw_entry_dates = [d.isoformat() for d in entries[entries].index]
    raw_exit_dates  = [d.isoformat() for d in exits[exits].index]

    if next_bar_open:
        # Execution happens 1 bar after the signal
        def _shift_dates_by_one(dates, all_dates):
            idx_map = {d: i for i, d in enumerate(all_dates)}
            out = []
            for d in dates:
                i = idx_map.get(d)
                if i is not None and i + 1 < len(all_dates):
                    out.append(all_dates[i + 1])
            return out
        all_iso = [d.isoformat() for d in close.index]
        entry_dates = _shift_dates_by_one(raw_entry_dates, all_iso)
        exit_dates  = _shift_dates_by_one(raw_exit_dates,  all_iso)
    else:
        entry_dates = raw_entry_dates
        exit_dates  = raw_exit_dates

    # Execution prices: open of the execution bar
    open_map  = {d.isoformat(): safe_val(v) for d, v in zip(close.index, open_.values)}
    entry_prices = [open_map.get(d) for d in entry_dates]
    exit_prices  = [open_map.get(d) for d in exit_dates]

    # ── OHLCV for candlestick price chart ──
    ohlcv_data = {
        "dates": dates,
        "open":  [safe_val(v) for v in open_.values],
        "high":  [safe_val(v) for v in high.values],
        "low":   [safe_val(v) for v in low.values],
        "close": [safe_val(v) for v in close.values],
    }

    # ── Candlestick pattern detections (dates + marker prices for overlay) ──
    pattern_detections = {}
    for key, val in indicators.items():
        if not key.startswith("cdl_"):
            continue
        mask_arr = val.fillna(False).astype(bool).values
        if not mask_arr.any():
            continue
        detected_dates = [d.isoformat() for d, m in zip(close.index, mask_arr) if m]
        if key in BULLISH_PATTERNS:
            marker_prices = [safe_val(v * 0.997) for v, m in zip(low.values, mask_arr) if m]
        elif key in BEARISH_PATTERNS:
            marker_prices = [safe_val(v * 1.003) for v, m in zip(high.values, mask_arr) if m]
        else:  # neutral (Doji)
            marker_prices = [safe_val((h + l) / 2) for h, l, m in zip(high.values, low.values, mask_arr) if m]
        meta = CDL_PATTERN_META.get(key, {"label": key, "type": "neutral"})
        pattern_detections[key] = {
            "dates": detected_dates,
            "prices": marker_prices,
            "label": meta["label"],
            "type": meta["type"],
            "count": len(detected_dates),
        }

    # ── Benchmark (buy & hold) ──
    interval = params.get("_interval", "1d")
    freq = INTERVAL_TO_FREQ.get(interval, "1D")
    pf_bh = vbt.Portfolio.from_holding(close, init_cash=params.get("init_cash", 10000), freq=freq)
    benchmark = {
        "equity": [safe_val(v) for v in pf_bh.value().values],
        "total_return_pct": round(safe_val(pf_bh.total_return()) * 100, 2),
        "sharpe": safe_val(pf_bh.stats().get("Sharpe Ratio", None)),
        "max_drawdown": round(safe_val(pf_bh.max_drawdown()) * 100, 2),
    }

    return {
        "metrics": metrics,
        "equity": equity_data,
        "ohlcv": ohlcv_data,
        "drawdown": dd_data,
        "monthly_returns": monthly_data,
        "trade_log": trade_log,
        "indicators": indicator_data,
        "patterns": pattern_detections,
        "entry_dates": entry_dates,
        "exit_dates": exit_dates,
        "entry_prices": entry_prices,
        "exit_prices": exit_prices,
        "signal_dates": {"entry": raw_entry_dates, "exit": raw_exit_dates},
        "next_bar_open": next_bar_open,
        "benchmark": benchmark,
    }


# ─── API Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/diagnose", methods=["GET"])
def diagnose():
    """Diagnostic endpoint — visit http://localhost:PORT/api/diagnose in your browser."""
    import sys
    results = {
        "python_version": sys.version,
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "vectorbt_version": getattr(vbt, "__version__", "unknown"),
        "tests": [],
    }

    test_ticker = "AAPL"

    # Test 1: Ticker object creation
    try:
        t = yf.Ticker(test_ticker)
        info = getattr(t, "info", None)
        results["tests"].append({
            "name": "Ticker object creation",
            "status": "OK",
            "detail": f"Ticker created; info keys: {list((info or {}).keys())[:5]}",
        })
    except Exception as e:
        results["tests"].append({
            "name": "Ticker object creation",
            "status": "FAIL",
            "detail": str(e),
        })

    # Test 2: Ticker.history()
    try:
        t = yf.Ticker(test_ticker)
        data = t.history(period="5d")
        results["tests"].append({
            "name": f"Ticker.history(period='5d') for {test_ticker}",
            "status": "OK" if data is not None and not data.empty else "EMPTY",
            "detail": f"shape={getattr(data, 'shape', None)}, cols={list(getattr(data, 'columns', []))}",
        })
    except Exception as e:
        results["tests"].append({
            "name": f"Ticker.history() for {test_ticker}",
            "status": "FAIL",
            "detail": str(e),
        })

    # Test 3: yf.download
    try:
        data = yf.download(test_ticker, period="5d", progress=False)
        results["tests"].append({
            "name": f"yf.download(period='5d') for {test_ticker}",
            "status": "OK" if data is not None and not data.empty else "EMPTY",
            "detail": f"shape={getattr(data, 'shape', None)}, cols={list(getattr(data, 'columns', []))}, "
                      f"col_type={type(getattr(data, 'columns', None)).__name__}",
        })
    except Exception as e:
        results["tests"].append({
            "name": f"yf.download() for {test_ticker}",
            "status": "FAIL",
            "detail": str(e),
        })

    # Test 4: BTC-USD specifically
    try:
        t = yf.Ticker("BTC-USD")
        data = t.history(period="5d")
        results["tests"].append({
            "name": "Ticker.history(period='5d') for BTC-USD",
            "status": "OK" if data is not None and not data.empty else "EMPTY",
            "detail": f"shape={getattr(data, 'shape', None)}, cols={list(getattr(data, 'columns', []))}",
        })
    except Exception as e:
        results["tests"].append({
            "name": "Ticker.history() for BTC-USD",
            "status": "FAIL",
            "detail": str(e),
        })

    return jsonify(results)


@app.route("/api/fetch-data", methods=["POST"])
def fetch_data():
    """Fetch OHLCV data for a given ticker and date range."""
    try:
        body = request.json
        ticker = body.get("ticker", "SPY")
        start = body.get("start", (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d"))
        end = body.get("end", datetime.now().strftime("%Y-%m-%d"))
        interval = body.get("interval", "1d")

        data = fetch_ohlcv(ticker, start, end, interval=interval)

        result = {
            "dates": [d.isoformat() for d in data.index],
            "open": [safe_val(v) for v in data["Open"].values],
            "high": [safe_val(v) for v in data["High"].values],
            "low": [safe_val(v) for v in data["Low"].values],
            "close": [safe_val(v) for v in data["Close"].values],
            "volume": [safe_val(v) for v in data["Volume"].values],
            "ticker": ticker,
            "interval": interval,
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ─── Correlation benchmark definitions ──────────────────────────────────────────

CORRELATION_BENCHMARKS = [
    {"key": "gold",     "label": "Gold",         "ticker": "GLD"},
    {"key": "silver",   "label": "Silver",        "ticker": "SLV"},
    {"key": "bitcoin",  "label": "Bitcoin",       "ticker": "BTC-USD"},
    {"key": "sp500",    "label": "S&P 500",       "ticker": "SPY"},
    {"key": "nasdaq",   "label": "Nasdaq 100",    "ticker": "QQQ"},
    {"key": "usd",      "label": "USD Index",     "ticker": "UUP"},
    {"key": "set",      "label": "SET Index",     "ticker": "^SET.BK"},
    {"key": "set100",   "label": "SET100",        "ticker": "^SET100.BK"},
    {"key": "oil",      "label": "Oil (WTI)",     "ticker": "USO"},
    {"key": "bonds",    "label": "US 20Y Bonds",  "ticker": "TLT"},
]


@app.route("/api/correlation", methods=["POST"])
def correlation():
    """Compute correlation of the user's asset against multiple global benchmarks."""
    try:
        body = request.json
        ticker = body.get("ticker", "SPY")
        start  = body.get("start",  (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d"))
        end    = body.get("end",    datetime.now().strftime("%Y-%m-%d"))
        interval = body.get("interval", "1d")

        # ── 1. Fetch user asset ──
        try:
            user_data = fetch_ohlcv(ticker, start, end, interval)
            closes = {ticker: user_data["Close"]}
        except Exception as e:
            return jsonify({"error": f"Failed to fetch {ticker}: {e}"}), 500

        # ── 2. Fetch benchmarks (skip silently if unavailable) ──
        fetched_meta = []
        for bm in CORRELATION_BENCHMARKS:
            if bm["ticker"].upper() == ticker.upper():
                continue  # don't correlate with itself
            try:
                d = fetch_ohlcv(bm["ticker"], start, end, interval)
                if d is not None and not d.empty and "Close" in d.columns:
                    closes[bm["label"]] = d["Close"]
                    fetched_meta.append({"key": bm["key"], "label": bm["label"], "ticker": bm["ticker"]})
            except Exception:
                pass  # silently skip unavailable tickers

        # ── 3. Align all series on a common date index ──
        df = pd.DataFrame(closes)
        df = df.dropna(how="all")
        df = df.ffill().bfill()  # forward/backward fill small gaps

        if df.empty or len(df.columns) < 2:
            return jsonify({"error": "Not enough data to compute correlations."}), 400

        # ── 4. Compute daily log returns ──
        returns = df.pct_change().dropna()

        # ── 5. Correlation matrix ──
        corr = returns.corr()

        # ── 6. Rolling 30-day correlation vs user asset ──
        rolling_window = 30
        rolling_corr = {}
        for col in returns.columns:
            if col == ticker:
                continue
            rc = returns[ticker].rolling(rolling_window).corr(returns[col]).dropna()
            rolling_corr[col] = {
                "dates":  [d.isoformat() for d in rc.index],
                "values": [safe_val(v) for v in rc.values],
            }

        # ── 7. Normalised price series (rebased to 100 at start) ──
        norm = (df / df.iloc[0]) * 100
        normalised = {}
        for col in norm.columns:
            normalised[col] = {
                "dates":  [d.isoformat() for d in norm.index],
                "values": [safe_val(v) for v in norm[col].values],
            }

        # ── 8. Annualised stats per asset ──
        stats = {}
        for col in returns.columns:
            r = returns[col].dropna()
            ann_ret  = safe_val(r.mean() * 252)
            ann_vol  = safe_val(r.std() * np.sqrt(252))
            sr       = safe_val(ann_ret / ann_vol) if ann_vol and ann_vol != 0 else None
            stats[col] = {
                "ann_return": round(ann_ret * 100, 2) if ann_ret is not None else None,
                "ann_vol":    round(ann_vol * 100, 2) if ann_vol is not None else None,
                "sharpe":     round(sr, 2) if sr is not None else None,
            }

        return jsonify({
            "ticker":      ticker,
            "assets":      list(df.columns),
            "benchmarks":  fetched_meta,
            "normalised":  normalised,
            "corr_matrix": {
                "labels": list(corr.columns),
                "values": [[safe_val(v) for v in row] for row in corr.values],
            },
            "rolling_corr":  rolling_corr,
            "rolling_window": rolling_window,
            "stats":         stats,
        })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/backtest", methods=["POST"])
def backtest():
    """Run a full backtest with the given parameters."""
    try:
        body = request.json
        ticker = body.get("ticker", "SPY")
        start = body.get("start", (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d"))
        end = body.get("end", datetime.now().strftime("%Y-%m-%d"))
        interval = body.get("interval", "1d")
        params = body.get("params", {})
        params["_interval"] = interval  # pass interval for freq mapping

        # Fetch data
        data = fetch_ohlcv(ticker, start, end, interval=interval)

        open_ = data["Open"]
        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        # Compute indicators (includes candlestick patterns)
        indicators = compute_indicators(open_, close, high, low, params)

        # Generate signals
        entries, exits = generate_signals(close, high, low, indicators, params)

        # Run backtest (next_bar_open=True by default → execute at open of next bar)
        pf = run_backtest(open_, close, entries, exits, params)

        # Extract results
        results = extract_results(pf, open_, close, high, low, entries, exits, indicators, params)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/optimize", methods=["POST"])
def optimize():
    """Run parameter optimization grid search."""
    try:
        body = request.json
        ticker = body.get("ticker", "SPY")
        start = body.get("start", (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d"))
        end = body.get("end", datetime.now().strftime("%Y-%m-%d"))
        interval = body.get("interval", "1d")
        base_params = body.get("params", {})
        base_params["_interval"] = interval
        opt_config = body.get("optimization", {})

        # Fetch data
        data = fetch_ohlcv(ticker, start, end, interval=interval)

        open_ = data["Open"]
        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        # Build parameter grid
        param_ranges = {}
        for param_name, config in opt_config.items():
            start_val = config.get("start", 5)
            end_val = config.get("end", 50)
            step = config.get("step", 5)
            param_ranges[param_name] = list(range(int(start_val), int(end_val) + 1, int(step)))

        # Run grid search
        results = []
        param_keys = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        # Generate all combinations
        from itertools import product as iter_product
        combinations = list(iter_product(*param_values))

        # Limit to 500 combinations max
        if len(combinations) > 500:
            step_mult = max(2, len(combinations) // 500)
            combinations = combinations[::step_mult]

        for combo in combinations:
            test_params = {**base_params}
            for i, key in enumerate(param_keys):
                test_params[key] = combo[i]

            try:
                indicators = compute_indicators(open_, close, high, low, test_params)
                entries, exits = generate_signals(close, high, low, indicators, test_params)
                pf = run_backtest(open_, close, entries, exits, test_params)

                total_return = safe_val(pf.total_return())
                stats = pf.stats()
                sharpe = safe_val(stats.get("Sharpe Ratio", None))
                max_dd = safe_val(pf.max_drawdown())
                trades = safe_val(pf.trades.count())
                win_rate = safe_val(pf.trades.win_rate()) if trades and trades > 0 else None

                result_entry = {k: combo[i] for i, k in enumerate(param_keys)}
                result_entry.update({
                    "total_return_pct": round(total_return * 100, 2) if total_return else None,
                    "sharpe_ratio": sharpe,
                    "max_drawdown_pct": round(max_dd * 100, 2) if max_dd else None,
                    "total_trades": trades,
                    "win_rate": round(win_rate * 100, 2) if win_rate else None,
                })
                results.append(result_entry)
            except Exception:
                continue

        # Sort by Sharpe ratio (descending)
        results.sort(key=lambda x: x.get("sharpe_ratio") or -999, reverse=True)

        return jsonify({
            "results": results,
            "total_combinations": len(combinations),
            "param_keys": param_keys,
        })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None, help="Port to run the server on")
    args = parser.parse_args()

    # PORT env var takes priority (Railway/Render/Fly.io inject this automatically)
    port = args.port or int(os.environ.get("PORT", 8501))

    print("\n" + "=" * 60)
    print("  VectorBT Backtesting Dashboard")
    print(f"  Open http://localhost:{port} in your browser")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)
