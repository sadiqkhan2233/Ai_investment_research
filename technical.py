"""Technical indicators (Pandas/NumPy only). Descriptive, never predictive."""
from __future__ import annotations

import numpy as np
import pandas as pd


def has_ohlc(df: pd.DataFrame) -> bool:
    """True only if real High/Low exist (not just a copy of Close)."""
    if df is None or df.empty or not {"High", "Low"}.issubset(df.columns):
        return False
    h, l = df["High"].dropna(), df["Low"].dropna()
    if h.empty or l.empty:
        return False
    return not bool((df["High"] == df["Close"]).all() and (df["Low"] == df["Close"]).all())


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("Date").reset_index(drop=True)
    c = d["Close"].astype(float)
    for n in (20, 50, 200):
        d[f"SMA{n}"] = c.rolling(n).mean()
        d[f"EMA{n}"] = c.ewm(span=n, adjust=False).mean()
    delta = c.diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_dn = dn.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    d["RSI14"] = 100 - 100 / (1 + rs)
    d.loc[(avg_dn == 0) & (avg_up > 0), "RSI14"] = 100.0
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["BB_mid"], d["BB_upper"], d["BB_lower"] = mid, mid + 2 * sd, mid - 2 * sd
    if has_ohlc(d):
        pc = c.shift(1)
        tr = pd.concat([d["High"] - d["Low"], (d["High"] - pc).abs(), (d["Low"] - pc).abs()], axis=1).max(axis=1)
        d["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    else:
        d["ATR14"] = np.nan
    if "Volume" in d:
        d["VOL_SMA20"] = d["Volume"].rolling(20).mean()
    return d


def _f(x, nd=2):
    try:
        return None if pd.isna(x) else round(float(x), nd)
    except (TypeError, ValueError):
        return None


def summarize(d: pd.DataFrame) -> dict:
    """Facts (observations) separated from signals (interpretations)."""
    if d is None or d.empty:
        return {}
    last = d.iloc[-1]
    close = float(last["Close"])
    ohlc = has_ohlc(d)
    win = d.tail(120)
    support = float(win["Low"].min()) if ohlc else float(win["Close"].min())
    resist = float(win["High"].max()) if ohlc else float(win["Close"].max())
    obs = {
        "data_start": str(d["Date"].iloc[0].date()),
        "last_date": str(d["Date"].iloc[-1].date()),
        "rows": int(len(d)),
        "last_close": _f(close),
        "sma20": _f(last["SMA20"]), "sma50": _f(last["SMA50"]), "sma200": _f(last["SMA200"]),
        "ema20": _f(last["EMA20"]), "ema50": _f(last["EMA50"]), "ema200": _f(last["EMA200"]),
        "rsi14": _f(last["RSI14"]), "macd": _f(last["MACD"], 3), "macd_signal": _f(last["MACD_signal"], 3),
        "bb_upper": _f(last["BB_upper"]), "bb_lower": _f(last["BB_lower"]), "atr14": _f(last["ATR14"]),
        "support_120d": _f(support), "resistance_120d": _f(resist),
        "ohlc_available": ohlc,
    }
    if "Volume" in d and len(d) >= 60:
        v20, v60 = d["Volume"].tail(20).mean(), d["Volume"].tail(60).mean()
        obs["volume_20d_vs_60d_ratio"] = _f(v20 / v60) if v60 else None
    sig = []
    if not pd.isna(last["SMA50"]):
        sig.append("Price above SMA50 (short/medium trend up)" if close > last["SMA50"] else "Price below SMA50 (short/medium trend down)")
    if not pd.isna(last["SMA200"]):
        sig.append("Price above SMA200 (long-term trend up)" if close > last["SMA200"] else "Price below SMA200 (long-term trend down)")
        if not pd.isna(last["SMA50"]):
            sig.append("SMA50 above SMA200 (golden-cross regime)" if last["SMA50"] > last["SMA200"] else "SMA50 below SMA200 (death-cross regime)")
    if not pd.isna(last["RSI14"]):
        r = last["RSI14"]
        sig.append("RSI > 70: overbought zone" if r > 70 else "RSI < 30: oversold zone" if r < 30 else "RSI neutral (30-70)")
    if not pd.isna(last["MACD_hist"]):
        sig.append("MACD histogram positive (bullish momentum)" if last["MACD_hist"] > 0 else "MACD histogram negative (bearish momentum)")
    if not pd.isna(last["BB_upper"]):
        sig.append("Close above upper Bollinger Band" if close > last["BB_upper"] else "Close below lower Bollinger Band" if close < last["BB_lower"] else "Close inside Bollinger Bands")
    return {"observations": obs, "signals": sig}
