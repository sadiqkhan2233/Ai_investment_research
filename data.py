"""Data providers (modular - swap these functions to change data source).

Sources, in order: uploaded CSV -> Yahoo Finance (<TICKER>.KA) -> PSX data portal
(dps.psx.com.pk, close/volume only). Nothing is ever fabricated: if a source has
no data the dataset is simply marked unavailable.
"""
from __future__ import annotations

import io
import re
from datetime import date

import pandas as pd
import requests
import yfinance as yf

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def clean_ticker(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]", "", t or "").upper()[:12]


def _yf_prices(t: str, period: str = "5y") -> pd.DataFrame:
    try:
        df = yf.Ticker(f"{t}.KA").history(period=period, auto_adjust=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    first = df.columns[0]
    df = df.rename(columns={first: "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume", "Dividends"] if c in df.columns]
    return df[cols].dropna(subset=["Close"])


def _psx_prices(t: str) -> pd.DataFrame:
    """PSX data-portal EOD feed. Assumed row layout [timestamp, close, volume, open];
    only timestamp/close/volume are used. Treat as a fallback."""
    try:
        r = requests.get(f"https://dps.psx.com.pk/timeseries/eod/{t}", timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        rows = r.json().get("data") or []
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).iloc[:, :3]
        df.columns = ["ts", "Close", "Volume"]
        df["Date"] = pd.to_datetime(df["ts"], unit="s")
        df = df.drop(columns="ts")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        return df.dropna(subset=["Close"]).sort_values("Date")[["Date", "Close", "Volume"]]
    except Exception:
        return pd.DataFrame()


def load_uploaded_prices(file) -> tuple[pd.DataFrame, str | None]:
    """Validate an uploaded CSV. Returns (df, error)."""
    try:
        raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            return pd.DataFrame(), "File larger than 5 MB."
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        return pd.DataFrame(), f"Could not read CSV: {e}"
    df.columns = [str(c).strip().title() for c in df.columns]
    if "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame(), "CSV needs at least 'Date' and 'Close' columns."
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").tail(20000)
    if len(df) < 30:
        return pd.DataFrame(), "Need at least 30 valid rows."
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep], None


def fetch_prices(t: str, uploaded: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str]:
    if uploaded is not None and not uploaded.empty:
        return uploaded, "User-uploaded CSV"
    df = _yf_prices(t)
    if not df.empty:
        return df, f"Yahoo Finance ({t}.KA)"
    df = _psx_prices(t)
    if not df.empty:
        return df, "PSX data portal (close/volume only)"
    return pd.DataFrame(), "unavailable"


def _news(tk) -> list[dict]:
    items = []
    try:
        for n in (tk.news or [])[:10]:
            c = n.get("content", n)
            url = (c.get("canonicalUrl") or {}).get("url") or c.get("link") or ""
            items.append({
                "title": c.get("title", ""),
                "publisher": (c.get("provider") or {}).get("displayName") or c.get("publisher", ""),
                "date": str(c.get("pubDate") or c.get("providerPublishTime") or ""),
                "url": url,
            })
    except Exception:
        pass
    return [i for i in items if i["title"]]


def fetch_company(t: str, uploaded: pd.DataFrame | None = None) -> dict:
    """Everything fetched for one ticker, with errors recorded instead of raised."""
    out = {"ticker": t, "retrieved": str(date.today()), "errors": [], "info": {}, "news": []}
    tk = yf.Ticker(f"{t}.KA")
    for key, attr in (("income", "financials"), ("balance", "balance_sheet"),
                      ("cashflow", "cashflow"), ("q_income", "quarterly_financials")):
        try:
            df = getattr(tk, attr)
            out[key] = df if df is not None and not df.empty else pd.DataFrame()
        except Exception as e:  # noqa: BLE001
            out[key] = pd.DataFrame()
            out["errors"].append(f"{attr}: {e}")
    try:
        info = tk.info or {}
        out["info"] = {k: info.get(k) for k in ("longName", "sector", "industry", "currency", "financialCurrency",
                                                  "marketCap", "sharesOutstanding", "website")}
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"info: {e}")
    out["news"] = _news(tk)
    out["prices"], out["price_source"] = fetch_prices(t, uploaded)
    return out
