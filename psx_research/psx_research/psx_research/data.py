"""Data layer: PSX Data Portal + Yahoo Finance (.KA). No Streamlit imports (thread-safe for CrewAI)."""
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PSXResearch/1.0)"}


def fmt(x, d=2):
    try:
        if x is None or pd.isna(x) or np.isinf(x):
            return "Unavailable"
        return f"{x:,.{d}f}"
    except Exception:
        return "Unavailable"


def clear_caches():
    get_yahoo.cache_clear()
    get_prices.cache_clear()


# ----------------------------------------------------------------------------- prices
def _psx_prices(symbol):
    try:
        r = requests.get(f"https://dps.psx.com.pk/timeseries/eod/{symbol}", headers=HEADERS, timeout=20)
        r.raise_for_status()
        rows = r.json().get("data", [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df.columns = ["ts", "close", "volume", "open"][: df.shape[1]]
        df["date"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[["close", "volume"]].dropna(subset=["close"])
    except Exception:
        return None


@lru_cache(maxsize=32)
def get_yahoo(symbol):
    out = {"prices": None, "info": {}, "income": None, "balance": None,
           "cashflow": None, "dividends": [], "error": None}
    try:
        import yfinance as yf

        t = yf.Ticker(f"{symbol}.KA")
        try:
            h = t.history(period="5y", auto_adjust=False)
            if h is not None and not h.empty:
                h.index = pd.to_datetime(h.index).tz_localize(None)
                out["prices"] = h[["Close", "Volume"]].rename(columns=str.lower)
        except Exception:
            pass
        try:
            out["info"] = t.info or {}
        except Exception:
            pass
        for key, attr in [("income", "financials"), ("balance", "balance_sheet"), ("cashflow", "cashflow")]:
            try:
                df = getattr(t, attr)
                if df is not None and not df.empty:
                    out[key] = df
            except Exception:
                pass
        try:
            dv = t.dividends
            if dv is not None and len(dv) > 0:
                dv.index = pd.to_datetime(dv.index).tz_localize(None)
                out["dividends"] = [(str(i.date()), float(v)) for i, v in dv.tail(8).items()]
        except Exception:
            pass
    except Exception as e:
        out["error"] = str(e)
    return out


@lru_cache(maxsize=32)
def get_prices(symbol):
    df = _psx_prices(symbol)
    if df is not None and len(df) > 30:
        return df, "PSX Data Portal (dps.psx.com.pk)"
    y = get_yahoo(symbol)
    if y["prices"] is not None and len(y["prices"]) > 30:
        return y["prices"][["close", "volume"]], "Yahoo Finance (.KA)"
    return None, "No price data available"


def add_indicators(df):
    d = df.copy()
    c = d["close"]
    for n in (20, 50, 200):
        d[f"ma{n}"] = c.rolling(n).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    d["macd"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    return d


def price_summary(df):
    price = df["close"].iloc[-1]
    lines = [f"Last close: PKR {fmt(price)} on {df.index[-1].date()}"]
    w = df.tail(252)
    lines.append(f"52-week closing range: {fmt(w['close'].min())} - {fmt(w['close'].max())}")
    for label, days in [("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 252), ("3Y", 756)]:
        if len(df) > days:
            lines.append(f"Return {label}: {(price / df['close'].iloc[-days - 1] - 1) * 100:.1f}%")
    return "\n".join(lines)


def technical_summary(df):
    d = add_indicators(df)
    last = d.iloc[-1]
    price = last["close"]
    lines = [price_summary(df)]
    for n in (20, 50, 200):
        v = last[f"ma{n}"]
        rel = "n/a" if pd.isna(v) else ("above" if price > v else "below")
        lines.append(f"MA{n}: {fmt(v)} (price is {rel})")
    rsi = last["rsi"]
    zone = "n/a" if pd.isna(rsi) else ("overbought (>70)" if rsi > 70 else "oversold (<30)" if rsi < 30 else "neutral")
    lines.append(f"RSI(14): {fmt(rsi)} -> {zone}")
    lines.append(
        f"MACD: {fmt(last['macd'], 3)} | Signal: {fmt(last['macd_signal'], 3)} | "
        f"Histogram: {fmt(last['macd'] - last['macd_signal'], 3)}"
    )
    r = d.tail(60)
    lines.append(f"Approx. support (60-day closing low): {fmt(r['close'].min())}")
    lines.append(f"Approx. resistance (60-day closing high): {fmt(r['close'].max())}")
    lines.append(
        f"Volume: last {fmt(d['volume'].iloc[-1], 0)} | avg 20d {fmt(d['volume'].tail(20).mean(), 0)} "
        f"| avg 100d {fmt(d['volume'].tail(100).mean(), 0)}"
    )
    # Simple crossover context
    if not pd.isna(last["ma50"]) and not pd.isna(last["ma200"]):
        lines.append("MA50 vs MA200: " + ("golden-cross regime (50 above 200)" if last["ma50"] > last["ma200"] else "death-cross regime (50 below 200)"))
    return "\n".join(lines)


# ----------------------------------------------------------------------------- fundamentals
_MAP = {
    "revenue": ("income", ["Total Revenue", "Operating Revenue"]),
    "cogs": ("income", ["Cost Of Revenue"]),
    "gross": ("income", ["Gross Profit"]),
    "ebit": ("income", ["EBIT", "Operating Income"]),
    "ebitda": ("income", ["EBITDA", "Normalized EBITDA"]),
    "interest": ("income", ["Interest Expense", "Interest Expense Non Operating"]),
    "pbt": ("income", ["Pretax Income"]),
    "ni": ("income", ["Net Income", "Net Income Common Stockholders"]),
    "eps": ("income", ["Diluted EPS", "Basic EPS"]),
    "ta": ("balance", ["Total Assets"]),
    "ca": ("balance", ["Current Assets"]),
    "cl": ("balance", ["Current Liabilities"]),
    "inv": ("balance", ["Inventory"]),
    "rec": ("balance", ["Accounts Receivable", "Receivables"]),
    "debt": ("balance", ["Total Debt"]),
    "eq": ("balance", ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]),
    "tl": ("balance", ["Total Liabilities Net Minority Interest"]),
    "cash": ("balance", ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]),
    "cfo": ("cashflow", ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]),
    "capex": ("cashflow", ["Capital Expenditure"]),
    "fcf": ("cashflow", ["Free Cash Flow"]),
}

_LABELS = {
    "revenue": "Revenue", "cogs": "Cost of sales", "gross": "Gross profit", "ebit": "Operating profit (EBIT)",
    "ebitda": "EBITDA", "interest": "Finance cost", "pbt": "Profit before tax", "ni": "Net profit",
    "eps": "EPS (PKR)", "ta": "Total assets", "ca": "Current assets", "cl": "Current liabilities",
    "inv": "Inventory", "rec": "Receivables", "debt": "Total debt", "eq": "Equity", "tl": "Total liabilities",
    "cash": "Cash", "cfo": "Operating cash flow", "capex": "Capex", "fcf": "Free cash flow",
}


def _series(df, names):
    if df is None or getattr(df, "empty", True):
        return None
    for n in names:
        if n in df.index:
            s = df.loc[n]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            s = pd.to_numeric(s, errors="coerce").dropna()
            if not s.empty:
                s.index = pd.to_datetime(s.index)
                return s.sort_index()
    return None


def _collect(y):
    cols = {}
    for key, (stmt, names) in _MAP.items():
        s = _series(y.get(stmt), names)
        if s is not None:
            cols[key] = s
    if not cols:
        return None
    return pd.concat(cols, axis=1).sort_index()


def _cagr(s):
    s = s.dropna()
    if len(s) < 3 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return None
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return ((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100 if yrs > 0 else None


_NO_FIN = ("UNAVAILABLE: Yahoo Finance returned no financial statements for this ticker. "
           "Use figures from uploaded documents if present; otherwise state 'Unavailable'.")


def company_profile(symbol):
    info = get_yahoo(symbol)["info"]
    keys = ["longName", "sector", "industry", "country", "fullTimeEmployees", "website", "financialCurrency"]
    lines = [f"{k}: {info[k]}" for k in keys if info.get(k)]
    if info.get("longBusinessSummary"):
        lines.append("Business summary (Yahoo, unverified): " + info["longBusinessSummary"][:700])
    return "\n".join(lines) if lines else "UNAVAILABLE: no company profile from Yahoo."


def statements_text(symbol):
    y = get_yahoo(symbol)
    t = _collect(y)
    if t is None:
        return _NO_FIN
    tt = t.tail(5).copy()
    for c in tt.columns:
        if c != "eps":
            tt[c] = tt[c] / 1e6
    tt = tt.rename(columns=_LABELS).T
    tt.columns = [d.strftime("%Y-%m") for d in tt.columns]
    cur = y["info"].get("financialCurrency", "unknown (assumed PKR)")
    return (f"Annual statements from Yahoo Finance. Currency: {cur}. Values in millions (EPS in currency units).\n"
            + tt.to_string(float_format=lambda x: f"{x:,.2f}", na_rep="n/a"))


def compute_ratios(symbol):
    y = get_yahoo(symbol)
    t = _collect(y)
    if t is None:
        return _NO_FIN

    def c(k):
        return t[k] if k in t.columns else pd.Series(np.nan, index=t.index)

    def avg(k):
        return ((c(k) + c(k).shift(1)) / 2).fillna(c(k))

    fcf = c("fcf").fillna(c("cfo") + c("capex"))
    r = pd.DataFrame({
        "Revenue (m)": c("revenue") / 1e6,
        "Revenue growth %": c("revenue").pct_change() * 100,
        "Net profit (m)": c("ni") / 1e6,
        "Net profit growth %": c("ni").pct_change() * 100,
        "EPS": c("eps"),
        "Gross margin %": c("gross") / c("revenue") * 100,
        "Operating margin %": c("ebit") / c("revenue") * 100,
        "Net margin %": c("ni") / c("revenue") * 100,
        "ROE % (avg equity)": c("ni") / avg("eq") * 100,
        "ROA % (avg assets)": c("ni") / avg("ta") * 100,
        "Current ratio": c("ca") / c("cl"),
        "Quick ratio": (c("ca") - c("inv")) / c("cl"),
        "Debt/Equity": c("debt") / c("eq"),
        "Debt ratio": c("tl") / c("ta"),
        "Interest coverage (x)": c("ebit") / c("interest").abs(),
        "Asset turnover (x)": c("revenue") / avg("ta"),
        "Inventory turnover (x)": c("cogs") / avg("inv"),
        "Receivables turnover (x)": c("revenue") / avg("rec"),
        "Operating cash flow (m)": c("cfo") / 1e6,
        "Capex (m)": c("capex") / 1e6,
        "Free cash flow (m)": fcf / 1e6,
        "CFO / Net profit (x)": c("cfo") / c("ni"),
    }, index=t.index).replace([np.inf, -np.inf], np.nan).tail(4)
    r.index = [d.strftime("%Y-%m") for d in r.index]
    cur = y["info"].get("financialCurrency", "unknown (assumed PKR)")
    out = [f"Ratios computed in Python from Yahoo annual statements. Currency: {cur}. (m) = millions.",
           "Return and turnover ratios use average balances where the prior year exists, otherwise ending balances.",
           r.T.to_string(float_format=lambda x: f"{x:,.2f}", na_rep="n/a")]
    for key, label in [("revenue", "Revenue"), ("ni", "Net profit"), ("eps", "EPS")]:
        g = _cagr(c(key))
        out.append(f"{label} CAGR over available annual data: {fmt(g)}%" if g is not None else f"{label} CAGR: Unavailable (need 3+ positive annual points)")
    out.append(f"Annual periods available: {len(t)} (Yahoo usually provides only about 4).")
    return "\n".join(out)


def valuation_inputs(symbol):
    y = get_yahoo(symbol)
    info = y["info"]
    prices, src = get_prices(symbol)
    t = _collect(y)
    lines = []
    price = None
    last_date = None
    if prices is not None:
        price = float(prices["close"].iloc[-1])
        last_date = prices.index[-1]
        lines.append(f"Last close: PKR {fmt(price)} on {last_date.date()} (source: {src})")
    else:
        lines.append("Price: Unavailable")
    shares = info.get("sharesOutstanding")
    lines.append(f"Shares outstanding (Yahoo): {fmt(shares, 0) if shares else 'Unavailable'}")
    if t is not None and not t.empty:
        last = t.iloc[-1]
        period = t.index[-1].strftime("%Y-%m")

        def g(k):
            return last[k] if k in last.index else np.nan

        eps = g("eps")
        if pd.isna(eps) and shares and not pd.isna(g("ni")):
            eps = g("ni") / shares
        lines.append(f"Latest annual period: {period} (values in PKR millions unless stated)")
        lines.append(f"Revenue {fmt(g('revenue') / 1e6)} | Net profit {fmt(g('ni') / 1e6)} | EBITDA {fmt(g('ebitda') / 1e6)} | EPS {fmt(eps)}")
        lines.append(f"Equity {fmt(g('eq') / 1e6)} | Total debt {fmt(g('debt') / 1e6)} | Cash {fmt(g('cash') / 1e6)}")
        fcf = g("fcf")
        if pd.isna(fcf):
            fcf = g("cfo") + g("capex")
        lines.append(f"Free cash flow (latest annual): {fmt(fcf / 1e6)} m")
        net_debt = (0 if pd.isna(g("debt")) else g("debt")) - (0 if pd.isna(g("cash")) else g("cash"))
        lines.append(f"Net debt: {fmt(net_debt / 1e6)} m (debt minus cash; treat as approximate)")
        if price and shares:
            mcap = price * shares
            ev = mcap + net_debt
            lines.append(f"Market cap: {fmt(mcap / 1e6)} m (price x shares)")
            lines.append(f"P/E: {fmt(price / eps) if not pd.isna(eps) and eps else 'Unavailable'}")
            if not pd.isna(g("eq")):
                lines.append(f"Book value/share: {fmt(g('eq') / shares)} | P/B: {fmt(price / (g('eq') / shares))}")
            if not pd.isna(g("revenue")):
                lines.append(f"P/S: {fmt(mcap / g('revenue'))}")
            lines.append(f"Enterprise value: {fmt(ev / 1e6)} m (market cap + net debt; not the same as market cap)")
            if not pd.isna(g("ebitda")) and g("ebitda"):
                lines.append(f"EV/EBITDA: {fmt(ev / g('ebitda'))}")
    else:
        lines.append("Annual statements: Unavailable (multiples cannot be computed from Yahoo data)")
    divs = y["dividends"]
    if divs:
        lines.append("Recent dividends per share (date, PKR): " + "; ".join(f"{d}: {a}" for d, a in divs))
        if price and last_date is not None:
            ttm = sum(a for d, a in divs if pd.Timestamp(d) >= last_date - pd.Timedelta(days=365))
            lines.append(f"Trailing-12-month dividends/share: {fmt(ttm)} | Dividend yield: {fmt(ttm / price * 100)}%")
    else:
        lines.append("Dividend history: Unavailable")
    return "\n".join(lines)


def peer_metrics(symbols):
    rows = []
    for s in symbols:
        info = get_yahoo(s)["info"]
        p, _ = get_prices(s)
        ret = np.nan
        price = np.nan
        if p is not None:
            price = p["close"].iloc[-1]
            if len(p) > 252:
                ret = (p["close"].iloc[-1] / p["close"].iloc[-253] - 1) * 100
        rows.append({
            "Ticker": s,
            "Price": price,
            "MktCap (PKR bn)": (info.get("marketCap") or np.nan) / 1e9 if info.get("marketCap") else np.nan,
            "P/E": info.get("trailingPE", np.nan),
            "P/B": info.get("priceToBook", np.nan),
            "ROE": info.get("returnOnEquity", np.nan),
            "Net margin": info.get("profitMargins", np.nan),
            "Debt/Equity": info.get("debtToEquity", np.nan),
            "1Y return %": ret,
        })
    df = pd.DataFrame(rows)
    return ("Peer table from Yahoo Finance (unverified, may be stale or incomplete; ROE and margin are fractions, "
            "e.g. 0.15 = 15%):\n" + df.to_string(index=False, float_format=lambda x: f"{x:,.2f}", na_rep="n/a"))
