"""Financial ratio engine.

Pure Python / Pandas. No API calls and no LLM: the LLM is never the source of
truth for any number. Every function returns None when inputs are missing or a
division by zero would occur (never a guessed value).
Ratios use year-end balances (not averages) - stated so the user is not misled.
"""
from __future__ import annotations

import math

import pandas as pd


def _ok(x) -> bool:
    if x is None:
        return False
    try:
        return not math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def safe_div(a, b):
    """a / b, or None if either is missing or b == 0."""
    if not _ok(a) or not _ok(b) or float(b) == 0:
        return None
    return float(a) / float(b)


def pct(x):
    return None if x is None else x * 100.0


def growth(prev, cur):
    """Period growth as a fraction; uses abs(prev) so sign changes behave sanely."""
    if not _ok(prev) or not _ok(cur) or float(prev) == 0:
        return None
    return (float(cur) - float(prev)) / abs(float(prev))


def cagr(first, last, years):
    """Compound annual growth rate as a fraction. Needs positive endpoints."""
    if not (_ok(first) and _ok(last)) or years <= 0:
        return None
    if float(first) <= 0 or float(last) <= 0:
        return None
    return (float(last) / float(first)) ** (1.0 / years) - 1.0


# field -> (statement, possible row labels in yfinance statements)
FIELDS = {
    "revenue": ("income", ["Total Revenue", "Operating Revenue", "Revenue"]),
    "cogs": ("income", ["Cost Of Revenue", "Cost of Revenue"]),
    "gross_profit": ("income", ["Gross Profit"]),
    "operating_income": ("income", ["Operating Income", "Total Operating Income As Reported"]),
    "ebit": ("income", ["EBIT"]),
    "ebitda": ("income", ["EBITDA", "Normalized EBITDA"]),
    "net_income": ("income", ["Net Income", "Net Income Common Stockholders"]),
    "eps": ("income", ["Diluted EPS", "Basic EPS"]),
    "interest_expense": ("income", ["Interest Expense", "Interest Expense Non Operating"]),
    "interest_income": ("income", ["Interest Income", "Interest Income Non Operating"]),
    "total_assets": ("balance", ["Total Assets"]),
    "equity": ("balance", ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]),
    "current_assets": ("balance", ["Current Assets"]),
    "current_liabilities": ("balance", ["Current Liabilities"]),
    "inventory": ("balance", ["Inventory"]),
    "receivables": ("balance", ["Accounts Receivable", "Receivables"]),
    "cash": ("balance", ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]),
    "total_debt": ("balance", ["Total Debt"]),
    "operating_cf": ("cashflow", ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]),
    "capex": ("cashflow", ["Capital Expenditure"]),
}


def _frame_by_year(df):
    m = {}
    if df is None or getattr(df, "empty", True):
        return m
    for c in df.columns:
        try:
            m[pd.Timestamp(c).year] = c
        except Exception:
            continue
    return m


def _lookup(df, names, col):
    for n in names:
        if n in df.index:
            v = df.loc[n, col]
            if isinstance(v, pd.Series):
                v = v.iloc[0]
            if _ok(v):
                return float(v)
    return None


def build_periods(income, balance, cashflow) -> pd.DataFrame:
    """Merge the three statements into one table: rows = fiscal year, cols = fields."""
    frames = {"income": income, "balance": balance, "cashflow": cashflow}
    ymaps = {k: _frame_by_year(v) for k, v in frames.items()}
    years = sorted(set().union(*[set(m) for m in ymaps.values()]))
    rows = {}
    for y in years:
        row = {}
        for field, (fr, names) in FIELDS.items():
            col = ymaps[fr].get(y)
            row[field] = _lookup(frames[fr], names, col) if col is not None else None
        rows[y] = row
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def compute_ratios(p: pd.DataFrame) -> pd.DataFrame:
    """Ratios per fiscal year (percent values are in %, others are plain multiples)."""
    if p is None or p.empty:
        return pd.DataFrame()
    out = {}
    prev = None
    for y, r in p.iterrows():
        def g(k):
            return float(r[k]) if k in r.index and _ok(r[k]) else None

        rev, ni, ta, eq = g("revenue"), g("net_income"), g("total_assets"), g("equity")
        gp, cogs, ca, cl = g("gross_profit"), g("cogs"), g("current_assets"), g("current_liabilities")
        inv, rec, cash, debt = g("inventory"), g("receivables"), g("cash"), g("total_debt")
        ebit, ie, eps = g("ebit"), g("interest_expense"), g("eps")
        opi = g("operating_income") if g("operating_income") is not None else ebit
        if gp is None and rev is not None and cogs is not None:
            gp = rev - cogs
        if cogs is None and rev is not None and gp is not None:
            cogs = rev - gp
        quick = safe_div(ca - inv, cl) if (ca is not None and inv is not None) else None
        row = {
            "revenue": rev, "net_income": ni, "eps": eps,
            "gross_margin_pct": pct(safe_div(gp, rev)),
            "operating_margin_pct": pct(safe_div(opi, rev)),
            "net_margin_pct": pct(safe_div(ni, rev)),
            "roe_pct": pct(safe_div(ni, eq)),
            "roa_pct": pct(safe_div(ni, ta)),
            "current_ratio": safe_div(ca, cl),
            "quick_ratio": quick,
            "cash_ratio": safe_div(cash, cl),
            "debt_to_equity": safe_div(debt, eq),
            "debt_to_assets": safe_div(debt, ta),
            "interest_coverage": safe_div(ebit, abs(ie)) if ie is not None else None,
            "asset_turnover": safe_div(rev, ta),
            "inventory_turnover": safe_div(cogs, inv),
            "receivables_turnover": safe_div(rev, rec),
        }
        if prev is not None:
            row["revenue_growth_pct"] = pct(growth(prev.get("revenue"), rev))
            row["net_income_growth_pct"] = pct(growth(prev.get("net_income"), ni))
            row["eps_growth_pct"] = pct(growth(prev.get("eps"), eps))
            row["operating_profit_growth_pct"] = pct(growth(prev.get("opi"), opi))
        out[y] = row
        prev = {"revenue": rev, "net_income": ni, "eps": eps, "opi": opi}
    df = pd.DataFrame.from_dict(out, orient="index").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def cagr_summary(p: pd.DataFrame) -> dict:
    res = {}
    if p is None or p.empty:
        return res
    for k in ("revenue", "net_income", "eps"):
        if k not in p:
            continue
        s = p[k].dropna()
        if len(s) >= 2:
            yrs = int(s.index[-1] - s.index[0])
            res[f"{k}_cagr_pct"] = pct(cagr(s.iloc[0], s.iloc[-1], yrs))
            res[f"{k}_cagr_period"] = f"{s.index[0]}-{s.index[-1]}"
    return res


def valuation(price, market_cap, latest: dict, div_ttm=None) -> dict:
    """Valuation multiples from a price and the latest annual statement row."""
    g = lambda k: latest.get(k) if latest else None
    debt, cash = g("total_debt"), g("cash")
    ev = None
    if _ok(market_cap) and _ok(debt) and _ok(cash):
        ev = float(market_cap) + float(debt) - float(cash)
    fcf = None
    if _ok(g("operating_cf")) and _ok(g("capex")):
        fcf = float(g("operating_cf")) - abs(float(g("capex")))
    return {
        "price": price,
        "market_cap": market_cap,
        "pe": safe_div(price, g("eps")),
        "pb": safe_div(market_cap, g("equity")),
        "ps": safe_div(market_cap, g("revenue")),
        "ev_ebitda": safe_div(ev, g("ebitda")),
        "dividend_yield_pct": pct(safe_div(div_ttm, price)),
        "fcf_yield_pct": pct(safe_div(fcf, market_cap)),
    }


def red_flags(p: pd.DataFrame, r: pd.DataFrame) -> list[str]:
    """Rule-based warning signs computed in Python (not by the LLM)."""
    flags = []
    if p is None or p.empty or r is None or r.empty:
        return ["Insufficient statement data to run red-flag checks."]
    last, y = p.iloc[-1], p.index[-1]
    lr = r.iloc[-1]
    if _ok(last.get("net_income")) and _ok(last.get("operating_cf")) and last["net_income"] > 0 > last["operating_cf"]:
        flags.append(f"{y}: positive net income but negative operating cash flow (earnings quality check).")
    if _ok(lr.get("current_ratio")) and lr["current_ratio"] < 1:
        flags.append(f"{y}: current ratio below 1 ({lr['current_ratio']:.2f}) - liquidity pressure.")
    if _ok(lr.get("debt_to_equity")) and lr["debt_to_equity"] > 2:
        flags.append(f"{y}: debt-to-equity above 2 ({lr['debt_to_equity']:.2f}).")
    if _ok(lr.get("interest_coverage")) and lr["interest_coverage"] < 1.5:
        flags.append(f"{y}: interest coverage below 1.5x ({lr['interest_coverage']:.2f}).")
    if _ok(last.get("equity")) and last["equity"] < 0:
        flags.append(f"{y}: negative shareholders' equity.")
    if "revenue_growth_pct" in r and _ok(lr.get("revenue_growth_pct")) and lr["revenue_growth_pct"] < 0:
        flags.append(f"{y}: revenue declined {lr['revenue_growth_pct']:.1f}% year on year.")
    if len(p) >= 2:
        prv = p.iloc[-2]
        rg = growth(prv.get("revenue"), last.get("revenue"))
        cg = growth(prv.get("receivables"), last.get("receivables"))
        if rg is not None and cg is not None and cg - rg > 0.20:
            flags.append(f"{y}: receivables grew much faster than revenue (possible collection/recognition issue).")
    return flags or ["No rule-based red flags triggered on the available data."]
