"""Turns raw fetched data into a validated 'bundle' of calculated results."""
from __future__ import annotations

import pandas as pd

from engine import ratios as R
from engine import technical as T


def _round(o, nd=2):
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    if isinstance(o, float):
        return None if o != o else round(o, nd)
    return o


def build_bundle(company: dict) -> dict:
    t = company["ticker"]
    periods = R.build_periods(company.get("income"), company.get("balance"), company.get("cashflow"))
    ratios = R.compute_ratios(periods)
    prices = company.get("prices", pd.DataFrame())
    tech = T.add_indicators(prices) if prices is not None and not prices.empty else pd.DataFrame()
    tsum = T.summarize(tech) if not tech.empty else {}
    info = company.get("info", {})

    price = tsum.get("observations", {}).get("last_close") if tsum else None
    mcap = info.get("marketCap")
    if mcap is None and price and info.get("sharesOutstanding"):
        mcap = price * info["sharesOutstanding"]
    div_ttm = None
    if not prices.empty and "Dividends" in prices:
        cutoff = prices["Date"].max() - pd.Timedelta(days=365)
        div_ttm = float(prices.loc[prices["Date"] > cutoff, "Dividends"].sum())
    latest = periods.iloc[-1].to_dict() if not periods.empty else {}
    val = R.valuation(price, mcap, latest, div_ttm)
    flags = R.red_flags(periods, ratios)
    cg = R.cagr_summary(periods)

    availability = {
        "annual_statements": bool(not periods.empty),
        "fiscal_years": [int(y) for y in periods.index] if not periods.empty else [],
        "prices": bool(not prices.empty),
        "price_source": company.get("price_source"),
        "news_items": len(company.get("news", [])),
    }
    last_year = int(periods.index[-1]) if not periods.empty else None
    latest_ratios = ratios.iloc[-1].to_dict() if not ratios.empty else {}
    ccy = info.get("financialCurrency") or info.get("currency")

    views = {
        "data": _round({"ticker": t, "company": info.get("longName"), "sector": info.get("sector"),
                         "industry": info.get("industry"), "currency": ccy, "retrieved": company["retrieved"],
                         "availability": availability, "fetch_errors": company.get("errors", [])[:5]}),
        "fundamental": _round({"company": info.get("longName"), "currency": ccy, "latest_fiscal_year": last_year,
                                "ratios_by_year": ratios.tail(4).to_dict(orient="index") if not ratios.empty else {},
                                "cagr": cg, "valuation": val}),
        "technical": _round(tsum),
        "market": _round({"company": info.get("longName"), "sector": info.get("sector"),
                           "industry": info.get("industry"), "headlines": company.get("news", [])[:8]}),
        "risk": _round({"latest_ratios": latest_ratios, "red_flags": flags, "valuation": val,
                         "technical_signals": tsum.get("signals") if tsum else None}),
    }
    return {
        "ticker": t, "company": info.get("longName") or t, "info": info, "currency": ccy,
        "retrieved": company["retrieved"], "periods": periods, "ratios": ratios, "valuation": val,
        "tech": tech, "tech_summary": tsum, "flags": flags, "cagr": cg, "news": company.get("news", []),
        "availability": availability, "latest_period": latest, "views": views, "errors": company.get("errors", []),
    }
