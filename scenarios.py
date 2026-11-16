"""Simple, transparent scenario table. Assumption-driven - NOT a forecast."""
from __future__ import annotations

import pandas as pd

from engine.ratios import safe_div


def scenarios(price, eps, base_growth_pct, years, base_pe, growth_spread_pct=5.0, pe_spread_pct=20.0) -> pd.DataFrame:
    """Project EPS for `years` and apply an exit P/E. All inputs are user assumptions."""
    if price is None or eps is None or eps <= 0 or base_pe is None or years <= 0:
        return pd.DataFrame()
    rows = []
    for name, dg, dpe in (("Bearish", -1, -1), ("Base", 0, 0), ("Bullish", 1, 1)):
        g = (base_growth_pct + dg * growth_spread_pct) / 100.0
        pe = base_pe * (1 + dpe * pe_spread_pct / 100.0)
        eps_f = eps * (1 + g) ** years
        implied = eps_f * pe
        up = safe_div(implied - price, price)
        rows.append({
            "Scenario": name, "EPS growth %/yr": round(g * 100, 1), "Exit P/E": round(pe, 1),
            f"EPS in {years}y": round(eps_f, 2), "Implied price": round(implied, 2),
            "Change vs current %": None if up is None else round(up * 100, 1),
        })
    return pd.DataFrame(rows)
