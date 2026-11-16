"""Preliminary Shariah screening support. NOT a fatwa or certification."""
from __future__ import annotations

from engine.ratios import pct, safe_div

METHODS = {
    "Simplified KMI-style (debt/assets < 37%, interest income/revenue < 5%)": {"debt_assets": 37.0, "interest_rev": 5.0},
    "Stricter (debt/assets < 30%, interest income/revenue < 5%)": {"debt_assets": 30.0, "interest_rev": 5.0},
}


def shariah_screen(method: str, total_debt, total_assets, interest_income, revenue, business_ok: bool) -> dict:
    th = METHODS[method]
    checks = []
    da = pct(safe_div(total_debt, total_assets))
    ir = pct(safe_div(interest_income, revenue))

    def status(val, limit):
        if val is None:
            return "Insufficient data"
        return "Pass" if val < limit else "Fail"

    checks.append({"Check": "Core business activity permissible (user-confirmed)", "Value": "Confirmed" if business_ok else "Not confirmed",
                   "Threshold": "Must be permissible", "Result": "Pass" if business_ok else "Not confirmed"})
    checks.append({"Check": "Total debt / total assets", "Value": None if da is None else round(da, 2),
                   "Threshold": f"< {th['debt_assets']}%", "Result": status(da, th["debt_assets"])})
    checks.append({"Check": "Interest income / revenue", "Value": None if ir is None else round(ir, 2),
                   "Threshold": f"< {th['interest_rev']}%", "Result": status(ir, th["interest_rev"])})
    results = [c["Result"] for c in checks]
    if "Fail" in results:
        outcome = "Preliminary: does NOT pass the selected ratio screen"
    elif "Insufficient data" in results or "Not confirmed" in results:
        outcome = "Inconclusive: missing data or unconfirmed business activity"
    else:
        outcome = "Preliminary: passes the selected simplified ratios (NOT a certification)"
    return {"method": method, "checks": checks, "outcome": outcome,
            "limitations": ["Uses annual statements from a public data feed, not audited screening data.",
                            "Other tests (illiquid assets, non-compliant investments, market-price test) are not covered.",
                            "Verify with a qualified Shariah adviser or a recognised screening provider."]}
