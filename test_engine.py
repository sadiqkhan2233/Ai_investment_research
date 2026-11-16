"""Unit tests. All data below is MOCK data for testing only - never live market data."""
import numpy as np
import pandas as pd

from engine import ratios as R
from engine import technical as T
from engine.scenarios import scenarios
from engine.shariah import METHODS, shariah_screen


def test_safe_div_handles_zero_and_none():
    assert R.safe_div(10, 0) is None
    assert R.safe_div(None, 5) is None
    assert R.safe_div(float("nan"), 5) is None
    assert R.safe_div(10, 4) == 2.5


def test_growth_and_cagr():
    assert R.growth(100, 120) == 0.2
    assert R.growth(0, 5) is None
    assert abs(R.cagr(100, 121, 2) - 0.10) < 1e-9
    assert R.cagr(-1, 5, 2) is None


def _mock_periods():
    return pd.DataFrame({
        "revenue": [100.0, 120.0], "cogs": [60.0, 70.0], "gross_profit": [None, None],
        "operating_income": [20.0, 30.0], "ebit": [20.0, 30.0], "ebitda": [25.0, 35.0],
        "net_income": [10.0, 15.0], "eps": [1.0, 1.5], "interest_expense": [2.0, 3.0],
        "interest_income": [1.0, 1.0], "total_assets": [200.0, 220.0], "equity": [100.0, 110.0],
        "current_assets": [80.0, 90.0], "current_liabilities": [40.0, 45.0], "inventory": [20.0, 20.0],
        "receivables": [10.0, 12.0], "cash": [15.0, 20.0], "total_debt": [50.0, 55.0],
        "operating_cf": [12.0, 18.0], "capex": [-5.0, -6.0],
    }, index=[2023, 2024])


def test_compute_ratios():
    r = R.compute_ratios(_mock_periods())
    assert round(r.loc[2024, "gross_margin_pct"], 2) == round(50 / 120 * 100, 2)  # derived from revenue - cogs
    assert round(r.loc[2024, "roe_pct"], 2) == round(15 / 110 * 100, 2)
    assert round(r.loc[2024, "current_ratio"], 2) == 2.0
    assert round(r.loc[2024, "quick_ratio"], 2) == round(70 / 45, 2)
    assert round(r.loc[2024, "revenue_growth_pct"], 1) == 20.0


def test_missing_data_returns_nan_not_crash():
    p = _mock_periods()
    p["current_liabilities"] = [0.0, np.nan]
    r = R.compute_ratios(p)
    assert pd.isna(r.loc[2023, "current_ratio"]) and pd.isna(r.loc[2024, "current_ratio"])


def test_valuation():
    v = R.valuation(15.0, 1500.0, _mock_periods().iloc[-1].to_dict(), div_ttm=0.75)
    assert v["pe"] == 10.0
    assert round(v["dividend_yield_pct"], 2) == 5.0
    assert R.valuation(None, None, {}, None)["pe"] is None


def _mock_prices(n=300):
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=n), "Close": close,
                         "High": close + 1, "Low": close - 1, "Volume": rng.integers(1000, 5000, n)})


def test_indicators():
    d = T.add_indicators(_mock_prices())
    assert d["RSI14"].dropna().between(0, 100).all()
    assert d["ATR14"].notna().any()
    s = T.summarize(d)
    assert s["observations"]["rows"] == 300 and s["signals"]


def test_close_only_has_no_atr():
    p = _mock_prices()[["Date", "Close", "Volume"]]
    d = T.add_indicators(p)
    assert d["ATR14"].isna().all()


def test_scenarios_and_shariah():
    s = scenarios(100, 10, 8, 3, 8)
    assert list(s["Scenario"]) == ["Bearish", "Base", "Bullish"]
    assert scenarios(100, -1, 8, 3, 8).empty
    m = list(METHODS)[0]
    out = shariah_screen(m, 30, 100, 1, 100, True)
    assert out["outcome"].startswith("Preliminary: passes")
    assert shariah_screen(m, 50, 100, 1, 100, True)["outcome"].startswith("Preliminary: does NOT")
    assert shariah_screen(m, None, 100, 1, 100, True)["outcome"].startswith("Inconclusive")
