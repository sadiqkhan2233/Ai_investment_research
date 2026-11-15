"""CrewAI tools. Each returns compact text so token usage stays low."""
import ast
import operator

from crewai.tools import tool

from . import data

try:
    from ddgs import DDGS
except Exception:  # pragma: no cover
    try:
        from duckduckgo_search import DDGS
    except Exception:
        DDGS = None


def _sym(s: str) -> str:
    return s.strip().upper().replace(".KA", "").replace(" ", "")


def _cap(text: str, n: int = 4500) -> str:
    return text if len(text) <= n else text[:n] + "\n[TRUNCATED]"


# ----------------------------------------------------------------------------- Agent 1 tools
@tool("Company profile")
def company_profile(symbol: str) -> str:
    """Get company name, sector, industry and business summary for a PSX ticker such as OGDC."""
    try:
        return _cap(data.company_profile(_sym(symbol)))
    except Exception as e:
        return f"ERROR: {e}"


@tool("Price snapshot")
def price_snapshot(symbol: str) -> str:
    """Get latest close, price date, 52-week range and 1M/3M/6M/1Y/3Y returns for a PSX ticker."""
    try:
        s = _sym(symbol)
        df, src = data.get_prices(s)
        if df is None:
            return f"UNAVAILABLE: no price data for {s}"
        return f"Source: {src}\n" + data.price_summary(df)
    except Exception as e:
        return f"ERROR: {e}"


@tool("Financial statements")
def financial_statements(symbol: str) -> str:
    """Get compact annual income statement, balance sheet and cash flow items (in millions) for a PSX ticker."""
    try:
        return _cap(data.statements_text(_sym(symbol)), 5000)
    except Exception as e:
        return f"ERROR: {e}"


# ----------------------------------------------------------------------------- Agent 2 tools
@tool("Financial ratios")
def financial_ratios(symbol: str) -> str:
    """Compute profitability, liquidity, leverage, efficiency, cash-flow ratios, growth and CAGR for a PSX ticker."""
    try:
        return _cap(data.compute_ratios(_sym(symbol)), 5000)
    except Exception as e:
        return f"ERROR: {e}"


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 20:
            raise ValueError("Exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool("Calculator")
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression like '(1250.5 - 980) / 980 * 100'. Supports + - * / % ** and parentheses."""
    try:
        return str(round(_eval(ast.parse(expression.strip(), mode="eval")), 6))
    except Exception as e:
        return f"ERROR: {e}"


# ----------------------------------------------------------------------------- Agent 3 tools
@tool("Peer comparison")
def peer_comparison(symbols_csv: str) -> str:
    """Compare PSX tickers given as a comma-separated string, e.g. 'OGDC,PPL,POL,MARI' (max 6)."""
    try:
        syms = [_sym(s) for s in symbols_csv.split(",") if s.strip()][:6]
        return _cap(data.peer_metrics(syms), 3500)
    except Exception as e:
        return f"ERROR: {e}"


@tool("Web search")
def web_search(query: str) -> str:
    """Search the web for Pakistan economy, sector or company news. Results are unverified third-party snippets."""
    if DDGS is None:
        return "UNAVAILABLE: web search library not installed."
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No results."
        lines = [f"- {r.get('title', '')} | {r.get('href', '')} | {str(r.get('body', ''))[:240]}" for r in results]
        return _cap("Unverified web snippets:\n" + "\n".join(lines), 2500)
    except Exception as e:
        return f"UNAVAILABLE: search failed ({e})"


# ----------------------------------------------------------------------------- Agent 4 tools
@tool("Technical indicators")
def technical_indicators(symbol: str) -> str:
    """Compute MA20/50/200, RSI, MACD, support, resistance, volume and returns for a PSX ticker."""
    try:
        s = _sym(symbol)
        df, src = data.get_prices(s)
        if df is None:
            return f"UNAVAILABLE: no price history for {s}; technical analysis cannot be performed."
        return f"Source: {src}\n" + data.technical_summary(df)
    except Exception as e:
        return f"ERROR: {e}"


# ----------------------------------------------------------------------------- Agent 5 tools
@tool("Valuation inputs")
def valuation_inputs(symbol: str) -> str:
    """Get price, market cap, EPS, P/E, P/B, P/S, EV, EV/EBITDA, net debt, free cash flow and dividend yield."""
    try:
        return _cap(data.valuation_inputs(_sym(symbol)), 3500)
    except Exception as e:
        return f"ERROR: {e}"


@tool("DCF valuation")
def dcf_valuation(base_fcf: float, growth_rate_pct: float, discount_rate_pct: float,
                  terminal_growth_pct: float, net_debt: float, shares_outstanding: float,
                  years: int = 5) -> str:
    """Discounted cash flow. Use consistent units (FCF and net_debt in PKR millions, shares_outstanding in millions).
    Rates are percentages, e.g. 12 for 12%. Returns enterprise value, equity value and value per share."""
    try:
        r, g, tg = discount_rate_pct / 100, growth_rate_pct / 100, terminal_growth_pct / 100
        if base_fcf <= 0:
            return "INVALID: base free cash flow must be positive for a DCF; use multiples instead."
        if r <= tg:
            return "INVALID: discount rate must exceed terminal growth."
        if shares_outstanding <= 0:
            return "INVALID: shares_outstanding must be positive."
        years = max(1, min(int(years), 10))
        pv = 0.0
        fcf = base_fcf
        for t in range(1, years + 1):
            fcf *= (1 + g)
            pv += fcf / (1 + r) ** t
        tv = fcf * (1 + tg) / (r - tg)
        pv_tv = tv / (1 + r) ** years
        ev = pv + pv_tv
        eq = ev - net_debt
        return (f"DCF result | years={years}, growth={growth_rate_pct}%, discount={discount_rate_pct}%, terminal={terminal_growth_pct}%\n"
                f"PV of explicit FCF: {pv:,.1f} | PV of terminal value: {pv_tv:,.1f} ({pv_tv / ev * 100:.0f}% of EV)\n"
                f"Enterprise value: {ev:,.1f} | Equity value: {eq:,.1f} | Value per share: {eq / shares_outstanding:,.2f}")
    except Exception as e:
        return f"ERROR: {e}"


@tool("Investment score calculator")
def score_total(financial_strength: float, growth: float, profitability: float, valuation: float,
                governance: float, industry: float, technical: float, risk: float) -> str:
    """Sum category scores. Maximums: financial_strength 20, growth 20, profitability 15, valuation 15,
    governance 10, industry 10, technical 5, risk 5 (higher risk score = lower, better-managed risk)."""
    try:
        caps = {"Financial Strength": (financial_strength, 20), "Growth Potential": (growth, 20),
                "Profitability": (profitability, 15), "Valuation": (valuation, 15),
                "Management & Governance": (governance, 10), "Industry Outlook": (industry, 10),
                "Technical Trend": (technical, 5), "Risk Profile": (risk, 5)}
        total = 0.0
        rows = []
        for name, (val, mx) in caps.items():
            v = max(0.0, min(float(val), mx))
            total += v
            rows.append(f"{name}: {v:g}/{mx}")
        if total >= 80:
            label = "Strong research profile"
        elif total >= 70:
            label = "Attractive research profile"
        elif total >= 60:
            label = "Mixed research profile"
        elif total >= 50:
            label = "Weak research profile"
        else:
            label = "High concern"
        return "\n".join(rows) + f"\nTOTAL: {total:g}/100 -> {label}"
    except Exception as e:
        return f"ERROR: {e}"
