"""Multi-agent research pipeline (8 specialised agents, sequential, explicit dependencies).

Each agent = a role + instructions + a task, run through Groq. Agents only INTERPRET
numbers that Python already calculated; they never calculate or invent data.
(Written without the CrewAI package on purpose: CrewAI + LiteLLM + chromadb often
fail to install on Streamlit Cloud. The Agent/Task structure below maps 1:1 to
CrewAI if you want to migrate later.)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime

from core.llm import chat

RULES = (
    "STRICT RULES: Use ONLY the data in the DATA block. Never invent numbers, prices, news, dates or sources. "
    "If a value is null or missing, say 'not available'. Do not give Buy/Sell/Hold orders and never promise returns. "
    "Label clearly: **Facts** (from data), **Interpretation** (your analysis), **Limitations**. "
    "Be concise (max ~350 words), plain English, markdown."
)


@dataclass
class Agent:
    key: str
    role: str
    persona: str
    task: str


AGENTS = [
    Agent("data", "Financial Data Researcher", "Financial data collection specialist.",
          "Summarise what data was collected: company, sector, currency, fiscal years available, price history period and source, "
          "retrieval date, and what is MISSING. Give a data-quality verdict."),
    Agent("fundamental", "Fundamental Analyst", "Senior equity research analyst.",
          "Interpret profitability, liquidity, solvency, efficiency, growth (incl. CAGR) and valuation using the ratios given. "
          "Describe trends across years, strengths and weaknesses. Do not recompute ratios."),
    Agent("technical", "Technical Analyst", "Technical market analysis specialist.",
          "Describe short, medium and long-term trend, momentum, volatility, volume and support/resistance. Separate observations from "
          "signals. State that indicators do not predict future prices."),
    Agent("market", "Market & Industry Researcher", "Financial market intelligence analyst.",
          "You have NO web access. Using only the headlines/notes provided, summarise company/industry developments (with publisher and date). "
          "For Pakistan macro factors (inflation, policy rate, PKR, regulation) do NOT state any figures; list what the investor should verify "
          "and the official sources (SBP, PBS, SECP, PSX), and explain generally how each could affect this sector."),
    Agent("risk", "Risk Analyst", "Senior financial risk analyst.",
          "Assess business, financial, liquidity, debt, earnings-volatility, valuation, currency, regulatory and governance risks that the data supports. "
          "Give bullish / base / bearish narrative scenarios (qualitative), key warning signs, information gaps, and what would invalidate a thesis."),
    Agent("shariah", "Shariah Screening Analyst", "Islamic investment screening researcher.",
          "Explain the preliminary screening result, methodology and thresholds, missing information and limitations. Never declare the company Halal or Haram. "
          "Recommend verification with a qualified Shariah adviser."),
    Agent("writer", "Investment Research Writer", "Senior investment research report writer.",
          "Write the final report in markdown with these sections: 1 Company overview, 2 Executive summary, 3 Financial performance & ratios, "
          "4 Valuation, 5 Technical view, 6 Industry & economic outlook, 7 Risk assessment, 8 Bullish/Base/Bearish scenarios, "
          "9 Shariah screening (only if provided), 10 Investment thesis & catalysts, 11 Red flags, 12 Long-term (3-5 yr) considerations, "
          "13 Data limitations, 14 Final research conclusion (balanced; NO buy/sell instruction). Use only the agent notes in DATA."),
    Agent("qa", "Quality Assurance Analyst", "Financial research verification specialist.",
          "Check the REPORT against the FACTS. List: numbers in the report that do not match the facts, unsupported claims, contradictions between sections, "
          "missing information. Be specific and short. END with exactly one line: STATUS: VERIFIED, STATUS: VERIFIED WITH FLAGS, or STATUS: REJECTED."),
]
BY_KEY = {a.key: a for a in AGENTS}


def _j(obj, limit: int = 6500) -> str:
    s = json.dumps(obj, default=str, separators=(",", ":"), ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "...[truncated]"


def _run(agent: Agent, data, extra: str = "") -> str:
    system = f"You are the {agent.role}: {agent.persona}\n{RULES}"
    user = f"TASK: {agent.task}\n{extra}\nDATA:\n{_j(data)}"
    return chat(system, user, max_tokens=1800 if agent.key != "writer" else 3200)


def key_figures_md(b: dict) -> str:
    """Deterministic (Python-only) key figures block appended to the report."""
    lines = []
    if not b["ratios"].empty:
        y = b["ratios"].index[-1]
        r = b["ratios"].iloc[-1]
        lines.append(f"**Latest fiscal year in data: {y}** (currency: {b['currency'] or 'n/a'})")
        for k, lab in (("revenue_growth_pct", "Revenue growth %"), ("net_margin_pct", "Net margin %"),
                       ("roe_pct", "ROE %"), ("roa_pct", "ROA %"), ("current_ratio", "Current ratio"),
                       ("debt_to_equity", "Debt/Equity"), ("eps", "EPS")):
            v = r.get(k)
            lines.append(f"- {lab}: {'n/a' if v is None or v != v else round(float(v), 2)}")
    v = b["valuation"]
    for k, lab in (("price", "Last price"), ("pe", "P/E"), ("pb", "P/B"), ("ps", "P/S"),
                   ("dividend_yield_pct", "Dividend yield %")):
        x = v.get(k)
        lines.append(f"- {lab}: {'n/a' if x is None else round(float(x), 2)}")
    ts = b["tech_summary"].get("observations", {}) if b["tech_summary"] else {}
    if ts:
        lines.append(f"- Price data: {ts['data_start']} to {ts['last_date']} ({ts['rows']} sessions), source: {b['availability']['price_source']}")
    return "\n".join(lines) or "No calculated figures available."


def run_research(b: dict, opts: dict, on_progress=lambda m: None) -> dict:
    sections, errors = {}, []
    pause = float(opts.get("pause", 6))

    def step(key: str, data, extra: str = "") -> None:
        ag = BY_KEY[key]
        on_progress(f"Running: {ag.role}")
        try:
            sections[key] = _run(ag, data, extra)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{ag.role}: {str(e)[:200]}")
            sections[key] = f"_{ag.role} failed: {str(e)[:200]}_"
        time.sleep(pause)

    v = b["views"]
    step("data", v["data"])
    step("fundamental", v["fundamental"])
    step("technical", v["technical"])
    notes = opts.get("notes", "").strip()[:1500]
    step("market", v["market"], f"USER-PROVIDED NOTES (unverified): {notes}" if notes else "")
    step("risk", v["risk"])
    if opts.get("shariah_result"):
        step("shariah", opts["shariah_result"])

    brief = {k: sections[k][:2200] for k in ("data", "fundamental", "technical", "market", "risk", "shariah") if k in sections}
    step("writer", {"company": b["company"], "ticker": b["ticker"], "agent_notes": brief},
         "Use ONLY agent_notes below; keep each section short.")

    report = sections.get("writer", "")
    facts = {"ratios_latest": b["views"]["risk"]["latest_ratios"], "valuation": b["views"]["fundamental"]["valuation"],
             "technical": b["views"]["technical"].get("observations") if b["views"]["technical"] else None,
             "flags": b["flags"], "availability": b["availability"]}
    step("qa", {"FACTS": facts, "REPORT": report[:7000]})

    m = re.search(r"STATUS:\s*(VERIFIED WITH FLAGS|VERIFIED|REJECTED)", sections.get("qa", ""), re.I)
    qa_status = m.group(1).upper() if m else "NOT CONFIRMED"
    status = "INCOMPLETE" if errors else qa_status

    md = (
        f"# Investment Research Report: {b['company']} ({b['ticker']})\n\n"
        f"*Generated {datetime.now():%Y-%m-%d %H:%M} | Data retrieved {b['retrieved']} | Verification status: **{status}***\n\n"
        f"> Educational research only. Not investment advice. No returns are guaranteed. AI text may contain errors; verify all figures against official filings.\n\n"
        f"## Key figures (calculated in Python)\n{key_figures_md(b)}\n\n"
        f"## Red flags (rule-based)\n" + "\n".join(f"- {f}" for f in b["flags"]) + "\n\n"
        f"---\n\n{report}\n\n---\n\n## Quality assurance\n{sections.get('qa', '')}\n"
    )
    if errors:
        md += "\n## Incomplete components\n" + "\n".join(f"- {e}" for e in errors) + "\n"
    return {"sections": sections, "errors": errors, "status": status, "markdown": md}
