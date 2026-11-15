"""Sequential 5-agent pipeline. Each stage is its own CrewAI Crew so progress is visible and rate limits are retried per stage."""
import datetime as dt
import time

from .config import find_peers  # keep config first (env flags)
from crewai import Crew, Process, Task

from .agents import build_agents


def _clip(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + "\n[TRUNCATED]"


def _run_stage(agent, description, expected, retries=3):
    for attempt in range(retries + 1):
        try:
            task = Task(description=description, expected_output=expected, agent=agent)
            crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
            out = crew.kickoff()
            return str(getattr(out, "raw", out))
        except Exception as e:
            msg = str(e).lower()
            if attempt < retries and any(k in msg for k in ("rate", "429", "tpm", "limit", "timeout", "overloaded")):
                time.sleep(25 * (attempt + 1))
                continue
            raise


def run_pipeline(api_key, main_model, light_model, symbol, peers, shariah, notes, pdf_text):
    """Generator yielding events: {'i','name','state': running|done, 'output'?}; final event state='complete'."""
    agents = build_agents(api_key, main_model, light_model)
    today = dt.date.today().isoformat()
    sector, auto_peers = find_peers(symbol)
    peer_list = peers or auto_peers
    peer_csv = ",".join([symbol] + [p for p in peer_list if p != symbol])
    outputs = {}

    doc_block = (f"\nUSER-UPLOADED DOCUMENTS (primary source; extract key figures with period and units):\n{_clip(pdf_text, 9000)}\n"
                 if pdf_text else "\nNo documents uploaded.\n")
    notes_block = f"\nUser notes: {notes}\n" if notes else ""

    stages = [
        ("Data Researcher", "researcher",
         lambda: (
             f"Today is {today}. Research PSX ticker {symbol} (sector guess: {sector}).\n"
             f"Use your tools to collect: (1) company profile, (2) price snapshot, (3) annual financial statements.{doc_block}{notes_block}"
             "Do NOT analyse. Output (max 500 words) with headings: 1 Company identification (name, ticker, sector, business model, "
             "main products, revenue sources if known); 2 Data collected (each item with source, period, units, retrieval date); "
             "3 Key raw figures; 4 Data gaps and inconsistencies.",
             "A concise sourced data summary with explicit data gaps.")),
        ("Financial Analyst", "financial",
         lambda: (
             f"Today is {today}. Analyse the financial statements of {symbol}.\n"
             f"Data Researcher output:\n{_clip(outputs['researcher'], 2500)}\n\n"
             f"Call the Financial ratios tool for {symbol}. Do not do mental arithmetic; use the Calculator for any extra maths. "
             "If the tool says data is unavailable, use figures from the researcher output and say the analysis is limited. "
             "Cover: profitability, liquidity, leverage, efficiency, cash-flow quality (profit vs operating cash flow), growth and CAGR, "
             "trend direction (improving/deteriorating) for each key ratio, and red flags. Use sector-appropriate benchmarks "
             "(banks need different logic). Max 600 words, with a short key-ratio table.",
             "Ratio-based financial analysis with trends, strengths, weaknesses and red flags.")),
        ("Market Analyst", "market",
         lambda: (
             f"Today is {today}. Research the industry, competitors and macro backdrop for {symbol} ({sector}).\n"
             f"Company context:\n{_clip(outputs['researcher'], 1500)}\n\n"
             f"Use Peer comparison with '{peer_csv}'. Use Web search (2-4 queries) for: Pakistan SBP policy rate and inflation latest, "
             f"PKR trend, {sector} sector outlook Pakistan, and recent news/governance/regulatory issues about {symbol}. "
             "Web results are unverified: label them, include the URL and date if shown, and say 'Unavailable' if nothing is found. "
             "Cover: industry drivers and threats, competitive position vs peers (growth, margins, ROE, leverage, valuation, dividend), "
             "macro factors, management/governance/regulatory red flags. Max 600 words.",
             "Industry, competitor and macro analysis with sourced, labelled evidence.")),
        ("Technical Analyst", "technical",
         lambda: (
             f"Today is {today}. Call the Technical indicators tool for {symbol}. Report long-, medium- and short-term trend, "
             "position vs MA20/50/200, RSI zone, MACD, support and resistance, volume confirmation, possible breakout/breakdown, and "
             "overbought/oversold status. Use only tool values. Indicators are supporting evidence, not signals. Max 350 words. "
             "End with: 'Technical trend assessment: bullish / neutral / bearish' plus one sentence why.",
             "Technical analysis from computed indicators only.")),
        ("Investment Analyst", "investment",
         lambda: (
             f"Today is {today}. Write the final institutional-style research report for {symbol} (3-5 year horizon, PKR).\n\n"
             f"=== DATA RESEARCHER ===\n{_clip(outputs['researcher'], 1800)}\n\n"
             f"=== FINANCIAL ANALYST ===\n{_clip(outputs['financial'], 3500)}\n\n"
             f"=== MARKET ANALYST ===\n{_clip(outputs['market'], 3000)}\n\n"
             f"=== TECHNICAL ANALYST ===\n{_clip(outputs['technical'], 1800)}\n\n"
             "Steps: (1) call Valuation inputs; (2) if base FCF is positive and data allows, call DCF valuation THREE times for bear, base and bull "
             "(use PKR millions for FCF and net debt, millions for shares; state every assumption); otherwise value with multiples and say DCF "
             "was not possible; (3) assess key risks (what, why it matters, impact, how to monitor) plus early-warning signs; "
             "(4) call Investment score calculator once with category scores and justify each; mark the score PROVISIONAL if material data is missing; "
             "(5) write the report. Valuation is an estimate, not a price target.\n"
             + ("Shariah screening was requested: add section 11 as a preliminary screen (business nature, debt, interest income) naming the "
                "methodology; do not state thresholds you cannot verify; if evidence is insufficient write 'Further verification required'; "
                "it is not a religious ruling.\n" if shariah else "Shariah screening was not requested: omit section 11.\n")
             + notes_block +
             "Report format (max 2200 words): # Investment Research Report; 1 Executive Summary; 2 Business Overview; 3 Financial Performance (table); "
             "4 Fundamental Analysis; 5 Growth Potential; 6 Management and Governance; 7 Industry and Competitor Comparison; 8 Technical Analysis; "
             "9 Valuation (bear/base/bull); 10 Risk Assessment; 11 Shariah Screening (only if requested); 12 Investment Score (table); "
             "13 Final Research Conclusion (exactly one of: Strong potential for further research / Potentially attractive / Watchlist candidate / "
             "Neutral / mixed outlook / High risk / unattractive); 14 Sources and Data Limitations (sources, periods, retrieval date, missing data). "
             "End with a one-line disclaimer that this is research, not financial advice.",
             "A complete 14-section investment research report in markdown.")),
    ]

    for i, (name, key, build) in enumerate(stages, start=1):
        yield {"i": i, "name": name, "state": "running"}
        description, expected = build()
        outputs[key] = _run_stage(agents[key], description, expected)
        yield {"i": i, "name": name, "state": "done", "output": outputs[key]}
        if i < len(stages):
            time.sleep(6)  # gentle pacing for Groq token-per-minute limits
    yield {"state": "complete", "outputs": outputs}
