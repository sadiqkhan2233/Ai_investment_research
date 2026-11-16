"""AI Investment Research System - PSX (Streamlit + Groq multi-agent)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import db
from core.agents import run_research
from core.analysis import build_bundle
from core.data import clean_ticker, fetch_company, load_uploaded_prices
from core.llm import MODEL, get_api_key
from engine.scenarios import scenarios
from engine.shariah import METHODS, shariah_screen
from engine.technical import has_ohlc
from reports.pdf_generator import markdown_to_pdf

st.set_page_config(page_title="AI Investment Research - PSX", page_icon="📈", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_company(t: str):
    return fetch_company(t)


def fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:,.{nd}f}"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📈 PSX AI Research")
    ticker = clean_ticker(st.text_input("PSX ticker", value="OGDC", help="e.g. OGDC, LUCK, ENGRO, HBL, MARI"))
    uploaded = st.file_uploader("Optional: price CSV (Date, Close, Volume, ...)", type=["csv"])
    use_shariah = st.checkbox("Include Shariah screening")
    method, biz_ok = None, False
    if use_shariah:
        method = st.selectbox("Methodology", list(METHODS))
        biz_ok = st.checkbox("I confirm the core business is permissible")
    notes = st.text_area("Your own notes (optional, treated as unverified)", height=80)
    pause = st.slider("Pause between agents (sec)", 0, 20, 6, help="Helps stay inside Groq free-tier rate limits.")
    run = st.button("🚀 Start multi-agent research", type="primary", use_container_width=True)
    if not get_api_key():
        st.error("GROQ_API_KEY missing. Add it in Settings -> Secrets.")
    st.caption(f"Model: {MODEL}")

# ------------------------------------------------------------------ run
if run:
    if not ticker:
        st.sidebar.error("Enter a ticker.")
    elif not get_api_key():
        st.sidebar.error("Cannot run agents without GROQ_API_KEY.")
    else:
        up_df = None
        if uploaded is not None:
            up_df, err = load_uploaded_prices(uploaded)
            if err:
                st.sidebar.error(err)
                st.stop()
        with st.spinner("Collecting data..."):
            company = cached_company(ticker)
            if up_df is not None and not up_df.empty:
                company = {**company, "prices": up_df, "price_source": "User-uploaded CSV"}
            bundle = build_bundle(company)
        if not bundle["availability"]["annual_statements"] and not bundle["availability"]["prices"]:
            st.error(f"No data found for '{ticker}'. Check the ticker or upload a price CSV.")
            st.stop()
        sh = None
        if use_shariah:
            p = bundle["latest_period"]
            sh = shariah_screen(method, p.get("total_debt"), p.get("total_assets"),
                                p.get("interest_income"), p.get("revenue"), biz_ok)
        with st.status("Multi-agent research in progress...", expanded=True) as status:
            result = run_research(bundle, {"pause": pause, "notes": notes, "shariah_result": sh},
                                  on_progress=lambda m: status.write(m))
            status.update(label=f"Finished - status: {result['status']}", state="complete")
        st.session_state.update(bundle=bundle, result=result, shariah=sh)
        db.save_report(ticker, result["status"], result["markdown"])

b, res, sh = st.session_state.get("bundle"), st.session_state.get("result"), st.session_state.get("shariah")

tabs = st.tabs(["Dashboard", "Report", "Fundamentals", "Technical", "Market", "Risk & Scenarios",
                "Shariah", "Compare & Watchlist", "Saved Reports"])

# ------------------------------------------------------------------ dashboard
with tabs[0]:
    st.header("AI Investment Research System")
    st.write("A multi-agent platform that researches PSX-listed companies: data collection, fundamentals, technicals, "
             "market context, risk, optional Shariah screening, report writing and quality assurance. "
             "**All numbers are calculated in Python; the AI only interprets them.**")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Recent reports")
        rows = db.list_reports(5)
        for rid, t, created, s in rows:
            st.write(f"**{t}** - {created} - {s}")
        if not rows:
            st.caption("None yet. Enter a ticker in the sidebar and press Start.")
    with c2:
        st.subheader("Watchlist")
        w = db.list_watch()
        st.write(", ".join(w) if w else "Empty")
    st.info("Educational research only. Not investment advice. Data from public feeds can be incomplete or delayed.")

if not b:
    for i in range(1, 7):
        with tabs[i]:
            st.info("Run research from the sidebar first.")
else:
    ratios, periods = b["ratios"], b["periods"]
    # -------------------------------------------------------------- report
    with tabs[1]:
        stt = res["status"]
        (st.success if stt.startswith("VERIFIED") else st.warning)(f"Verification status: {stt}")
        st.markdown(res["markdown"])
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download PDF", markdown_to_pdf(res["markdown"]), f"{b['ticker']}_research.pdf", "application/pdf")
        d2.download_button("⬇️ Download Markdown", res["markdown"], f"{b['ticker']}_research.md")
        with st.expander("Individual agent outputs"):
            for k, v in res["sections"].items():
                st.markdown(f"**{k.upper()}**\n\n{v}")
                st.divider()
    # -------------------------------------------------------------- fundamentals
    with tabs[2]:
        st.subheader(f"{b['company']} - fundamentals ({b['currency'] or 'currency n/a'})")
        st.caption(f"Source: Yahoo Finance annual statements, retrieved {b['retrieved']}. Years available: {b['availability']['fiscal_years']}")
        if ratios.empty:
            st.warning("No annual statements available for this ticker. Ratios were not calculated (nothing is estimated).")
        else:
            disp = ratios.round(2).T
            disp.columns = [str(c) for c in disp.columns]
            st.dataframe(disp, use_container_width=True)
            x = [str(i) for i in ratios.index]
            fig = go.Figure()
            fig.add_bar(x=x, y=ratios["revenue"], name="Revenue")
            fig.add_bar(x=x, y=ratios["net_income"], name="Net income")
            fig.update_layout(barmode="group", title="Revenue vs net income", height=350)
            st.plotly_chart(fig, use_container_width=True)
            fig = go.Figure()
            for c in ("gross_margin_pct", "operating_margin_pct", "net_margin_pct", "roe_pct"):
                fig.add_scatter(x=x, y=ratios[c], name=c.replace("_pct", " %"), mode="lines+markers")
            fig.update_layout(title="Profitability trends", height=350)
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(go.Figure(go.Bar(x=x, y=ratios["eps"])).update_layout(title="EPS", height=300), use_container_width=True)
        st.subheader("Valuation")
        st.dataframe(pd.DataFrame([b["valuation"]]).T.rename(columns={0: "value"}), use_container_width=True)
        if b["cagr"]:
            st.write("CAGR:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in b["cagr"].items()})
    # -------------------------------------------------------------- technical
    with tabs[3]:
        d = b["tech"]
        if d.empty:
            st.warning("No price history available. Upload a CSV in the sidebar.")
        else:
            rng = st.radio("Range", ["6M", "1Y", "3Y", "All"], horizontal=True, index=1)
            n = {"6M": 126, "1Y": 252, "3Y": 756, "All": len(d)}[rng]
            v = d.tail(n)
            o = b["tech_summary"]["observations"]
            st.caption(f"Data: {o['data_start']} to {o['last_date']} | last close {o['last_close']} on {o['last_date']} | source: {b['availability']['price_source']}")
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.15, 0.17, 0.18], vertical_spacing=0.03)
            if has_ohlc(d):
                fig.add_trace(go.Candlestick(x=v["Date"], open=v["Open"], high=v["High"], low=v["Low"], close=v["Close"], name="Price"), 1, 1)
            else:
                fig.add_trace(go.Scatter(x=v["Date"], y=v["Close"], name="Close"), 1, 1)
            for c in ("SMA20", "SMA50", "SMA200", "EMA20", "BB_upper", "BB_lower"):
                fig.add_trace(go.Scatter(x=v["Date"], y=v[c], name=c, line=dict(width=1), visible=True if c in ("SMA50", "SMA200", "BB_upper", "BB_lower") else "legendonly"), 1, 1)
            if "Volume" in v:
                fig.add_trace(go.Bar(x=v["Date"], y=v["Volume"], name="Volume"), 2, 1)
            fig.add_trace(go.Scatter(x=v["Date"], y=v["RSI14"], name="RSI14"), 3, 1)
            fig.add_hline(y=70, line_dash="dot", row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", row=3, col=1)
            fig.add_trace(go.Scatter(x=v["Date"], y=v["MACD"], name="MACD"), 4, 1)
            fig.add_trace(go.Scatter(x=v["Date"], y=v["MACD_signal"], name="Signal"), 4, 1)
            fig.add_trace(go.Bar(x=v["Date"], y=v["MACD_hist"], name="Hist"), 4, 1)
            fig.update_layout(height=850, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Observations")
            st.json(o)
            st.subheader("Signals (interpretation, not prediction)")
            for s_ in b["tech_summary"]["signals"]:
                st.write("- " + s_)
            if not o["ohlc_available"]:
                st.caption("This source has close/volume only, so candlesticks and ATR are not shown.")
    # -------------------------------------------------------------- market
    with tabs[4]:
        st.subheader("Latest available company news")
        if not b["news"]:
            st.info("No headlines returned by the data feed. Check PSX announcements, SBP, PBS and SECP directly.")
        for n_ in b["news"]:
            st.markdown(f"- [{n_['title']}]({n_['url']}) - {n_['publisher']} {n_['date'][:10]}")
        st.caption("The app has no live web-search tool. For inflation, policy rate and PKR data use sbp.org.pk, pbs.gov.pk, secp.gov.pk and dps.psx.com.pk.")
        if res:
            st.markdown(res["sections"].get("market", ""))
    # -------------------------------------------------------------- risk
    with tabs[5]:
        st.subheader("Rule-based warning indicators")
        for f in b["flags"]:
            st.write("- " + f)
        st.subheader("Scenario table (your assumptions - not a forecast)")
        eps = b["latest_period"].get("eps") if b["latest_period"] else None
        price = b["valuation"].get("price")
        c1, c2, c3 = st.columns(3)
        g = c1.number_input("Base EPS growth %/yr", value=8.0)
        yrs = c2.slider("Years", 1, 5, 3)
        pe0 = c3.number_input("Base exit P/E", value=float(round(b["valuation"]["pe"], 1)) if b["valuation"].get("pe") else 8.0)
        sc = scenarios(price, eps, g, yrs, pe0)
        if sc.empty:
            st.info("Needs a positive latest EPS and a price. Not available for this company.")
        else:
            st.dataframe(sc, use_container_width=True, hide_index=True)
        if res:
            st.markdown(res["sections"].get("risk", ""))
    # -------------------------------------------------------------- shariah
    with tabs[6]:
        if not sh:
            st.info("Enable 'Include Shariah screening' in the sidebar and run research.")
        else:
            st.write(f"**Methodology:** {sh['method']}")
            st.dataframe(pd.DataFrame(sh["checks"]), hide_index=True, use_container_width=True)
            st.warning(sh["outcome"])
            for l in sh["limitations"]:
                st.write("- " + l)
            st.markdown(res["sections"].get("shariah", ""))

# ------------------------------------------------------------------ compare & watchlist
with tabs[7]:
    st.subheader("Watchlist")
    wc1, wc2 = st.columns(2)
    if wc1.button(f"Add {ticker} to watchlist") and ticker:
        db.add_watch(ticker)
    rm = wc2.selectbox("Remove", [""] + db.list_watch())
    if rm and st.button("Remove selected"):
        db.remove_watch(rm)
        st.rerun()
    st.write("Saved:", ", ".join(db.list_watch()) or "none")
    st.subheader("Compare 2-5 companies (no AI calls)")
    raw = st.text_input("Tickers (comma separated)", value=", ".join(db.list_watch()[:5]) or "OGDC, PPL")
    if st.button("Compare"):
        tl = [clean_ticker(x) for x in raw.split(",") if clean_ticker(x)][:5]
        if len(tl) < 2:
            st.warning("Enter at least 2 tickers.")
        else:
            rows = []
            for t in tl:
                with st.spinner(f"Loading {t}..."):
                    bb = build_bundle(cached_company(t))
                lr = bb["ratios"].iloc[-1] if not bb["ratios"].empty else {}
                ob = bb["tech_summary"].get("observations", {}) if bb["tech_summary"] else {}
                g_ = lambda k: lr.get(k) if len(lr) else None
                rows.append({"Ticker": t, "FY": int(bb["ratios"].index[-1]) if not bb["ratios"].empty else None,
                             "Rev growth %": g_("revenue_growth_pct"), "EPS growth %": g_("eps_growth_pct"),
                             "Net margin %": g_("net_margin_pct"), "ROE %": g_("roe_pct"), "ROA %": g_("roa_pct"),
                             "D/E": g_("debt_to_equity"), "P/E": bb["valuation"]["pe"], "P/B": bb["valuation"]["pb"],
                             "Div yield %": bb["valuation"]["dividend_yield_pct"],
                             "Price > SMA200": (ob.get("last_close") > ob["sma200"]) if ob.get("sma200") else None,
                             "Red flags": sum(1 for f in bb["flags"] if not f.startswith(("No rule", "Insufficient")))})
            cdf = pd.DataFrame(rows)
            st.dataframe(cdf.round(2), hide_index=True, use_container_width=True)
            if cdf["FY"].nunique() > 1:
                st.warning("Companies report different latest fiscal years - comparison is not like-for-like. No ranking is given.")
            metric = st.selectbox("Chart metric", [c for c in cdf.columns if c not in ("Ticker", "FY", "Price > SMA200")])
            st.plotly_chart(go.Figure(go.Bar(x=cdf["Ticker"], y=cdf[metric])).update_layout(title=metric, height=350), use_container_width=True)

# ------------------------------------------------------------------ saved reports
with tabs[8]:
    st.subheader("Saved reports")
    st.caption("Stored in a temporary SQLite file: on Streamlit Cloud they disappear when the app restarts. Download what you need.")
    rows = db.list_reports()
    if not rows:
        st.info("No saved reports yet.")
    else:
        pick = st.selectbox("Report", rows, format_func=lambda r: f"#{r[0]} {r[1]} - {r[2]} - {r[3]}")
        md = db.get_report(pick[0])
        st.download_button("⬇️ Download PDF", markdown_to_pdf(md), f"{pick[1]}_{pick[0]}.pdf", "application/pdf", key="dlsaved")
        st.markdown(md)
