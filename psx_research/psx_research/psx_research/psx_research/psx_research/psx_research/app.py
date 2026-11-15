# Use a modern SQLite on Streamlit Cloud (needed by crewai's dependencies)
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import datetime as dt
import io

import plotly.graph_objects as go
import streamlit as st
from pypdf import PdfReader

from psx_research import config  # noqa: F401  (sets env flags before crewai loads)
from psx_research import data, pipeline

st.set_page_config(page_title="PSX Multi-Agent Research", page_icon="📈", layout="wide")
st.title("📈 PSX AI Investment Research System")
st.caption("5 CrewAI agents: Data Researcher → Financial Analyst → Market Analyst → Technical Analyst → Investment Analyst. "
           "Educational research, not financial advice.")


def read_pdfs(files, max_chars=12000):
    text = ""
    for f in files or []:
        try:
            reader = PdfReader(io.BytesIO(f.read()))
            text += f"\n--- {f.name} ---\n" + "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as e:
            text += f"\n[Could not read {f.name}: {e}]\n"
    return text[:max_chars] + ("\n[TRUNCATED]" if len(text) > max_chars else "")


def price_chart(symbol):
    df, _ = data.get_prices(symbol)
    if df is None:
        return None
    d = data.add_indicators(df).tail(500)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["close"], name="Close", line=dict(width=2)))
    for n in (20, 50, 200):
        fig.add_trace(go.Scatter(x=d.index, y=d[f"ma{n}"], name=f"MA{n}", line=dict(width=1)))
    fig.update_layout(title=f"{symbol} price with moving averages", height=400, margin=dict(l=10, r=10, t=40, b=10))
    return fig


with st.sidebar:
    st.header("Settings")
    try:
        default_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        default_key = ""
    api_key = st.text_input("Groq API key", value=default_key, type="password")
    main_model = st.selectbox("Model for Financial + Investment agents", config.MODEL_CHOICES, index=0)
    light_model = st.selectbox("Model for Researcher, Market, Technical agents", config.MODEL_CHOICES, index=0)
    st.caption("Runs 5 agents in sequence (about 3-8 minutes on Groq free tier). Rate limits are retried automatically.")

col1, col2 = st.columns([1, 2])
with col1:
    symbol = st.text_input("PSX ticker (e.g. OGDC, HBL, LUCK)", value="OGDC").strip().upper().replace(".KA", "")
    sector, auto_peers = config.find_peers(symbol)
    peers_text = st.text_input("Peers (comma-separated, optional)", value=", ".join(auto_peers))
    shariah = st.checkbox("Include Shariah screening")
with col2:
    notes = st.text_area("Extra notes (optional)", height=90)
    files = st.file_uploader("Upload annual/quarterly report PDFs (optional)", type=["pdf"], accept_multiple_files=True)

if st.button("Run 5-agent research", type="primary"):
    if not api_key:
        st.error("Add your Groq API key in the sidebar or Streamlit Secrets.")
        st.stop()
    if not symbol:
        st.error("Enter a PSX ticker.")
        st.stop()

    data.clear_caches()
    peers = [p.strip().upper().replace(".KA", "") for p in peers_text.split(",") if p.strip()]
    pdf_text = read_pdfs(files)
    boxes = {}
    try:
        for ev in pipeline.run_pipeline(api_key, main_model, light_model, symbol, peers, shariah, notes, pdf_text):
            if ev["state"] == "running":
                boxes[ev["i"]] = st.status(f"Agent {ev['i']}: {ev['name']} working...", state="running")
            elif ev["state"] == "done":
                boxes[ev["i"]].update(label=f"Agent {ev['i']}: {ev['name']} finished", state="complete")
            elif ev["state"] == "complete":
                st.session_state["result"] = {"symbol": symbol, "outputs": ev["outputs"], "date": dt.date.today().isoformat()}
    except Exception as e:
        st.error(f"Pipeline stopped: {e}")
        st.info("Common causes: Groq rate limit (wait a minute and retry, or use the 8B model), invalid API key, or Yahoo/PSX data being unavailable.")

res = st.session_state.get("result")
if res:
    out = res["outputs"]
    fig = price_chart(res["symbol"])
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    tabs = st.tabs(["Final Report", "1 Data", "2 Financial", "3 Market", "4 Technical"])
    with tabs[0]:
        st.markdown(out["investment"])
        st.download_button("Download report (.md)", out["investment"],
                           file_name=f"{res['symbol']}_research_{res['date']}.md", mime="text/markdown")
    for tab, key in zip(tabs[1:], ["researcher", "financial", "market", "technical"]):
        with tab:
            st.markdown(out[key])
