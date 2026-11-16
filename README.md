# AI Investment Research System (PSX)

Multi-agent investment research on Pakistan Stock Exchange companies.
**Streamlit + Groq (`openai/gpt-oss-120b`) + Pandas/Plotly + SQLite.**

All numbers (ratios, indicators, valuation, scenarios, Shariah ratios) are calculated in Python (`engine/`).
The 8 agents (`core/agents.py`) only interpret those numbers. Nothing is fabricated: missing data shows as `n/a`.

## Deploy (no local setup needed)
1. Create a GitHub repo and upload all these files (keep the folder structure).
2. Go to https://share.streamlit.io -> **New app** -> pick the repo, branch `main`, main file `app.py`.
3. **Advanced settings -> Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   ```
4. Deploy. Never commit real keys (`.env` and `secrets.toml` are in `.gitignore`).

## Agents (sequential)
Data Researcher -> Fundamental -> Technical -> Market -> Risk -> Shariah (optional) -> Writer -> Quality Assurance.
If one agent fails the others still run and the report is marked **INCOMPLETE**.

## Data sources
Price CSV upload -> Yahoo Finance (`TICKER.KA`) -> PSX data portal (close/volume only, fallback).
Yahoo's PSX financial statements can be incomplete; ratios are skipped when inputs are missing.
Providers live in `core/data.py` so they can be swapped.

## Known limits
- The Market agent has **no live web search**; it analyses only returned headlines + your notes and never states macro figures.
- SQLite on Streamlit Cloud is temporary (watchlist/reports reset on restart). Replace `core/db.py` with a hosted DB for persistence.
- Groq free tier has token-per-minute limits; use the "Pause between agents" slider (retries are automatic).
- Not investment advice. Shariah output is preliminary screening, not certification.

## Tests
`pytest -q tests` (engine tests use clearly-labelled MOCK data, no API calls).

## Troubleshooting
| Problem | Fix |
|---|---|
| "GROQ_API_KEY missing" | Add it in Streamlit Settings -> Secrets, then reboot app |
| "No data found" | Check ticker or upload a price CSV (Date, Close, Volume) |
| 429 / rate limit | Increase pause slider; wait a minute |
| Empty fundamentals | Statements not available for that ticker on the data feed |
