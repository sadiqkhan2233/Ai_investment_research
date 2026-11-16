"""Groq client wrapper with retries, rate-limit handling and safe key lookup."""
from __future__ import annotations

import os
import re
import time

from groq import Groq

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def get_api_key() -> str | None:
    """Env var first (local), then Streamlit secrets (Streamlit Cloud)."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def _retry_after(msg: str) -> float | None:
    m = re.search(r"try again in (?:(\d+)m)?\s*([\d.]+)s", msg)
    if not m:
        return None
    return int(m.group(1) or 0) * 60 + float(m.group(2))


def chat(system: str, user: str, *, temperature: float = 0.2, max_tokens: int = 2000, retries: int = 4) -> str:
    key = get_api_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it in Streamlit Cloud -> Settings -> Secrets.")
    client = Groq(api_key=key, timeout=90.0, max_retries=0)
    use_reasoning, last = True, None
    for attempt in range(1, retries + 1):
        kwargs = dict(
            model=MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        if use_reasoning:
            kwargs["reasoning_effort"] = "low"  # keeps token use low on the free tier
        try:
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content
            if text and text.strip():
                return text.strip()
            last = RuntimeError("Empty response (reasoning may have used all tokens)")
            max_tokens = int(max_tokens * 1.5)
        except Exception as e:  # noqa: BLE001
            msg, last = str(e), e
            low = msg.lower()
            if use_reasoning and "reasoning" in low:
                use_reasoning = False
                continue
            if "401" in msg or "invalid api key" in low:
                raise RuntimeError("Groq rejected the API key (401). Check GROQ_API_KEY.") from e
            if "429" in msg or "rate limit" in low:
                time.sleep(min(_retry_after(msg) or 15 * attempt, 60) + 1)
                continue
            time.sleep(2 * attempt)
    raise RuntimeError(f"Groq request failed after {retries} attempts: {last}")
