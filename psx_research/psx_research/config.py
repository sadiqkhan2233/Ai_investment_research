"""Configuration, sector peer map and LLM factory.
Environment flags must be set BEFORE crewai is imported."""
import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import LLM  # noqa: E402

MODEL_CHOICES = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

# Common PSX sector peers (starting points only; the user can override in the UI).
SECTOR_PEERS = {
    "Commercial Banks": ["HBL", "UBL", "MCB", "MEBL", "BAHL", "NBP", "ABL", "BAFL"],
    "Cement": ["LUCK", "DGKC", "MLCF", "FCCL", "CHCC", "KOHC", "PIOC"],
    "Oil & Gas Exploration": ["OGDC", "PPL", "POL", "MARI"],
    "Oil Marketing": ["PSO", "APL", "SHEL", "HASCOL"],
    "Fertilizer": ["ENGRO", "FFC", "EFERT", "FATIMA"],
    "Automobile": ["INDU", "HCAR", "MTL", "PSMC", "ATLH"],
    "Technology": ["SYS", "TRG", "NETSOL", "AVN"],
    "Pharmaceuticals": ["SEARL", "GLAXO", "ABOT", "HINOON"],
    "Power": ["HUBC", "KAPCO", "NPL", "NCPL"],
    "Textile": ["NML", "NCL", "GATM", "ILP"],
}


def find_peers(symbol: str):
    for sector, tickers in SECTOR_PEERS.items():
        if symbol in tickers:
            return sector, [t for t in tickers if t != symbol][:5]
    return "Unknown", []


def make_llm(model: str, api_key: str, max_tokens: int = 2000, temperature: float = 0.2) -> LLM:
    return LLM(
        model=f"groq/{model}",
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
