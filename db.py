"""SQLite storage. NOTE: on Streamlit Community Cloud the disk is temporary -
reports/watchlist vanish when the app restarts. Swap this module for a hosted DB
(e.g. Supabase/Postgres) if you need permanent storage."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime

DB_PATH = os.environ.get("DB_PATH", "data/app.db")


def _conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute("CREATE TABLE IF NOT EXISTS reports(id INTEGER PRIMARY KEY, ticker TEXT, created TEXT, status TEXT, markdown TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS watchlist(ticker TEXT PRIMARY KEY, added TEXT)")
    return c


def save_report(ticker: str, status: str, markdown: str) -> None:
    with _conn() as c:
        c.execute("INSERT INTO reports(ticker,created,status,markdown) VALUES(?,?,?,?)",
                  (ticker, datetime.now().strftime("%Y-%m-%d %H:%M"), status, markdown))


def list_reports(limit: int = 30) -> list[tuple]:
    with _conn() as c:
        return c.execute("SELECT id,ticker,created,status FROM reports ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def get_report(rid: int) -> str:
    with _conn() as c:
        row = c.execute("SELECT markdown FROM reports WHERE id=?", (rid,)).fetchone()
    return row[0] if row else ""


def add_watch(ticker: str) -> None:
    with _conn() as c:
        c.execute("INSERT OR IGNORE INTO watchlist VALUES(?,?)", (ticker, datetime.now().strftime("%Y-%m-%d")))


def remove_watch(ticker: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM watchlist WHERE ticker=?", (ticker,))


def list_watch() -> list[str]:
    with _conn() as c:
        return [r[0] for r in c.execute("SELECT ticker FROM watchlist ORDER BY ticker")]
