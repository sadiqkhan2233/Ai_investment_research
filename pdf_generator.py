"""Markdown-ish text -> simple professional PDF using fpdf2 (core fonts, latin-1 safe)."""
from __future__ import annotations

import re

from fpdf import FPDF

_REPL = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
         "\u2022": "-", "\u2192": "->", "\u2265": ">=", "\u2264": "<=", "\u00a0": " "}


def _clean(s: str) -> str:
    for k, v in _REPL.items():
        s = s.replace(k, v)
    s = re.sub(r"[*_`]{1,3}", "", s)
    return s.encode("latin-1", "replace").decode("latin-1")


def markdown_to_pdf(md: str, title: str = "Investment Research Report") -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    def line(txt: str, size: int = 10, bold: bool = False, h: float = 5.5):
        pdf.set_font("Helvetica", "B" if bold else "", size)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, h, _clean(txt), new_x="LMARGIN", new_y="NEXT")

    for raw in md.splitlines():
        t = raw.rstrip()
        if not t.strip() or t.strip() == "---":
            pdf.ln(2)
        elif t.startswith("# "):
            line(t[2:], 16, True, 8)
            pdf.ln(1)
        elif t.startswith("## "):
            pdf.ln(2)
            line(t[3:], 13, True, 7)
        elif t.startswith("### "):
            line(t[4:], 11, True, 6)
        elif t.lstrip().startswith(("- ", "* ")):
            line("  - " + t.lstrip()[2:])
        elif t.startswith("|"):
            if set(t.replace("|", "").strip()) <= set("-: "):
                continue
            line("  ".join(c.strip() for c in t.strip("|").split("|")), 9)
        else:
            line(t.lstrip("> "))
    return bytes(pdf.output())
