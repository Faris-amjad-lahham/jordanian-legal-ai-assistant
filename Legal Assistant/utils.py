"""
utils.py
--------
Shared utilities for the Jordanian Legal AI Assistant:
  - RateLimiter   : session-based token budget (prevents runaway API costs)
  - export_conversation_pdf : convert chat history to downloadable PDF
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_QA_LIMIT = 30_000        # tokens per session for Q&A
_CONTRACT_LIMIT = 20_000  # tokens per session for contract ops

_DISCLAIMER = (
    "DISCLAIMER: This document is for informational / educational purposes only "
    "and does not constitute legal advice. Always consult a licensed Jordanian "
    "lawyer before making any legal decisions."
)


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiter
# ─────────────────────────────────────────────────────────────────────────────
class RateLimiter:
    """
    Simple session-based token budget.

    Usage:
        if not RateLimiter.check("qa"):
            RateLimiter.show_warning("qa")
        else:
            # call OpenAI...
            RateLimiter.increment("qa", tokens_used)
    """

    @staticmethod
    def _key(mode: str) -> str:
        return f"_rl_{mode}"

    @staticmethod
    def _limit(mode: str) -> int:
        return _QA_LIMIT if mode == "qa" else _CONTRACT_LIMIT

    @staticmethod
    def used(mode: str) -> int:
        return int(st.session_state.get(RateLimiter._key(mode), 0))

    @staticmethod
    def remaining(mode: str) -> int:
        return max(0, RateLimiter._limit(mode) - RateLimiter.used(mode))

    @staticmethod
    def check(mode: str, estimated: int = 1_500) -> bool:
        """True if the request is within the session budget."""
        return (RateLimiter.used(mode) + estimated) <= RateLimiter._limit(mode)

    @staticmethod
    def increment(mode: str, tokens: int) -> None:
        key = RateLimiter._key(mode)
        st.session_state[key] = RateLimiter.used(mode) + max(0, tokens)

    @staticmethod
    def show_warning(mode: str) -> None:
        rem = RateLimiter.remaining(mode)
        lim = RateLimiter._limit(mode)
        used = lim - rem
        pct = (used / lim) * 100 if lim else 100

        if rem == 0:
            st.error(
                "⚠️ **Session limit reached.** You've used all available tokens "
                "for this session. Please refresh the page to start a new session."
            )
        elif pct >= 80:
            st.warning(
                f"⚠️ **{rem:,} tokens remaining** this session "
                f"({pct:.0f}% used). Consider starting a new session soon."
            )

    @staticmethod
    def sidebar_meter(mode: str) -> None:
        """Show a compact token usage meter in the sidebar."""
        rem = RateLimiter.remaining(mode)
        lim = RateLimiter._limit(mode)
        used = lim - rem
        pct = used / lim if lim else 1.0

        color = "#4ade80" if pct < 0.6 else "#f6c90e" if pct < 0.85 else "#f87171"
        st.markdown(
            f"""
            <div style="margin:4px 0 8px">
                <div style="font-size:11px;color:#718096;margin-bottom:3px">
                    Session tokens: {used:,} / {lim:,}
                </div>
                <div style="background:#1a1f2e;border-radius:6px;height:5px;overflow:hidden">
                    <div style="width:{pct*100:.1f}%;height:100%;background:{color};
                                border-radius:6px;transition:width 0.3s"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Conversation PDF export
# ─────────────────────────────────────────────────────────────────────────────
def export_conversation_pdf(
    messages: List[Dict[str, Any]],
    title: str = "Legal Q&A Session",
) -> Optional[bytes]:
    """
    Convert a list of chat messages to a PDF.

    Each message dict: {"role": "user"|"assistant", "content": str,
                        "articles": list, "low_confidence": bool}

    Returns raw PDF bytes, or None on failure.
    """
    try:
        from fpdf import FPDF  # type: ignore
    except ImportError:
        return None

    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # ── Title block ───────────────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 17)
        pdf.set_text_color(26, 31, 46)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align="C")

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(113, 128, 150)
        pdf.cell(
            0, 5,
            f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}  |  "
            "Jordanian Legal Assistant  |  For informational purposes only",
            new_x="LMARGIN", new_y="NEXT", align="C",
        )
        pdf.ln(3)
        pdf.set_draw_color(45, 55, 72)
        pdf.set_line_width(0.4)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # ── Messages ─────────────────────────────────────────────────────────
        for msg in messages:
            role = msg.get("role", "")
            raw = msg.get("content", "")
            # Encode to latin-1 safely (fpdf default encoding)
            content = raw.encode("latin-1", errors="replace").decode("latin-1")

            if role == "user":
                pdf.set_fill_color(30, 58, 95)
                pdf.set_text_color(232, 244, 253)
                pdf.set_font("Helvetica", "B", 8)
                pdf.cell(0, 5, "You", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(232, 244, 253)
                pdf.multi_cell(0, 5, content)
            else:
                pdf.set_text_color(26, 31, 46)
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(78, 180, 161)
                pdf.cell(0, 5, "⚖ Legal Assistant", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(45, 55, 72)
                pdf.multi_cell(0, 5, content)

                # Article references
                arts = msg.get("articles")
                if arts:
                    pdf.set_font("Helvetica", "I", 8)
                    pdf.set_text_color(113, 128, 150)
                    pdf.cell(
                        0, 4,
                        f"Articles cited: {', '.join(str(a) for a in arts)}",
                        new_x="LMARGIN", new_y="NEXT",
                    )

                # Confidence
                if msg.get("low_confidence"):
                    pdf.set_font("Helvetica", "I", 8)
                    pdf.set_text_color(248, 113, 113)
                    pdf.cell(0, 4, "[Low confidence — verify with a lawyer]",
                             new_x="LMARGIN", new_y="NEXT")

            pdf.ln(2)
            # Separator
            pdf.set_draw_color(226, 232, 240)
            pdf.set_line_width(0.2)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)

        # ── Footer disclaimer ─────────────────────────────────────────────────
        pdf.ln(4)
        pdf.set_draw_color(45, 55, 72)
        pdf.set_line_width(0.4)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(160, 174, 192)
        pdf.multi_cell(0, 4, _DISCLAIMER)

        return bytes(pdf.output())

    except Exception as exc:
        print(f"[utils] PDF export error: {exc}")
        return None
