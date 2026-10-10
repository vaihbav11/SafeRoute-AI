"""
chat_assistant.py — SafeRoute AI
Gemini-powered chat assistant that can answer any question.

Key design decisions:
- Uses Google Gemini (gemini-1.5-flash) via google-generativeai — free tier available.
- No restrictions on topics; answers any question freely.
- API key sourced from env var GEMINI_API_KEY → st.secrets["GEMINI_API_KEY"].
- No key → graceful message instead of a crash.
"""

from __future__ import annotations

import os
from typing import Optional


# ── API-key resolution ────────────────────────────────────────────────────────
def get_gemini_api_key() -> Optional[str]:
    """
    Return the Gemini API key or None.
    Priority: environment variable GEMINI_API_KEY → st.secrets["GEMINI_API_KEY"].
    """
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


# ── Chat API call ─────────────────────────────────────────────────────────────
def chat(messages: list[dict], api_key: str) -> str:
    """
    Call Gemini API with the full conversation history.
    messages: list of {"role": "user"|"assistant", "content": "..."}
    Returns the assistant reply string.
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Convert messages to Gemini history format
    history = []
    for msg in messages[:-1]:  # all but the last message
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})

    chat_session = model.start_chat(history=history)

    # Send the last user message
    last_message = messages[-1]["content"]
    response = chat_session.send_message(last_message)
    return response.text
