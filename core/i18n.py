import streamlit as st

from .config import TEXT


def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return TEXT.get(lang, TEXT["en"]).get(key, key)


def get_text(lang: str, key: str) -> str:
    return TEXT.get(lang, TEXT["en"]).get(key, key)


def toggle_language() -> None:
    current = st.session_state.get("lang", "en")
    st.session_state["lang"] = "ko" if current == "en" else "en"
