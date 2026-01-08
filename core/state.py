import streamlit as st


def init_state(build_prompt_func) -> None:
    if "lang" not in st.session_state:
        st.session_state["lang"] = "ko"
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "db_context" not in st.session_state:
        st.session_state["db_context"] = ""
    if "company_data" not in st.session_state:
        st.session_state["company_data"] = None
    if "chat_system_prompt" not in st.session_state:
        st.session_state["chat_system_prompt"] = ""
    if "last_generation_response" not in st.session_state:
        st.session_state["last_generation_response"] = ""
    if "last_papers_response" not in st.session_state:
        st.session_state["last_papers_response"] = ""
    if "agent_intent" not in st.session_state:
        st.session_state["agent_intent"] = ""
    if "agent_draft_edit" not in st.session_state:
        st.session_state["agent_draft_edit"] = ""

    if not st.session_state.get("chat_system_prompt"):
        st.session_state["chat_system_prompt"] = build_prompt_func(
            st.session_state.get("lang", "en"),
            None,
        )
