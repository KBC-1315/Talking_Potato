from datetime import datetime
import os

import streamlit as st

from core.chat import run_chat
from core.config import (
    APP_TITLE,
    DB_PATH,
    DB_SIZE_VALUES,
    MODEL_OPTIONS,
    PAPER_COUNT_PRESETS,
    TABLES,
)
from core.db import (
    build_db_context,
    cached_examples_text,
    cached_schema_text,
    get_db_overview,
    replace_company_data,
    replace_papers_data,
)
from core.i18n import t, toggle_language
from core.llm import (
    build_chat_system_prompt,
    build_company_prompts,
    build_papers_prompts,
    generate_agent_draft,
    generate_company_profile,
    generate_papers_profile,
    get_openai_client,
)
from core.state import init_state


def main() -> None:
    init_state(build_chat_system_prompt)
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(t("title"))
    st.caption(t("subtitle"))

    if os.path.exists(DB_PATH) and not st.session_state.get("db_context"):
        st.session_state["db_context"] = build_db_context()

    with st.sidebar:
        st.button(t("lang_toggle"), on_click=toggle_language)
        st.markdown(f"**{t('api_section')}**")

        api_key = st.text_input(
            t("api_key"),
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help=t("api_key_help"),
        )

        model = st.selectbox(t("model"), MODEL_OPTIONS, help=t("model_help"))

        st.text_area(
            t("chat_system_prompt_label"),
            key="chat_system_prompt",
            help=t("chat_system_prompt_help"),
            height=140,
        )

        base_url = st.text_input(t("base_url"), help=t("base_url_help"))

    tab_setup, tab_db, tab_chat = st.tabs([t("tab_setup"), t("tab_db"), t("tab_chat")])

    with tab_setup:
        st.markdown(f"### {t('company_section')}")
        col1, col2 = st.columns(2)
        with col1:
            company_name = st.text_input(t("company_name"))
            industry = st.text_input(t("industry"), value="Technology")
            location = st.text_input(t("location"), value="Seoul")
        with col2:
            size = st.slider(t("size"), min_value=30, max_value=500, value=120, step=10)
            tone = st.selectbox(t("tone"), options=t("tone_options"))
            size_labels = t("db_size_options")
            size_map = (
                dict(zip(DB_SIZE_VALUES, size_labels))
                if isinstance(size_labels, list) and len(size_labels) == len(DB_SIZE_VALUES)
                else {}
            )
            db_size = st.selectbox(
                t("db_size"),
                options=DB_SIZE_VALUES,
                index=1,
                format_func=lambda value: size_map.get(value, value),
                help=t("db_size_help"),
            )
            overwrite = st.checkbox(t("overwrite_db"), value=True)

        generate_clicked = st.button(t("generate"))

        system_prompt, user_prompt = build_company_prompts(
            company_name,
            industry,
            location,
            size,
            tone,
            st.session_state["lang"],
            db_size,
        )

        st.markdown(f"### {t('prompt_section')}")
        st.text_area(t("system_prompt_label"), value=system_prompt, height=120)
        st.text_area(t("user_prompt_label"), value=user_prompt, height=300)
        st.text_area(
            t("llm_response_label"),
            value=st.session_state.get("last_generation_response", ""),
            height=240,
        )

        if generate_clicked:
            if os.path.exists(DB_PATH) and not overwrite:
                st.warning(t("db_exists"))
            else:
                client, error_key = get_openai_client(api_key, base_url if base_url else None)
                if error_key:
                    st.error(t(error_key))
                else:
                    data, error_key, raw_response = generate_company_profile(
                        client,
                        model,
                        company_name,
                        industry,
                        location,
                        size,
                        tone,
                        st.session_state["lang"],
                        db_size,
                    )
                    if raw_response is not None:
                        st.session_state["last_generation_response"] = raw_response
                    if error_key:
                        st.error(t(error_key))
                    else:
                        replace_company_data(data)
                        cached_schema_text.clear()
                        cached_examples_text.clear()
                        st.session_state["company_data"] = data
                        st.session_state["db_context"] = build_db_context()
                        st.success(t("db_created"))


    with tab_db:
        st.markdown(f"### {t('db_status_title')}")
        if not os.path.exists(DB_PATH):
            st.info(t("no_db"))
        else:
            st.caption(f"{t('db_path_label')}: {DB_PATH}")
            counts, samples = get_db_overview()
            st.markdown(f"**{t('db_status_counts')}**")
            st.table(counts)
            for table in TABLES:
                st.markdown(f"#### {table}")
                rows = samples.get(table, [])
                if rows:
                    st.dataframe(rows, width="stretch")
                else:
                    st.write(t("db_status_empty"))

    with tab_chat:
        if not os.path.exists(DB_PATH):
            st.info(t("no_db"))
            return

        st.markdown(f"### {t('chat_ready')}")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(t("refresh_context")):
                st.session_state["db_context"] = build_db_context()
                st.success(t("context_updated"))
        with col2:
            if st.button(t("reset_chat")):
                st.session_state["messages"] = []

        def render_sql_debug(info):
            st.markdown(f"**{t('sql_debug_title')}**")
            if info.get("skipped"):
                st.write(t("sql_debug_skipped"))
                return
            st.markdown(f"{t('sql_debug_query')}:")
            sql_text = info.get("executed_sql") or info.get("sql") or "N/A"
            st.code(sql_text, language="sql")
            sql_error = info.get("sql_error")
            if sql_error:
                st.warning(f"{t('sql_debug_error')}: {sql_error}")
            st.markdown(f"{t('sql_debug_results')}:")
            results = info.get("results") or []
            if results:
                st.dataframe(results, width="stretch")
            else:
                st.write(t("sql_debug_empty"))

        def render_chat_turn(message_text: str) -> None:
            st.session_state["messages"].append({"role": "user", "content": message_text})
            with st.chat_message("user"):
                st.write(message_text)

            client, error_key = get_openai_client(api_key, base_url if base_url else None)
            if error_key:
                reply = t(error_key)
                debug_info = None
            else:
                try:
                    reply, debug_info = run_chat(
                        client,
                        model,
                        st.session_state["lang"],
                        message_text,
                        st.session_state.get("chat_system_prompt"),
                    )
                except Exception:
                    reply = t("llm_error")
                    debug_info = None

            message_payload = {"role": "assistant", "content": reply}
            if debug_info:
                message_payload["sql_info"] = debug_info
            st.session_state["messages"].append(message_payload)
            with st.chat_message("assistant"):
                st.write(reply)
                if debug_info:
                    render_sql_debug(debug_info)

        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and message.get("sql_info"):
                    render_sql_debug(message["sql_info"])

        agent_enabled = st.checkbox(t("agent_toggle"), value=False, key="agent_enabled")
        if agent_enabled:
            st.markdown(f"#### {t('agent_section')}")
            agent_model = st.selectbox(
                t("agent_model"),
                options=MODEL_OPTIONS,
                key="agent_model",
            )
            st.text_area(
                t("agent_intent"),
                key="agent_intent",
                height=120,
            )
            if st.button(t("agent_generate")):
                if not st.session_state["agent_intent"].strip():
                    st.warning(t("agent_missing_intent"))
                else:
                    client, error_key = get_openai_client(api_key, base_url if base_url else None)
                    if error_key:
                        st.error(t(error_key))
                    else:
                        draft, draft_error = generate_agent_draft(
                            client,
                            agent_model,
                            st.session_state["lang"],
                            st.session_state["agent_intent"],
                            st.session_state.get("db_context", ""),
                        )
                        if draft_error:
                            st.error(t(draft_error))
                        else:
                            st.session_state["agent_draft_edit"] = draft or ""

            st.text_area(
                t("agent_draft"),
                key="agent_draft_edit",
                height=140,
            )
            if st.button(t("agent_send")):
                final_message = st.session_state.get("agent_draft_edit", "").strip()
                if not final_message:
                    st.warning(t("agent_empty_draft"))
                else:
                    render_chat_turn(final_message)
        else:
            user_input = st.chat_input(t("user_input"))
            if user_input:
                render_chat_turn(user_input)


if __name__ == "__main__":
    main()
