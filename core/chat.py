from typing import Any, Dict, Optional, Tuple

import streamlit as st

from .config import (
    ENABLE_EXAMPLES_IN_PROMPT,
    MAX_LLM_ROWS,
    MAX_SQL_ROWS,
    SCHEMA_TOPK_COLUMNS,
    SCHEMA_TOPK_TABLES,
    SQL_RETRY_ON_EMPTY,
    SQL_RETRY_ON_ERROR,
)
from .db import (
    build_schema_context_topk,
    cached_examples_text,
    cached_schema_text,
    run_sql_query,
)
from .i18n import get_text
from .llm import (
    generate_fallback_answer,
    generate_final_answer,
    generate_sql_query,
    generate_sql_retry,
    should_use_sql,
)
from .sql_utils import enforce_limit, is_safe_sql, truncate_results


def format_history(messages, limit: int = 6) -> str:
    recent = messages[-limit:] if messages else []
    lines = []
    for message in recent:
        role = message.get("role", "user")
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "N/A"


def sanitize_history_messages(messages) -> list:
    cleaned = []
    for message in messages or []:
        cleaned.append(
            {
                "role": message.get("role", "user"),
                "content": message.get("content", ""),
            }
        )
    return cleaned


def run_chat(
    client,
    model: str,
    lang: str,
    user_input: str,
    system_prompt_override: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    1) SQL 필요 여부 판단
    2) 필요 없으면 SQL 없이 답변 생성
    3) 필요하면 SQL 생성 + 실행 + 결과 기반 답변
    """
    history_text = format_history(st.session_state.get("messages", []))

    full_schema = cached_schema_text()
    use_sql, use_sql_error = should_use_sql(
        client=client,
        model=model,
        lang=lang,
        question=user_input,
        schema_context=full_schema or "N/A",
    )

    history_messages = sanitize_history_messages(st.session_state.get("messages", []))

    if use_sql is False:
        print("[DEBUG][Chat] SQL not required -> non-SQL response")
        fallback, _ = generate_fallback_answer(
            client,
            model,
            lang,
            user_input,
            "no_sql_needed",
            history_messages=history_messages,
        )
        reply = fallback or get_text(lang, "no_sql_response")
        return reply, {
            "sql": None,
            "executed_sql": None,
            "results": [],
            "truncated": False,
            "sql_error": None,
            "skipped": True,
        }

    if use_sql_error:
        print(f"[DEBUG][Chat] SQL need classification error: {use_sql_error}. Proceeding with SQL.")

    schema_context = build_schema_context_topk(
        question=user_input,
        history_text=history_text,
        topk_tables=SCHEMA_TOPK_TABLES,
        topk_columns=SCHEMA_TOPK_COLUMNS,
    ) or full_schema

    examples_text = (
        cached_examples_text() if ENABLE_EXAMPLES_IN_PROMPT else None
    )

    sql, sql_error = generate_sql_query(
        client=client,
        model=model,
        lang=lang,
        question=user_input,
        schema_context=schema_context or "N/A",
        history_text=history_text,
        allow_no_sql=True,
        examples_text=examples_text,
    )

    results = None
    execution_error = None
    executed_sql = None

    if sql_error:
        execution_error = sql_error
    elif sql and sql.strip().lower().startswith("no_sql"):
        fallback, _ = generate_fallback_answer(
            client,
            model,
            lang,
            user_input,
            "no_sql",
            history_messages=history_messages,
        )
        reply = fallback or get_text(lang, "no_sql_response")
        return reply, {
            "sql": sql,
            "executed_sql": None,
            "results": [],
            "truncated": False,
            "sql_error": "no_sql",
        }
    elif not sql or not is_safe_sql(sql):
        fallback, _ = generate_fallback_answer(
            client,
            model,
            lang,
            user_input,
            "no_sql",
            history_messages=history_messages,
        )
        reply = fallback or get_text(lang, "no_sql_response")
        return reply, {
            "sql": sql,
            "executed_sql": None,
            "results": [],
            "truncated": False,
            "sql_error": "unsafe_or_empty_sql",
        }
    else:
        executed_sql = enforce_limit(sql, MAX_SQL_ROWS)
        results, execution_error = run_sql_query(executed_sql)

    limited_results, truncated = truncate_results(results, MAX_LLM_ROWS)

    did_retry = False

    if execution_error and SQL_RETRY_ON_ERROR and executed_sql:
        retry_sql, retry_err = generate_sql_retry(
            client=client,
            model=model,
            lang=lang,
            question=user_input,
            schema_context=schema_context or "N/A",
            history_text=history_text,
            prev_sql=executed_sql,
            exec_error=execution_error,
            had_no_rows=False,
            examples_text=examples_text,
        )
        did_retry = True

        if retry_err is None and retry_sql and is_safe_sql(retry_sql):
            executed_sql = enforce_limit(retry_sql, MAX_SQL_ROWS)
            results, execution_error = run_sql_query(executed_sql)
            limited_results, truncated = truncate_results(results, MAX_LLM_ROWS)
            sql = retry_sql

    if (not execution_error) and (not limited_results) and SQL_RETRY_ON_EMPTY and executed_sql:
        retry_sql, retry_err = generate_sql_retry(
            client=client,
            model=model,
            lang=lang,
            question=user_input,
            schema_context=schema_context or "N/A",
            history_text=history_text,
            prev_sql=executed_sql,
            exec_error=None,
            had_no_rows=True,
            examples_text=examples_text,
        )
        did_retry = True

        if retry_err is None and retry_sql and is_safe_sql(retry_sql):
            executed_sql = enforce_limit(retry_sql, MAX_SQL_ROWS)
            results, execution_error = run_sql_query(executed_sql)
            limited_results, truncated = truncate_results(results, MAX_LLM_ROWS)
            sql = retry_sql

    print(f"[DEBUG][Chat] did_retry={did_retry}, sql_error={execution_error}, rows={len(limited_results)}")

    if execution_error:
        fallback, _ = generate_fallback_answer(
            client,
            model,
            lang,
            user_input,
            "sql_error",
            history_messages=history_messages,
        )
        reply = fallback or get_text(lang, "sql_error_response")
        return reply, {
            "sql": sql,
            "executed_sql": executed_sql,
            "results": limited_results,
            "truncated": truncated,
            "sql_error": execution_error,
        }

    if not limited_results:
        fallback, _ = generate_fallback_answer(
            client,
            model,
            lang,
            user_input,
            "no_data",
            history_messages=history_messages,
        )
        reply = fallback or get_text(lang, "no_data_response")
        return reply, {
            "sql": sql,
            "executed_sql": executed_sql,
            "results": [],
            "truncated": truncated,
            "sql_error": None,
        }

    reply = generate_final_answer(
        client,
        model,
        lang,
        system_prompt_override,
        user_input,
        sql,
        limited_results,
        truncated,
        execution_error,
        history_messages=history_messages,
    )
    return reply, {
        "sql": sql,
        "executed_sql": executed_sql,
        "results": limited_results,
        "truncated": truncated,
        "sql_error": execution_error,
    }
