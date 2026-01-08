import json
import os
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# App / DB 설정
# ============================================================
APP_TITLE = "SQL + LLM Company Chatbot"
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "company.db")
TABLES = ["company", "departments", "employees", "products", "policies"]
DB_SIZE_VALUES = ["small", "medium", "large"]

# SQL 실행 결과 제한(보안/성능 목적)
MAX_SQL_ROWS = 200
MAX_LLM_ROWS = 200

# ============================================================
# (개선) LLM-to-SQL 품질/비용 개선 옵션
# - 보안 관련 로직(is_safe_sql 등)은 그대로 유지
# ============================================================
SCHEMA_TOPK_TABLES = 3          # 질문과 관련 있어 보이는 테이블 Top-K만 스키마 컨텍스트에 포함
SCHEMA_TOPK_COLUMNS = 12        # 선택된 테이블에서 컬럼 Top-K만 포함(너무 길어지는 것 방지)
SQL_RETRY_ON_EMPTY = True       # 결과 0 rows일 때 1회 재시도할지
SQL_RETRY_ON_ERROR = True       # SQL 실행 에러일 때 1회 재시도할지
ENABLE_EXAMPLES_IN_PROMPT = True  # examples(샘플 행)를 SQL 생성 프롬프트에 넣을지

# 간단 스몰톡/사용법 질문 감지용 패턴
SMALLTALK_PATTERNS = [
    r"^\s*(hi|hello|hey)\b",
    r"^\s*(안녕|하이|헬로)\b",
    r"\b(너\s*누구|무슨\s*기능|사용법|help|도움말)\b",
]


# ============================================================
# UI 텍스트
# ============================================================
TEXT = {
    "en": {
        "title": "Talking Potatoes",
        "subtitle": "Create a virtual company database with an LLM, then chat over it using SQL.",
        "tab_setup": "Setup",
        "tab_db": "DB Status",
        "tab_chat": "Chatbot",
        "lang_toggle": "한국어로 전환",
        "lang_label": "Language",
        "api_section": "LLM Settings",
        "api_key": "OpenAI API Key",
        "api_key_help": "Stored only in session state. Required to call the LLM.",
        "model": "Model",
        "model_help": "Choose a model for generation and chat.",
        "chat_system_prompt_label": "Chatbot system prompt",
        "chat_system_prompt_help": "Optional. You can use {guardrails}, {language}, {examples} placeholders.",
        "base_url": "Base URL (optional)",
        "base_url_help": "Use this to point to a compatible API endpoint.",
        "company_section": "Company Seed",
        "prompt_section": "Generation Prompts",
        "system_prompt_label": "Generation system prompt",
        "user_prompt_label": "Generation user prompt",
        "llm_response_label": "LLM response",
        "db_size": "DB size",
        "db_size_help": "Controls how much data the LLM generates.",
        "db_size_options": ["Small", "Medium", "Large"],
        "agent_section": "LLM Agent Draft",
        "agent_toggle": "Enable LLM agent drafting",
        "agent_model": "Agent model",
        "agent_intent": "Intent (what you want to ask)",
        "agent_generate": "Generate draft",
        "agent_draft": "Draft message",
        "agent_send": "Confirm and send",
        "agent_missing_intent": "Please enter your intent before generating.",
        "agent_empty_draft": "Draft is empty. Generate or write a message first.",
        "sql_debug_title": "SQL Execution",
        "sql_debug_query": "Executed SQL",
        "sql_debug_results": "SQL Results",
        "sql_debug_error": "SQL Error",
        "sql_debug_empty": "No rows returned.",
        "small_talk_response": "Hi! Ask me about the company database.",
        "no_sql_response": "I couldn't generate a database query for that. Please ask about the company data.",
        "no_data_response": "No matching data was found in the database.",
        "sql_error_response": "I couldn't run the database query. Please try again.",
        "company_name": "Company name (optional)",
        "industry": "Industry",
        "location": "HQ location",
        "size": "Approx. employee count",
        "tone": "Communication tone",
        "tone_options": ["Professional", "Friendly", "Formal", "Casual"],
        "generate": "Generate Company + Database",
        "overwrite_db": "Overwrite existing database",
        "db_created": "Database created successfully.",
        "db_exists": "Database already exists. Enable overwrite to regenerate.",
        "missing_key": "Please provide an API key to generate with the LLM.",
        "missing_sdk": "OpenAI SDK is not available. Install requirements first.",
        "llm_error": "LLM call failed. Check your API key and model.",
        "parse_error": "Could not parse LLM response as JSON. Try again.",
        "chat_ready": "Chat with your company assistant.",
        "no_db": "No database yet. Please generate one in the Setup tab.",
        "refresh_context": "Refresh DB context",
        "reset_chat": "Reset conversation",
        "user_input": "Ask about the company, people, products, or policies...",
        "context_updated": "DB context refreshed.",
        "db_status_title": "Database Overview",
        "db_status_counts": "Table row counts",
        "db_status_empty": "No rows available.",
        "db_path_label": "DB path",
    },
    "ko": {
        "title": "말하는 감자들",
        "subtitle": "LLM으로 가상 회사를 만들고 SQL로 읽어 대화하는 챗봇입니다.",
        "tab_setup": "설정",
        "tab_db": "DB 상태",
        "tab_chat": "챗봇",
        "lang_toggle": "Switch to English",
        "lang_label": "언어",
        "api_section": "LLM 설정",
        "api_key": "OpenAI API Key",
        "api_key_help": "세션에만 저장됩니다. LLM 호출에 필요합니다.",
        "model": "모델",
        "model_help": "생성과 챗봇에 사용할 모델을 선택하세요.",
        "chat_system_prompt_label": "챗봇 시스템 프롬프트",
        "chat_system_prompt_help": "선택 사항입니다. {guardrails}, {language}, {examples} 치환자를 사용할 수 있습니다.",
        "base_url": "Base URL (선택)",
        "base_url_help": "호환 API 엔드포인트를 사용할 때 입력하세요.",
        "company_section": "회사 기본 정보",
        "prompt_section": "생성 프롬프트",
        "system_prompt_label": "생성 시스템 프롬프트",
        "user_prompt_label": "생성 유저 프롬프트",
        "llm_response_label": "LLM 응답",
        "db_size": "DB 크기",
        "db_size_help": "LLM이 생성하는 데이터 양을 조절합니다.",
        "db_size_options": ["작게", "보통", "크게"],
        "agent_section": "LLM 에이전트 초안",
        "agent_toggle": "LLM 에이전트로 메시지 초안 생성",
        "agent_model": "에이전트 모델",
        "agent_intent": "의도(무엇을 물어보고 싶은가요)",
        "agent_generate": "초안 생성",
        "agent_draft": "메시지 초안",
        "agent_send": "확인 후 전송",
        "agent_missing_intent": "초안을 생성하려면 의도를 입력하세요.",
        "agent_empty_draft": "초안이 비어있습니다. 생성하거나 직접 작성하세요.",
        "sql_debug_title": "SQL 실행",
        "sql_debug_query": "실행된 SQL",
        "sql_debug_results": "SQL 결과",
        "sql_debug_error": "SQL 오류",
        "sql_debug_empty": "반환된 데이터가 없습니다.",
        "small_talk_response": "안녕하세요! 회사 데이터에 대해 질문해 주세요.",
        "no_sql_response": "해당 질문에 대한 DB 쿼리를 만들 수 없었습니다. 회사 데이터에 대해 질문해 주세요.",
        "no_data_response": "DB에서 일치하는 데이터를 찾지 못했습니다.",
        "sql_error_response": "DB 쿼리를 실행하지 못했습니다. 다시 시도해 주세요.",
        "company_name": "회사명 (선택)",
        "industry": "산업",
        "location": "본사 위치",
        "size": "직원 수(대략)",
        "tone": "커뮤니케이션 톤",
        "tone_options": ["프로페셔널", "친근함", "격식", "캐주얼"],
        "generate": "회사 + 데이터베이스 생성",
        "overwrite_db": "기존 DB 덮어쓰기",
        "db_created": "데이터베이스가 생성되었습니다.",
        "db_exists": "이미 DB가 있습니다. 덮어쓰기를 체크하세요.",
        "missing_key": "LLM 생성을 위해 API Key를 입력하세요.",
        "missing_sdk": "OpenAI SDK를 사용할 수 없습니다. requirements를 설치하세요.",
        "llm_error": "LLM 호출에 실패했습니다. API Key와 모델을 확인하세요.",
        "parse_error": "LLM 응답을 JSON으로 해석하지 못했습니다. 다시 시도하세요.",
        "chat_ready": "회사 어시스턴트와 대화하세요.",
        "no_db": "DB가 없습니다. 설정 탭에서 먼저 생성하세요.",
        "refresh_context": "DB 컨텍스트 새로고침",
        "reset_chat": "대화 초기화",
        "user_input": "회사, 사람, 제품, 정책에 대해 물어보세요...",
        "context_updated": "DB 컨텍스트가 업데이트되었습니다.",
        "db_status_title": "데이터베이스 현황",
        "db_status_counts": "테이블 행 개수",
        "db_status_empty": "데이터가 없습니다.",
        "db_path_label": "DB 경로",
    },
}


# ============================================================
# 다국어 / 상태 관리
# ============================================================
def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return TEXT.get(lang, TEXT["en"]).get(key, key)


def get_text(lang: str, key: str) -> str:
    return TEXT.get(lang, TEXT["en"]).get(key, key)


def toggle_language() -> None:
    current = st.session_state.get("lang", "en")
    st.session_state["lang"] = "ko" if current == "en" else "en"


def init_state() -> None:
    """
    Streamlit session_state 초기화.
    - 최초 실행 시 필요한 키들이 없다면 기본값을 세팅한다.
    """
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
    if "agent_intent" not in st.session_state:
        st.session_state["agent_intent"] = ""
    if "agent_draft_edit" not in st.session_state:
        st.session_state["agent_draft_edit"] = ""

    # 기본 시스템 프롬프트가 비어있으면 앱 기본 가드레일로 채움
    if not st.session_state.get("chat_system_prompt"):
        st.session_state["chat_system_prompt"] = build_chat_system_prompt(
            st.session_state.get("lang", "en"),
            None,
        )


# ============================================================
# OpenAI Client
# ============================================================
def get_openai_client(api_key: str, base_url: Optional[str]):
    """
    OpenAI SDK 클라이언트를 생성한다.
    - base_url이 있으면 호환 엔드포인트로도 사용 가능
    """
    if OpenAI is None:
        return None, "missing_sdk"
    if not api_key:
        return None, "missing_key"
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url), None
    return OpenAI(api_key=api_key), None


# ============================================================
# JSON 파싱 유틸
# ============================================================
def extract_json(text: str) -> dict | None:
    """
    텍스트에서 JSON 객체를 추출한다.
    - LLM 응답이 순수 JSON이 아니어도, 가장 바깥 { ... } 구간을 찾아 파싱 시도
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


# ============================================================
# Company 생성 관련 (기존 유지)
# ============================================================
def get_size_constraints(size_key: str) -> dict:
    presets = {
        "small": {"departments": (3, 5), "employees": (8, 12), "products": (2, 4), "policies": (3, 4)},
        "medium": {"departments": (4, 8), "employees": (12, 20), "products": (3, 6), "policies": (4, 6)},
        "large": {"departments": (6, 10), "employees": (20, 35), "products": (5, 8), "policies": (6, 8)},
    }
    return presets.get(size_key, presets["medium"])


def build_login_id(name: Optional[str], index: int) -> str:
    if name:
        cleaned = "".join(ch for ch in name.lower() if ch.isalnum())
        if cleaned:
            return cleaned[:12]
    return f"user{index + 1}"


def build_password(index: int) -> str:
    return f"pw{index + 1}!"


# ============================================================
# (보안/정책) 챗봇 시스템 프롬프트
# - 사용자가 override를 주더라도 현재는 기본 가드레일을 반환(현 상태 유지)
# ============================================================
def build_chat_system_prompt(lang: str, override: Optional[str]) -> str:
    base_guardrails = (
        "You are a helpful company assistant. Use only the database context provided. "
        "If a detail is not in the database, say you don't have it. "
        "Use SQL results as the source of truth. "
        "Example data is illustrative only and must not be treated as factual answers."
    )
    language = "Korean" if lang == "ko" else "English"
    return f"{base_guardrails} Answer in {language}."


# ============================================================
# (개선) fallback 프롬프트/답변
# ============================================================
def build_fallback_system_prompt(lang: str, reason: str) -> str:
    language = "Korean" if lang == "ko" else "English"
    reason_note = {
        "no_sql": "A suitable SQL query could not be generated.",
        "no_data": "SQL ran but returned no rows.",
        "sql_error": "SQL execution failed.",
    }.get(reason, "No database results are available.")
    return (
        "You are the assistant for a SQL + LLM company chatbot app. "
        "You do NOT have database results, and you must NOT invent company facts. "
        "If the user asks about app capabilities or how to use the app, answer clearly. "
        "Otherwise, explain that the data is unavailable and ask them to rephrase with a "
        "company-related question that can be answered from the database.\n\n"
        f"Reason: {reason_note}\n"
        f"Answer in {language}."
    )


def generate_fallback_answer(
    client,
    model: str,
    lang: str,
    user_input: str,
    reason: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    DB 결과가 없거나/실패했을 때, 환각을 최소화하는 fallback 답변을 생성한다.
    """
    if client is None:
        return None, "missing_key"
    system_prompt = build_fallback_system_prompt(lang, reason)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
    except Exception as exc:
        print(f"[DEBUG][Fallback] LLM call failed: {exc}")
        return None, "llm_error"
    content = response.choices[0].message.content if response.choices else ""
    return content, None


# ============================================================
# (개선) 대화 히스토리 축약
# ============================================================
def format_history(messages: List[Dict[str, Any]], limit: int = 6) -> str:
    """
    최근 대화 일부만 문자열로 요약해 SQL 생성 프롬프트에 제공한다.
    - 너무 많은 히스토리를 넣으면 SQL 생성이 흔들릴 수 있어 제한한다.
    """
    recent = messages[-limit:] if messages else []
    lines = []
    for message in recent:
        role = message.get("role", "user")
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "N/A"


# ============================================================
# SQL 정규화/검증/제한 (보안 로직 유지)
# ============================================================
def normalize_sql(raw_sql: str) -> str:
    """
    LLM이 반환한 SQL 텍스트를 최대한 'SELECT/WITH 단일문' 형태로 정리한다.
    """
    text = raw_sql.strip()
    if not text:
        return text

    lowered = text.lower()
    if re.fullmatch(r"no_sql[.!]?", lowered.strip()):
        return "NO_SQL"

    # 코드블록 제거(```sql ... ```)
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
            if text.lower().startswith("sql"):
                text = text[3:].strip()

    # SELECT/WITH 시작점 찾기
    match = re.search(r"\b(select|with)\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start() :].strip()

    # 세미콜론 이후 제거(다중문 방지)
    if ";" in text:
        text = text.split(";")[0].strip()

    if re.fullmatch(r"no_sql[.!]?", text.strip().lower()):
        return "NO_SQL"

    return text.strip()


def is_safe_sql(sql: str) -> bool:
    """
    안전한 단일 SELECT/WITH인지 검사한다.
    - 보안 관련 사항: 현재 로직 유지(금칙어, 단일문, SELECT/WITH만 허용)
    """
    lowered = sql.strip().lower()
    if not lowered:
        return False
    if lowered.startswith("no_sql"):
        return False
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return False

    banned = ["insert", "update", "delete", "drop", "alter", "create", "pragma", "attach", "detach"]
    for keyword in banned:
        if re.search(rf"\b{keyword}\b", lowered):
            return False

    # 세미콜론으로 여러 문장 분리 여부 확인
    statements = [stmt for stmt in lowered.split(";") if stmt.strip()]
    return len(statements) == 1


def enforce_limit(sql: str, limit: int = MAX_SQL_ROWS) -> str:
    """
    SELECT 결과가 과도하게 커지는 것을 막기 위해 LIMIT을 강제한다.
    """
    trimmed = sql.strip().rstrip(";")
    if " limit " in trimmed.lower():
        return trimmed
    return f"{trimmed} LIMIT {limit}"


def truncate_results(
    results: Optional[List[Dict[str, Any]]],
    limit: int = MAX_LLM_ROWS,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    LLM에 전달할 결과 행 수를 제한한다.
    - 너무 많은 결과를 LLM에 주면 비용/지연/품질이 악화될 수 있다.
    """
    rows = results or []
    if len(rows) > limit:
        return rows[:limit], True
    return rows, False


# ============================================================
# DB Query
# ============================================================
def run_sql_query(sql: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    SQLite에 SQL을 실행하고 Row list(dict) 형태로 반환한다.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        return results, None
    except sqlite3.Error as exc:
        return None, str(exc)
    finally:
        if conn:
            conn.close()


# ============================================================
# (개선) 스키마/예시 컨텍스트 생성
# ============================================================
def _tokenize(text: str) -> List[str]:
    """
    질문/히스토리에서 간단 키워드를 뽑기 위한 토큰화 함수.
    - 완전한 NLP가 아니라, 빠르고 안전한 룰 기반 방식
    - 알파벳/숫자/한글/언더스코어를 토큰으로 취급
    """
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z0-9가-힣_]+", text.lower())
    tokens = [t for t in tokens if len(t) >= 2]  # 너무 짧은 토큰 제거
    return tokens


def get_schema_map() -> Dict[str, List[str]]:
    """
    DB에서 테이블별 컬럼 목록을 읽어 {table: [columns...]} 형태로 반환.
    """
    if not os.path.exists(DB_PATH):
        return {}

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    schema_map: Dict[str, List[str]] = {}

    for table in TABLES:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        schema_map[table] = [col[1] for col in columns] if columns else []

    conn.close()
    return schema_map


def _score_schema_items(question: str, history_text: str, schema_map: Dict[str, List[str]]) -> Dict[str, float]:
    """
    질문/히스토리 토큰과 테이블/컬럼명 매칭 정도로 '관련도 점수'를 계산한다.
    - 스키마를 줄이기 위한 휴리스틱 스코어링
    """
    q_tokens = _tokenize(question)
    h_tokens = _tokenize(history_text)
    all_tokens = q_tokens + (h_tokens[:50])  # 히스토리 토큰을 무한정 쓰지 않도록 제한

    scores: Dict[str, float] = {}
    for table, cols in schema_map.items():
        table_l = table.lower()
        score = 0.0

        # 테이블명 매칭 가중치
        for tok in all_tokens:
            if tok in table_l:
                score += 3.0

        # 컬럼명 매칭 가중치
        for col in cols:
            col_l = col.lower()
            for tok in all_tokens:
                if tok in col_l:
                    score += 1.0

        # 이 앱의 도메인에서 자주 쓰일 법한 테이블에 약간 가산(너무 큰 가산은 피함)
        if table_l in ("employees", "departments"):
            score += 0.5

        scores[table] = score

    return scores


def build_schema_context_topk(
    question: str,
    history_text: str,
    topk_tables: int = SCHEMA_TOPK_TABLES,
    topk_columns: int = SCHEMA_TOPK_COLUMNS,
) -> str:
    """
    질문에 맞춰 관련 있어 보이는 테이블 Top-K만 골라 schema_context를 만든다.
    - 전체 스키마를 넣는 것보다 토큰/비용/정확도 측면에서 유리한 경우가 많다.
    """
    schema_map = get_schema_map()
    if not schema_map:
        return "Tables: N/A"

    scores = _score_schema_items(question, history_text, schema_map)

    # 점수 높은 순으로 정렬 후 Top-K 선택
    ranked_tables = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected_tables = [t for (t, _s) in ranked_tables[:topk_tables]]

    # 점수가 모두 0이면 기본 핵심 테이블을 포함
    if selected_tables and all(scores[t] <= 0.0 for t in selected_tables):
        selected_tables = ["company", "employees", "departments"][:topk_tables]

    lines = ["Tables:"]
    for table in selected_tables:
        cols = schema_map.get(table, [])
        cols_trimmed = cols[:topk_columns]
        lines.append(f"- {table}: {', '.join(cols_trimmed)}")

    lines.append("Relationships: employees.department = departments.name")
    lines.append("Notes: company has a single row.")
    return "\n".join(lines)


def get_db_schema_text() -> str:
    """
    전체 스키마 텍스트를 생성한다(디버그/표시용).
    - SQL 생성에는 Top-K 스키마를 쓰지만, 전체도 필요할 수 있어 유지
    """
    if not os.path.exists(DB_PATH):
        return ""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    lines = ["Tables:"]
    for table in TABLES:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        if columns:
            column_names = ", ".join(col[1] for col in columns)
            lines.append(f"- {table}: {column_names}")
    conn.close()
    lines.append("Relationships: employees.department = departments.name")
    lines.append("Notes: company has a single row.")
    return "\n".join(lines)


def get_db_examples_text() -> str:
    """
    DB의 샘플 행을 examples 텍스트로 생성한다.
    - "사실 데이터"가 아니라, LLM이 스키마 의미를 잡는 데 도움을 주는 용도
    """
    if not os.path.exists(DB_PATH):
        return "Example data: N/A"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    samples: Dict[str, List[Dict[str, Any]]] = {}
    queries = {
        "company": "SELECT name, industry, mission, location, founded_year, size FROM company LIMIT 1",
        "departments": "SELECT name, purpose, headcount FROM departments LIMIT 2",
        "employees": "SELECT name, role, department, email, login_id FROM employees LIMIT 2",
        "products": "SELECT name, category, pricing, status FROM products LIMIT 2",
        "policies": "SELECT title FROM policies LIMIT 2",
    }

    for table, sql in queries.items():
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        samples[table] = [dict(zip(columns, row)) for row in rows]

    conn.close()

    lines = ["Example data (sample rows):"]
    for table in TABLES:
        payload = json.dumps(samples.get(table, []), ensure_ascii=False)
        lines.append(f"- {table}: {payload}")
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def cached_schema_text() -> str:
    """
    스키마 텍스트 캐시.
    - 매 요청마다 PRAGMA를 호출하지 않도록 성능 개선
    """
    return get_db_schema_text()


@st.cache_data(show_spinner=False)
def cached_examples_text() -> str:
    """
    예시 데이터 텍스트 캐시.
    - 예시는 DB 샘플 몇 줄이므로 캐싱 효과가 좋다.
    """
    return get_db_examples_text()


def build_db_context() -> str:
    """
    기존 db_context는 스키마 요약을 담도록 유지.
    """
    if not os.path.exists(DB_PATH):
        return ""
    lines = [get_db_schema_text()]
    lines.append(f"Context updated at {datetime.utcnow().isoformat()} UTC")
    return "\n".join(lines)


# ============================================================
# (개선) SQL 생성: JSON 출력 강제 + 안전 추출
# ============================================================
def extract_sql_from_llm(content: str) -> Optional[str]:
    """
    LLM 응답에서 SQL을 안전하게 추출한다.
    - 우선 JSON 파싱으로 {"sql": "..."}를 기대
    - 실패하면 기존 normalize_sql 로직으로 폴백
    """
    data = extract_json(content)
    if isinstance(data, dict) and "sql" in data:
        sql_val = str(data.get("sql") or "").strip()
        return normalize_sql(sql_val)
    return normalize_sql(content)


def generate_sql_query(
    client,
    model: str,
    lang: str,
    question: str,
    schema_context: str,
    history_text: str,
    allow_no_sql: bool = True,
    examples_text: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    질문을 SQLite SELECT로 변환하는 LLM 호출.
    - JSON 형태로만 출력하도록 강제하여 파싱 안정성을 높임
    - allow_no_sql=True면 답할 수 없을 때 {"sql":"NO_SQL"} 허용
    """
    if client is None:
        return None, "missing_key"

    language = "Korean" if lang == "ko" else "English"

    system_prompt = (
        "You are a senior data analyst writing a single SQLite SELECT query. "
        "Use only the provided schema. "
        "Return ONLY valid JSON with the key 'sql'. "
        "Do not include markdown, comments, or extra keys."
    )
    if allow_no_sql:
        system_prompt += " If the question cannot be answered from the schema, set sql to 'NO_SQL'."
    else:
        system_prompt += " Always return a valid SQL SELECT (or WITH) query in sql."

    examples_block = ""
    if ENABLE_EXAMPLES_IN_PROMPT and examples_text:
        examples_block = f"\n\nExamples (illustrative only):\n{examples_text}\n"

    user_prompt = (
        f"Schema:\n{schema_context}\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"User question:\n{question}\n"
        f"{examples_block}\n"
        "Rules:\n"
        "- Only one SQLite SELECT statement (WITH is allowed).\n"
        "- Do not use INSERT/UPDATE/DELETE/DDL/PRAGMA.\n"
        "- Avoid selecting password unless explicitly requested.\n"
        "- Keep it minimal: select only necessary columns.\n"
        f"- Consider {language} context for string filters.\n\n"
        "Output format:\n"
        '{"sql":"<your SQLite query or NO_SQL>"}'
    )

    # 디버그 출력: 문제 발생 시 원인 파악 용도
    print(f"[DEBUG][SQLGen] allow_no_sql={allow_no_sql}, lang={lang}, model={model}")
    print(f"[DEBUG][SQLGen] schema_len={len(schema_context)}, history_len={len(history_text)}, examples_len={len(examples_text or '')}")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
    except Exception as exc:
        print(f"[DEBUG][SQLGen] LLM call failed: {exc}")
        return None, "llm_error"

    content = response.choices[0].message.content if response.choices else ""
    sql = extract_sql_from_llm(content)

    print(f"[DEBUG][SQLGen] raw_content={content[:300]}")
    print(f"[DEBUG][SQLGen] parsed_sql={sql}")

    return sql, None


# ============================================================
# Final answer / Direct answer
# ============================================================
def generate_final_answer(
    client,
    model: str,
    lang: str,
    system_prompt_override: Optional[str],
    user_input: str,
    sql: Optional[str],
    results: List[Dict[str, Any]],
    truncated: bool,
    sql_error: Optional[str],
) -> str:
    """
    SQL 실행 결과를 LLM에 주고 자연어 최종 답변을 생성한다.
    - 보안은 유지하되, "SQL 결과가 source of truth"라는 시스템 프롬프트를 활용한다.
    """
    system_prompt = build_chat_system_prompt(lang, system_prompt_override)

    error_message = None
    if sql_error == "no_sql":
        error_message = "No suitable SQL could be generated for the question."
    elif sql_error == "unsafe_or_empty_sql":
        error_message = "The generated SQL was unsafe or empty."
    elif sql_error:
        error_message = sql_error

    results_payload = json.dumps(results, ensure_ascii=False)
    tool_context = [
        f"SQL: {sql or 'N/A'}",
        f"SQL error: {error_message or 'None'}",
        f"Results (JSON, truncated={truncated}): {results_payload}",
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        # DB 결과(도구 출력물)를 system 메시지로 제공
        {"role": "system", "content": "\n".join(tool_context)},
    ]
    messages.extend(st.session_state.get("messages", []))
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
    )
    return response.choices[0].message.content if response.choices else ""


def generate_direct_answer(
    client,
    model: str,
    lang: str,
    system_prompt_override: Optional[str],
    user_input: str,
) -> str:
    """
    DB를 치지 않고 일반 답변을 생성하는 경로.
    - 현재 앱에서는 주로 NO_SQL, unsafe, smalltalk 등에 사용
    """
    system_prompt = build_chat_system_prompt(lang, system_prompt_override)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(st.session_state.get("messages", []))
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
    )
    return response.choices[0].message.content if response.choices else ""


# ============================================================
# (개선) 스몰톡 감지 / SQL 재시도 1회 로직
# ============================================================
def should_treat_as_smalltalk(user_input: str) -> bool:
    """
    스몰톡/사용법 질문 등 DB를 굳이 치지 않아도 되는 입력을 감지한다.
    - 비용/오답 감소 목적
    """
    if not user_input:
        return True
    text = user_input.strip().lower()
    for pat in SMALLTALK_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False


def generate_sql_retry(
    client,
    model: str,
    lang: str,
    question: str,
    schema_context: str,
    history_text: str,
    prev_sql: str,
    exec_error: Optional[str],
    had_no_rows: bool,
    examples_text: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    실행 실패(에러) 또는 0 rows일 때, 1회만 SQL을 수정해 재시도하도록 유도한다.
    - 무한 재시도는 금지(품질/비용/예측불가를 줄이기 위함)
    """
    if client is None:
        return None, "missing_key"

    reason_lines = [f"Previous SQL:\n{prev_sql}"]
    if exec_error:
        reason_lines.append(f"Execution error:\n{exec_error}")
    if had_no_rows:
        reason_lines.append("The query returned 0 rows. Adjust filters/joins/conditions.")

    retry_hint = "\n\n".join(reason_lines)

    retry_question = (
        f"{question}\n\n"
        "Retry instructions:\n"
        "- Fix the SQL to answer the question using the schema.\n"
        "- Prefer broader matching if filters were too strict.\n"
        "- Keep it a single SELECT/WITH.\n\n"
        f"{retry_hint}"
    )

    print("[DEBUG][SQLRetry] attempting one retry based on failure context")

    return generate_sql_query(
        client=client,
        model=model,
        lang=lang,
        question=retry_question,
        schema_context=schema_context,
        history_text=history_text,
        allow_no_sql=False,  # 재시도에서는 NO_SQL을 허용하지 않고 최대한 쿼리를 만들도록 유도
        examples_text=examples_text,
    )


# ============================================================
# Agent Draft / Company generation (기존 유지)
# ============================================================
def generate_agent_draft(
    client,
    model: str,
    lang: str,
    intent: str,
    db_context: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    사용자 의도를 바탕으로 '사용자 메시지 초안'만 생성한다(답변 생성 금지).
    """
    if client is None:
        return None, "missing_key"
    language = "Korean" if lang == "ko" else "English"
    system_prompt = (
        "You draft a user's message to a company assistant. "
        "Output only the message text. No markdown, no quotes."
    )
    user_prompt = (
        f"User intent: {intent.strip()}\n\n"
        f"Language: {language}\n\n"
        "Use the database context only if it helps refine the user's question. "
        "Do not answer the question; only draft the user's message.\n\n"
        f"Database context:\n{db_context or 'N/A'}"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
        )
    except Exception as exc:
        print(f"[DEBUG][AgentDraft] LLM call failed: {exc}")
        return None, "llm_error"
    content = response.choices[0].message.content if response.choices else ""
    return content.strip(), None


def build_company_prompts(
    company_name: str,
    industry: str,
    location: str,
    size: int,
    tone: str,
    lang: str,
    db_size: str,
    system_prompt_override: Optional[str] = None,
) -> Tuple[str, str]:
    default_system_prompt = (
        "You are a business analyst generating structured company data. "
        "Return ONLY valid JSON that matches the schema. No markdown."
    )
    system_prompt = (
        system_prompt_override.strip() if system_prompt_override and system_prompt_override.strip() else default_system_prompt
    )
    constraints = get_size_constraints(db_size)
    dept_min, dept_max = constraints["departments"]
    emp_min, emp_max = constraints["employees"]
    prod_min, prod_max = constraints["products"]
    pol_min, pol_max = constraints["policies"]
    user_prompt = f"""
Create a detailed virtual company profile based on the hints below.
Language for text fields: { "Korean" if lang == "ko" else "English" }.
Database size target: {db_size}.

Hints:
- company_name: {company_name or "Invent a strong brand name"}
- industry: {industry}
- location: {location}
- size: {size}
- tone: {tone}

Schema:
{{
  "company": {{
    "name": "...",
    "industry": "...",
    "mission": "...",
    "location": "...",
    "founded_year": 2010,
    "size": {size},
    "description": "..."
  }},
  "departments": [
    {{"name": "...", "purpose": "...", "headcount": 10}}
  ],
  "employees": [
    {{
      "name": "...",
      "role": "...",
      "department": "...",
      "bio": "...",
      "email": "...",
      "login_id": "...",
      "password": "..."
    }}
  ],
  "products": [
    {{"name": "...", "category": "...", "description": "...", "pricing": "...", "status": "..."}}
  ],
  "policies": [
    {{"title": "...", "content": "..."}}
  ]
}}

Constraints:
- departments: {dept_min}-{dept_max} items
- employees: {emp_min}-{emp_max} items (match the company size roughly)
- products: {prod_min}-{prod_max} items
- policies: {pol_min}-{pol_max} items
- employees must include login_id and password
- include exactly one admin account with login_id "ADMIN" and password "admin123"
"""
    return system_prompt, user_prompt


def generate_company_profile(
    client,
    model: str,
    company_name: str,
    industry: str,
    location: str,
    size: int,
    tone: str,
    lang: str,
    db_size: str,
    system_prompt_override: Optional[str] = None,
) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
    if client is None:
        return None, "missing_key", None

    system_prompt, user_prompt = build_company_prompts(
        company_name,
        industry,
        location,
        size,
        tone,
        lang,
        db_size,
        system_prompt_override,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
    except Exception as exc:
        print(f"[DEBUG][CompanyGen] LLM call failed: {exc}")
        return None, "llm_error", None

    content = response.choices[0].message.content if response.choices else ""
    data = extract_json(content)
    if not data:
        return None, "parse_error", content
    return data, None, content


# ============================================================
# DB 생성/현황(기존 유지)
# ============================================================
def ensure_db_dir() -> None:
    if not os.path.isdir(DB_DIR):
        os.makedirs(DB_DIR, exist_ok=True)


def create_database(data: dict) -> None:
    """
    LLM이 생성한 company JSON을 SQLite DB로 생성한다.
    """
    ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(
        """
        DROP TABLE IF EXISTS company;
        DROP TABLE IF EXISTS departments;
        DROP TABLE IF EXISTS employees;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS policies;

        CREATE TABLE company (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            industry TEXT,
            mission TEXT,
            location TEXT,
            founded_year INTEGER,
            size INTEGER,
            description TEXT
        );

        CREATE TABLE departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            purpose TEXT,
            headcount INTEGER
        );

        CREATE TABLE employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            role TEXT,
            department TEXT,
            bio TEXT,
            email TEXT,
            login_id TEXT,
            password TEXT
        );

        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            category TEXT,
            description TEXT,
            pricing TEXT,
            status TEXT
        );

        CREATE TABLE policies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT
        );
        """
    )

    company = data.get("company", {})
    cursor.execute(
        """
        INSERT INTO company (name, industry, mission, location, founded_year, size, description)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            company.get("name"),
            company.get("industry"),
            company.get("mission"),
            company.get("location"),
            company.get("founded_year"),
            company.get("size"),
            company.get("description"),
        ),
    )

    for department in data.get("departments", []):
        cursor.execute(
            """
            INSERT INTO departments (name, purpose, headcount)
            VALUES (?, ?, ?)
            """,
            (
                department.get("name"),
                department.get("purpose"),
                department.get("headcount"),
            ),
        )

    for index, employee in enumerate(data.get("employees", [])):
        login_id = employee.get("login_id")
        password = employee.get("password") or employee.get("login_pw")
        if not login_id:
            login_id = build_login_id(employee.get("name"), index)
        if not password:
            password = build_password(index)
        cursor.execute(
            """
            INSERT INTO employees (name, role, department, bio, email, login_id, password)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                employee.get("name"),
                employee.get("role"),
                employee.get("department"),
                employee.get("bio"),
                employee.get("email"),
                login_id,
                password,
            ),
        )

    for product in data.get("products", []):
        cursor.execute(
            """
            INSERT INTO products (name, category, description, pricing, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                product.get("name"),
                product.get("category"),
                product.get("description"),
                product.get("pricing"),
                product.get("status"),
            ),
        )

    for policy in data.get("policies", []):
        cursor.execute(
            """
            INSERT INTO policies (title, content)
            VALUES (?, ?)
            """,
            (
                policy.get("title"),
                policy.get("content"),
            ),
        )

    # 관리자 계정 보장
    cursor.execute("SELECT id FROM employees WHERE login_id = ?", ("ADMIN",))
    admin_row = cursor.fetchone()
    if admin_row:
        cursor.execute(
            """
            UPDATE employees
            SET password = ?, role = ?
            WHERE login_id = ?
            """,
            ("admin123", "Administrator", "ADMIN"),
        )
    else:
        cursor.execute(
            """
            INSERT INTO employees (name, role, department, bio, email, login_id, password)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "Admin Account",
                "Administrator",
                "Executive",
                "System administrator account for internal management.",
                "admin@company.local",
                "ADMIN",
                "admin123",
            ),
        )

    conn.commit()
    conn.close()


def get_db_overview() -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    DB 상태 탭에서 보여줄 overview(테이블별 row count + 샘플)를 만든다.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    counts: List[Dict[str, Any]] = []
    samples: Dict[str, List[Dict[str, Any]]] = {}

    for table in TABLES:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        counts.append({"table": table, "rows": count})

        cursor.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        samples[table] = [dict(zip(columns, row)) for row in rows]

    conn.close()
    return counts, samples


# ============================================================
# (개선) run_chat: Top-K 스키마 + JSON SQL + 재시도 1회
# ============================================================
def run_chat(
    client,
    model: str,
    lang: str,
    user_input: str,
    system_prompt_override: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    사용자 입력을 받아:
    1) 스몰톡이면 즉시 응답
    2) SQL 생성(Top-K schema + examples optional)
    3) 안전성 체크 후 SQL 실행
    4) 에러/0 rows면 1회 재시도
    5) 결과를 바탕으로 최종 답변 생성
    """
    # 0) 스몰톡/사용법 질문이면 DB/SQL 생성을 생략
    if should_treat_as_smalltalk(user_input):
        print("[DEBUG][Chat] smalltalk detected -> direct response")
        reply = get_text(lang, "small_talk_response")
        return reply, {
            "sql": None,
            "executed_sql": None,
            "results": [],
            "truncated": False,
            "sql_error": "no_sql",
        }

    # 1) 히스토리 문자열(최근 일부)
    history_text = format_history(st.session_state.get("messages", []))

    # 2) 스키마 컨텍스트는 Top-K로 축소
    #    - 전체 스키마는 cached_schema_text로 유지 가능하지만,
    #      실제 SQL 생성에는 build_schema_context_topk 결과를 사용
    full_schema = cached_schema_text() if os.path.exists(DB_PATH) else ""
    schema_context = build_schema_context_topk(
        question=user_input,
        history_text=history_text,
        topk_tables=SCHEMA_TOPK_TABLES,
        topk_columns=SCHEMA_TOPK_COLUMNS,
    ) or (full_schema or st.session_state.get("db_context", ""))

    # 3) examples 컨텍스트(선택)
    examples_text = cached_examples_text() if (ENABLE_EXAMPLES_IN_PROMPT and os.path.exists(DB_PATH)) else None

    print(f"[DEBUG][Chat] schema_context_len={len(schema_context)}")

    # 4) 1차 SQL 생성
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

    # 5) SQL 실행(보안 로직 유지)
    if sql_error:
        execution_error = sql_error
    elif sql and sql.strip().lower().startswith("no_sql"):
        fallback, _ = generate_fallback_answer(client, model, lang, user_input, "no_sql")
        reply = fallback or get_text(lang, "no_sql_response")
        return reply, {
            "sql": sql,
            "executed_sql": None,
            "results": [],
            "truncated": False,
            "sql_error": "no_sql",
        }
    elif not sql or not is_safe_sql(sql):
        fallback, _ = generate_fallback_answer(client, model, lang, user_input, "no_sql")
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

    # 6) 결과 truncate
    limited_results, truncated = truncate_results(results, MAX_LLM_ROWS)

    # 7) (개선) 에러/0 rows면 SQL 재시도 1회
    did_retry = False

    # 7-1) 실행 에러 재시도
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
            sql = retry_sql  # 디버그/표시 목적

    # 7-2) 0 rows 재시도(에러가 없을 때만)
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

    # 8) 에러면 fallback
    if execution_error:
        fallback, _ = generate_fallback_answer(client, model, lang, user_input, "sql_error")
        reply = fallback or get_text(lang, "sql_error_response")
        return reply, {
            "sql": sql,
            "executed_sql": executed_sql,
            "results": limited_results,
            "truncated": truncated,
            "sql_error": execution_error,
        }

    # 9) 결과가 없으면 fallback
    if not limited_results:
        fallback, _ = generate_fallback_answer(client, model, lang, user_input, "no_data")
        reply = fallback or get_text(lang, "no_data_response")
        return reply, {
            "sql": sql,
            "executed_sql": executed_sql,
            "results": [],
            "truncated": truncated,
            "sql_error": None,
        }

    # 10) 최종 답변 생성
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
    )
    return reply, {
        "sql": sql,
        "executed_sql": executed_sql,
        "results": limited_results,
        "truncated": truncated,
        "sql_error": execution_error,
    }


# ============================================================
# Streamlit UI
# ============================================================
def main() -> None:
    init_state()
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(t("title"))
    st.caption(t("subtitle"))

    # DB가 존재하고 context가 비어있으면 스키마 context를 채움
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

        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-3.5-turbo"]
        model = st.selectbox(t("model"), model_options, help=t("model_help"))

        # 사용자 커스텀 프롬프트 입력 UI는 유지(현재 build_chat_system_prompt는 override를 무시하는 정책)
        st.text_area(
            t("chat_system_prompt_label"),
            key="chat_system_prompt",
            help=t("chat_system_prompt_help"),
            height=140,
        )

        base_url = st.text_input(t("base_url"), help=t("base_url_help"))

    tab_setup, tab_db, tab_chat = st.tabs([t("tab_setup"), t("tab_db"), t("tab_chat")])

    # --------------------------
    # Setup Tab
    # --------------------------
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
                        create_database(data)

                        # (개선) DB 재생성 시 스키마/예시 캐시 무효화
                        cached_schema_text.clear()
                        cached_examples_text.clear()

                        st.session_state["company_data"] = data
                        st.session_state["db_context"] = build_db_context()
                        st.success(t("db_created"))

    # --------------------------
    # DB Status Tab
    # --------------------------
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

    # --------------------------
    # Chat Tab
    # --------------------------
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

        def render_sql_debug(info: Dict[str, Any]) -> None:
            """
            각 턴에서 실행된 SQL/결과/에러를 UI에 표시한다.
            - 디버깅을 매우 쉽게 해준다.
            """
            st.markdown(f"**{t('sql_debug_title')}**")
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
            """
            사용자 메시지를 추가하고, run_chat을 호출해 응답을 생성한 뒤 UI에 표시한다.
            """
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
                except Exception as exc:
                    print(f"[DEBUG][Chat] run_chat crashed: {exc}")
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

        # 기존 메시지 렌더
        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and message.get("sql_info"):
                    render_sql_debug(message["sql_info"])

        # Agent draft 기능(기존 유지)
        agent_enabled = st.checkbox(t("agent_toggle"), value=False, key="agent_enabled")
        if agent_enabled:
            st.markdown(f"#### {t('agent_section')}")
            agent_model = st.selectbox(
                t("agent_model"),
                options=model_options,
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
