import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


APP_TITLE = "SQL + LLM Company Chatbot"
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "company.db")
TABLES = ["company", "departments", "employees", "products", "policies"]
DB_SIZE_VALUES = ["small", "medium", "large"]


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
        "chat_system_prompt_help": "Optional. You can use {guardrails} and {language} placeholders.",
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
        "chat_system_prompt_help": "선택 사항입니다. {guardrails} 및 {language} 치환자를 사용할 수 있습니다.",
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


def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return TEXT.get(lang, TEXT["en"]).get(key, key)


def toggle_language() -> None:
    current = st.session_state.get("lang", "en")
    st.session_state["lang"] = "ko" if current == "en" else "en"


def init_state() -> None:
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


def get_openai_client(api_key: str, base_url: Optional[str]):
    if OpenAI is None:
        return None, "missing_sdk"
    if not api_key:
        return None, "missing_key"
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url), None
    return OpenAI(api_key=api_key), None


def extract_json(text: str) -> dict | None:
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


def build_chat_system_prompt(lang: str, override: Optional[str]) -> str:
    base_guardrails = (
        "You are a helpful company assistant. Use only the database context provided. "
        "If a detail is not in the database, say you don't have it."
    )
    language = "Korean" if lang == "ko" else "English"
    if override and override.strip():
        prompt = override.strip()
        if "{guardrails}" in prompt:
            prompt = prompt.replace("{guardrails}", base_guardrails)
        else:
            prompt = f"{base_guardrails}\n\n{prompt}"
        if "{language}" in prompt:
            prompt = prompt.replace("{language}", language)
        else:
            prompt = f"{prompt}\n\nAnswer in {language}."
        return prompt
    return f"{base_guardrails} Answer in {language}."


def generate_agent_draft(
    client,
    model: str,
    lang: str,
    intent: str,
    db_context: str,
) -> Tuple[Optional[str], Optional[str]]:
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
    except Exception:
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
    except Exception:
        return None, "llm_error", None

    content = response.choices[0].message.content if response.choices else ""
    data = extract_json(content)
    if not data:
        return None, "parse_error", content
    return data, None, content


def ensure_db_dir() -> None:
    if not os.path.isdir(DB_DIR):
        os.makedirs(DB_DIR, exist_ok=True)


def create_database(data: dict) -> None:
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
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    counts: List[Dict[str, Any]] = []
    samples: Dict[str, List[Dict[str, Any]]] = {}

    for table in TABLES:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        counts.append({"table": table, "rows": count})

        cursor.execute(f"SELECT * FROM {table} LIMIT 5")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        samples[table] = [dict(zip(columns, row)) for row in rows]

    conn.close()
    return counts, samples


def build_db_context() -> str:
    if not os.path.exists(DB_PATH):
        return ""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    def query(sql: str):
        cursor.execute(sql)
        return cursor.fetchall()

    company = query(
        "SELECT name, industry, mission, location, founded_year, size, description FROM company LIMIT 1"
    )
    departments = query("SELECT name, purpose, headcount FROM departments")
    employees = query("SELECT name, role, department, bio, email, login_id, password FROM employees")
    products = query("SELECT name, category, description, pricing, status FROM products")
    policies = query("SELECT title, content FROM policies")

    conn.close()

    lines = []
    if company:
        name, industry, mission, location, founded_year, size, description = company[0]
        lines.append(
            f"Company: {name} | Industry: {industry} | Mission: {mission} | Location: {location} | "
            f"Founded: {founded_year} | Size: {size} | Description: {description}"
        )
    if departments:
        lines.append("Departments:")
        for name, purpose, headcount in departments:
            lines.append(f"- {name} (Headcount: {headcount}) Purpose: {purpose}")
    if employees:
        lines.append("Employees:")
        for name, role, department, bio, email, login_id, password in employees:
            lines.append(
                f"- {name} | {role} | {department} | {email} | {login_id} / {password} | {bio}"
            )
    if products:
        lines.append("Products:")
        for name, category, description, pricing, status in products:
            lines.append(f"- {name} ({category}) {description} | Pricing: {pricing} | Status: {status}")
    if policies:
        lines.append("Policies:")
        for title, content in policies:
            lines.append(f"- {title}: {content}")

    lines.append(f"Context updated at {datetime.utcnow().isoformat()} UTC")
    return "\n".join(lines)


def run_chat(
    client,
    model: str,
    lang: str,
    user_input: str,
    system_prompt_override: Optional[str] = None,
) -> str:
    system_prompt = build_chat_system_prompt(lang, system_prompt_override)
    messages = [{"role": "system", "content": system_prompt}]

    db_context = st.session_state.get("db_context", "")
    if db_context:
        messages.append(
            {
                "role": "system",
                "content": f"Database context:\n{db_context}",
            }
        )
    messages.extend(st.session_state.get("messages", []))
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
    )
    return response.choices[0].message.content if response.choices else ""


def main() -> None:
    init_state()
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
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-3.5-turbo"]
        model = st.selectbox(t("model"), model_options, help=t("model_help"))
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
                        create_database(data)
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
                    st.dataframe(rows, use_container_width=True)
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

        def render_chat_turn(message_text: str) -> None:
            st.session_state["messages"].append({"role": "user", "content": message_text})
            with st.chat_message("user"):
                st.write(message_text)

            client, error_key = get_openai_client(api_key, base_url if base_url else None)
            if error_key:
                reply = t(error_key)
            else:
                try:
                    reply = run_chat(
                        client,
                        model,
                        st.session_state["lang"],
                        message_text,
                        st.session_state.get("chat_system_prompt"),
                    )
                except Exception:
                    reply = t("llm_error")

            st.session_state["messages"].append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.write(reply)

        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.write(message["content"])

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
