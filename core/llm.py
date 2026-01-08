import json
from typing import Any, Dict, List, Optional, Tuple

from .config import ENABLE_EXAMPLES_IN_PROMPT
from .db import get_db_examples_text
from .sql_utils import normalize_sql

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


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


def extract_sql_from_llm(content: str) -> Optional[str]:
    data = extract_json(content)
    if isinstance(data, dict) and "sql" in data:
        sql_val = str(data.get("sql") or "").strip()
        return normalize_sql(sql_val)
    return normalize_sql(content)


def build_chat_system_prompt(lang: str, override: Optional[str]) -> str:
    base_guardrails = (
        "You are a helpful research assistant for a company + papers database. "
        "Use only SQL results as the source of truth. "
        "If a detail is not in SQL results, say you don't have it. "
        "When results include academic papers, write in an academic tone and cite as [Title, Year]. "
        "Do not invent citations. Examples are illustrative only."
    )
    language = "Korean" if lang == "ko" else "English"
    examples = get_db_examples_text()
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
        if "{examples}" in prompt:
            prompt = prompt.replace("{examples}", examples)
        else:
            prompt = f"{prompt}\n\n{examples}"
        return prompt
    return f"{base_guardrails} Answer in {language}."


def build_fallback_system_prompt(lang: str, reason: str) -> str:
    language = "Korean" if lang == "ko" else "English"
    reason_note = {
        "no_sql_needed": "This question does not require a database query.",
        "no_sql": "A suitable SQL query could not be generated.",
        "no_data": "SQL ran but returned no rows.",
        "sql_error": "SQL execution failed.",
    }.get(reason, "No database results are available.")
    return (
        "You are the assistant for a SQL + LLM company chatbot app. "
        "You do NOT have database results, and you must NOT invent company facts. "
        "If the user asks about app capabilities or how to use the app, answer clearly. "
        "Otherwise, explain that the data is unavailable and ask them to rephrase with a "
        "company- or research-related question that can be answered from the database.\n\n"
        f"Reason: {reason_note}\n"
        f"Answer in {language}."
    )


def generate_fallback_answer(
    client,
    model: str,
    lang: str,
    user_input: str,
    reason: str,
    history_messages: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    if client is None:
        return None, "missing_key"
    system_prompt = build_fallback_system_prompt(lang, reason)
    messages = [{"role": "system", "content": system_prompt}]
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": user_input})
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
    except Exception:
        return None, "llm_error"
    content = response.choices[0].message.content if response.choices else ""
    return content, None


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
    history_messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
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
        {"role": "system", "content": "\n".join(tool_context)},
    ]
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
    )
    return response.choices[0].message.content if response.choices else ""


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
        system_prompt_override.strip()
        if system_prompt_override and system_prompt_override.strip()
        else default_system_prompt
    )
    size_constraints = {
        "small": {"departments": (3, 5), "employees": (8, 12), "products": (2, 4), "policies": (3, 4)},
        "medium": {"departments": (4, 8), "employees": (12, 20), "products": (3, 6), "policies": (4, 6)},
        "large": {"departments": (6, 10), "employees": (20, 35), "products": (5, 8), "policies": (6, 8)},
    }
    constraints = size_constraints.get(db_size, size_constraints["medium"])
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


def build_papers_prompts(
    topic: str,
    start_year: int,
    end_year: int,
    paper_count: int,
    lang: str,
) -> Tuple[str, str]:
    system_prompt = (
        "You are a research analyst generating structured metadata for academic papers. "
        "Return ONLY valid JSON that matches the schema. No markdown."
    )
    user_prompt = f"""
Create a list of recent academic papers for the topic below.
Language for abstract: { "Korean" if lang == "ko" else "English" }.

Topic: {topic}
Years: {start_year} to {end_year}
Count: {paper_count}

Schema:
{{
  "papers": [
    {{
      "title": "...",
      "authors": "...",
      "year": {end_year},
      "venue": "...",
      "abstract": "...",
      "keywords": "...",
      "doi": "...",
      "url": "..."
    }}
  ]
}}

Constraints:
- years must fall within the given range
- papers should look realistic for recent academic work
- keep authors as a comma-separated string
- keywords as a comma-separated string
"""
    return system_prompt, user_prompt


def generate_papers_profile(
    client,
    model: str,
    topic: str,
    start_year: int,
    end_year: int,
    paper_count: int,
    lang: str,
) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
    if client is None:
        return None, "missing_key", None
    system_prompt, user_prompt = build_papers_prompts(
        topic,
        start_year,
        end_year,
        paper_count,
        lang,
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
        return None, "llm_error", None

    content = response.choices[0].message.content if response.choices else ""
    data = extract_json(content)
    if not data:
        return None, "parse_error", content
    return data, None, content


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

    print(f"[DEBUG][SQLGen] allow_no_sql={allow_no_sql}, lang={lang}, model={model}")
    print(
        f"[DEBUG][SQLGen] schema_len={len(schema_context)}, history_len={len(history_text)}, "
        f"examples_len={len(examples_text or '')}"
    )

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
        allow_no_sql=False,
        examples_text=examples_text,
    )


def should_use_sql(
    client,
    model: str,
    lang: str,
    question: str,
    schema_context: str,
) -> Tuple[Optional[bool], Optional[str]]:
    if client is None:
        return None, "missing_key"

    language = "Korean" if lang == "ko" else "English"
    system_prompt = "You decide if a database query is required. Reply ONLY JSON."
    user_prompt = (
        f"Schema:\n{schema_context}\n\n"
        f"User question:\n{question}\n\n"
        "Rules:\n"
        "- Set use_sql=true if the question needs database facts (company or papers).\n"
        "- Set use_sql=false if it is about app usage, capabilities, or general conversation.\n"
        f"Answer in {language} but output only JSON.\n"
        '{"use_sql": true|false}'
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
    except Exception as exc:
        print(f"[DEBUG][SQLNeed] LLM call failed: {exc}")
        return None, "llm_error"

    content = response.choices[0].message.content if response.choices else ""
    data = extract_json(content)
    if isinstance(data, dict) and "use_sql" in data:
        return bool(data.get("use_sql")), None

    normalized = content.strip().lower()
    if "true" in normalized or "yes" in normalized:
        return True, None
    if "false" in normalized or "no" in normalized:
        return False, None
    return None, "parse_error"
