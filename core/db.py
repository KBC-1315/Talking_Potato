import json
import os
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from .config import DB_DIR, DB_PATH, TABLES


def ensure_db_dir() -> None:
    if not os.path.isdir(DB_DIR):
        os.makedirs(DB_DIR, exist_ok=True)


def init_db_schema() -> None:
    ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS company (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            industry TEXT,
            mission TEXT,
            location TEXT,
            founded_year INTEGER,
            size INTEGER,
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            purpose TEXT,
            headcount INTEGER
        );

        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            role TEXT,
            department TEXT,
            bio TEXT,
            email TEXT,
            login_id TEXT,
            password TEXT
        );

        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            category TEXT,
            description TEXT,
            pricing TEXT,
            status TEXT
        );

        CREATE TABLE IF NOT EXISTS policies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT
        );

        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            authors TEXT,
            year INTEGER,
            venue TEXT,
            abstract TEXT,
            keywords TEXT,
            doi TEXT,
            url TEXT
        );
        """
    )
    conn.commit()
    conn.close()


def ensure_schema_if_exists() -> None:
    if os.path.exists(DB_PATH):
        init_db_schema()


def build_login_id(name: Optional[str], index: int) -> str:
    if name:
        cleaned = "".join(ch for ch in name.lower() if ch.isalnum())
        if cleaned:
            return cleaned[:12]
    return f"user{index + 1}"


def build_password(index: int) -> str:
    return f"pw{index + 1}!"


def replace_company_data(data: dict) -> None:
    init_db_schema()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM company")
    cursor.execute("DELETE FROM departments")
    cursor.execute("DELETE FROM employees")
    cursor.execute("DELETE FROM products")
    cursor.execute("DELETE FROM policies")

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


def replace_papers_data(data: dict) -> None:
    init_db_schema()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM papers")

    for paper in data.get("papers", []):
        cursor.execute(
            """
            INSERT INTO papers (title, authors, year, venue, abstract, keywords, doi, url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                paper.get("title"),
                paper.get("authors"),
                paper.get("year"),
                paper.get("venue"),
                paper.get("abstract"),
                paper.get("keywords"),
                paper.get("doi"),
                paper.get("url"),
            ),
        )

    conn.commit()
    conn.close()


def get_db_overview() -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    ensure_schema_if_exists()
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


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z0-9가-힣_]+", text.lower())
    tokens = [t for t in tokens if len(t) >= 2]
    return tokens


def get_schema_map() -> Dict[str, List[str]]:
    if not os.path.exists(DB_PATH):
        return {}
    ensure_schema_if_exists()

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
    q_tokens = _tokenize(question)
    h_tokens = _tokenize(history_text)
    all_tokens = q_tokens + (h_tokens[:50])
    has_paper_intent = any(tok in {"paper", "papers", "논문", "연구"} for tok in all_tokens)

    scores: Dict[str, float] = {}
    for table, cols in schema_map.items():
        table_l = table.lower()
        score = 0.0

        for tok in all_tokens:
            if tok in table_l:
                score += 3.0

        for col in cols:
            col_l = col.lower()
            for tok in all_tokens:
                if tok in col_l:
                    score += 1.0

        if table_l in ("employees", "departments", "papers"):
            score += 0.5
        if has_paper_intent and table_l == "papers":
            score += 2.0

        scores[table] = score

    return scores


def build_schema_context_topk(
    question: str,
    history_text: str,
    topk_tables: int,
    topk_columns: int,
) -> str:
    schema_map = get_schema_map()
    if not schema_map:
        return "Tables: N/A"

    scores = _score_schema_items(question, history_text, schema_map)
    ranked_tables = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected_tables = [t for (t, _s) in ranked_tables[:topk_tables]]

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
    if not os.path.exists(DB_PATH):
        return ""
    ensure_schema_if_exists()
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
    if not os.path.exists(DB_PATH):
        return "Example data: N/A"
    ensure_schema_if_exists()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    samples: Dict[str, List[Dict[str, Any]]] = {}
    queries = {
        "company": "SELECT name, industry, mission, location, founded_year, size FROM company LIMIT 1",
        "departments": "SELECT name, purpose, headcount FROM departments LIMIT 2",
        "employees": "SELECT name, role, department, email, login_id FROM employees LIMIT 2",
        "products": "SELECT name, category, pricing, status FROM products LIMIT 2",
        "policies": "SELECT title FROM policies LIMIT 2",
        "papers": "SELECT title, authors, year, venue FROM papers LIMIT 2",
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
    return get_db_schema_text()


@st.cache_data(show_spinner=False)
def cached_examples_text() -> str:
    return get_db_examples_text()


def build_db_context() -> str:
    if not os.path.exists(DB_PATH):
        return ""
    ensure_schema_if_exists()
    lines = [get_db_schema_text()]
    lines.append(f"Context updated at {datetime.utcnow().isoformat()} UTC")
    return "\n".join(lines)


def run_sql_query(sql: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    ensure_schema_if_exists()
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
