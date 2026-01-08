import re
from typing import Any, Dict, List, Optional, Tuple

from .config import MAX_LLM_ROWS, MAX_SQL_ROWS


def normalize_sql(raw_sql: str) -> str:
    text = raw_sql.strip()
    if not text:
        return text
    lowered = text.lower()
    if re.fullmatch(r"no_sql[.!]?", lowered.strip()):
        return "NO_SQL"
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
            if text.lower().startswith("sql"):
                text = text[3:].strip()
    match = re.search(r"\b(select|with)\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start() :].strip()
    if ";" in text:
        text = text.split(";")[0].strip()
    if re.fullmatch(r"no_sql[.!]?", text.strip().lower()):
        return "NO_SQL"
    return text.strip()


def is_safe_sql(sql: str) -> bool:
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
    statements = [stmt for stmt in lowered.split(";") if stmt.strip()]
    return len(statements) == 1


def enforce_limit(sql: str, limit: int = MAX_SQL_ROWS) -> str:
    trimmed = sql.strip().rstrip(";")
    if " limit " in trimmed.lower():
        return trimmed
    return f"{trimmed} LIMIT {limit}"


def truncate_results(
    results: Optional[List[Dict[str, Any]]],
    limit: int = MAX_LLM_ROWS,
) -> Tuple[List[Dict[str, Any]], bool]:
    rows = results or []
    if len(rows) > limit:
        return rows[:limit], True
    return rows, False
