from __future__ import annotations

import re
from typing import Dict, List

DISALLOWED = re.compile(r"\b(drop|delete|truncate|alter|update|insert|create)\b", re.I)


def basic_sql_safety_check(sql: str) -> List[str]:
    issues = []
    if DISALLOWED.search(sql or ""):
        issues.append("Detected forbidden DML/DDL command (DROP/DELETE/UPDATE/INSERT/...).")

    if "select" in (sql or "").lower() and "where" not in (sql or "").lower():
        issues.append("Query without a WHERE clause (may scan the entire table).")

    return issues


def extract_sql_block(raw: str) -> str:
    """Extract the content between the [SQL] tag and the next section.

    If a [SQL] section is not found, return the full raw text.
    """
    if not raw:
        return ""

    lower = raw.lower()
    start = lower.find("[sql]")
    if start == -1:
        return raw.strip()

    after = raw[start + len("[SQL]") :]

    next_markers = ["[explicacao]", "[checks]"]
    end_positions = []
    for m in next_markers:
        p = after.lower().find(m)
        if p != -1:
            end_positions.append(p)
    end = min(end_positions) if end_positions else len(after)

    return after[:end].strip()


def split_structured_output(raw: str) -> Dict[str, str]:
    """Split the model output into [SQL], [EXPLICACAO]/[EXPLICAÇÃO] and [CHECKS] sections.

    Returns a dictionary with keys "sql", "explicacao" and "checks".
    If a section does not exist, the corresponding value is an empty string.
    """

    sections: Dict[str, List[str]] = {"sql": [], "explicacao": [], "checks": []}
    current: str | None = None

    for line in (raw or "").splitlines():
        tag = line.strip().lower()
        if tag == "[sql]":
            current = "sql"
            continue
        if tag in ("[explicacao]", "[explicação]"):
            current = "explicacao"
            continue
        if tag == "[checks]":
            current = "checks"
            continue

        if current is not None:
            sections[current].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items()}
