from __future__ import annotations

from typing import List, Dict


def system_prompt(dialect: str) -> str:
    return f"""
You must write read-only SQL queries (SELECT only).
Target dialect: {dialect}

Mandatory rules:
- NEVER use: DROP, DELETE, UPDATE, INSERT, TRUNCATE, ALTER, CREATE.
- Prefer queries with a date filter whenever there is a temporal column (e.g., created_at).
- Avoid cardinality explosion: when joining 1:N tables, deduplicate to 1 row when appropriate.
- At the end, return exactly 3 sections:

[SQL]
<only the query>

[EXPLANATION]
- 3 bullet points describing what the query does

[CHECKS]
- 2 short SQL checks to validate duplicates/volume
""".strip()


MAX_TOTAL_CONTEXT_CHARS = 9000
MAX_CARD_CHARS = 3000


def _compact_card_text(text: str, max_chars: int = MAX_CARD_CHARS) -> str:
    """Simple heuristic to reduce the size of each card.

    Strategy:
    - Keep the first block (title/introduction).
    - Prioritize paragraphs that appear to contain SQL (SELECT, JOIN, WHERE...).
    - Fill with other initial paragraphs until the character limit is reached.
    """

    if len(text) <= max_chars:
        return text

    paragraphs = text.split("\n\n")
    if not paragraphs:
        return text[:max_chars]

    title = paragraphs[0]
    rest = paragraphs[1:]

    sql_keywords = [
        "select ",
        " join ",
        " where ",
        " group by ",
        " having ",
        " window ",
        " over(",
    ]

    important: List[str] = []
    for p in rest:
        low = p.lower()
        if any(k in low for k in sql_keywords):
            important.append(p)

    chunks: List[str] = [title]

    def total_len(parts: List[str]) -> int:
        return len("\n\n".join(parts))

    for p in important:
        if total_len(chunks + [p]) > max_chars:
            break
        chunks.append(p)

    if total_len(chunks) < max_chars:
        for p in rest:
            if p in important:
                continue
            if total_len(chunks + [p]) > max_chars:
                break
            chunks.append(p)

    compact = "\n\n".join(chunks)
    if len(compact) > max_chars:
        compact = compact[:max_chars]
    return compact


def user_prompt(question: str, retrieved_cards: List[Dict], dialect: str) -> str:
    context_parts: List[str] = []
    current_chars = 0

    for c in retrieved_cards:
        raw_text = str(c.get("text", ""))
        compact_text = _compact_card_text(raw_text)

        card_block = f"### CARD: {c['id']}\n{compact_text}"
        new_len = current_chars + len(card_block)

        if new_len > MAX_TOTAL_CONTEXT_CHARS:
            break

        context_parts.append(card_block)
        current_chars = new_len

    context = "\n\n ---\n\n".join(context_parts)

    return f"""
{question}

Banco: {dialect}

Use os cards abaixo como referencia. Adapte nomes de tabelas/colunas quando necessario; se estiver faltando uma tabela/coluna, coloque nomes genericos e deixe o usuario completar.

Cards:
{context}
""".strip()
