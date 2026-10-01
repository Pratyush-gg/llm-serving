"""Render a database schema as CREATE TABLE, a compact list, or plain English."""
import sqlite3
from typing import List, Tuple

STYLES = ["create", "compact", "prose"]


def _objects(setup_sql: str) -> List[Tuple[str, str, str, List[Tuple[str, str]]]]:
    """(kind, name, create_sql, [(column, declared_type)]) for every table and view, in creation order."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(setup_sql)
        rows = conn.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE type IN ('table', 'view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        ).fetchall()
        out = []
        for kind, name, sql in rows:
            cols = [(c[1], (c[2] or "").strip()) for c in conn.execute(f'PRAGMA table_info("{name}")')]
            out.append((kind, name, (sql or "").strip(), cols))
        return out
    finally:
        conn.close()


def _join_words(items: List[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def render_schema(setup_sql: str, style: str) -> str:
    objs = _objects(setup_sql)
    if style == "create":
        return "\n".join(sql.rstrip(";") + ";" for _, _, sql, _ in objs)
    if style == "compact":
        lines = []
        for kind, name, _, cols in objs:
            cols_txt = ", ".join(f"{c} {t}".strip() for c, t in cols)
            lines.append(f"{name}({cols_txt})" + (" [view]" if kind == "view" else ""))
        return "Tables:\n" + "\n".join(lines)
    if style == "prose":
        sentences = []
        for kind, name, _, cols in objs:
            cols_txt = _join_words([f"{c} ({t or 'any type'})" for c, t in cols])
            noun = "view" if kind == "view" else "table"
            word = "column" if len(cols) == 1 else "columns"
            sentences.append(f"The {noun} {name} has the {word} {cols_txt}.")
        return " ".join(sentences)
    raise ValueError(f"Unknown schema style: {style}")
