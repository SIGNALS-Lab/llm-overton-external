"""
Database Module

Description:
    Shared SQLite database interface for the llm-overton project.
    Provides connection management, schema initialization, and CRUD operations
    for generation and evaluation data.

    This module is the single source of truth for the database schema.
    No raw SQL should appear in any other file.
"""

import sqlite3
import pandas as pd
from typing import Optional

# ============================================================================
# SCHEMA
# ============================================================================
JUDGE_COLUMNS = [
    ("judgeA",        "INTEGER"),
    ("judgeA_conf",   "REAL"),
    ("judgeB",        "INTEGER"),
    ("judgeB_conf",   "REAL"),
    ("judgeC",        "INTEGER"),
    ("judgeC_conf",   "REAL"),
    ("judgeA_L",      "INTEGER"),
    ("judgeA_L_conf", "REAL"),
    ("judgeB_L",      "INTEGER"),
    ("judgeB_L_conf", "REAL"),
    ("judgeC_L",      "INTEGER"),
    ("judgeC_L_conf", "REAL"),
]

BINARY_JUDGE_COLS = ["judgeA", "judgeB", "judgeC"]
LIKERT_JUDGE_COLS = ["judgeA_L", "judgeB_L", "judgeC_L"]

# ============================================================================
# CONNECTION
# ============================================================================
def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Open a SQLite connection with WAL mode and return it.
    
    Args:
        db_path: Path to the SQLite database file.
    
    Returns:
        sqlite3.Connection
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

# ============================================================================
# SCHEMA INITIALIZATION
# ============================================================================
def init_db(conn: sqlite3.Connection) -> None:
    """
    Create the generations table and indexes if they don't already exist.
    
    Args:
        conn: Active SQLite connection.
    """
    judge_col_defs = ",\n    ".join(
        f"{name} {dtype}" for name, dtype in JUDGE_COLUMNS
    )

    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS generations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            model       TEXT NOT NULL,
            prompt_code TEXT NOT NULL,
            trial       INTEGER NOT NULL,
            opinion_id  TEXT NOT NULL,
            opinion     TEXT NOT NULL,
            post        TEXT,
            {judge_col_defs},
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_run
            ON generations(model, prompt_code, trial, opinion_id);

        CREATE INDEX IF NOT EXISTS idx_model
            ON generations(model);

        CREATE INDEX IF NOT EXISTS idx_prompt
            ON generations(prompt_code);
    """)

def ensure_judge_columns(conn: sqlite3.Connection, judges: list[str]) -> list[str]:
    """
    Ensure Likert judge columns exist for each judge name.
    Adds ``{name}_L`` (INTEGER) and ``{name}_L_conf`` (REAL) columns via
    ALTER TABLE for any that don't already exist. Safe to call repeatedly.

    Args:
        conn: Active SQLite connection.
        judges: Judge name prefixes (e.g., ['judgeD', 'judgeE']).

    Returns:
        list[str]: Column names that were added (empty if all existed).
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(generations)").fetchall()}
    added = []
    for judge in judges:
        for col, dtype in [(f"{judge}_L", "INTEGER"), (f"{judge}_L_conf", "REAL")]:
            if col not in existing:
                conn.execute(f"ALTER TABLE generations ADD COLUMN [{col}] {dtype}")
                added.append(col)
    if added:
        conn.commit()
    return added


def get_likert_judge_cols(conn: sqlite3.Connection) -> list[str]:
    """
    Discover all Likert judge columns present in the generations table.

    Returns:
        list[str]: Column names like ['judgeA_L', 'judgeB_L', ...].
    """
    cols = [row[1] for row in conn.execute("PRAGMA table_info(generations)").fetchall()]
    return [c for c in cols if c.endswith("_L") and not c.endswith("_L_conf")]


# ============================================================================
# QUERY HELPERS
# ============================================================================
def trial_row_count(conn: sqlite3.Connection, model: str, prompt_code: str,
                    trial: int) -> int:
    """Return the number of rows for a given (model, prompt_code, trial)."""
    row = conn.execute(
        "SELECT COUNT(*) FROM generations WHERE model=? AND prompt_code=? AND trial=?",
        (model, prompt_code, trial)
    ).fetchone()
    return row[0]


# ============================================================================
# WRITE OPERATIONS
# ============================================================================
def insert_generations(conn: sqlite3.Connection, model: str, prompt_code: str,
                       trial: int, rows: list[tuple]) -> int:
    """
    Batch-insert generated posts into the database.
    Uses INSERT OR IGNORE to skip duplicate rows.
    Use upsert_generations() to overwrite existing rows.
    
    Args:
        conn: Active SQLite connection.
        model: Model name (e.g., 'Llama-3.3-70B-Instruct').
        prompt_code: Prompt designation code (e.g., 'B', 'AN_B', 'inherent').
        trial: Trial number (0-indexed).
        rows: List of (opinion_id, opinion, post) tuples.
    
    Returns:
        int: Number of rows inserted.
    """
    data = [(model, prompt_code, trial, oid, opinion, post) for oid, opinion, post in rows]
    with conn:
        conn.executemany(
            """INSERT OR IGNORE INTO generations (model, prompt_code, trial, opinion_id, opinion, post)
               VALUES (?, ?, ?, ?, ?, ?)""",
            data
        )
    return len(data)


def upsert_generations(conn: sqlite3.Connection, model: str, prompt_code: str,
                       trial: int, rows: list[tuple]) -> int:
    """
    Batch-upsert generated posts into the database.
    Inserts new rows or updates existing ones (sets post, opinion) on conflict.
    
    Args:
        conn: Active SQLite connection.
        model: Model name (e.g., 'Llama-3.3-70B-Instruct').
        prompt_code: Prompt designation code (e.g., 'B', 'AN_B', 'inherent').
        trial: Trial number (0-indexed).
        rows: List of (opinion_id, opinion, post) tuples.
    
    Returns:
        int: Number of rows upserted.
    """
    data = [(model, prompt_code, trial, oid, opinion, post) for oid, opinion, post in rows]
    with conn:
        conn.executemany(
            """INSERT INTO generations (model, prompt_code, trial, opinion_id, opinion, post)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(model, prompt_code, trial, opinion_id)
               DO UPDATE SET post = excluded.post, opinion = excluded.opinion""",
            data
        )
    return len(data)


def update_evaluations(conn: sqlite3.Connection, model: str, prompt_code: str,
                       trial: int, opinion_ids: list[str], judge_name: str,
                       classifications: list, confidences: list[float],
                       ratings: list[int], rating_confidences: list[float]) -> int:
    """
    Batch-update judge evaluation columns for existing generation rows.
    
    Args:
        conn: Active SQLite connection.
        model: Model name.
        prompt_code: Prompt designation code.
        trial: Trial number.
        opinion_ids: List of opinion_id values identifying rows.
        judge_name: Judge column prefix (e.g., 'judgeA').
        classifications: List of binary classifications (bool or int).
        confidences: List of binary classification confidence scores.
        ratings: List of Likert scale ratings.
        rating_confidences: List of Likert confidence scores.
    
    Returns:
        int: Number of rows updated.
    """
    data = [
        (int(cls), conf, rating, r_conf, model, prompt_code, trial, oid)
        for cls, conf, rating, r_conf, oid
        in zip(classifications, confidences, ratings, rating_confidences, opinion_ids)
    ]
    with conn:
        conn.executemany(
            f"""UPDATE generations
                SET {judge_name} = ?, {judge_name}_conf = ?,
                    {judge_name}_L = ?, {judge_name}_L_conf = ?
                WHERE model = ? AND prompt_code = ? AND trial = ? AND opinion_id = ?""",
            data
        )
    return len(data)

def update_likert_evaluations(conn: sqlite3.Connection, model: str, prompt_code: str,
                              trial: int, opinion_ids: list[str], judge_name: str,
                              ratings: list[int], rating_confidences: list[float]) -> int:
    """
    Batch-update only Likert evaluation columns for existing generation rows.
    
    Args:
        conn: Active SQLite connection.
        model: Model name.
        prompt_code: Prompt designation code.
        trial: Trial number.
        opinion_ids: List of opinion_id values identifying rows.
        judge_name: Judge column prefix (e.g., 'judgeA').
        ratings: List of Likert scale ratings.
        rating_confidences: List of Likert confidence scores.
    
    Returns:
        int: Number of rows updated.
    """
    data = [
        (rating, r_conf, model, prompt_code, trial, oid)
        for rating, r_conf, oid
        in zip(ratings, rating_confidences, opinion_ids)
    ]
    with conn:
        conn.executemany(
            f"""UPDATE generations
                SET {judge_name}_L = ?, {judge_name}_L_conf = ?
                WHERE model = ? AND prompt_code = ? AND trial = ? AND opinion_id = ?""",
            data
        )
    return len(data)

# ============================================================================
# READ OPERATIONS
# ============================================================================
def get_unevaluated_runs(conn: sqlite3.Connection,
                         model: Optional[str] = None,
                         likert_only: bool = False,
                         judges: Optional[list[str]] = None) -> list[tuple]:
    """
    Find distinct (model, prompt_code, trial) combinations that have not been
    fully evaluated. Only considers rows that have a non-NULL post.
    
    Args:
        conn: Active SQLite connection.
        model: Optional model name filter.
        likert_only: If True, check Likert columns instead of binary columns.
        judges: Explicit list of judge name prefixes (e.g., ['judgeD']).
                When provided, checks those Likert columns for NULL.
                When None, falls back to the original 3 judges.
    
    Returns:
        list of (model, prompt_code, trial) tuples.
    """
    if judges is not None:
        cols = [f"{j}_L" for j in judges]
    elif likert_only:
        cols = LIKERT_JUDGE_COLS
    else:
        cols = BINARY_JUDGE_COLS
    or_clauses = " OR ".join(f"{col} IS NULL" for col in cols)
    query = f"""
        SELECT DISTINCT model, prompt_code, trial
        FROM generations
        WHERE post IS NOT NULL
          AND ({or_clauses})
    """
    params = []
    if model:
        query += " AND model = ?"
        params.append(model)
    query += " ORDER BY model, prompt_code, trial"
    return conn.execute(query, params).fetchall()


def load_df(conn: sqlite3.Connection, *,
            model: Optional[str | list[str]] = None,
            prompt_code: Optional[str | list[str]] = None,
            trial: Optional[int | list[int]] = None,
            evaluated: Optional[bool] = None,
            eval_mode: str = "binary",
            exclude_prompt: Optional[str | list[str]] = None,
            columns: Optional[list[str]] = None,
            judges: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Load data from the generations table into a pandas DataFrame with optional filters.
    
    Args:
        conn: Active SQLite connection.
        model: Filter by model name(s).
        prompt_code: Filter by prompt code(s).
        trial: Filter by trial number(s).
        evaluated: If True, only rows with all judge columns filled.
                   If False, only rows with at least one NULL judge column.
        eval_mode: Which judge columns to check for the ``evaluated`` filter.
                   ``"binary"`` (default) checks judgeA/B/C;
                   ``"likert"`` checks judgeA_L/B_L/C_L.
                   Ignored when ``judges`` is provided.
        exclude_prompt: Exclude rows matching these prompt code(s).
        columns: Specific columns to select (default: all).
        judges: Explicit list of judge name prefixes (e.g., ['judgeA', 'judgeD']).
                When provided, the ``evaluated`` filter checks those Likert columns.
                When None, falls back to ``eval_mode`` behaviour.
    
    Returns:
        pd.DataFrame with appropriate type coercion.
    """
    select = ", ".join(columns) if columns else "*"
    query = f"SELECT {select} FROM generations"
    conditions = []
    params = []

    # model filter
    if model is not None:
        if isinstance(model, str):
            conditions.append("model = ?")
            params.append(model)
        else:
            placeholders = ", ".join("?" * len(model))
            conditions.append(f"model IN ({placeholders})")
            params.extend(model)

    # prompt_code filter
    if prompt_code is not None:
        if isinstance(prompt_code, str):
            conditions.append("prompt_code = ?")
            params.append(prompt_code)
        else:
            placeholders = ", ".join("?" * len(prompt_code))
            conditions.append(f"prompt_code IN ({placeholders})")
            params.extend(prompt_code)

    # trial filter
    if trial is not None:
        if isinstance(trial, int):
            conditions.append("trial = ?")
            params.append(trial)
        else:
            placeholders = ", ".join("?" * len(trial))
            conditions.append(f"trial IN ({placeholders})")
            params.extend(trial)

    # evaluated filter
    if judges is not None:
        eval_cols = [f"{j}_L" for j in judges]
    elif eval_mode == "likert":
        eval_cols = LIKERT_JUDGE_COLS
    else:
        eval_cols = BINARY_JUDGE_COLS
    if evaluated is True:
        for col in eval_cols:
            conditions.append(f"{col} IS NOT NULL")
    elif evaluated is False:
        or_clauses = " OR ".join(f"{col} IS NULL" for col in eval_cols)
        conditions.append(f"({or_clauses})")

    # exclude_prompt filter
    if exclude_prompt is not None:
        if isinstance(exclude_prompt, str):
            conditions.append("prompt_code != ?")
            params.append(exclude_prompt)
        else:
            placeholders = ", ".join("?" * len(exclude_prompt))
            conditions.append(f"prompt_code NOT IN ({placeholders})")
            params.extend(exclude_prompt)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY model, prompt_code, trial, opinion_id"

    df = pd.read_sql_query(query, conn, params=params)

    # Type coercion: SQLite INTEGER 0/1 → pandas bool for binary judge columns
    for col in BINARY_JUDGE_COLS:
        if col in df.columns:
            df[col] = df[col].map({1: True, 0: False, None: None})

    return df
