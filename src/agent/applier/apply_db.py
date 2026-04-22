"""SQLite store for seen jobs and application history."""

import os
import sqlite3
from datetime import datetime

_DEFAULT_DB = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "jobs.db"))


def _conn() -> sqlite3.Connection:
    path = os.getenv("DB_PATH", _DEFAULT_DB)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _init(conn)
    return conn


def _init(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_jobs (
            url        TEXT PRIMARY KEY,
            title      TEXT,
            company    TEXT,
            first_seen TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            url        TEXT PRIMARY KEY,
            title      TEXT,
            company    TEXT,
            score      INTEGER,
            status     TEXT,
            applied_at TEXT,
            notes      TEXT
        )
    """)
    conn.commit()


def filter_new(jobs: list[dict]) -> list[dict]:
    """Return only jobs not yet seen in previous runs."""
    with _conn() as conn:
        return [
            j for j in jobs
            if not conn.execute("SELECT 1 FROM seen_jobs WHERE url=?", (j["link"],)).fetchone()
        ]


def mark_seen(jobs: list[dict]) -> None:
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO seen_jobs (url,title,company,first_seen) VALUES (?,?,?,?)",
            [(j["link"], j["title"], j["company"], now) for j in jobs],
        )
        conn.commit()


def was_applied(url: str) -> bool:
    with _conn() as conn:
        row = conn.execute("SELECT status FROM applications WHERE url=?", (url,)).fetchone()
        return bool(row and row["status"] == "applied")


def record_application(job: dict, status: str, notes: str = "") -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO applications
               (url,title,company,score,status,applied_at,notes)
               VALUES (?,?,?,?,?,?,?)""",
            (
                job["link"], job["title"], job["company"],
                job.get("score", 0), status,
                datetime.utcnow().isoformat(), notes,
            ),
        )
        conn.commit()
