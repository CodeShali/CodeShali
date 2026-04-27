import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime

logger = logging.getLogger(__name__)

_default_db = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "codeshali.db",
)
DB_PATH = os.getenv("DB_PATH", _default_db)


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                url          TEXT UNIQUE NOT NULL,
                title        TEXT,
                company      TEXT,
                location     TEXT,
                salary       TEXT,
                site         TEXT,
                date_posted  TEXT,
                description  TEXT,
                score        REAL,
                score_reason TEXT,
                status       TEXT DEFAULT 'new',
                found_date   TEXT NOT NULL,
                emailed      INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS cover_letters (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id     INTEGER NOT NULL REFERENCES jobs(id),
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
    logger.info("Database ready at %s", DB_PATH)


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def is_seen(url: str) -> bool:
    with _conn() as conn:
        return conn.execute("SELECT 1 FROM jobs WHERE url=?", (url,)).fetchone() is not None


def upsert_job(job: dict) -> int:
    today = date.today().isoformat()
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO jobs
               (url, title, company, location, salary, site, date_posted, description, found_date)
               VALUES (:url,:title,:company,:location,:salary,:site,:date_posted,:description,:found_date)
               ON CONFLICT(url) DO UPDATE SET
                 title=excluded.title, company=excluded.company
               RETURNING id""",
            {**job, "found_date": today},
        )
        return cur.fetchone()[0]


def update_job_score(job_id: int, score: float, reason: str) -> None:
    with _conn() as conn:
        conn.execute("UPDATE jobs SET score=?, score_reason=? WHERE id=?", (score, reason, job_id))


def mark_emailed(job_ids: list[int]) -> None:
    with _conn() as conn:
        conn.executemany("UPDATE jobs SET emailed=1 WHERE id=?", [(i,) for i in job_ids])


def set_status(job_id: int, status: str) -> None:
    with _conn() as conn:
        conn.execute("UPDATE jobs SET status=? WHERE id=?", (status, job_id))


def get_job(job_id: int) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None


def save_cover_letter(job_id: int, content: str) -> int:
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO cover_letters (job_id, content, created_at) VALUES (?,?,?)",
            (job_id, content, datetime.utcnow().isoformat()),
        )
        return cur.lastrowid


def delete_cover_letters(job_id: int) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM cover_letters WHERE job_id=?", (job_id,))


def get_cover_letter(job_id: int) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM cover_letters WHERE job_id=? ORDER BY created_at DESC LIMIT 1",
            (job_id,),
        ).fetchone()
        return dict(row) if row else None


def get_jobs(filter_by: str = "all", days: int = 30, limit: int = 300) -> list[dict]:
    if filter_by == "today":
        where, params = "WHERE j.found_date = date('now')", []
    elif filter_by in ("interested", "skipped", "applied"):
        where, params = "WHERE j.status = ?", [filter_by]
    else:
        where, params = f"WHERE j.found_date >= date('now', '-{days} days')", []

    with _conn() as conn:
        rows = conn.execute(
            f"""SELECT j.*,
                   (SELECT id FROM cover_letters WHERE job_id=j.id ORDER BY created_at DESC LIMIT 1)
                   AS has_cover_letter
               FROM jobs j
               {where}
               ORDER BY j.found_date DESC, j.score DESC
               LIMIT ?""",
            [*params, limit],
        ).fetchall()
        return [dict(r) for r in rows]


def get_stats() -> dict:
    with _conn() as conn:
        def count(sql, *args):
            return conn.execute(sql, args).fetchone()[0]
        return {
            "total":         count("SELECT COUNT(*) FROM jobs"),
            "interested":    count("SELECT COUNT(*) FROM jobs WHERE status='interested'"),
            "skipped":       count("SELECT COUNT(*) FROM jobs WHERE status='skipped'"),
            "applied":       count("SELECT COUNT(*) FROM jobs WHERE status='applied'"),
            "today":         count("SELECT COUNT(*) FROM jobs WHERE found_date=date('now')"),
            "cover_letters": count("SELECT COUNT(*) FROM cover_letters"),
        }
