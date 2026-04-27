import logging
import os

import pandas as pd
from jobspy import scrape_jobs

from agent import database as db

logger = logging.getLogger(__name__)

SITES = ["indeed", "linkedin", "glassdoor", "google", "zip_recruiter"]


def search_all_boards(profile: dict) -> list[dict]:
    job_titles = profile.get("job_titles", [])[:5]
    location = profile.get("location", os.getenv("JOB_LOCATION", "Remote"))
    max_per_board = int(os.getenv("MAX_JOBS_PER_BOARD", "15"))

    all_jobs: list[dict] = []
    seen_urls: set[str] = set()

    for title in job_titles:
        logger.info("Searching '%s' in '%s' across all boards...", title, location)
        try:
            df = scrape_jobs(
                site_name=SITES,
                search_term=title,
                location=location,
                results_wanted=max_per_board,
                hours_old=24,
                country_indeed="USA",
            )
            _collect(df, all_jobs, seen_urls)
            logger.info("  → %d new jobs collected so far", len(all_jobs))
        except Exception as exc:
            logger.warning("Board search failed for '%s': %s", title, exc)

    logger.info("Total new jobs this run: %d", len(all_jobs))
    return all_jobs


def _collect(df: pd.DataFrame, jobs: list[dict], seen: set[str]) -> None:
    for _, row in df.iterrows():
        url = str(row.get("job_url") or "").strip()
        if not url or url in seen:
            continue
        if db.is_seen(url):
            continue
        seen.add(url)

        job = {
            "title":       str(row.get("title") or ""),
            "company":     str(row.get("company") or ""),
            "location":    str(row.get("location") or ""),
            "salary":      _format_salary(row),
            "url":         url,
            "link":        url,
            "description": str(row.get("description") or "")[:1500],
            "site":        str(row.get("site") or ""),
            "date_posted": str(row.get("date_posted") or ""),
        }
        job["db_id"] = db.upsert_job(job)
        jobs.append(job)


def _format_salary(row: pd.Series) -> str:
    min_amt = row.get("min_amount")
    max_amt = row.get("max_amount")
    interval = row.get("interval") or ""
    currency = row.get("currency") or "USD"

    try:
        if min_amt and max_amt:
            return f"{currency} {float(min_amt):,.0f} – {float(max_amt):,.0f} / {interval}"
        if min_amt:
            return f"{currency} {float(min_amt):,.0f}+ / {interval}"
    except (TypeError, ValueError):
        pass

    return "Not disclosed"
