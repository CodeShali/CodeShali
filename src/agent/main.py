"""
AI Job Search Agent — entry point.

Usage:
  python -m agent.main            # Start scheduler (runs every evening)
  python -m agent.main --run-now  # Run once immediately, then exit
"""

import argparse
import logging
import os
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("job_agent")

from agent.database import init_db
from agent.resume_parser import parse_resume
from agent.job_analyzer import analyze_resume
from agent.job_searcher import search_all_boards
from agent.relevance_scorer import score_and_filter
from agent.email_sender import send_digest
from agent.alert_sender import send_alerts
from agent.web_server import start_web_server
from agent import database as db


def run_pipeline(resume_text: str) -> None:
    logger.info("=== Job search pipeline started ===")

    if not resume_text:
        logger.error("No resume text available — aborting pipeline.")
        return

    try:
        profile = analyze_resume(resume_text)
    except Exception as exc:
        logger.error("Resume analysis failed: %s", exc)
        return

    try:
        jobs = search_all_boards(profile)
    except Exception as exc:
        logger.error("Job search failed: %s", exc)
        return

    if not jobs:
        logger.warning("No new jobs found — skipping email.")
        return

    try:
        scored_jobs = score_and_filter(jobs, resume_text)
    except Exception as exc:
        logger.error("Job scoring failed: %s", exc)
        return

    if not scored_jobs:
        logger.warning("No jobs passed relevance threshold — skipping email.")
        return

    # Immediate alert for high-scoring jobs
    if os.getenv("ENABLE_ALERTS", "true").lower() == "true":
        threshold = int(os.getenv("ALERT_SCORE_THRESHOLD", "9"))
        alert_jobs = [j for j in scored_jobs if (j.get("score") or 0) >= threshold]
        if alert_jobs:
            try:
                send_alerts(alert_jobs)
            except Exception as exc:
                logger.error("Alert sending failed: %s", exc)

    try:
        send_digest(scored_jobs)
        db.mark_emailed([j["db_id"] for j in scored_jobs if j.get("db_id")])
    except Exception as exc:
        logger.error("Email sending failed: %s", exc)
        return

    logger.info("=== Pipeline complete: %d jobs sent ===", len(scored_jobs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run the pipeline once immediately and exit",
    )
    args = parser.parse_args()

    init_db()

    _default_resume = os.path.join(os.path.dirname(__file__), "myresume")
    resume_path = os.getenv("RESUME_PATH", _default_resume)
    resume_text = ""

    if os.path.exists(resume_path):
        try:
            resume_text = parse_resume(resume_path)
            logger.info("Resume loaded: %d characters", len(resume_text))
        except Exception as exc:
            logger.error("Failed to parse resume at '%s': %s", resume_path, exc)
    else:
        logger.error("Resume not found at '%s'. Check RESUME_PATH.", resume_path)

    start_web_server(resume_text)

    if args.run_now:
        run_pipeline(resume_text)
        return

    hour = int(os.getenv("SCHEDULE_HOUR", "18"))
    minute = int(os.getenv("SCHEDULE_MINUTE", "0"))

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        lambda: run_pipeline(resume_text),
        trigger=CronTrigger(hour=hour, minute=minute),
        name="daily_job_search",
        misfire_grace_time=3600,
    )

    logger.info("Scheduler started — daily at %02d:%02d UTC", hour, minute)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
