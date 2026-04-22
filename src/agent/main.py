"""
AI Job Search Agent — entry point.

Usage:
  python -m agent.main                      # Start scheduler (runs every evening)
  python -m agent.main --run-now            # Run once immediately, then exit
  python -m agent.main --run-now --apply    # Run once + auto-apply on Indeed
"""

import argparse
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

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

from agent.resume_parser import parse_resume
from agent.job_analyzer import analyze_resume
from agent.job_searcher import search_all_boards
from agent.relevance_scorer import score_and_filter
from agent.email_sender import send_digest
from agent.applier.apply_db import filter_new, mark_seen


def run_pipeline(auto_apply: bool = False) -> None:
    logger.info("=== Job search pipeline started (auto_apply=%s) ===", auto_apply)

    _default_resume = os.path.join(os.path.dirname(__file__), "myresume")
    resume_path = os.getenv("RESUME_PATH", _default_resume)
    resume_pdf = os.getenv("RESUME_PDF_PATH")  # optional PDF for upload

    if not os.path.exists(resume_path):
        logger.error("Resume not found at '%s'. Check RESUME_PATH.", resume_path)
        return

    try:
        resume_text = parse_resume(resume_path)
    except Exception as exc:
        logger.error("Failed to parse resume: %s", exc)
        return

    try:
        profile = analyze_resume(resume_text)
    except Exception as exc:
        logger.error("GPT resume analysis failed: %s", exc)
        return

    try:
        jobs = search_all_boards(profile)
    except Exception as exc:
        logger.error("Job search failed: %s", exc)
        return

    if not jobs:
        logger.warning("No jobs found — skipping.")
        return

    # Drop jobs already seen in previous runs
    new_jobs = filter_new(jobs)
    logger.info("%d new jobs (out of %d found, %d already seen).", len(new_jobs), len(jobs), len(jobs) - len(new_jobs))

    if not new_jobs:
        logger.info("No new jobs since last run — skipping email.")
        return

    try:
        scored_jobs = score_and_filter(new_jobs, resume_text)
    except Exception as exc:
        logger.error("Job scoring failed: %s", exc)
        return

    if not scored_jobs:
        logger.warning("No jobs passed the relevance threshold — skipping email.")
        mark_seen(new_jobs)
        return

    if auto_apply:
        try:
            from agent.applier.indeed_applier import apply_to_indeed_jobs
            apply_to_indeed_jobs(scored_jobs, resume_text, resume_pdf)
        except Exception as exc:
            logger.error("Auto-apply failed: %s", exc)

    try:
        send_digest(scored_jobs)
    except Exception as exc:
        logger.error("Email sending failed: %s", exc)
        return

    mark_seen(new_jobs)
    logger.info("=== Pipeline complete: %d jobs sent ===", len(scored_jobs))


# ── Minimal health-check server (K8s liveness / readiness probe) ────────────

class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *_):
        pass


def _start_health_server() -> None:
    port = int(os.getenv("HEALTH_PORT", "5000"))
    try:
        server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    except OSError:
        logger.warning("Port %d already in use — health server skipped.", port)
        return
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    logger.info("Health endpoint listening on port %d", port)


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-now", action="store_true",
                        help="Run the pipeline once immediately and exit")
    parser.add_argument("--apply", action="store_true",
                        help="Auto-apply to high-scoring Indeed Easy Apply jobs")
    args = parser.parse_args()

    _start_health_server()

    if args.run_now:
        run_pipeline(auto_apply=args.apply)
        return

    hour = int(os.getenv("SCHEDULE_HOUR", "18"))
    minute = int(os.getenv("SCHEDULE_MINUTE", "0"))
    apply_scheduled = os.getenv("AUTO_APPLY", "false").lower() == "true"

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        run_pipeline,
        kwargs={"auto_apply": apply_scheduled},
        trigger=CronTrigger(hour=hour, minute=minute),
        name="daily_job_search",
        misfire_grace_time=3600,
    )

    next_run = scheduler.get_jobs()[0].next_run_time
    logger.info("Scheduler started — next run at %s UTC (every day at %02d:%02d)", next_run, hour, minute)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
