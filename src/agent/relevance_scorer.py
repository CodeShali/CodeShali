import logging
import os
import time

from agent.ai_client import chat_json
from agent import database as db

logger = logging.getLogger(__name__)

BATCH_SIZE = 10

SYSTEM_PROMPT = """You are a strict recruiter. Given a candidate's resume summary and a batch of job listings, score each job's relevance to the candidate on a scale of 1–10.

Scoring guide:
- 9-10: Near-perfect match (title, seniority, and most skills align)
- 7-8:  Good match (title fits, majority of skills match)
- 5-6:  Partial match (related field, some skill overlap)
- 1-4:  Poor match (unrelated role or major skill gaps)

Return ONLY valid JSON — an array of objects, one per job, in the same order:
[{"index": 0, "score": 8, "reason": "one-line reason"}, ...]"""


def score_and_filter(jobs: list[dict], resume_text: str) -> list[dict]:
    min_score = int(os.getenv("MIN_RELEVANCE_SCORE", "7"))
    max_jobs = int(os.getenv("MAX_EMAIL_JOBS", "30"))

    resume_summary = resume_text[:3000]
    scored: list[dict] = []

    batches = [jobs[i: i + BATCH_SIZE] for i in range(0, len(jobs), BATCH_SIZE)]
    for batch_idx, batch in enumerate(batches):
        logger.info("Scoring batch %d/%d (%d jobs)...", batch_idx + 1, len(batches), len(batch))
        try:
            results = _score_batch(batch, resume_summary)
            for item in results:
                idx = item.get("index", -1)
                score = item.get("score", 0)
                if 0 <= idx < len(batch):
                    job_id = batch[idx].get("db_id")
                    if job_id:
                        db.update_job_score(job_id, score, item.get("reason", ""))
                    if score >= min_score:
                        job = dict(batch[idx])
                        job["score"] = score
                        job["score_reason"] = item.get("reason", "")
                        scored.append(job)
        except Exception as exc:
            logger.warning("Scoring batch %d failed: %s", batch_idx + 1, exc)
        time.sleep(0.5)

    scored.sort(key=lambda j: j["score"], reverse=True)
    logger.info("%d jobs passed relevance threshold (>= %d)", len(scored), min_score)
    return scored[:max_jobs]


def _score_batch(batch: list[dict], resume_summary: str) -> list[dict]:
    job_list = "\n".join(
        f'[{i}] {j["title"]} at {j["company"]} ({j["location"]})\n{j["description"][:400]}'
        for i, j in enumerate(batch)
    )
    user_content = (
        f"RESUME SUMMARY:\n{resume_summary}\n\n"
        f"JOB LISTINGS:\n{job_list}"
    )

    raw = chat_json(SYSTEM_PROMPT, user_content)
    if isinstance(raw, list):
        return raw
    for key in raw:
        if isinstance(raw[key], list):
            return raw[key]
    return []
