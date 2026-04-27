import logging
import os

from agent.ai_client import chat_text
from agent import database as db

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a professional career coach. Write a concise, compelling cover letter tailored "
    "to the specific job posting and the candidate's background. Keep it to 3-4 paragraphs. "
    "Be specific and personalized — no generic filler. Start with a strong opening paragraph "
    "that references the specific role and company by name."
)


def generate_cover_letter(job_id: int, resume_text: str) -> str | None:
    if os.getenv("ENABLE_COVER_LETTERS", "true").lower() != "true":
        return None

    existing = db.get_cover_letter(job_id)
    if existing:
        return existing["content"]

    job = db.get_job(job_id)
    if not job:
        logger.warning("Job %d not found for cover letter generation", job_id)
        return None

    user_content = (
        f"CANDIDATE RESUME:\n{resume_text[:3000]}\n\n"
        f"JOB POSTING:\n"
        f"Title: {job['title']}\n"
        f"Company: {job['company']}\n"
        f"Location: {job['location']}\n"
        f"Description: {(job.get('description') or '')[:2000]}\n\n"
        "Write a tailored cover letter for this specific role."
    )

    try:
        content = chat_text(_SYSTEM, user_content)
        db.save_cover_letter(job_id, content)
        logger.info("Cover letter generated: job %d — %s at %s", job_id, job["title"], job["company"])
        return content
    except Exception as exc:
        logger.error("Cover letter generation failed for job %d: %s", job_id, exc)
        return None
