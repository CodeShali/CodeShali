import logging
import os

from agent.ai_client import chat_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a career expert. Given a resume, extract the candidate's job search profile.

Return ONLY valid JSON with this exact structure:
{
  "job_titles": ["<top 5 relevant job titles to search for, most specific first>"],
  "skills": ["<top 15 technical skills>"],
  "experience_years": <integer>,
  "location": "<city, state or 'Remote'>",
  "remote_ok": <true|false>,
  "seniority": "<entry|mid|senior|lead|principal>"
}"""


def analyze_resume(resume_text: str) -> dict:
    location_override = os.getenv("JOB_LOCATION", "")

    profile = chat_json(SYSTEM_PROMPT, resume_text)

    if location_override:
        profile["location"] = location_override

    logger.info(
        "Resume analyzed — titles: %s | seniority: %s | location: %s",
        profile.get("job_titles", [])[:3],
        profile.get("seniority"),
        profile.get("location"),
    )
    return profile
