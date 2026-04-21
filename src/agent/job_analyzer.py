import json
import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


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
    client = _get_client()
    location_override = os.getenv("JOB_LOCATION", "")

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": resume_text},
        ],
        temperature=0.2,
    )

    profile = json.loads(response.choices[0].message.content)

    if location_override:
        profile["location"] = location_override

    logger.info(
        "Resume analyzed — titles: %s | seniority: %s | location: %s",
        profile.get("job_titles", [])[:3],
        profile.get("seniority"),
        profile.get("location"),
    )
    return profile
