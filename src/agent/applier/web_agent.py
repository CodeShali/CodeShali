"""
Vision-based browser agent for job applications.

Works on any ATS or career site — no site-specific code needed.
The agent sees a screenshot at each step, decides the next action,
and Playwright executes it. Loops until submitted or stuck.
"""

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI
from playwright.sync_api import Page, TimeoutError as PWTimeout

logger = logging.getLogger(__name__)

MAX_STEPS = 20
SCREENSHOT_WIDTH = 1280
SCREENSHOT_HEIGHT = 800

# ── Candidate profile (from env) ─────────────────────────────────────────────

def _candidate() -> dict:
    return {
        "name":  os.getenv("APPLICANT_NAME", ""),
        "email": os.getenv("APPLICANT_EMAIL", os.getenv("GMAIL_USER", "")),
        "phone": os.getenv("APPLICANT_PHONE", ""),
        "location": os.getenv("APPLICANT_LOCATION", ""),
        "linkedin": os.getenv("APPLICANT_LINKEDIN", ""),
        "github":   os.getenv("APPLICANT_GITHUB", ""),
        "website":  os.getenv("APPLICANT_WEBSITE", ""),
    }


# ── Action dataclass ──────────────────────────────────────────────────────────

@dataclass
class Action:
    action: str          # click | fill | select | upload | scroll_down | scroll_up | wait | done | failed | needs_human
    selector: str = ""
    value: str = ""
    thought: str = ""
    reason: str = ""


# ── GPT-4o vision interface ───────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert job application assistant controlling a real web browser.

Candidate:
  Name     : {name}
  Email    : {email}
  Phone    : {phone}
  Location : {location}
  LinkedIn : {linkedin}
  GitHub   : {github}
  Website  : {website}

Resume summary:
{resume}

Job: {title} at {company}

Your goal: complete and submit the job application visible in the screenshot.

Respond with ONLY a valid JSON object — no markdown, no explanation outside JSON:
{{
  "thought":   "<brief reasoning about what you see and why you chose this action>",
  "action":    "<one of: click | fill | select | upload | scroll_down | scroll_up | wait | done | failed | needs_human>",
  "selector":  "<CSS selector of the target element (empty for scroll/wait/done/failed)>",
  "value":     "<text to type, option text to select, or empty>",
  "reason":    "<only for done/failed/needs_human: explain what happened>"
}}

Action rules:
- click    : click a button, link, checkbox, or radio button
- fill     : type into a text input or textarea (clear it first implicitly)
- select   : choose a <select> dropdown option by visible text
- upload   : set the resume PDF on a file <input type="file">
- scroll_down / scroll_up : scroll the viewport to reveal more content
- wait     : pause for the page to finish loading (use sparingly, max 2×)
- done     : the application was successfully submitted — use only when you
             see a confirmation/thank-you message
- failed   : you are unable to complete the application; explain in reason
- needs_human : CAPTCHA, 2FA, or mandatory login wall encountered

Policy:
- Never invent contact details; use only the candidate profile above.
- For "years of experience" infer from the resume summary.
- For work authorisation questions answer Yes / I am authorised.
- For salary fields use a reasonable market-rate range for the role.
- If a field is already filled correctly, skip it.
- Prefer semantic selectors (aria-label, name, id) over positional ones.
- If the page shows a confirmation or "Application submitted" message → done.
- If you are stuck after several attempts → failed.
"""


def _screenshot_b64(page: Page) -> str:
    png = page.screenshot(full_page=False)
    return base64.b64encode(png).decode()


def _decide(
    client: OpenAI,
    page: Page,
    job: dict,
    resume_text: str,
    history: list[str],
) -> Action:
    cand = _candidate()
    system = SYSTEM_PROMPT.format(
        name=cand["name"], email=cand["email"], phone=cand["phone"],
        location=cand["location"], linkedin=cand["linkedin"],
        github=cand["github"], website=cand["website"],
        resume=resume_text[:2500],
        title=job["title"], company=job["company"],
    )

    history_text = "\n".join(f"Step {i+1}: {h}" for i, h in enumerate(history[-6:]))
    user_text = f"Recent actions:\n{history_text}\n\nDecide the next action." if history else "Decide the next action."

    img_b64 = _screenshot_b64(page)

    response = client.chat.completions.create(
        model="gpt-4o",                 # vision required — mini doesn't support images reliably
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "high",
                }},
            ]},
        ],
        temperature=0.2,
        max_tokens=300,
        response_format={"type": "json_object"},
    )

    raw = json.loads(response.choices[0].message.content)
    return Action(
        action=raw.get("action", "failed"),
        selector=raw.get("selector", ""),
        value=raw.get("value", ""),
        thought=raw.get("thought", ""),
        reason=raw.get("reason", ""),
    )


# ── Action executor ───────────────────────────────────────────────────────────

def _execute(page: Page, action: Action, resume_pdf: Optional[str]) -> bool:
    """Execute an action on the page. Returns False if execution failed."""
    try:
        if action.action == "click":
            page.locator(action.selector).first.click(timeout=8_000)
            page.wait_for_load_state("networkidle", timeout=10_000)

        elif action.action == "fill":
            loc = page.locator(action.selector).first
            loc.clear()
            loc.fill(action.value, timeout=8_000)

        elif action.action == "select":
            page.locator(action.selector).first.select_option(label=action.value, timeout=8_000)

        elif action.action == "upload":
            path = resume_pdf or ""
            if not path or not os.path.exists(path):
                logger.warning("upload requested but RESUME_PDF_PATH not set or file missing.")
                return False
            page.locator(action.selector).first.set_input_files(path)
            time.sleep(1)

        elif action.action == "scroll_down":
            page.evaluate("window.scrollBy(0, 600)")
            time.sleep(0.5)

        elif action.action == "scroll_up":
            page.evaluate("window.scrollBy(0, -600)")
            time.sleep(0.5)

        elif action.action == "wait":
            page.wait_for_load_state("networkidle", timeout=10_000)

        return True

    except PWTimeout:
        logger.warning("Action timed out: %s on '%s'", action.action, action.selector)
        return False
    except Exception as exc:
        logger.warning("Action failed (%s on '%s'): %s", action.action, action.selector, exc)
        return False


# ── Public entry point ────────────────────────────────────────────────────────

def run_web_agent(
    page: Page,
    job: dict,
    resume_text: str,
    resume_pdf: Optional[str] = None,
    client: Optional[OpenAI] = None,
) -> str:
    """
    Run the vision agent on the current page to complete a job application.
    Returns: 'applied' | 'failed' | 'needs_human' | 'skipped'
    """
    if client is None:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    history: list[str] = []
    consecutive_failures = 0

    logger.info("Web agent starting for '%s' at %s (max %d steps).", job["title"], job["company"], MAX_STEPS)

    for step in range(1, MAX_STEPS + 1):
        try:
            action = _decide(client, page, job, resume_text, history)
        except Exception as exc:
            logger.error("Vision call failed at step %d: %s", step, exc)
            return "failed"

        log_entry = f"{action.action}({action.selector!r}, {action.value!r}) — {action.thought}"
        history.append(log_entry)
        logger.info("[Step %d/%d] %s", step, MAX_STEPS, log_entry)

        # Terminal actions
        if action.action == "done":
            logger.info("Agent confirmed application submitted: %s", action.reason)
            return "applied"

        if action.action == "failed":
            logger.warning("Agent gave up: %s", action.reason)
            return "failed"

        if action.action == "needs_human":
            logger.warning("Human needed: %s", action.reason)
            return "needs_human"

        # Execute and track failures
        ok = _execute(page, action, resume_pdf)
        if not ok:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                logger.warning("3 consecutive execution failures — aborting.")
                return "failed"
        else:
            consecutive_failures = 0

        time.sleep(0.8)

    logger.warning("Max steps (%d) reached without submission.", MAX_STEPS)
    return "failed"
