"""Indeed Easy Apply automation using Playwright."""

import logging
import os
import time
from typing import Optional

from openai import OpenAI
from playwright.sync_api import sync_playwright, Page, TimeoutError as PWTimeout

from agent.applier.apply_db import record_application, was_applied

logger = logging.getLogger(__name__)
_client: Optional[OpenAI] = None

# Seconds to wait between consecutive applications
_APPLY_DELAY = 4

ANSWER_PROMPT = """You are filling out a job application.

Job: {title} at {company}
Resume summary:
{resume}

Answer this question concisely and professionally (1-3 sentences max):
{question}

Reply with ONLY the answer text."""


def _gpt(question: str, job: dict, resume: str) -> str:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": ANSWER_PROMPT.format(
            title=job["title"], company=job["company"],
            resume=resume[:2000], question=question,
        )}],
        temperature=0.3,
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip()


# ── Public entry point ───────────────────────────────────────────────────────

def apply_to_indeed_jobs(
    jobs: list[dict],
    resume_text: str,
    resume_pdf_path: Optional[str] = None,
) -> dict:
    """
    Attempt Indeed Easy Apply for scored jobs above AUTO_APPLY_SCORE_THRESHOLD.

    Returns {"applied": [...], "skipped": [...], "failed": [...]}.
    """
    email = os.getenv("INDEED_EMAIL")
    password = os.getenv("INDEED_PASSWORD")
    threshold = int(os.getenv("AUTO_APPLY_SCORE_THRESHOLD", "9"))
    headless = os.getenv("HEADLESS", "false").lower() == "true"

    summary: dict[str, list] = {"applied": [], "skipped": [], "failed": []}

    if not email or not password:
        logger.warning("INDEED_EMAIL / INDEED_PASSWORD not set — skipping auto-apply.")
        summary["skipped"] = jobs
        return summary

    candidates = [
        j for j in jobs
        if "indeed" in j.get("site", "").lower()
        and j.get("score", 0) >= threshold
        and not was_applied(j["link"])
    ]

    if not candidates:
        logger.info("No new Indeed jobs at or above score threshold %d.", threshold)
        summary["skipped"] = jobs
        return summary

    logger.info(
        "Indeed auto-apply: %d candidate(s) (score >= %d, headless=%s).",
        len(candidates), threshold, headless,
    )

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = ctx.new_page()

        if not _login(page, email, password):
            logger.error("Indeed login failed — aborting auto-apply.")
            browser.close()
            summary["failed"] = candidates
            return summary

        for job in candidates:
            try:
                status = _apply_one(page, job, resume_text, resume_pdf_path)
            except Exception as exc:
                logger.error("Unhandled error on '%s': %s", job["title"], exc)
                status = "failed"
                record_application(job, "failed", str(exc))

            summary[status].append(job)
            job["auto_applied"] = status == "applied"
            time.sleep(_APPLY_DELAY)

        browser.close()

    logger.info(
        "Auto-apply done — applied: %d  skipped: %d  failed: %d",
        len(summary["applied"]), len(summary["skipped"]), len(summary["failed"]),
    )
    return summary


# ── Login ────────────────────────────────────────────────────────────────────

def _login(page: Page, email: str, password: str) -> bool:
    try:
        page.goto("https://secure.indeed.com/account/login", timeout=30_000)
        page.wait_for_load_state("networkidle", timeout=15_000)

        if _already_logged_in(page):
            logger.info("Indeed: already logged in.")
            return True

        page.fill('input[name="__email"]', email)
        page.click('button[type="submit"]')
        page.wait_for_load_state("networkidle", timeout=10_000)

        pw_field = page.query_selector('input[name="__password"]')
        if pw_field:
            pw_field.fill(password)
            page.click('button[type="submit"]')
            page.wait_for_load_state("networkidle", timeout=15_000)

        if _has_captcha(page):
            logger.warning("CAPTCHA on Indeed login. Solve it in the browser, then press Enter.")
            input(">>> Press Enter after solving the CAPTCHA <<<")

        return _already_logged_in(page)

    except PWTimeout:
        logger.error("Indeed login timed out.")
        return False


def _already_logged_in(page: Page) -> bool:
    return (
        "myaccount" in page.url
        or "dashboard" in page.url
        or page.query_selector('[data-gnav-element-name="SignIn"]') is None
    )


# ── Single application ────────────────────────────────────────────────────────

def _apply_one(
    page: Page,
    job: dict,
    resume_text: str,
    resume_pdf_path: Optional[str],
) -> str:
    url = job["link"]
    title = job["title"]

    try:
        page.goto(url, timeout=30_000)
        page.wait_for_load_state("networkidle", timeout=15_000)

        apply_btn = _find_apply_button(page)
        if not apply_btn:
            logger.info("No Apply button found — skipping '%s'.", title)
            record_application(job, "skipped", "no apply button")
            return "skipped"

        with page.expect_popup(timeout=5_000) as popup_info:
            apply_btn.click()
        new_page = popup_info.value
        new_page.wait_for_load_state("networkidle", timeout=15_000)
        apply_page = new_page
    except Exception:
        # No popup — apply stays on same page
        try:
            apply_btn = _find_apply_button(page)
            if apply_btn:
                apply_btn.click()
                page.wait_for_load_state("networkidle", timeout=15_000)
            apply_page = page
        except Exception as exc:
            logger.warning("Could not click apply for '%s': %s", title, exc)
            record_application(job, "failed", str(exc))
            return "failed"

    # If redirected off Indeed it's an external application
    if "indeed.com" not in apply_page.url:
        logger.info("External ATS detected for '%s' — skipping.", title)
        record_application(job, "skipped", "external ATS")
        return "skipped"

    if _has_captcha(apply_page):
        logger.warning("CAPTCHA on application for '%s' — skipping.", title)
        record_application(job, "skipped", "captcha")
        return "skipped"

    # Work through multi-step form (max 10 steps)
    for _ in range(10):
        if _is_submitted(apply_page):
            logger.info("Applied to '%s' at %s.", title, job["company"])
            record_application(job, "applied")
            return "applied"

        _fill_step(apply_page, job, resume_text, resume_pdf_path)

        next_btn = (
            apply_page.query_selector('button:has-text("Continue")')
            or apply_page.query_selector('button:has-text("Next")')
            or apply_page.query_selector('button:has-text("Submit")')
            or apply_page.query_selector('button[type="submit"]')
        )
        if not next_btn:
            break

        next_btn.click()
        apply_page.wait_for_load_state("networkidle", timeout=10_000)

    if _is_submitted(apply_page):
        logger.info("Applied to '%s' at %s.", title, job["company"])
        record_application(job, "applied")
        return "applied"

    logger.warning("Could not complete application for '%s'.", title)
    record_application(job, "failed", "form not completed")
    return "failed"


# ── Form helpers ──────────────────────────────────────────────────────────────

def _fill_step(
    page: Page,
    job: dict,
    resume_text: str,
    resume_pdf_path: Optional[str],
) -> None:
    # Resume upload
    if resume_pdf_path and os.path.exists(resume_pdf_path):
        file_input = page.query_selector('input[type="file"]')
        if file_input:
            file_input.set_input_files(resume_pdf_path)
            time.sleep(1)

    # Text inputs and textareas with visible labels (screening questions)
    for field in page.query_selector_all('input[type="text"]:visible, textarea:visible'):
        if field.input_value():
            continue
        label = _label_for(page, field)
        if not label:
            continue
        answer = _gpt(label, job, resume_text)
        field.fill(answer)
        time.sleep(0.3)

    # Number inputs (years of experience etc.)
    for field in page.query_selector_all('input[type="number"]:visible'):
        if field.input_value():
            continue
        label = _label_for(page, field)
        if not label:
            continue
        answer = _gpt(label, job, resume_text)
        digits = "".join(c for c in answer if c.isdigit())
        if digits:
            field.fill(digits[:3])

    # Select dropdowns — pick first non-placeholder option
    for sel in page.query_selector_all('select:visible'):
        try:
            options = sel.query_selector_all('option')
            for opt in options[1:]:
                val = opt.get_attribute("value")
                if val:
                    sel.select_option(val)
                    break
        except Exception:
            pass

    # Radio groups — prefer "Yes" where it makes sense
    for radio in page.query_selector_all('input[type="radio"]:visible'):
        label = _label_for(page, radio)
        if label and "yes" in label.lower():
            try:
                radio.check()
            except Exception:
                pass


def _label_for(page: Page, element) -> str:
    el_id = element.get_attribute("id")
    if el_id:
        lbl = page.query_selector(f'label[for="{el_id}"]')
        if lbl:
            return lbl.inner_text().strip()
    try:
        return element.evaluate("""el => {
            const p = el.closest('label')
                     || el.closest('[class*="field"]')
                     || el.closest('[class*="question"]')
                     || el.parentElement;
            return p ? p.innerText.trim().split('\\n')[0] : '';
        }""")
    except Exception:
        return ""


def _find_apply_button(page: Page):
    selectors = [
        'button:has-text("Apply now")',
        'a:has-text("Apply now")',
        '[id*="applyButton"]',
        '[data-testid*="apply"]',
        'button:has-text("Easy Apply")',
    ]
    for sel in selectors:
        btn = page.query_selector(sel)
        if btn:
            return btn
    return None


def _has_captcha(page: Page) -> bool:
    return bool(
        page.query_selector('iframe[src*="recaptcha"]')
        or page.query_selector('[class*="captcha"]')
        or page.query_selector('iframe[title*="CAPTCHA"]')
    )


def _is_submitted(page: Page) -> bool:
    phrases = [
        "application submitted",
        "application was sent",
        "you applied",
        "successfully applied",
        "thank you for applying",
    ]
    try:
        content = page.content().lower()
        return any(p in content for p in phrases)
    except Exception:
        return False
