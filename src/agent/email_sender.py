import logging
import os
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

SCORE_COLORS = {10: "#1a7f37", 9: "#1a7f37", 8: "#2d7dd2", 7: "#f0a500"}


def send_digest(jobs: list[dict]) -> None:
    gmail_user = os.environ["GMAIL_USER"]
    app_password = os.environ["GMAIL_APP_PASSWORD"]
    recipient = os.environ["RECIPIENT_EMAIL"]

    subject = f"Job Digest — {len(jobs)} matches found ({date.today().strftime('%b %d, %Y')})"
    html = _build_html(jobs)
    plain = _build_plain(jobs)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Job Search Agent <{gmail_user}>"
    msg["To"] = recipient
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_user, app_password)
        server.sendmail(gmail_user, recipient, msg.as_string())

    logger.info("Digest sent to %s (%d jobs)", recipient, len(jobs))


def _build_html(jobs: list[dict]) -> str:
    today = date.today().strftime("%B %d, %Y")
    cards = "".join(_job_card(j) for j in jobs)
    dashboard_url = os.getenv("DASHBOARD_URL", "http://localhost:5000")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f5f5; margin: 0; padding: 20px; color: #333; }}
  .wrapper {{ max-width: 680px; margin: 0 auto; }}
  .header {{ background: #1a1a2e; color: #fff; padding: 28px 32px; border-radius: 12px 12px 0 0; }}
  .header h1 {{ margin: 0 0 4px; font-size: 22px; }}
  .header p  {{ margin: 0; opacity: .75; font-size: 14px; }}
  .card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 10px;
           padding: 20px 24px; margin: 12px 0; }}
  .card-title {{ font-size: 17px; font-weight: 700; margin: 0 0 4px; color: #1a1a2e; }}
  .card-meta  {{ font-size: 13px; color: #666; margin: 0 0 10px; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px;
            font-size: 12px; font-weight: 700; color: #fff; margin-right: 6px; }}
  .salary {{ display: inline-block; padding: 3px 10px; border-radius: 20px;
             font-size: 12px; background: #f0f0f0; color: #555; }}
  .reason {{ font-size: 13px; color: #555; margin: 10px 0 14px;
             border-left: 3px solid #e0e0e0; padding-left: 10px; font-style: italic; }}
  .actions {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }}
  .btn {{ display: inline-block; text-decoration: none; padding: 9px 18px;
          border-radius: 6px; font-size: 14px; font-weight: 600; }}
  .btn-green {{ background: #1a7f37; color: #fff; }}
  .btn-blue  {{ background: #2d7dd2; color: #fff; }}
  .btn-gray  {{ background: #f0f0f0; color: #666; border: 1px solid #ddd; }}
  .footer {{ text-align: center; font-size: 12px; color: #aaa;
             padding: 20px; border-top: 1px solid #e8e8e8; margin-top: 8px; }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <h1>Your Daily Job Digest</h1>
    <p>{today} &nbsp;·&nbsp; {len(jobs)} relevant opening{"s" if len(jobs) != 1 else ""} found</p>
  </div>
  {cards}
  <div class="footer">
    Powered by CodeShali &nbsp;·&nbsp;
    <a href="{dashboard_url}" style="color:#2d7dd2;">Open Dashboard</a>
  </div>
</div>
</body>
</html>"""


def _job_card(job: dict) -> str:
    score = job.get("score", 0)
    color = SCORE_COLORS.get(score, "#888")
    salary = job.get("salary", "Not disclosed")
    salary_html = f'<span class="salary">{salary}</span>' if salary != "Not disclosed" else ""
    reason = job.get("score_reason", "")
    reason_html = f'<p class="reason">{reason}</p>' if reason else ""
    site = job.get("site", "").capitalize()
    date_posted = job.get("date_posted", "")
    meta_parts = [p for p in [job.get("company", ""), job.get("location", ""), site, date_posted] if p]

    dashboard_url = os.getenv("DASHBOARD_URL", "http://localhost:5000")
    job_id = job.get("db_id", "")
    interested_url = f"{dashboard_url}/feedback?id={job_id}&action=interested" if job_id else ""
    skip_url = f"{dashboard_url}/feedback?id={job_id}&action=skipped" if job_id else ""

    feedback_btns = ""
    if interested_url:
        feedback_btns = (
            f'<a class="btn btn-blue" href="{interested_url}">★ Interested</a>'
            f'<a class="btn btn-gray" href="{skip_url}">Skip</a>'
        )

    return f"""
<div class="card">
  <p class="card-title">{_esc(job.get("title",""))}</p>
  <p class="card-meta">{" &nbsp;·&nbsp; ".join(_esc(p) for p in meta_parts)}</p>
  <span class="badge" style="background:{color}">Match {score}/10</span>
  {salary_html}
  {reason_html}
  <div class="actions">
    <a class="btn btn-green" href="{job.get('link','#')}" target="_blank">View &amp; Apply →</a>
    {feedback_btns}
  </div>
</div>"""


def _build_plain(jobs: list[dict]) -> str:
    today = date.today().strftime("%B %d, %Y")
    lines = [f"Your Daily Job Digest — {today}", f"{len(jobs)} relevant openings found", "=" * 60, ""]
    for j in jobs:
        lines += [
            f"{j.get('title','')} at {j.get('company','')}",
            f"Location : {j.get('location','')}",
            f"Salary   : {j.get('salary','Not disclosed')}",
            f"Score    : {j.get('score',0)}/10 — {j.get('score_reason','')}",
            f"Apply    : {j.get('link','')}",
            f"Source   : {j.get('site','')}  |  Posted: {j.get('date_posted','')}",
            "-" * 60,
            "",
        ]
    return "\n".join(lines)


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )
