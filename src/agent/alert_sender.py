import json
import logging
import os
import smtplib
import urllib.request
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


def send_alerts(high_score_jobs: list[dict]) -> None:
    if not high_score_jobs:
        return
    slack_url = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if slack_url:
        _send_slack(high_score_jobs, slack_url)
    else:
        _send_email_alert(high_score_jobs)


def _send_slack(jobs: list[dict], webhook_url: str) -> None:
    dashboard_url = os.getenv("DASHBOARD_URL", "http://localhost:5000")
    blocks: list[dict] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*🔥 {len(jobs)} high-match job{'s' if len(jobs)!=1 else ''} found today!*",
            },
        }
    ]
    for j in jobs:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*{j.get('title','')}* at *{j.get('company','')}*\n"
                    f"{j.get('location','')} · Score: *{j.get('score',0)}/10*\n"
                    f"_{j.get('score_reason','')}_"
                ),
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "View Job"},
                "url": j.get("link", dashboard_url),
            },
        })

    payload = json.dumps({"blocks": blocks}).encode()
    req = urllib.request.Request(
        webhook_url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        logger.info("Slack alert sent (%d jobs)", len(jobs))
    except Exception as exc:
        logger.warning("Slack alert failed: %s", exc)


def _send_email_alert(jobs: list[dict]) -> None:
    gmail_user = os.environ["GMAIL_USER"]
    app_password = os.environ["GMAIL_APP_PASSWORD"]
    recipient = os.getenv("ALERT_EMAIL", os.environ["RECIPIENT_EMAIL"])
    dashboard_url = os.getenv("DASHBOARD_URL", "http://localhost:5000")
    threshold = os.getenv("ALERT_SCORE_THRESHOLD", "9")

    subject = f"🔥 {len(jobs)} High-Match Job{'s' if len(jobs)!=1 else ''} — {date.today().strftime('%b %d, %Y')}"

    cards_html = ""
    for j in jobs:
        job_id = j.get("db_id", "")
        interested_url = f"{dashboard_url}/feedback?id={job_id}&action=interested" if job_id else "#"
        cards_html += f"""
<div style="background:#fff;border:2px solid #1a7f37;border-radius:10px;padding:20px 24px;margin:12px 0;">
  <p style="font-size:17px;font-weight:700;margin:0 0 4px;color:#1a1a2e;">{j.get('title','')}</p>
  <p style="font-size:13px;color:#666;margin:0 0 10px;">
    {j.get('company','')} &nbsp;·&nbsp; {j.get('location','')} &nbsp;·&nbsp; {j.get('site','').capitalize()}
  </p>
  <span style="background:#1a7f37;color:#fff;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700;">
    Match {j.get('score',0)}/10
  </span>
  <p style="font-size:13px;color:#555;margin:10px 0 14px;border-left:3px solid #1a7f37;padding-left:10px;">
    {j.get('score_reason','')}
  </p>
  <a href="{j.get('link','#')}"
     style="background:#1a7f37;color:#fff;text-decoration:none;padding:9px 18px;border-radius:6px;font-size:14px;font-weight:600;margin-right:8px;">
    View &amp; Apply →
  </a>
  <a href="{interested_url}"
     style="background:#2d7dd2;color:#fff;text-decoration:none;padding:9px 18px;border-radius:6px;font-size:14px;font-weight:600;">
    ★ Mark Interested
  </a>
</div>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f5;padding:20px;color:#333;">
<div style="max-width:680px;margin:0 auto;">
  <div style="background:#1a7f37;color:#fff;padding:24px 32px;border-radius:12px 12px 0 0;">
    <h1 style="margin:0 0 4px;font-size:20px;">🔥 High-Match Job Alert</h1>
    <p style="margin:0;opacity:.85;font-size:14px;">
      {len(jobs)} job{'s' if len(jobs)!=1 else ''} scored {threshold}+ out of 10
    </p>
  </div>
  {cards_html}
  <p style="text-align:center;font-size:12px;color:#aaa;padding:20px;">
    <a href="{dashboard_url}" style="color:#2d7dd2;">Open Dashboard</a>
  </p>
</div>
</body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Job Alert Agent <{gmail_user}>"
    msg["To"] = recipient
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_user, app_password)
        server.sendmail(gmail_user, recipient, msg.as_string())

    logger.info("Alert email sent for %d high-score jobs to %s", len(jobs), recipient)
