"""Flask web server: job dashboard + feedback endpoint."""

import logging
import os
import threading

from flask import Flask, redirect, render_template_string, request

from agent import database as db

logger = logging.getLogger(__name__)

app = Flask(__name__)
_resume_text: str = ""

_gen_lock = threading.Lock()
_generating: set[int] = set()

# ── Dashboard template ────────────────────────────────────────────────────────

_DASH = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CodeShali — Job Dashboard</title>
<style>
  *,*::before,*::after{box-sizing:border-box}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       background:#f0f2f5;margin:0;color:#333}
  a{color:inherit}
  header{background:#1a1a2e;color:#fff;padding:20px 32px;
         display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px}
  header h1{margin:0;font-size:20px;font-weight:700}
  .stats{display:flex;gap:24px;flex-wrap:wrap}
  .stat{text-align:center}
  .stat-n{font-size:22px;font-weight:700}
  .stat-l{font-size:11px;opacity:.65;text-transform:uppercase;letter-spacing:.5px}
  nav{background:#fff;border-bottom:1px solid #e0e0e0;padding:0 32px;
      display:flex;gap:2px;overflow-x:auto}
  nav a{display:block;padding:13px 16px;text-decoration:none;color:#666;font-size:14px;
        font-weight:500;border-bottom:3px solid transparent;white-space:nowrap}
  nav a:hover{color:#1a1a2e}
  nav a.on{color:#1a1a2e;border-bottom-color:#1a7f37;font-weight:700}
  .wrap{max-width:860px;margin:0 auto;padding:24px 20px}
  .lbl{font-size:13px;color:#999;margin:0 0 14px}
  .card{background:#fff;border:1px solid #e0e0e0;border-radius:10px;
        padding:20px 24px;margin-bottom:12px;transition:box-shadow .15s}
  .card:hover{box-shadow:0 2px 12px rgba(0,0,0,.07)}
  .card.interested{border-left:4px solid #1a7f37}
  .card.skipped{border-left:4px solid #ccc;opacity:.55}
  .card.applied{border-left:4px solid #2d7dd2}
  .ch{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:6px}
  .ct{font-size:16px;font-weight:700;margin:0;color:#1a1a2e}
  .cm{font-size:13px;color:#666;margin:2px 0 10px}
  .chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}
  .chip{display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600;color:#fff}
  .chip-s{background:#f0f0f0;color:#555;font-weight:500}
  .chip-i{background:#e8f5e9;color:#1a7f37}
  .chip-a{background:#e3f2fd;color:#1565c0}
  .chip-cl{background:#f3e5f5;color:#7b1fa2}
  .reason{font-size:13px;color:#555;margin:6px 0 12px;
          border-left:3px solid #e8e8e8;padding-left:10px;font-style:italic}
  .acts{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
  .btn{display:inline-block;padding:7px 15px;border-radius:6px;font-size:13px;
       font-weight:600;text-decoration:none;white-space:nowrap}
  .btn-g{background:#1a7f37;color:#fff}
  .btn-b{background:#2d7dd2;color:#fff}
  .btn-o{border:1px solid #ddd;color:#555;background:#fff}
  .btn-p{border:1px solid #d0a0e0;color:#7b1fa2;background:#fff}
  .date{font-size:11px;background:#f5f5f5;color:#999;padding:2px 8px;border-radius:20px;white-space:nowrap}
  .empty{text-align:center;padding:60px 20px;color:#bbb}
  .empty h3{font-size:18px;color:#ccc;margin-bottom:6px}
  .flash{background:#e8f5e9;border:1px solid #a5d6a7;border-radius:8px;
         padding:12px 20px;margin-bottom:16px;font-size:14px;color:#2e7d32}
</style>
</head>
<body>
<header>
  <h1>CodeShali — Job Dashboard</h1>
  <div class="stats">
    <div class="stat"><div class="stat-n">{{ stats.today }}</div><div class="stat-l">Today</div></div>
    <div class="stat"><div class="stat-n">{{ stats.total }}</div><div class="stat-l">Total</div></div>
    <div class="stat"><div class="stat-n">{{ stats.interested }}</div><div class="stat-l">Interested</div></div>
    <div class="stat"><div class="stat-n">{{ stats.applied }}</div><div class="stat-l">Applied</div></div>
    <div class="stat"><div class="stat-n">{{ stats.cover_letters }}</div><div class="stat-l">Cover Letters</div></div>
  </div>
</header>
<nav>
  <a href="/?filter=all" class="{{ 'on' if filter=='all' else '' }}">All ({{ stats.total }})</a>
  <a href="/?filter=today" class="{{ 'on' if filter=='today' else '' }}">Today ({{ stats.today }})</a>
  <a href="/?filter=interested" class="{{ 'on' if filter=='interested' else '' }}">Interested ({{ stats.interested }})</a>
  <a href="/?filter=skipped" class="{{ 'on' if filter=='skipped' else '' }}">Skipped ({{ stats.skipped }})</a>
  <a href="/?filter=applied" class="{{ 'on' if filter=='applied' else '' }}">Applied ({{ stats.applied }})</a>
</nav>
<div class="wrap">
  {% if msg %}<div class="flash">{{ msg | e }}</div>{% endif %}
  <p class="lbl">{{ jobs|length }} job{{ 's' if jobs|length != 1 else '' }}</p>
  {% if not jobs %}
  <div class="empty">
    <h3>No jobs here yet</h3>
    <p>Jobs will appear after the next pipeline run.</p>
  </div>
  {% endif %}
  {% for j in jobs %}
  {% set score = j.score or 0 %}
  {% if score >= 9 %}{% set sc = '#1a7f37' %}{% elif score >= 7 %}{% set sc = '#2d7dd2' %}{% else %}{% set sc = '#f0a500' %}{% endif %}
  <div class="card {{ j.status if j.status in ('interested','skipped','applied') else '' }}">
    <div class="ch">
      <div>
        <p class="ct">{{ j.title | e }}</p>
        <p class="cm">{{ j.company | e }}{% if j.location %} &nbsp;·&nbsp; {{ j.location | e }}{% endif %}{% if j.site %} &nbsp;·&nbsp; {{ j.site | capitalize }}{% endif %}{% if j.date_posted %} &nbsp;·&nbsp; posted {{ j.date_posted | e }}{% endif %}</p>
      </div>
      <span class="date">{{ j.found_date }}</span>
    </div>
    <div class="chips">
      {% if score %}<span class="chip" style="background:{{ sc }}">Match {{ score }}/10</span>{% endif %}
      {% if j.salary and j.salary != 'Not disclosed' %}<span class="chip chip-s">{{ j.salary | e }}</span>{% endif %}
      {% if j.status == 'interested' %}<span class="chip chip-i">★ Interested</span>{% elif j.status == 'applied' %}<span class="chip chip-a">✓ Applied</span>{% endif %}
      {% if j.has_cover_letter %}<span class="chip chip-cl">Cover Letter Ready</span>{% endif %}
    </div>
    {% if j.score_reason %}<p class="reason">{{ j.score_reason | e }}</p>{% endif %}
    <div class="acts">
      <a href="{{ j.url | e }}" class="btn btn-g" target="_blank" rel="noopener">View &amp; Apply →</a>
      {% if j.status != 'interested' %}
      <a href="/feedback?id={{ j.id }}&action=interested&ref={{ filter }}" class="btn btn-b">★ Interested</a>
      {% endif %}
      {% if j.status == 'interested' %}
      <a href="/cover-letter/{{ j.id }}" class="btn btn-p">
        {{ 'View Cover Letter' if j.has_cover_letter else 'Generate Cover Letter' }}
      </a>
      {% endif %}
      {% if j.status == 'interested' %}
      <a href="/feedback?id={{ j.id }}&action=applied&ref={{ filter }}" class="btn btn-o" style="border-color:#90caf9;color:#1565c0;">Mark Applied</a>
      {% endif %}
      {% if j.status not in ('skipped','applied') %}
      <a href="/feedback?id={{ j.id }}&action=skipped&ref={{ filter }}" class="btn btn-o">Skip</a>
      {% endif %}
    </div>
  </div>
  {% endfor %}
</div>
</body>
</html>"""

# ── Cover letter template ─────────────────────────────────────────────────────

_CL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{% if generating %}<meta http-equiv="refresh" content="4">{% endif %}
<title>Cover Letter — {{ job.title | e }} at {{ job.company | e }}</title>
<style>
  *,*::before,*::after{box-sizing:border-box}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       background:#f0f2f5;margin:0;color:#333}
  header{background:#1a1a2e;color:#fff;padding:18px 32px;
         display:flex;align-items:center;gap:18px}
  header a{color:#aaa;text-decoration:none;font-size:14px}
  header a:hover{color:#fff}
  header h2{margin:0;font-size:18px}
  .wrap{max-width:860px;margin:0 auto;padding:28px 20px}
  .jinfo{background:#fff;border:1px solid #e0e0e0;border-radius:10px;
         padding:16px 24px;margin-bottom:20px;
         display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}
  .jinfo h3{margin:0 0 4px;font-size:16px;color:#1a1a2e}
  .jinfo p{margin:0;font-size:13px;color:#666}
  .cl-box{background:#fff;border:1px solid #e0e0e0;border-radius:10px;
          padding:32px;white-space:pre-wrap;font-size:15px;line-height:1.75;color:#333}
  .acts{display:flex;gap:10px;margin-top:20px;flex-wrap:wrap}
  .btn{padding:9px 20px;border-radius:6px;font-size:14px;font-weight:600;
       text-decoration:none;border:none;cursor:pointer;display:inline-block}
  .btn-g{background:#1a7f37;color:#fff}
  .btn-o{border:1px solid #ddd;color:#555;background:#fff}
  .btn-r{border:1px solid #f0a500;color:#f0a500;background:#fff}
  .loading{text-align:center;padding:60px 20px;color:#aaa}
  .loading p{font-size:16px;margin:0 0 8px}
  .spinner{display:inline-block;width:32px;height:32px;border:3px solid #e0e0e0;
           border-top-color:#1a7f37;border-radius:50%;animation:spin 1s linear infinite;margin-bottom:16px}
  @keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<header>
  <a href="/">← Dashboard</a>
  <h2>Cover Letter</h2>
</header>
<div class="wrap">
  <div class="jinfo">
    <div>
      <h3>{{ job.title | e }}</h3>
      <p>{{ job.company | e }}{% if job.location %} &nbsp;·&nbsp; {{ job.location | e }}{% endif %}</p>
    </div>
    <a href="{{ job.url | e }}" target="_blank" rel="noopener" class="btn btn-g">View Job →</a>
  </div>

  {% if generating %}
  <div class="loading">
    <div class="spinner"></div>
    <p>Generating your cover letter…</p>
    <p style="font-size:13px;color:#bbb">This page will refresh automatically in a few seconds.</p>
  </div>
  {% elif cover_letter %}
  <div class="cl-box">{{ cover_letter | e }}</div>
  <div class="acts">
    <button onclick="navigator.clipboard.writeText(document.querySelector('.cl-box').innerText).then(()=>this.textContent='Copied!')" class="btn btn-o">Copy to Clipboard</button>
    <a href="/cover-letter/{{ job.id }}/regenerate" class="btn btn-r">↺ Regenerate</a>
    <a href="/" class="btn btn-o">← Back</a>
  </div>
  {% else %}
  <div class="loading">
    <p>No cover letter yet.</p>
    <a href="/cover-letter/{{ job.id }}/generate" class="btn btn-g" style="margin-top:12px;">Generate Now</a>
  </div>
  {% endif %}
</div>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return "OK", 200


@app.route("/")
def dashboard():
    f = request.args.get("filter", "all")
    msg = request.args.get("msg", "")
    jobs = db.get_jobs(filter_by=f)
    stats = db.get_stats()
    return render_template_string(_DASH, jobs=jobs, stats=stats, filter=f, msg=msg)


@app.route("/feedback")
def feedback():
    job_id = request.args.get("id", type=int)
    action = request.args.get("action", "")
    ref = request.args.get("ref", "all")

    if not job_id or action not in ("interested", "skipped", "applied"):
        return "Invalid request", 400

    db.set_status(job_id, action)

    if action == "interested" and os.getenv("ENABLE_COVER_LETTERS", "true").lower() == "true":
        _trigger_cover_letter(job_id)

    msgs = {
        "interested": "Marked as interested — cover letter being generated.",
        "skipped": "Job skipped.",
        "applied": "Marked as applied!",
    }
    return redirect(f"/?filter={ref}&msg={msgs[action]}")


@app.route("/cover-letter/<int:job_id>")
@app.route("/cover-letter/<int:job_id>/generate")
def cover_letter(job_id: int):
    job = db.get_job(job_id)
    if not job:
        return "Job not found", 404

    cl = db.get_cover_letter(job_id)
    generating = job_id in _generating

    if not cl and not generating:
        _trigger_cover_letter(job_id)
        generating = True

    return render_template_string(
        _CL,
        job=job,
        cover_letter=cl["content"] if cl else None,
        generating=generating,
    )


@app.route("/cover-letter/<int:job_id>/regenerate")
def regenerate_cover_letter(job_id: int):
    if not db.get_job(job_id):
        return "Job not found", 404
    db.delete_cover_letters(job_id)
    _trigger_cover_letter(job_id)
    return redirect(f"/cover-letter/{job_id}")


# ── Background cover-letter generation ───────────────────────────────────────

def _trigger_cover_letter(job_id: int) -> None:
    with _gen_lock:
        if job_id in _generating:
            return
        _generating.add(job_id)

    def _run():
        try:
            from agent.cover_letter_generator import generate_cover_letter
            generate_cover_letter(job_id, _resume_text)
        finally:
            with _gen_lock:
                _generating.discard(job_id)

    threading.Thread(target=_run, daemon=True).start()


# ── Server startup ────────────────────────────────────────────────────────────

def start_web_server(resume_text: str = "") -> None:
    global _resume_text
    _resume_text = resume_text

    port = int(os.getenv("DASHBOARD_PORT", "5000"))

    def _run():
        import logging as _log
        _log.getLogger("werkzeug").setLevel(_log.WARNING)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    try:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        logger.info("Dashboard at http://0.0.0.0:%d", port)
    except Exception as exc:
        logger.warning("Web server failed to start: %s", exc)
