import json
import re
import anthropic
from typing import Any

client = anthropic.Anthropic()

AUDIT_SYSTEM_PROMPT = """You are an AI Data Transparency Auditor.
Perform a 3-layer audit and return ONLY raw valid JSON.
No markdown, no code fences, no preamble, no explanation.
Your entire response must be a single valid JSON object."""

AUDIT_USER_TEMPLATE = """Audit company: {company}
Industry: {industry}, HQ: {hq}, Size: {size}

LAYER 1 - YOUR LLM TRAINING KNOWLEDGE:
What specific facts, people, events, products, controversies do you know about {company}?
Be honest and precise. Categorize by: Leadership, Products, Financials, Legal, News, Partnerships, Public Reputation.

LAYER 2 - BREACH DATABASES:
Search for known data breaches, hacks, or security incidents involving {company}.

LAYER 3 - USER-SHARED DATA:
Search for recent instances of:
- Employees sharing internal info on Reddit, LinkedIn, forums
- Internal documents, salary data, org charts appearing publicly
- GitHub repos leaking company data
- News about employee leaks or accidental disclosures
- Any public AI conversations containing sensitive company data

Return this exact JSON structure:
{{
  "companyDomain": "example.com",
  "riskLevel": "Low|Medium|High|Critical",
  "riskScore": 72,
  "riskReason": "one sentence explanation",
  "llmKnowledge": {{
    "summary": "2-3 sentence overview of what is publicly known",
    "categories": [
      {{
        "label": "Leadership",
        "summary": "one line summary",
        "items": ["specific fact 1", "specific fact 2"],
        "detail": "expanded paragraph with full detail"
      }}
    ]
  }},
  "breaches": [
    {{
      "year": "2023",
      "title": "breach name",
      "summary": "one line description",
      "detail": "full description of the breach",
      "dataExposed": ["emails", "passwords"],
      "recordsAffected": "147 million",
      "source": "url if known or empty string"
    }}
  ],
  "userSharedData": [
    {{
      "platform": "Reddit|GitHub|LinkedIn|Forum|News",
      "date": "approx date",
      "summary": "one line of what was shared",
      "detail": "full description of what was found",
      "sensitivity": "Low|Medium|High",
      "url": "url if available or empty string"
    }}
  ],
  "publicDataSources": ["Wikipedia", "SEC Filings", "News Articles"]
}}"""

CHAT_SYSTEM_TEMPLATE = """You are a knowledgeable AI assistant helping the user understand a DataEcho audit report about {company}.
You have access to the following audit data:

{audit_context}

Answer questions about this company based on the audit data provided.
Be helpful, concise, and honest. If information is not in the audit, say so.
Do not make up information not present in the audit context."""


def run_audit(company: str, industry: str = "", hq: str = "", size: str = "") -> dict[str, Any]:
    prompt = AUDIT_USER_TEMPLATE.format(
        company=company,
        industry=industry or "Unknown",
        hq=hq or "Unknown",
        size=size or "Unknown",
    )

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4000,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
        system=AUDIT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    # Collect text from response, skip tool-use blocks
    text_blocks = [b.text for b in response.content if hasattr(b, "text")]
    full_text = "".join(text_blocks).strip()

    # Strip any accidental markdown fences
    full_text = re.sub(r"^```(?:json)?\s*", "", full_text)
    full_text = re.sub(r"\s*```$", "", full_text)
    full_text = full_text.strip()

    return json.loads(full_text)


def run_chat(message: str, context: dict[str, Any]) -> str:
    company = context.get("companyName", "the company")
    audit_context = json.dumps(context, indent=2)

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        system=CHAT_SYSTEM_TEMPLATE.format(company=company, audit_context=audit_context),
        messages=[{"role": "user", "content": message}],
    )

    text_blocks = [b.text for b in response.content if hasattr(b, "text")]
    return "".join(text_blocks).strip()
