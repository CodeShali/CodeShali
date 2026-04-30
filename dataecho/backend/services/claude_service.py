import json
import re
import anthropic
from dataclasses import dataclass
from typing import Any

client = anthropic.Anthropic()

# Per-plan configuration: model, token limits, search uses
@dataclass
class PlanConfig:
    model: str
    max_tokens: int
    search_uses: int

PLAN_CONFIGS: dict[str, PlanConfig] = {
    "FREE":       PlanConfig(model="claude-haiku-4-5",  max_tokens=1500, search_uses=2),
    "PRO":        PlanConfig(model="claude-sonnet-4-6", max_tokens=3000, search_uses=5),
    "ENTERPRISE": PlanConfig(model="claude-opus-4-7",   max_tokens=6000, search_uses=10),
}

# Cost per million tokens (input, output) in USD
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5":  (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7":   (5.00, 25.00),
}

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


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    cost_per_m = MODEL_COSTS.get(model, (5.00, 25.00))
    return round(
        (input_tokens / 1_000_000) * cost_per_m[0] +
        (output_tokens / 1_000_000) * cost_per_m[1],
        6,
    )


def run_audit(
    company: str,
    industry: str = "",
    hq: str = "",
    size: str = "",
    plan: str = "FREE",
) -> dict[str, Any]:
    config = PLAN_CONFIGS.get(plan, PLAN_CONFIGS["FREE"])
    prompt = AUDIT_USER_TEMPLATE.format(
        company=company,
        industry=industry or "Unknown",
        hq=hq or "Unknown",
        size=size or "Unknown",
    )

    response = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": config.search_uses}],
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

    audit_data = json.loads(full_text)

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost_usd = _calculate_cost(config.model, input_tokens, output_tokens)

    return {
        "result": audit_data,
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "costUsd": cost_usd,
            "model": config.model,
        },
    }


def run_chat(message: str, context: dict[str, Any]) -> str:
    company = context.get("companyName", "the company")
    audit_context = json.dumps(context, indent=2)

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=800,
        system=CHAT_SYSTEM_TEMPLATE.format(company=company, audit_context=audit_context),
        messages=[{"role": "user", "content": message}],
    )

    text_blocks = [b.text for b in response.content if hasattr(b, "text")]
    return "".join(text_blocks).strip()
