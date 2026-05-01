import asyncio
import logging
import os
import socket
from typing import Any

import httpx

logger = logging.getLogger(__name__)

HIBP_API_KEY = os.getenv("HIBP_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

_HEADERS = {"user-agent": "DataEcho/1.0"}


async def fetch_hibp_breaches(domain: str) -> list[dict[str, Any]]:
    """Query HIBP for verified data breaches associated with a domain (free endpoint)."""
    if not domain:
        return []
    try:
        headers = dict(_HEADERS)
        if HIBP_API_KEY:
            headers["hibp-api-key"] = HIBP_API_KEY
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://haveibeenpwned.com/api/v3/breaches",
                params={"domain": domain},
                headers=headers,
            )
            if r.status_code == 404:
                return []
            r.raise_for_status()
            breaches = r.json()
            return [
                {
                    "name": b.get("Name", ""),
                    "title": b.get("Title", ""),
                    "date": b.get("BreachDate", ""),
                    "recordCount": b.get("PwnCount", 0),
                    "dataClasses": b.get("DataClasses", []),
                    "isVerified": b.get("IsVerified", False),
                    "description": b.get("Description", ""),
                }
                for b in breaches
                if isinstance(b, dict)
            ]
    except Exception as e:
        logger.warning("HIBP fetch failed for %s: %s", domain, e)
        return []


async def fetch_crt_subdomains(domain: str) -> list[str]:
    """Query crt.sh certificate transparency logs to discover exposed subdomains (free, no key)."""
    if not domain:
        return []
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get(
                "https://crt.sh/",
                params={"q": f"%.{domain}", "output": "json"},
                headers=_HEADERS,
            )
            r.raise_for_status()
            entries = r.json()
            seen: set[str] = set()
            for entry in entries:
                name = entry.get("name_value", "")
                for sub in name.splitlines():
                    sub = sub.strip().lstrip("*.")
                    if sub and sub.endswith(domain) and sub != domain:
                        seen.add(sub)
            # Return sorted unique subdomains, capped at 60
            return sorted(seen)[:60]
    except Exception as e:
        logger.warning("crt.sh fetch failed for %s: %s", domain, e)
        return []


async def fetch_github_leaks(company: str, domain: str) -> list[dict[str, str]]:
    """Search GitHub public code for files referencing the company domain (may reveal leaks)."""
    query = domain or company
    if not query:
        return []
    try:
        headers = dict(_HEADERS)
        headers["Accept"] = "application/vnd.github+json"
        if GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://api.github.com/search/code",
                params={"q": f"{query} in:file", "per_page": 10},
                headers=headers,
            )
            if r.status_code in (403, 422):
                return []
            r.raise_for_status()
            data = r.json()
            return [
                {
                    "repo": item.get("repository", {}).get("full_name", ""),
                    "file": item.get("path", ""),
                    "url": item.get("html_url", ""),
                    "repoUrl": item.get("repository", {}).get("html_url", ""),
                }
                for item in data.get("items", [])
            ]
    except Exception as e:
        logger.warning("GitHub search failed for %s: %s", query, e)
        return []


async def _resolve_ip(domain: str) -> str | None:
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, socket.getaddrinfo, domain, None, socket.AF_INET
        )
        return results[0][4][0]
    except Exception:
        return None


async def fetch_shodan_exposure(domain: str) -> dict[str, Any]:
    """Query Shodan InternetDB for open ports, CVEs, and services on the company's IP (free, no key)."""
    if not domain:
        return {}
    ip = await _resolve_ip(domain)
    if not ip:
        return {}
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(
                f"https://internetdb.shodan.io/{ip}",
                headers=_HEADERS,
            )
            if r.status_code == 404:
                return {"ip": ip, "openPorts": [], "knownVulns": [], "services": []}
            r.raise_for_status()
            data = r.json()
            return {
                "ip": ip,
                "openPorts": data.get("ports", []),
                "knownVulns": data.get("vulns", []),
                "services": data.get("cpes", []),
                "hostnames": data.get("hostnames", []),
            }
    except Exception as e:
        logger.warning("Shodan fetch failed for %s (%s): %s", domain, ip, e)
        return {"ip": ip}
