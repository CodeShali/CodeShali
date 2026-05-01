from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Any
import logging

from services.claude_service import run_audit, run_chat

logger = logging.getLogger(__name__)

router = APIRouter()


class AuditRequest(BaseModel):
    company: str
    domain: Optional[str] = None
    industry: Optional[str] = None
    hq: Optional[str] = None
    size: Optional[str] = None
    userId: Optional[str] = None
    plan: Optional[str] = "FREE"


class ChatRequest(BaseModel):
    message: str
    context: dict[str, Any]


@router.post("/audit")
async def create_audit(req: AuditRequest) -> dict[str, Any]:
    if not req.company.strip():
        raise HTTPException(status_code=400, detail="Company name is required")

    try:
        payload = await run_audit(
            company=req.company.strip(),
            domain=req.domain or "",
            industry=req.industry or "",
            hq=req.hq or "",
            size=req.size or "",
            plan=req.plan or "FREE",
        )
        return payload
    except ValueError as e:
        logger.error("JSON parse error for %s: %s", req.company, e)
        raise HTTPException(status_code=422, detail=f"Failed to parse audit result: {e}")
    except Exception as e:
        logger.exception("Audit error for %s", req.company)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat(req: ChatRequest) -> dict[str, str]:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")

    try:
        reply = await run_chat(message=req.message.strip(), context=req.context)
        return {"reply": reply}
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))
