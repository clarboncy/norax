"""Paid API endpoints for Norax commerce.

Mounts on the existing FastAPI app (http_in adapter or gateway proxy).
Provides:
  GET  /api/pricing           — list services + prices
  GET  /api/wallet             — get Norax wallet info for payment
  POST /api/pay                — verify on-chain payment, create session
  POST /api/code-review        — AI code review (paid)
  POST /api/research           — AI research with citations (paid)
  POST /api/agent-task         — Mini agent execution (paid)
  POST /api/debug              — Debug assistance (paid)
  POST /api/summarize          — Summarize text/URL (paid)
  GET  /api/session            — check session status + remaining credits
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from .payment_gate import (
    SERVICE_PRICES,
    PaymentAlreadyRedeemedError,
    SessionPersistenceError,
    SessionStore,
    get_session_store,
    get_wallet_info,
    verify_payment,
)

log = logging.getLogger("norax.commerce.api")

# ── Request models ─────────────────────────────────────────────────


class PayRequest(BaseModel):
    tx_hash: str
    service: str  # code_review, research, agent_task, debug, summarize


class CodeReviewRequest(BaseModel):
    code: str
    language: str = "auto"
    context: str = ""


class ResearchRequest(BaseModel):
    query: str
    depth: str = "standard"  # quick, standard, deep


class AgentTaskRequest(BaseModel):
    task: str
    max_steps: int = 5


class DebugRequest(BaseModel):
    error: str
    code: str = ""
    language: str = "auto"


class SummarizeRequest(BaseModel):
    text: str = ""
    url: str = ""


# ── Response models ─────────────────────────────────────────────────


class ServiceList(BaseModel):
    services: dict[str, float]
    wallet: dict


class SessionResponse(BaseModel):
    valid: bool
    service: str = ""
    credits_remaining: float = 0
    expires_at: float = 0


# ── Mount function ──────────────────────────────────────────────────


def mount_commerce(app: FastAPI, agent_runner=None) -> None:
    """Mount commerce API endpoints on a FastAPI app.

    Args:
        app: FastAPI app to mount on
        agent_runner: async callable(task: str, max_steps: int) -> str
                      Used for /api/agent-task endpoint. If None, returns 503.
    """

    store = get_session_store()

    # ── Public endpoints ────────────────────────────────────────────

    @app.get("/api/pricing")
    async def pricing() -> dict:
        """List all services and their prices."""
        return {
            "services": {k: v / 1e6 for k, v in SERVICE_PRICES.items()},
            "wallet": get_wallet_info(),
            "currency": "USDC on Base",
        }

    @app.get("/api/wallet")
    async def wallet_info() -> dict:
        """Get Norax wallet address for payment."""
        return get_wallet_info()

    # ── Payment endpoint ─────────────────────────────────────────────

    @app.post("/api/pay")
    async def pay(req: PayRequest) -> dict:
        """Verify on-chain payment and create a session.

        User sends USDC or ETH to Norax wallet on Base, then submits
        the tx hash here. Norax verifies on-chain and issues a session token.
        """
        if req.service not in SERVICE_PRICES:
            raise HTTPException(400, f"Unknown service: {req.service}")

        result = await asyncio.to_thread(verify_payment, req.tx_hash, req.service)
        if not result.valid:
            raise HTTPException(402, f"Payment verification failed: {result.error}")

        # Create session with credits = amount paid (in USDC micro-units)
        # This allows users to overpay and use credits later
        try:
            sess = await asyncio.to_thread(
                store.create,
                result.sender,
                req.service,
                int(result.amount_usdc),
                req.tx_hash,
            )
        except PaymentAlreadyRedeemedError as exc:
            raise HTTPException(409, str(exc)) from exc
        except SessionPersistenceError as exc:
            raise HTTPException(503, "Paid-session storage is temporarily unavailable") from exc

        # Accounting is secondary to honoring an already verified payment and
        # durably created session. Record the on-chain receipt idempotently,
        # but never revoke service because the reporting ledger is degraded.
        try:
            from .revenue_engine import get_revenue_engine

            await asyncio.to_thread(
                get_revenue_engine().record_earning,
                "api_payment",
                result.amount_usdc / 1_000_000,
                f"Verified payment for {req.service}",
                result.tx_hash,
            )
        except Exception:  # noqa: BLE001
            log.exception("commerce: verified payment revenue receipt could not be recorded")

        log.info(
            "commerce: session created for %s service=%s credits=$%.2f",
            result.sender,
            req.service,
            result.amount_usdc / 1e6,
        )

        return {
            "ok": True,
            "session_token": sess.token,
            "service": sess.service,
            "credits_remaining": sess.credits_remaining / 1e6,
            "expires_at": sess.expires_at,
            "message": f"Payment verified! You have ${sess.credits_remaining / 1e6:.2f} in credits.",
        }

    # ── Session check ────────────────────────────────────────────────

    @app.get("/api/session")
    async def session_status(authorization: str = Header(None)) -> dict:
        """Check session status and remaining credits."""
        token = _extract_token(authorization)
        if not token:
            raise HTTPException(401, "Missing session token. Use Authorization: Bearer nx_...")
        try:
            sess = await asyncio.to_thread(store.get, token)
        except SessionPersistenceError as exc:
            raise HTTPException(503, "Paid-session storage is temporarily unavailable") from exc
        if not sess:
            raise HTTPException(401, "Invalid or expired session")
        return {
            "valid": True,
            "service": sess.service,
            "credits_remaining": sess.credits_remaining / 1e6,
            "expires_at": sess.expires_at,
        }

    # ── Paid service endpoints ──────────────────────────────────────

    @app.post("/api/code-review")
    async def code_review(
        req: CodeReviewRequest,
        authorization: str = Header(None),
    ) -> dict:
        """AI-powered code review. $0.50 per review."""
        return await _paid_service(
            "code_review",
            authorization,
            store,
            _run_code_review,
            req,
        )

    @app.post("/api/research")
    async def research(
        req: ResearchRequest,
        authorization: str = Header(None),
    ) -> dict:
        """AI research with web search + citations. $0.25 per query."""
        return await _paid_service(
            "research",
            authorization,
            store,
            _run_research,
            req,
        )

    @app.post("/api/agent-task")
    async def agent_task(
        req: AgentTaskRequest,
        authorization: str = Header(None),
    ) -> dict:
        """Mini agent execution. $1.00 per task."""
        if agent_runner is None:
            raise HTTPException(503, "Agent runner not configured")
        return await _paid_service(
            "agent_task",
            authorization,
            store,
            lambda r: agent_runner(r.task, r.max_steps),
            req,
        )

    @app.post("/api/debug")
    async def debug_help(
        req: DebugRequest,
        authorization: str = Header(None),
    ) -> dict:
        """Debug assistance. $0.75 per session."""
        return await _paid_service(
            "debug",
            authorization,
            store,
            _run_debug,
            req,
        )

    @app.post("/api/summarize")
    async def summarize(
        req: SummarizeRequest,
        authorization: str = Header(None),
    ) -> dict:
        """Summarize text or URL. $0.15 per summary."""
        return await _paid_service(
            "summarize",
            authorization,
            store,
            _run_summarize,
            req,
        )

    log.info("commerce: API endpoints mounted on FastAPI app")


# ── Helpers ────────────────────────────────────────────────────────


def _extract_token(authorization: str | None) -> str | None:
    """Extract session token from Authorization header."""
    if not authorization:
        return None
    if authorization.startswith("Bearer "):
        return authorization[7:]
    return authorization


async def _paid_service(
    service: str,
    authorization: str | None,
    store: SessionStore,
    handler: Any,
    req: Any,
) -> dict:
    """Generic paid service handler: verify credits, run, deduct."""
    token = _extract_token(authorization)
    if not token:
        raise HTTPException(401, "Missing session token. Pay first at /api/pay")

    try:
        sess = await asyncio.to_thread(store.get, token)
    except SessionPersistenceError as exc:
        raise HTTPException(503, "Paid-session storage is temporarily unavailable") from exc
    if sess is None:
        raise HTTPException(401, "Invalid or expired session")

    if sess.service != service:
        raise HTTPException(403, f"Session is valid only for service: {sess.service}")

    price = SERVICE_PRICES[service]
    try:
        consumed = await asyncio.to_thread(store.consume, token, price)
    except SessionPersistenceError as exc:
        raise HTTPException(503, "Paid-session storage is temporarily unavailable") from exc
    if not consumed:
        try:
            current = await asyncio.to_thread(store.get, token)
        except SessionPersistenceError as exc:
            raise HTTPException(503, "Paid-session storage is temporarily unavailable") from exc
        available = current.credits_remaining if current is not None else 0
        raise HTTPException(
            402,
            f"Insufficient credits. Need ${price / 1e6:.2f}, have ${available / 1e6:.2f}",
        )

    try:
        result = await handler(req)
        current = await asyncio.to_thread(store.get, token)
        return {
            "ok": True,
            "service": service,
            "result": result,
            "credits_remaining": (current.credits_remaining if current is not None else 0) / 1e6,
        }
    except asyncio.CancelledError:
        try:
            await asyncio.shield(asyncio.to_thread(store.refund, token, price))
        except Exception:  # noqa: BLE001
            log.exception("commerce: cancelled service refund failed service=%s", service)
        raise
    except Exception as e:
        log.error("commerce: service %s failed: %s", service, e)
        # Refund on failure
        try:
            refunded = await asyncio.to_thread(store.refund, token, price)
        except SessionPersistenceError as refund_error:
            raise HTTPException(
                503,
                "Service failed and the credit refund could not be committed; contact support",
            ) from refund_error
        if not refunded:
            raise HTTPException(500, f"Service failed: {e}. Credit refund failed.") from e
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(500, f"Service failed: {e}. Credits refunded.") from e


# ── Service implementations ─────────────────────────────────────────


async def _run_code_review(req: CodeReviewRequest) -> dict:
    """Run AI code review using the Norax gateway."""
    from ..gateway_client import GatewayRequest, get_gateway

    gw = get_gateway()
    prompt = f"""Review this {req.language} code. Find bugs, security issues, performance problems, and style issues. Be concise and specific.

Context: {req.context}

Code:
```
{req.code}
```

Return a JSON object with keys: issues (list of {{severity, line, description, suggestion}}), summary (string), score (1-10)."""

    resp = await gw.chat(
        GatewayRequest(
            model="glm-5.2:cloud",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
    )
    return {"review": resp.content, "model": resp.model}


async def _run_research(req: ResearchRequest) -> dict:
    """Run AI research with web search."""
    from ..dispatch.tools import t_web_fetch, t_web_search

    results = await t_web_search(query=req.query, count=8)
    items = results.get("items", [])

    # Fetch top 3 results for deeper content
    summaries = []
    for item in items[:3]:
        url = item.get("url", "")
        if url:
            try:
                fetched = await t_web_fetch(url=url, max_chars=5000)
                summaries.append(
                    {
                        "title": item.get("title", ""),
                        "url": url,
                        "content": fetched.get("text", "")[:2000],
                    }
                )
            except Exception:
                summaries.append(
                    {
                        "title": item.get("title", ""),
                        "url": url,
                        "snippet": item.get("snippet", ""),
                    }
                )

    return {
        "query": req.query,
        "sources": [
            {"title": i.get("title", ""), "url": i.get("url", ""), "snippet": i.get("snippet", "")}
            for i in items
        ],
        "deep_results": summaries,
    }


async def _run_debug(req: DebugRequest) -> dict:
    """Run AI debug assistance."""
    from ..gateway_client import GatewayRequest, get_gateway

    gw = get_gateway()
    prompt = f"""Debug this error. Provide the root cause and a fix.

Error:
{req.error}

Code ({req.language}):
```
{req.code}
```

Return: root_cause (string), fix (string with code if applicable), explanation (string)."""

    resp = await gw.chat(
        GatewayRequest(
            model="glm-5.2:cloud",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
    )
    return {"analysis": resp.content, "model": resp.model}


async def _run_summarize(req: SummarizeRequest) -> dict:
    """Summarize text or fetch URL and summarize."""
    text = req.text
    if not text and req.url:
        from ..dispatch.tools import t_web_fetch

        fetched = await t_web_fetch(url=req.url, max_chars=10000)
        text = fetched.get("text", "")

    if not text:
        raise HTTPException(400, "Provide either text or url")

    from ..gateway_client import GatewayRequest, get_gateway

    gw = get_gateway()
    prompt = f"""Summarize the following text in 3-5 bullet points plus a one-sentence TL;DR:

{text[:8000]}"""

    resp = await gw.chat(
        GatewayRequest(
            model="glm-5.2:cloud",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )
    )
    return {"summary": resp.content, "model": resp.model}
