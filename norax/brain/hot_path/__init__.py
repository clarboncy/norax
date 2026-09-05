"""Brain hot-path pipeline (L0..L10).

Runs on every ingress envelope. Each layer is a pure function or a small
class; they mutate / return a BrainContext. No layer talks to the outside
world except L10 which calls the gateway.

L0  ingress        — normalize envelope, set trace_id
L1  identify       — attach sender tier, trust flag
L2  safety_gate    — enforce envelope trust and payload-size invariants
L3  state          — update valence/confidence/arousal (decay + signals)
L4  focus          — derive a 1-line summary of what this turn is about
L5  memory         — retrieve and normalize top-K memory evidence
L6  skills         — populate skills directory (always-on + on-request)
L7  tools          — compute allowed tool list for this turn
L8  plan           — decide emit_reply / silent / defer
L9  prompt         — assemble the 11-block prompt (prompt.assembler)
L10 call           — run the gateway (or no-op when decision != emit_reply)

The gateway dependency is injected, so the same pipeline runs against either
a deterministic test double or a configured live provider.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from ...envelope import BrainContext, SensoryInput
from ...gateway_client import GatewayClient, GatewayRequest, GatewayResponse
from ...prompt import assembler
from ...soul import Soul, load_soul
from ..policy import tool_names_for_turn
from ..skills import active_skills_filtered, retrieve_skills_by_trigger
from ..tool_retriever import infer_tools_from_memory_items
from .amygdala import score as amygdala_score
from .arousal import assess as assess_arousal
from .thalamus import classify as thalamic_classify

log = logging.getLogger("norax.brain.hot_path")


OWNER_IDS: set[str] = {
    value.strip() for value in os.environ.get("NORAX_OWNER_IDS", "").split(",") if value.strip()
}


# ---- individual layers ----------------------------------------------------


def l0_ingress(env: SensoryInput) -> BrainContext:
    return BrainContext(env=env)


def l1_identify(ctx: BrainContext) -> BrainContext:
    # Tier is already on the Principal from the adapter; we promote owner_id
    # matches here as a second line of defense.
    s = ctx.env.sender
    if s.id in OWNER_IDS and s.tier != "owner":
        # can't mutate a frozen Principal; we accept adapter-assigned tier
        log.info("sender matches OWNER_IDS but tier=%s (adapter-assigned)", s.tier)
    return ctx


def l2_safety_gate(ctx: BrainContext) -> BrainContext:
    """Enforce cheap, deterministic ingress invariants.

    Prompt text is not classified as "hostile" here: legitimate security work
    and quoted adversarial content routinely contain injection language. Tool
    authorization remains identity/tier based, and untrusted content is fenced
    by the assembler. This layer handles the objective hazards that can be
    decided without an LLM: forged trust, pathological payload size, NULs, and
    unsafe/oversized attachment metadata.
    """
    env = ctx.env
    actions: list[str] = []

    if env.trusted and not env.sender.trust:
        env.trusted = False
        actions.append("trust_downgraded_to_sender_identity")

    body = env.body if isinstance(env.body, str) else str(env.body or "")
    if "\x00" in body:
        body = body.replace("\x00", "")
        actions.append("nul_bytes_removed")

    try:
        configured_limit = int(os.environ.get("NORAX_MAX_INGRESS_CHARS", "250000"))
    except ValueError:
        configured_limit = 250_000
    max_chars = min(2_000_000, max(16_000, configured_limit))
    if len(body) > max_chars:
        marker = f"\n...[ingress truncated at {max_chars} of {len(body)} characters]"
        body = body[: max(0, max_chars - len(marker))] + marker
        actions.append("body_truncated")
    env.body = body

    sanitized_attachments: list[dict] = []
    dropped_attachments = 0
    for attachment in list(env.attachments or [])[:16]:
        if not isinstance(attachment, dict):
            dropped_attachments += 1
            continue
        url = str(attachment.get("url") or "")[:16_384]
        lower_url = url.lower()
        if url and not lower_url.startswith(("https://", "http://", "data:image/")):
            dropped_attachments += 1
            continue
        if lower_url.startswith("data:image/") and len(url) > 8_000_000:
            dropped_attachments += 1
            continue
        try:
            size = max(0, min(int(attachment.get("size") or 0), 10**12))
        except (TypeError, ValueError):
            size = 0
        sanitized_attachments.append(
            {
                "url": url,
                "filename": str(attachment.get("filename") or "")[:512],
                "content_type": str(attachment.get("content_type") or "")[:128],
                "size": size,
            }
        )
    dropped_attachments += max(0, len(env.attachments or []) - 16)
    if dropped_attachments:
        actions.append(f"attachments_dropped:{dropped_attachments}")
    env.attachments = sanitized_attachments

    ctx.metadata["safety_gate"] = {
        "applied": bool(actions),
        "actions": actions,
        "body_chars": len(env.body),
        "attachments": len(env.attachments),
    }
    return ctx


def l3_state(ctx: BrainContext) -> BrainContext:
    """L3 + L27 RAS + L14 Amygdala — real arousal/valence/confidence."""
    body = ctx.env.body or ""

    # L1 Thalamus: classify input
    thalamic = thalamic_classify(body)
    ctx.metadata["thalamic"] = {
        "type": thalamic.msg_type,
        "signal": thalamic.signal_strength,
        "pathway": thalamic.pathway,
        "complexity": thalamic.complexity,
    }

    # L27 RAS: arousal level
    arousal_state = assess_arousal(
        body,
        complexity=thalamic.complexity,
        signal_strength=thalamic.signal_strength,
    )
    ctx.state.arousal = arousal_state.level_int / 6.0  # normalize to 0-1
    ctx.metadata["arousal"] = {
        "level": arousal_state.level,
        "reason": arousal_state.reason,
    }

    # L14 Amygdala: emotional scoring
    emotion = amygdala_score(body)
    ctx.state.valence = emotion.valence / 10.0  # normalize to -1..+1
    ctx.metadata["emotion"] = {
        "valence": emotion.valence,
        "arousal": emotion.arousal,
        "threat": emotion.threat,
        "dominant": emotion.dominant,
        "tags": list(emotion.tags),
    }

    # Confidence: trust + emotional stability
    base_conf = 1.0 if ctx.env.sender.trust else 0.7
    if emotion.threat >= 5:
        base_conf *= 0.6  # reduce confidence under threat
    ctx.state.confidence = base_conf

    return ctx


def l4_focus(ctx: BrainContext) -> BrainContext:
    """L4 Focus — attention gate informed by thalamic classification."""
    body = ctx.env.body.strip()
    thalamic = ctx.metadata.get("thalamic", {})
    msg_type = thalamic.get("type", "unknown")
    pathway = thalamic.get("pathway", "executive")

    # Build focus summary with type prefix for the assembler
    if len(body) <= 120:
        summary = f"[{msg_type}] {body}"
    else:
        summary = f"[{msg_type}] {body[:117]}..."
    ctx.focus.summary = summary

    # Suggest skills based on pathway
    if pathway == "executive":
        ctx.focus.suggested_skills = ["exec", "write", "edit"]
    elif pathway == "retrieval":
        ctx.focus.suggested_skills = ["read", "search_memory", "web_fetch"]
    elif pathway == "reward":
        ctx.focus.suggested_skills = []
    elif pathway == "procedural_update":
        ctx.focus.suggested_skills = ["write", "append_memory"]
    else:
        ctx.focus.suggested_skills = []

    return ctx


async def l5_memory(ctx: BrainContext, *, retrieve=None) -> BrainContext:
    """Populate ctx.memory.items.

    If `retrieve` is supplied, it's called with (query:str, k:int) →
    iterable of (text:str, score:float) OR iterable of
    (neuron:Neuron, score:float, source:str) or
    (neuron:Neuron, score:float).

    retrieve may be async; it is awaited when necessary.
    """
    import asyncio

    if retrieve is None:
        ctx.memory.items = []
        return ctx
    try:
        raw = retrieve(ctx.env.body, k=5)
        if asyncio.iscoroutine(raw):
            raw = await raw
    except Exception as e:  # noqa: BLE001
        log.warning("memory.retrieve.error: %r", e)
        ctx.memory.items = []
        return ctx
    # Normalize to list[(text, score)] / (text, score, kind) for plain-text
    # rows, but PRESERVE Neuron objects as (neuron, score[, kind]) — downstream
    # plasticity (Hebbian co-firing, temporal sequences, episodic retrieval
    # hits) needs entity_id. Text consumers extract .text via getattr.
    out: list[tuple] = []
    for row in raw:
        if isinstance(row, tuple):
            if len(row) == 2:
                a, b = row
                if isinstance(a, str):
                    out.append((a, float(b)))
                else:
                    kind = getattr(a, "kind", "")
                    if kind:
                        out.append((a, float(b), kind))
                    else:
                        out.append((a, float(b)))
            elif len(row) == 3:
                n, s, src = row
                if isinstance(n, str):
                    kind = src if isinstance(src, str) else ""
                    out.append((n, float(s), kind) if kind else (n, float(s)))
                else:
                    kind = getattr(n, "kind", "") or (src if isinstance(src, str) else "")
                    if kind:
                        out.append((n, float(s), kind))
                    else:
                        out.append((n, float(s)))
    ctx.memory.items = out
    return ctx


def l6_skills(
    ctx: BrainContext,
    *,
    directory=None,
    memory_root: Path | None = None,
) -> BrainContext:
    if directory is None:
        user_prompt = ctx.env.body or ""
        task_type = ctx.metadata.get("task_type", "")
        # 1. Always-on + learned skills (filtered by task relevance)
        ctx.skills.entries = active_skills_filtered(
            memory_root=memory_root,
            user_prompt=user_prompt,
            task_type=task_type,
        )
        # 2. Retrieval-type skills (trigger-matched against user prompt).
        # These are the skills with inject=retrieval that were previously
        # disconnected — browser.control, staging.sync, twitter.engagement,
        # code.change, gateway.providers, etc. Their TRIGGERS are matched
        # against the user prompt and matching skills are injected with
        # their full body (RULES, FLOW, PURPOSE) into the SKILLS block.
        retrieval_skills = retrieve_skills_by_trigger(user_prompt, memory_root=memory_root)
        if retrieval_skills:
            # Convert 4-tuples to 3-tuples for entries, store body separately
            existing_ids = {e[0] for e in ctx.skills.entries}
            for sid, scope, inject, body in retrieval_skills:
                if sid not in existing_ids:
                    ctx.skills.entries.append((sid, scope, inject))
                    # Store body in metadata for assembler to render
                    if "skill_bodies" not in ctx.metadata:
                        ctx.metadata["skill_bodies"] = {}
                    ctx.metadata["skill_bodies"][sid] = body
    else:
        ctx.skills.entries = list(directory())
    return ctx


def l7_tools(ctx: BrainContext) -> BrainContext:
    tools = tool_names_for_turn(ctx.env.body or "", ctx.env.sender.tier)
    if ctx.env.sender.tier == "owner":
        tools += infer_tools_from_memory_items(ctx.memory.items)
    ctx.allowed_tools = list(dict.fromkeys(tools))
    return ctx


def l8_plan(ctx: BrainContext) -> BrainContext:
    body = ctx.env.body.strip()
    has_attachments = bool(ctx.env.attachments)
    if not body and not has_attachments:
        ctx.decision = "silent"
        ctx.silent_notes = "empty body"
    else:
        ctx.decision = "emit_reply"
    return ctx


def l9_prompt(ctx: BrainContext, *, soul: Soul | None = None, runtime_info: dict | None = None):
    ctx.runtime_info = dict(runtime_info or {})
    ctx.runtime_info.setdefault(
        "time",
        datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    )
    return ctx, assembler.render(ctx, soul or load_soul())


async def l10_call(
    ctx: BrainContext,
    rendered,
    *,
    gateway: GatewayClient,
    model: str,
) -> GatewayResponse | None:
    if ctx.decision != "emit_reply":
        return None
    req = GatewayRequest(
        model=model,
        messages=[
            {"role": "system", "content": rendered.system},
            {"role": "user", "content": rendered.user},
        ],
    )
    return await gateway.chat(req)


# ---- orchestration --------------------------------------------------------


async def run_turn(
    env: SensoryInput,
    *,
    gateway: GatewayClient,
    model: str = "qwen3.8-27b-fast:latest",
    runtime_info: dict | None = None,
    soul: Soul | None = None,
    window=None,  # norax.context.RollingWindow | None
    correction_gate=None,  # norax.context.CorrectionGate | None
    retrieve=None,  # async (query, k) -> list[(text|neuron, score, src)]
    memory_root: Path | None = None,
    user_model_block: str = "",
    max_correction_rounds: int = 1,
):
    """Run L0..L10 and (optionally) rolling-window + correction-gate post-pass.

    If `window` is provided:
      - the turn's user message is appended before L10
      - eviction runs BEFORE the LLM call if the window is over budget
      - assistant response (if any) is appended after L10
    If `correction_gate` is provided:
      - the draft is checked; on contradiction we do up to
        `max_correction_rounds` re-prompts with the correction block
        injected into the user message.
    """
    ctx = l0_ingress(env)
    ctx = l1_identify(ctx)
    ctx = l2_safety_gate(ctx)
    ctx = l3_state(ctx)
    ctx = l4_focus(ctx)
    ctx = await l5_memory(ctx, retrieve=retrieve)
    ctx = l6_skills(ctx, memory_root=memory_root)
    if user_model_block:
        ctx.metadata["user_model"] = user_model_block[:8_192]
    ctx = l7_tools(ctx)
    ctx = l8_plan(ctx)
    ctx, rendered = l9_prompt(ctx, soul=soul, runtime_info=runtime_info)

    # ---- rolling window ingest + eviction ----
    turn_id = None
    if window is not None:
        # Keep head in sync with the latest rendered system prompt
        if not window.head:
            window.add_system(rendered.system)
        turn_id = window.start_turn()
        window.add_user(env.body, turn_id=turn_id)
        window.evict_if_needed()

    if ctx.decision != "emit_reply":
        return ctx, rendered, None

    resp = await l10_call(ctx, rendered, gateway=gateway, model=model)

    # ---- correction gate loop ----
    if resp is not None and correction_gate is not None:
        for _ in range(max_correction_rounds):
            try:
                gate_res = await correction_gate.check(resp.content or "")
            except Exception as e:  # noqa: BLE001
                log.warning("correction_gate.error: %r", e)
                break
            if not gate_res.needs_revision:
                break
            block = gate_res.as_correction_block()
            if not block:
                break
            log.info("correction.retry contras=%d", len(gate_res.contradictions))
            # Re-prompt: inject CORRECTION above the user turn
            req = GatewayRequest(
                model=model,
                messages=[
                    {"role": "system", "content": rendered.system},
                    {"role": "user", "content": f"{block}\n\n---\nORIGINAL:\n{env.body}"},
                ],
            )
            resp = await gateway.chat(req)

    # ---- rolling window assistant-side ingest ----
    if window is not None and resp is not None and turn_id is not None:
        _clean = resp.content or ""
        try:
            from ..adapter.reply_tag import parse_reply_tag

            _clean = parse_reply_tag(_clean).text
        except Exception:  # noqa: BLE001
            pass
        _stripped = _clean.strip()
        if _stripped and _stripped not in ("NO_REPLY", "HEARTBEAT_OK"):
            window.add_assistant(_clean, turn_id=turn_id)
        window.evict_if_needed()

    return ctx, rendered, resp


async def plan_turn(
    env: SensoryInput,
    *,
    runtime_info: dict | None = None,
    soul: Soul | None = None,
    window=None,
    retrieve=None,
    memory_root: Path | None = None,
    user_model_block: str = "",
):
    """Run L0..L9 only (no LLM call). Returns (ctx, rendered).

    This is the planning-only variant for agentic flows where the caller
    (runtime) runs its own L10 + tool loop using `agent_loop.run_agent_loop`.

    Behaviour mirrors `run_turn` up through L9; rolling-window user-turn
    ingest + eviction happens here so the caller doesn't need to repeat it.
    """
    ctx = l0_ingress(env)
    ctx = l1_identify(ctx)
    ctx = l2_safety_gate(ctx)
    ctx = l3_state(ctx)
    ctx = l4_focus(ctx)
    ctx = await l5_memory(ctx, retrieve=retrieve)
    ctx = l6_skills(ctx, memory_root=memory_root)
    if user_model_block:
        ctx.metadata["user_model"] = user_model_block[:8_192]
    ctx = l7_tools(ctx)
    ctx = l8_plan(ctx)
    ctx, rendered = l9_prompt(ctx, soul=soul, runtime_info=runtime_info)

    if window is not None:
        if not window.head:
            window.add_system(rendered.system)
        turn_id = window.start_turn()
        window.add_user(env.body, turn_id=turn_id)
        window.evict_if_needed()

    return ctx, rendered
