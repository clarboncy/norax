"""Correction Gate — narrow post-draft checks against canonical memory.

Runs after the LLM generates a draft response, before it is sent. It checks
only facts whose structure supports deterministic comparison:

  1. Owned identifiers (wallet/email/id) against same-class stored values
  2. Explicit uppercase ``KEY=value`` / ``KEY:value`` facts against |W4+

Ordinary prose is deliberately excluded: lexical retrieval cannot establish
whether a general statement is true, and treating stale memory as an oracle
would make a strong model less reliable. Retrieval is bounded and concurrent.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

log = logging.getLogger("norax.context.correction")


# ---------- claim extraction ------------------------------------------------

# Identifier patterns that we treat as hard, checkable facts.
_ID_PATTERNS = [
    ("eth_addr", re.compile(r"\b0x[a-fA-F0-9]{40}\b")),
    ("sol_addr", re.compile(r"(?<![A-Za-z0-9])[1-9A-HJ-NP-Za-km-z]{32,44}(?![A-Za-z0-9])")),
    ("bankr_id", re.compile(r"\bbk_[A-Z0-9]{20,}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("cashtag", re.compile(r"\$[a-zA-Z][a-zA-Z0-9_]{2,30}\b")),
    ("url", re.compile(r"\bhttps?://[^\s)\]]+")),
    # "port" removed — ports are contextual, not hard identifiers.
    # Multiple services run on different ports; treating port as a
    # checkable identifier causes false contradictions (e.g. embed port
    # vs gateway port).
]
_CLASS_RX = dict(_ID_PATTERNS)

# Keywords we use to pull back stored neurons that reference an identifier
# class. We query by these instead of the claimed value itself — the claim
# might be wrong, but the stored canonical fact will still use one of these.
_CLASS_KEYWORDS: dict[str, list[str]] = {
    "eth_addr": [
        "wallet",
        "WALLET",
        "evm",
        "EVM",
        "address",
        "token",
        "TOKEN",
        "contract",
        "NORAX",
        "norax",
    ],
    "sol_addr": ["wallet", "WALLET", "sol", "SOL", "solana", "token", "mint"],
    "bankr_id": ["bankr", "BANKR"],
    "email": ["email", "EMAIL", "gmail", "mail"],
    "cashtag": ["cash", "CASH", "cashapp", "CASH_APP"],
    "url": ["url", "URL", "link"],
    "port": ["port", "PORT", "runtime", "gateway", "bind"],
}

# Role hints — when the draft sentence mentions one of these words, only
# match against stored neurons that mention the SAME role word. This
# prevents eth_addr(wallet) from colliding with eth_addr(token_contract).
_ROLE_HINTS: dict[str, list[list[str]]] = {
    "eth_addr": [
        ["wallet", "WALLET"],
        ["token", "TOKEN", "NORAX", "OCAE", "CLAW", "contract"],
        ["creator", "CREATOR"],
    ],
    "sol_addr": [
        ["wallet", "WALLET"],
        ["token", "TOKEN", "mint"],
    ],
}


def _role_for(
    subject: str, text: str, span: tuple[int, int] | None = None, window: int = 60
) -> frozenset[str] | None:
    """Which role-bucket does this text fall into for `subject`?

    If *span* is given, only look at ±window chars around the claim to avoid
    cross-contamination when a draft contains multiple identifiers of the
    same class (e.g. wallet addr AND token contract addr).
    """
    hints = _ROLE_HINTS.get(subject)
    if not hints:
        return None
    if span is not None:
        lo = max(0, span[0] - window)
        hi = min(len(text), span[1] + window)
        text = text[lo:hi]
    low = text.lower()
    for bucket in hints:
        if any(w.lower() in low for w in bucket):
            return frozenset(w.lower() for w in bucket)
    return None


def _is_owned_identifier_claim(claim: Claim, draft: str) -> bool:
    """Only hard-check identifiers presented as the agent/user's own fact.

    A support address, citation URL, recipient wallet, or other third-party
    identifier must not be contradicted merely because memory contains the
    owner's different identifier of the same syntactic class.
    """
    lo = max(0, claim.span[0] - 70)
    hi = min(len(draft), claim.span[1] + 30)
    context = draft[lo:hi]
    labels = {
        "email": r"(?:email|mail|contact|address)",
        "url": r"(?:url|link|site|website|endpoint)",
        "eth_addr": r"(?:wallet|address|token|contract|evm)",
        "sol_addr": r"(?:wallet|address|token|mint|solana)",
        "bankr_id": r"(?:bankr|account|id)",
        "cashtag": r"(?:cashtag|cashapp|account)",
    }
    label = labels.get(claim.subject, re.escape(claim.subject))
    ownership = re.compile(
        rf"\b(?:my|our|your|owner(?:'s)?|norax(?:'s)?|official)\b[^\n.!?]{{0,35}}\b{label}\b",
        re.IGNORECASE,
    )
    keyed = re.compile(rf"\b{label}\s*[:=]\s*$", re.IGNORECASE)
    prefix = draft[lo : claim.span[0]]
    return bool(ownership.search(context) or keyed.search(prefix))


# Claim patterns — "X is Y", "X = Y", "KEY:value"
_KV_CLAIM = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\s*[:=]\s*([^\s,.;]+)")
_CODE_SPAN = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)

# Internal protocol markers that the agent may echo in its output but which
# are NOT factual claims about the world. Treating these as checkable KV
# claims causes false contradictions (e.g. "GOAL: said 'answer' but stored
# 'find'") because stored neurons contain the same markers with different
# values from prior turns. These are runtime scaffolding, not user-facing
# facts, so the correction gate must skip them.
_INTERNAL_PROTOCOL_SUBJECTS = frozenset(
    {
        "GOAL",
        "TASK_STATE",
        "TASK_TYPE",
        "COMPLETED_ACTIONS",
        "PENDING_STEPS",
        "NEXT_STEP",
        "NEXT_STEP_ADVICE",
        "BLOCKERS",
        "FACTS",
        "FILES_TOUCHED",
        "FILE_SUMMARIES",
        "PROGRESS",
        "ESCALATE",
        "ESCALATE_IF_NEEDED",
        "ESCALATION_REASONS",
        "DIRECTIVE",
        "OLLAMA_EXEC_GUARD",
        "ARITHMETIC",
        "CORRECTION",
        "MISMATCH",
        "EVIDENCE",
        "RULE",
        "EVICTED_CONTEXT",
        "FEATURES",
        "TURN",
        "DONE_CRITERIA",
        "DONE",
        "CONSTRAINTS",
        "TOOL_CALLS",
        "TOOL_ROUNDS",
        "WEIGHT",
        "W1",
        "W2",
        "W3",
        "W4",
        "W5",
    }
)


@dataclass
class Claim:
    kind: str  # "identifier", "kv", "statement"
    subject: str  # what the claim is about ("wallet", "RUNTIME", etc.)
    value: str  # the claimed value
    span: tuple[int, int]  # (start, end) in draft

    def key(self) -> str:
        return f"{self.kind}:{self.subject}:{self.value}".lower()


def extract_claims(draft: str) -> list[Claim]:
    """Pull checkable claims out of a draft response.

    Intentionally narrow: we only want things we can actually verify against
    stored memory. Opinions, suggestions, questions — all ignored.
    """
    # Code/config examples routinely contain placeholder identifiers and
    # uppercase KEY=value assignments. They are artifacts, not assertions
    # about the user's canonical state, and checking them creates both false
    # corrections and needless retrieval work. Preserve offsets while masking.
    prose = _CODE_SPAN.sub(
        lambda match: "".join("\n" if char == "\n" else " " for char in match.group(0)),
        draft,
    )
    out: list[Claim] = []
    seen: set[str] = set()

    for kind, rx in _ID_PATTERNS:
        for m in rx.finditer(prose):
            c = Claim(kind="identifier", subject=kind, value=m.group(0), span=(m.start(), m.end()))
            if c.key() not in seen:
                out.append(c)
                seen.add(c.key())

    for m in _KV_CLAIM.finditer(prose):
        subject = m.group(1)
        # Skip internal protocol markers (GOAL:, TASK_STATE, COMPLETED_ACTIONS,
        # etc.) — these are runtime scaffolding the model may echo, not
        # factual claims. Treating them as checkable causes false
        # contradictions against stored neurons from prior turns.
        if subject in _INTERNAL_PROTOCOL_SUBJECTS:
            continue
        c = Claim(kind="kv", subject=subject, value=m.group(2), span=(m.start(), m.end()))
        if c.key() not in seen:
            out.append(c)
            seen.add(c.key())

    return out


# ---------- correction gate -------------------------------------------------

RetrieveFn = Callable[[str, int], Awaitable[list[tuple[object, float, str]]]]


@dataclass
class CorrectionResult:
    ok: bool  # True if draft passes
    needs_revision: bool  # True if any contradiction found
    checked: int  # number of claims checked
    contradictions: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    inconclusive: bool = False
    errors: list[str] = field(default_factory=list)
    total_claims: int = 0

    def as_correction_block(self) -> str:
        """Render a compressed-notation CORRECTION block the assembler can
        inject above the draft for a one-shot re-prompt."""
        if not self.contradictions:
            return ""
        lines = ["CORRECTION;needs=true"]
        for c in self.contradictions:
            lines.append(
                f"MISMATCH:{c['claim_subject']}:claimed={c['claim_value']};"
                f"stored={c['stored_value']}|W5"
            )
        for e in self.evidence[:3]:
            lines.append(f"EVIDENCE:{e['text'][:140]}")
        lines.append("RULE:prefer stored value; correct the draft before sending|W5")
        return "\n".join(lines)


@dataclass
class CorrectionGate:
    retrieve: RetrieveFn  # async (query, k) -> list[(neuron, score, source)]
    retrieve_k: int = 5
    contradiction_threshold: float = 0.25  # similarity score floor
    # If weight tag on stored fact is >= this, mismatches are hard-blocking.
    hard_weight: float = 1.0
    max_claims: int = 8

    async def check(self, draft: str) -> CorrectionResult:
        extracted_claims = extract_claims(draft)
        # Citation/support URLs and third-party addresses are not claims about
        # the owner. Filter them before applying the cap so they cannot crowd
        # out a later checkable value.
        eligible_claims = [
            claim
            for claim in extracted_claims
            if claim.kind != "identifier" or _is_owned_identifier_claim(claim, draft)
        ]
        total_claims = len(eligible_claims)
        if not eligible_claims:
            return CorrectionResult(
                ok=True,
                needs_revision=False,
                checked=0,
                total_claims=0,
            )
        max_claims = max(1, int(self.max_claims))
        claims = eligible_claims[:max_claims]

        contradictions: list[dict] = []
        evidence: list[dict] = []
        errors: list[str] = []
        checked = 0
        retrieval_cache: dict[str, list[tuple[object, float, str]] | None] = {}

        async def retrieve_once(query: str, claim: Claim):
            if query in retrieval_cache:
                return retrieval_cache[query]
            try:
                hits = await self.retrieve(query, self.retrieve_k)
            except Exception as exc:  # noqa: BLE001
                log.warning("correction.retrieve.error claim=%s err=%r", claim.key(), exc)
                errors.append(f"retrieval failed for {claim.subject}: {type(exc).__name__}")
                retrieval_cache[query] = None
                return None
            retrieval_cache[query] = hits
            return hits

        query_examples: dict[str, Claim] = {}
        for claim in claims:
            if claim.kind == "identifier":
                keywords = _CLASS_KEYWORDS.get(claim.subject, [claim.subject])
                draft_role = _role_for(claim.subject, draft, span=claim.span)
                query = " ".join(list(draft_role or ()) or keywords[:4])
            else:
                query = f"{claim.subject} {claim.value}"
            query_examples.setdefault(query, claim)

        semaphore = asyncio.Semaphore(4)

        async def prefetch(query: str, claim: Claim) -> None:
            async with semaphore:
                await retrieve_once(query, claim)

        await asyncio.gather(*(prefetch(query, claim) for query, claim in query_examples.items()))

        for claim in claims:
            # ---- identifier claims: class-based contradiction ----
            # For identifier claims (eth_addr, email, bankr_id, etc.) we don't
            # require the stored neuron to contain the literal class name.
            # Instead: if any W4+ neuron returned by retrieval contains an
            # identifier of the SAME CLASS but DIFFERENT VALUE → contradict.
            if claim.kind == "identifier":
                checked += 1
                rx = _CLASS_RX.get(claim.subject)
                if rx is not None:
                    # One class/role query is enough. The previous per-keyword
                    # loop performed up to ten retrievals for one address and
                    # amplified unrelated matches as well as latency.
                    keywords = _CLASS_KEYWORDS.get(claim.subject, [claim.subject])
                    # Determine the role of this claim from draft context
                    draft_role = _role_for(claim.subject, draft, span=claim.span)
                    query_terms = list(draft_role or ()) or keywords[:4]
                    hits_agg = await retrieve_once(" ".join(query_terms), claim)
                    if hits_agg is None:
                        continue
                    matched = False
                    contra = None
                    for n, score, src in hits_agg:
                        text = getattr(n, "text", str(n))
                        weight = getattr(n, "weight", 1.0)
                        if weight < self.hard_weight:
                            continue
                        stored_ids = rx.findall(text)
                        if not stored_ids:
                            continue
                        # Role-aware: skip if draft and stored fact are
                        # about different semantic roles (wallet vs token).
                        if draft_role is not None:
                            stored_role = _role_for(claim.subject, text)
                            if stored_role is not None and stored_role != draft_role:
                                continue
                        if any(claim.value.lower() == s.lower() for s in stored_ids):
                            evidence.append(
                                {
                                    "claim_subject": claim.subject,
                                    "claim_value": claim.value,
                                    "text": text,
                                    "score": score,
                                    "source": src,
                                    "weight": weight,
                                }
                            )
                            matched = True
                            break
                        if contra is None:
                            contra = {
                                "claim_subject": claim.subject,
                                "claim_value": claim.value,
                                "stored_value": stored_ids[0],
                                "stored_text": text,
                                "weight": weight,
                                "source": src,
                            }
                    if not matched and contra is not None:
                        contradictions.append(contra)
                    continue

            # ---- explicit uppercase KV claims: subject-match path ----
            checked += 1
            query = f"{claim.subject} {claim.value}"
            hits = await retrieve_once(query, claim)
            if hits is None:
                continue

            for n, score, src in hits:
                text = getattr(n, "text", str(n))
                weight = getattr(n, "weight", 1.0)
                # Does this stored fact mention the same subject?
                low = text.lower()
                subj_low = claim.subject.lower()
                if subj_low not in low:
                    continue
                evidence.append(
                    {
                        "claim_subject": claim.subject,
                        "claim_value": claim.value,
                        "text": text,
                        "score": score,
                        "source": src,
                        "weight": weight,
                    }
                )
                # Does the stored fact disagree with the claimed value?
                if claim.value.lower() not in low and weight >= self.hard_weight:
                    # Extract the stored value opposite the claim subject
                    stored = _extract_stored_value(text, claim.subject)
                    if stored and stored.lower() != claim.value.lower():
                        contradictions.append(
                            {
                                "claim_subject": claim.subject,
                                "claim_value": claim.value,
                                "stored_value": stored,
                                "stored_text": text,
                                "weight": weight,
                                "source": src,
                            }
                        )
                        break  # one contradiction per claim is enough

        ok = not contradictions
        return CorrectionResult(
            ok=ok,
            needs_revision=not ok,
            checked=checked,
            contradictions=contradictions,
            evidence=evidence,
            inconclusive=bool(errors or total_claims > len(claims)),
            errors=errors,
            total_claims=total_claims,
        )


def _extract_stored_value(text: str, subject: str) -> str | None:
    """Pull `<subject>:<value>` or `<subject>=<value>` out of a neuron line."""
    rx = re.compile(
        rf"\b{re.escape(subject)}\s*[:=]\s*([^\s|,;]+)",
        re.IGNORECASE,
    )
    m = rx.search(text)
    return m.group(1) if m else None
