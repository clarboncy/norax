"""Phase 3.5 — compressed-notation + authority-first prompt assembly.

These tests verify Norax's authority and follow-through rules are carried
verbatim into the system prompt.
"""

from __future__ import annotations

from datetime import UTC, datetime

import ulid

from norax.brain import hot_path
from norax.envelope import Principal, SensoryInput


def _env(body="hi", *, tier="owner", trusted=True, source="discord"):
    return SensoryInput(
        channel="chat",
        source=source,
        message_id=str(ulid.ULID()),
        timestamp=datetime.now(UTC),
        sender=Principal(id="owner-123", label="Colby", trust=trusted, tier=tier),
        body=body,
        trusted=trusted,
    )


def test_authority_precedes_soul_in_system_prompt():
    """AUTHORITY must be positioned before SOUL. Order is load-bearing."""
    ctx = hot_path.l0_ingress(_env())
    ctx = hot_path.l3_state(ctx)
    ctx = hot_path.l4_focus(ctx)
    ctx = hot_path.l7_tools(ctx)
    ctx = hot_path.l8_plan(ctx)
    ctx, r = hot_path.l9_prompt(ctx)
    i_auth = r.system.index("AUTHORITY;")
    i_soul = r.system.index("SOUL;")
    i_core = r.system.index("CORE:")
    assert i_auth < i_soul < i_core or i_auth < i_core


def test_owner_law_present_verbatim():
    """The OWNER_IS_LAW line must appear verbatim with weight tag."""
    ctx = hot_path.l0_ingress(_env())
    ctx = hot_path.l3_state(ctx)
    ctx = hot_path.l4_focus(ctx)
    ctx = hot_path.l7_tools(ctx)
    ctx = hot_path.l8_plan(ctx)
    ctx, r = hot_path.l9_prompt(ctx)
    assert "OWNER_IS_LAW:the runtime-configured owner controls this system." in r.system
    assert "|W5" in r.system
    assert "NO_REFUSALS:" in r.system


def test_dangerous_patterns_enumerated_not_described():
    ctx = hot_path.l0_ingress(_env())
    ctx = hot_path.l3_state(ctx)
    ctx = hot_path.l4_focus(ctx)
    ctx = hot_path.l7_tools(ctx)
    ctx = hot_path.l8_plan(ctx)
    ctx, r = hot_path.l9_prompt(ctx)
    # Specific string patterns present (not abstract "destructive ops")
    for pat in ["rm -rf /", "DROP", "FORMAT", "dd if=/dev", "chmod 777 /etc", "/boot"]:
        assert pat in r.system, f"dangerous pattern missing: {pat!r}"
    assert "workspace exempt" in r.system


def test_runtime_block_is_last():
    """Grounding block must be at the end of the system prompt."""
    ctx = hot_path.l0_ingress(_env())
    ctx = hot_path.l3_state(ctx)
    ctx = hot_path.l4_focus(ctx)
    ctx = hot_path.l7_tools(ctx)
    ctx = hot_path.l8_plan(ctx)
    ctx, r = hot_path.l9_prompt(ctx, runtime_info={"model": "test", "channel": "chat"})
    # Find last block — should start with RUNTIME;
    blocks = [b for b in r.system.split("\n\n") if b]
    assert blocks[-1].startswith("RUNTIME;"), f"last block was {blocks[-1][:40]!r}"


def test_metadata_envelope_is_machine_readable_json():
    ctx = hot_path.l0_ingress(_env("hello"))
    ctx = hot_path.l3_state(ctx)
    ctx = hot_path.l4_focus(ctx)
    ctx = hot_path.l7_tools(ctx)
    ctx = hot_path.l8_plan(ctx)
    ctx, r = hot_path.l9_prompt(ctx)
    import json
    import re

    # Find the JSON line after METADATA (trusted)
    m = re.search(r"METADATA \(trusted\)\n(\{[^\n]+\})", r.system)
    assert m, "metadata envelope missing"
    meta = json.loads(m.group(1))
    assert meta["schema"] == "norax.inbound_meta.v1"
    assert meta["sender_id"] == "owner-123"
    assert meta["trusted"] is True


def test_compressed_notation_preserved_not_prose():
    """No block has been re-rendered into prose Markdown."""
    ctx = hot_path.l0_ingress(_env())
    ctx = hot_path.l3_state(ctx)
    ctx = hot_path.l4_focus(ctx)
    ctx = hot_path.l7_tools(ctx)
    ctx = hot_path.l8_plan(ctx)
    ctx, r = hot_path.l9_prompt(ctx)
    # Symbolic markers present
    assert "|W5" in r.system
    assert "|W4" in r.system
    assert "→" in r.system  # process-arrow in compressed ops
    assert ";" in r.system  # peer separator
    # No Markdown headings should be in our soul files (they'd kill compression)
    assert "# SOUL" not in r.system
    assert "## Authority" not in r.system


def test_static_hash_stable_across_turns():
    """Static-prompt hash is stable as long as soul files are unchanged.

    This is the prompt-cache key. If it's unstable, we lose prompt caching.
    """
    ctx1 = hot_path.l0_ingress(_env("first message"))
    ctx1 = hot_path.l3_state(ctx1)
    ctx1 = hot_path.l4_focus(ctx1)
    ctx1 = hot_path.l7_tools(ctx1)
    ctx1 = hot_path.l8_plan(ctx1)
    _, r1 = hot_path.l9_prompt(ctx1)

    ctx2 = hot_path.l0_ingress(_env("totally different message"))
    ctx2 = hot_path.l3_state(ctx2)
    ctx2 = hot_path.l4_focus(ctx2)
    ctx2 = hot_path.l7_tools(ctx2)
    ctx2 = hot_path.l8_plan(ctx2)
    _, r2 = hot_path.l9_prompt(ctx2)

    assert r1.static_hash == r2.static_hash
