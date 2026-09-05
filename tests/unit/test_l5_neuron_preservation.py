"""Regression: l5_memory must preserve Neuron objects in ctx.memory.items.

Bug: l5_memory flattened (Neuron, score, tag) rows to (text, score, kind),
stripping entity_id. Downstream plasticity all keyed on entity_id:
  - HebbianLearner.record_cofiring  (core.py Sprint B)
  - TemporalGraph.record_sequence   (core.py graph recording)
  - Episodic retrieval_hits         (core.py Sprint C)
All three were silently dead in production (0 hebbian flushes in 23 days).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from norax.brain import hot_path
from norax.memory.store import Neuron


def _neuron(text: str, kind: str = "semantic") -> Neuron:
    return Neuron(text=text, path=Path("/tmp/x.md"), line=1, kind=kind)


class _Ctx:
    def __init__(self, body: str = "q") -> None:
        class Env:  # minimal env shim
            def __init__(self, b: str) -> None:
                self.body = b

        class Mem:
            items: list = []

        self.env = Env(body)
        self.memory = Mem()


@pytest.mark.asyncio
async def test_l5_memory_preserves_neuron_objects():
    neurons = [_neuron("fact one about ports"), _neuron("fact two about models")]

    async def retrieve(query: str, k: int):
        return [(neurons[0], 0.9, "kw+em"), (neurons[1], 0.8, "ent")]

    ctx = await hot_path.l5_memory(_Ctx(), retrieve=retrieve)
    assert len(ctx.memory.items) == 2
    first = ctx.memory.items[0]
    # The Neuron itself must survive — entity_id is what plasticity needs
    assert hasattr(first[0], "entity_id")
    assert first[0].entity_id == neurons[0].entity_id
    assert first[1] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_l5_memory_plain_text_rows_still_work():
    async def retrieve(query: str, k: int):
        return [("plain text hit", 0.5)]

    ctx = await hot_path.l5_memory(_Ctx(), retrieve=retrieve)
    assert ctx.memory.items == [("plain text hit", 0.5)]


@pytest.mark.asyncio
async def test_l5_memory_feeds_hebbian_cofiring():
    """End-to-end: preserved neurons must satisfy the core.py plasticity guard."""
    from norax.memory.hebbian import HebbianLearner

    neurons = [_neuron(f"co-fired fact {i}") for i in range(3)]

    async def retrieve(query: str, k: int):
        return [(n, 0.7, "kw") for n in neurons]

    ctx = await hot_path.l5_memory(_Ctx(), retrieve=retrieve)

    # Replicate the exact guard in core.py Sprint B
    retrieved = []
    for item in ctx.memory.items:
        if isinstance(item, tuple) and len(item) >= 2:
            n = item[0]
            if hasattr(n, "entity_id"):
                retrieved.append(n)
    assert len(retrieved) == 3

    hebb = HebbianLearner()
    hebb.record_cofiring(retrieved)
    assert hebb.turn_count == 1
    assert len(hebb._pending_strengthens) == 3


def test_memory_block_renders_neuron_text():
    """Prompt assembler must render .text, not a dataclass repr."""
    from norax.prompt.assembler import _block_memory

    ctx = _Ctx()
    ctx.memory.items = [(_neuron("IDENTITY:Norax test"), 1.35, "hot")]
    block = _block_memory(ctx)
    assert "IDENTITY:Norax test" in block
    assert "Neuron(" not in block


def test_infer_tools_from_memory_items_with_neurons():
    from norax.brain.tool_retriever import infer_tools_from_memory_items

    items = [(_neuron("use the search_memory tool for recalls"), 0.8, "semantic")]
    tools = infer_tools_from_memory_items(items)
    assert isinstance(tools, list)  # must not crash on Neuron objects


def test_temporal_session_id_from_str_channel():
    """core.py Sprint E must not crash on str channels (env.channel is a
    Literal string, not an object with .id). Regression for the latent
    AttributeError exposed once retrieval_hits became non-empty."""
    from norax.memory.temporal_graph import TemporalGraph

    class EnvShim:
        channel = "chat"  # agent_os/Discord ingress passes a plain string
        raw = {"channel_id": "channel-123"}

    env = EnvShim()
    # Exact resolution logic from core.py
    session_id = str((env.raw or {}).get("channel_id") or env.channel or "default")
    assert session_id == "channel-123"

    tg = TemporalGraph(root=Path("/tmp/tg_test"))
    added = tg.record_sequence(["a" * 16, "b" * 16, "c" * 16], session_id=session_id)
    assert added == 2

    # No channel_id in raw → falls back to the channel string itself
    env2 = type("E", (), {"channel": "chat", "raw": {}})()
    sid2 = str((env2.raw or {}).get("channel_id") or env2.channel or "default")
    assert sid2 == "chat"
