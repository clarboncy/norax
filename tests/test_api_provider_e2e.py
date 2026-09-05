"""End-to-end API provider test — all models, all transitions."""

# ruff: noqa: E402,I001

import asyncio
import sys
import time
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from norax.api_provider import NoraxProvider


TEST_MESSAGES = [{"role": "user", "content": "What is 2+2? Reply with just the number."}]

TEST_MESSAGES_CODING = [
    {
        "role": "user",
        "content": "Write a Python function that returns the sum of two numbers. Just the code, no explanation.",
    }
]

TEST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform a calculation",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }
]


async def _test_model(
    provider: NoraxProvider, model: str, label: str, *, messages=None, tools=None
):
    """Test one model and return (ok, ms, detail)."""
    msgs = messages or TEST_MESSAGES
    t0 = time.monotonic()
    try:
        resp = await provider.chat(model, messages=msgs, tools=tools)
        ms = (time.monotonic() - t0) * 1000
        ok = bool(resp.content.strip() or resp.tool_calls)
        detail = (
            f"{len(resp.content)} chars, {len(resp.tool_calls)} tool calls, {resp.routing_info}"
        )
        return ok, ms, detail
    except Exception as e:
        ms = (time.monotonic() - t0) * 1000
        return False, ms, str(type(e).__name__)


async def main():
    provider = NoraxProvider()

    # Quick health check
    print("=== Health Check ===")
    health = await provider.health()
    for k, v in health.items():
        status = "✅" if v else "❌"
        print(f"  {status} {k}: {v}")
    print()

    # Test all models (grouped by backend)
    tests: list[tuple[str, str, dict | None, list | None]] = [
        # Direct Ollama models
        ("kimi-k2.6:cloud", "Kimi K2.6", None, None),
        ("glm-5.1:cloud", "GLM 5.1", None, None),
        ("qwen3-coder-next:cloud", "Qwen Coder Next", None, TEST_TOOLS),
        ("deepseek-v4-pro:cloud", "DeepSeek V4 Pro", None, None),
        ("deepseek-v4-flash:cloud", "DeepSeek V4 Flash", None, None),
        ("qwen3.5:cloud", "Qwen 3.5", None, None),
        ("gemma4:31b-cloud", "Gemma 4", None, None),
        ("nemotron-3-super:cloud", "Nemotron 3 Super", None, None),
    ]

    results = []
    print("=== Model Tests ===")
    for model, label, msgs, tools in tests:
        print(f"  Testing {label} ({model})...", end=" ", flush=True)
        ok, ms, detail = await _test_model(provider, model, label, messages=msgs, tools=tools)
        status = "✅" if ok else "❌"
        print(f"{status} {ms:.0f}ms — {detail}")
        results.append((model, label, ok, ms))

    # Summary
    print("\n=== Summary ===")
    passed = sum(1 for r in results if r[2])
    total = len(results)
    print(f"  {passed}/{total} models passed")
    if passed == total:
        print("  🎉 All models working!")
    else:
        failed = [r for r in results if not r[2]]
        print(f"  Failed: {[f'{r[0]} ({r[1]})' for r in failed]}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
