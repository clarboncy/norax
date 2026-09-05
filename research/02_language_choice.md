# Research 02 — Language Choice for the Norax Runtime

**Date:** 2026-04-22
**Decision:** **Python 3.12+** (primary), with narrow Rust for one
specific hot-path component if/when profiling proves the need.

---

## 1. The real decision space

Only four serious candidates in 2025–2026 for an agent runtime:

| Language | Agent-ecosystem fit | Our fit |
|---|---|---|
| **Python** | Dominant. Letta, Mem0, MemGPT, PydanticAI, LangChain, LlamaIndex, Anthropic SDK, OpenAI SDK all native. | Native — **7,821 lines of brain stack already Python** |
| **TypeScript** | Strong (Mastra, Vercel AI SDK). Good for web-first stacks. | Poor — full rewrite; no brain reuse |
| **Rust** | Emerging (Aether, Kernex, GENT). Bleeding edge. | Poor — full rewrite; tiny ecosystem for our use case |
| **Go** | Weak for LLM work (Agno-Go, few others). Good concurrency. | Poor — full rewrite; limited AI libs |

---

## 2. Why Python wins for Norax specifically

### 2a. The brain stack is already Python

Our audit shows 25 numbered-layer modules totaling ~7,821 lines of pure
Python. Rewriting these in anything else is **weeks of work to reach
parity** — and the brain layers ARE the differentiator. Anything that
forces us to translate L1–L32 is a regression.

### 2b. Every upstream dependency is Python

From our audit's real usage:

- `gateway-proxy.py` — **ours, Python**
- `connect.py` — **Python** (stdlib only)
- `norax-demo` backend — **FastAPI (Python)**
- `brain-runner.py` (2,489 lines) — **Python**
- Ollama clients, OpenAI SDK, Anthropic SDK — **Python-first**
- `norax-embed-v3` (the embedding server) — **Python**

### 2c. `discord.py` is mature and covers everything we use

From audit §8, we use exactly: text send, reply tags, reactions, voice
passthrough. `discord.py` handles all of these natively with async/await.
Moving to `discord.js` would mean crossing a language boundary for
no gain.

### 2d. FastAPI is fast enough

The concern with Python is performance. Real benchmarks (Mustafic 2025
"FastAPI gives Rust a Good Run", CloudInsight 2025 FastAPI vs Go
comparison): for **I/O-bound LLM-call workloads** (which is 100% of
what we do), FastAPI + asyncio is within **5–15% of Go/Rust** throughput
because the bottleneck is the upstream model, not our code.

A Discord-bot + LLM-proxy runtime is I/O-bound by definition — we spend
~99% of wall-clock waiting on model tokens, not on CPU. Language speed
is effectively invisible here.

### 2e. The industry converged on Python for agents

Letta, Mem0 (50K GitHub stars), MemGPT, PydanticAI, Anthropic cookbooks
— all Python. When we want to audit against SOTA research (MemGPT 2.0
sleep-time, Mem0g graph memory), the reference implementations are in
our language.

---

## 3. When NOT to stay Python (the honest caveats)

Three narrow cases where Python is a real bottleneck:

1. **Embedding throughput on CPU** — if we ever generate >100k embeddings
   per minute. Today we embed on GPU via `norax-embed-v3` (separate
   process, not our concern).
2. **Discord gateway raw connection at >100k servers.** Discord itself
   famously rewrote in Rust. We have 1 server. Not relevant.
3. **Tight per-turn CPU-bound work** — the only thing close is dedup
   similarity math (L29 dentate gyrus). Even that runs in <10ms per call.

If any of these becomes a real profile bottleneck, the answer is
**drop a Rust extension via pyo3 or maturin for just that hot path** —
not rewrite the whole runtime. Same pattern NumPy/Tokenizers/Polars use.

---

## 4. Rust / Go / TS — why not

### Rust (Aether, Kernex, GENT)
- **Pros:** Real performance, memory safety, Kernex specifically offers
  OS-level agent isolation.
- **Cons:** Zero brain-stack reuse. ~3–5× dev time. Ecosystem is
  early (Aether has <100 stars; Kernex is v0.4). We'd be pioneering.
- **Verdict:** wrong cost/benefit for our single-user single-machine
  deployment.

### Go (Agno-Go)
- **Pros:** Cleaner concurrency than Python.
- **Cons:** Minimal LLM ecosystem (Agno-Go has 54 stars). No brain reuse.
  And per the "Go Is Fast. But Not Fast Enough for Discord" post,
  for chat-agent workloads even Go hits walls that Python-asyncio also
  hits — the bottleneck isn't the language.
- **Verdict:** zero upside.

### TypeScript (Mastra, Vercel AI SDK)
- **Pros:** Unified web + server stack; Mastra is polished.
- **Cons:** Full rewrite of L1–L32. Different mental model for math/
  memory pipelines. Our `noraxdev.org` frontend already calls into a
  Python FastAPI today.
- **Verdict:** only compelling if we were green-fielding. We're not.

---

## 5. Python-specific tactical choices

If Python is the language, these are the right libraries:

| Concern | Pick | Why |
|---|---|---|
| Async runtime | `asyncio` (stdlib) | mature, universal |
| HTTP server | `FastAPI` + `uvicorn` | already in use for noraxdev.org |
| Discord | `discord.py` 2.4+ | active fork (Rapptz returned), covers what we use |
| Validation | `pydantic` v2 | envelope schemas, tool schemas, fast Rust-backed core |
| LLM clients | `httpx` directly (no wrapper frameworks) | we have our own proxy + routing |
| DB | stdlib `sqlite3` + `sqlite-vss` for vectors | already our stack |
| Vector index | FAISS or sqlite-vss | pick per load |
| Embeddings | our own `norax-embed-v3` via HTTP | unchanged |
| MCP export face | `mcp` SDK (Python) | future-proof external surface |
| Process mgmt | `systemd` user units + `asyncio.subprocess` | OS-native |
| Logging | `structlog` | JSON logs, queryable |
| Typing | full `mypy` strict | eng discipline, not runtime cost |

**No-ops (explicit rejections):**
- LangChain / LangGraph — fights our brain layers
- CrewAI — same
- LlamaIndex — we have our own retrieval (L22)
- Pydantic AI — would subsume L12/L15 decisions; keep our own
- uv / poetry — `pip install -e .` + `pyproject.toml` is enough

---

## 6. Version and deployment

- **Python 3.12+** (3.12 has improved asyncio performance, typing
  generics, per-interpreter GIL work; 3.13 adds optional free-threaded
  mode we may want later for embedding batching).
- **Single venv** at `~/norax/.venv`, locked `requirements.txt`.
- **systemd user units** for `norax-runtime`, `norax-proxy`,
  `norax-sleep.timer`.
- **No Docker for local** (adds a layer, costs perf, we own the host).
  We'd containerize only if we deploy to a non-trusted host.

---

## 7. Final answer

**Python 3.12+**, single process for the runtime, separate process for
the proxy (already is), separate timer-triggered process for the sleep
worker. If profiling ever shows a real bottleneck, targeted Rust via
`pyo3` for that module — same approach NumPy and Polars use. **No full-
language-rewrite scenarios are on the table.**

This choice:
- preserves 7,821 lines of working brain code
- matches the language every referenced research system uses
- is faster than the model 99% of the time (I/O-bound)
- keeps one toolchain, one venv, one deploy story
- leaves a clean Rust-hot-path escape hatch

---

## Sources

- Letta SDK: https://github.com/letta-ai/letta-python (Python)
- Mem0 (50K stars, Python): https://github.com/mem0ai/mem0
- Mastra (TS): https://mastra.one/
- Aether (Rust): https://aether-agent.io/
- Kernex (Rust runtime): https://kernex.dev/
- Agno-Go: https://github.com/rexleimo/agno-Go
- FastAPI vs Rust benchmark: https://medium.com/@almirx101/fastapi-the-surprising-performance-workhorse-that-gives-rust-a-good-run-23fc52dd815c
- Python vs Go microservices 2026: https://freeacademy.ai/blog/python-vs-go-microservices-performance-comparison-2026
- Python vs Go high concurrency: https://cloudinsight.cc/en/blog/python-golang-high-concurrency
- Agent memory framework comparison: https://blog.appxlab.io/2026/04/13/ai-agent-memory-frameworks/
