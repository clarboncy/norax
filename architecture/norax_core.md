# Architecture 01 — `norax_core.md`

> The authoritative package layout, module boundaries, runtime invariants,
> and implementation architecture for Norax. This document records the design
> that the production Python runtime implements.

---

## 1. The package (`norax/`) — top-level layout

```
~/norax/                       (project root, where we already are)
├── pyproject.toml             # uv-managed; py 3.12+
├── uv.lock
├── .python-version            # 3.12
├── .env.example               # secrets template (never committed)
├── README.md                  # one paragraph + how to run
│
├── norax/                     # the package
│   ├── __init__.py
│   ├── __main__.py            # `python -m norax` entrypoint
│   │
│   ├── runtime/               # the message loop
│   │   ├── __init__.py
│   │   ├── core.py            # Runtime class, main async loop
│   │   ├── ingress_bus.py     # IngressBus: merges all sensory adapters
│   │   ├── heartbeat.py       # the only time-based emitter we keep
│   │   └── shutdown.py        # graceful drain on SIGTERM
│   │
│   ├── envelope/              # The shared shape every adapter speaks
│   │   ├── __init__.py
│   │   ├── sensory.py         # SensoryInput dataclass
│   │   ├── motor.py           # ToolCall, ToolResult dataclasses
│   │   ├── brain.py           # BrainContext, LayerOutput
│   │   └── principal.py       # Principal (caller identity + trust tier)
│   │
│   ├── adapter/               # 6 sensory adapters (S1..S6)
│   │   ├── __init__.py
│   │   ├── base.py            # Adapter Protocol
│   │   ├── discord_in.py      # S1 — native discord.py client
│   │   ├── http_in.py         # webhook + API ingress
│   │   ├── email_in.py        # S2 — Gmail monitor consumer
│   │   ├── payment_in.py      # S3 — Cash App monitor consumer
│   │   ├── schedule_in.py     # S4 — internal reminder emitter
│   │   ├── heartbeat_in.py    # S5 — idle-tick emitter
│   │   └── node_in.py         # S6 — paired-device events
│   │
│   ├── brain/                 # 36 layers, in-process
│   │   ├── __init__.py
│   │   ├── pipeline.py        # BrainPipeline: hot path + post hooks + in-turn bg + sleep
│   │   ├── layer.py           # Layer Protocol, LayerOutput
│   │   ├── context.py         # BrainContext (consumed by assembler)
│   │   ├── hot_path/          # 16 per-message layers (L27..L19)
│   │   │   ├── l27_ras.py
│   │   │   ├── l01_thalamus.py
│   │   │   ├── l22_retrieve.py
│   │   │   ├── l28_entorhinal.py
│   │   │   ├── l03_working.py
│   │   │   ├── l04_focus.py
│   │   │   ├── l14_amygdala.py
│   │   │   ├── l24_acc.py
│   │   │   ├── l15_basal_ganglia.py
│   │   │   ├── l25_vta.py
│   │   │   ├── l12_cerebellum.py
│   │   │   ├── l23_hypothalamus.py
│   │   │   ├── l16_motor.py
│   │   │   ├── l13_reconsolidation.py
│   │   │   ├── l26_metacog.py
│   │   │   └── l19_language.py
│   │   ├── post/              # post-action hooks (L25, L16, L15, L14)
│   │   │   ├── l25_rpe.py
│   │   │   ├── l16_habit.py
│   │   │   ├── l15_gate_update.py
│   │   │   └── l14_valence_update.py
│   │   ├── in_turn_bg/        # L17, L18, L20, L21, L13 (turn-decided)
│   │   │   ├── l17_replay.py
│   │   │   ├── l18_hemispheric.py
│   │   │   ├── l20_mirror.py
│   │   │   ├── l21_dmn.py
│   │   │   └── l13_reconsolidation_bg.py
│   │   └── sleep/             # L6→L7/L8/L9 transfer (turn-triggered)
│   │       └── consolidator.py
│   │
│   ├── prompt/                # System-prompt assembler (11 blocks)
│   │   ├── __init__.py
│   │   ├── assembler.py       # SystemPromptAssembler
│   │   ├── blocks/
│   │   │   ├── identity.py
│   │   │   ├── runtime.py
│   │   │   ├── persona.py
│   │   │   ├── state.py        # consumes brain.context.state
│   │   │   ├── focus.py        # consumes brain.context.focus
│   │   │   ├── memory.py       # consumes brain.context.memory
│   │   │   ├── tool_schema.py  # consumes dispatcher registry + caller
│   │   │   ├── skills_dir.py   # consumes filesystem scan
│   │   │   ├── safety.py
│   │   │   ├── caller.py
│   │   │   └── reply_rules.py
│   │   └── cache.py            # input-hash cache (no time expiry)
│   │
│   ├── dispatch/              # 12 motor tools, registry, gates
│   │   ├── __init__.py
│   │   ├── dispatcher.py       # Dispatcher class
│   │   ├── registry.py         # ToolRegistry
│   │   ├── gate.py             # RiskGate (SOUL DANGEROUS)
│   │   ├── budget.py           # BudgetEnforcer (per-caller $/tok/req)
│   │   └── tools/              # the 12 motor tools (T1..T12)
│   │       ├── read_tool.py
│   │       ├── write_tool.py
│   │       ├── edit_tool.py
│   │       ├── exec_tool.py
│   │       ├── web_fetch_tool.py
│   │       ├── web_search_tool.py
│   │       ├── tts_tool.py
│   │       ├── discord_send_tool.py
│   │       ├── skill_open_tool.py
│   │       ├── skill_add_tool.py
│   │       ├── skill_edit_tool.py
│   │       ├── skill_remove_tool.py
│   │       ├── code_launch_tool.py
│   │       ├── nodes_invoke_tool.py
│   │       └── schedule_tool.py
│   │
│   ├── gateway_client/        # Talks to norax-gateway:4100
│   │   ├── __init__.py
│   │   └── client.py
│   │
│   ├── memory/                # Event store + 11 derived stores
│   │   ├── __init__.py
│   │   ├── event_store.py     # append-only events.jsonl
│   │   ├── snapshots.py       # periodic derived-index snapshots
│   │   ├── replay.py          # replay events → rebuild indexes
│   │   ├── stores/
│   │   │   ├── scratchpad.py     # L3
│   │   │   ├── focus.py          # L4
│   │   │   ├── episodic.py       # L5
│   │   │   ├── exact_cache.py    # L6
│   │   │   ├── semantic_cache.py # L6b
│   │   │   ├── semantic.py       # L7
│   │   │   ├── procedural.py     # L8 — skills + procedures live here
│   │   │   ├── intel.py          # L9
│   │   │   ├── kg.py             # L11 (sqlite)
│   │   │   ├── vectors.py        # L22 (faiss in-process)
│   │   │   └── budget_db.py      # caller budget state
│   │   └── consolidate.py     # called by brain.sleep.consolidator
│   │
│   ├── safety/                # SOUL + risk
│   │   ├── __init__.py
│   │   ├── soul.py            # loads SOUL.md → policies
│   │   ├── owner.py           # Owner identity, OWNER_IS_LAW check
│   │   └── injection.py       # prompt-injection classifier
│   │
│   ├── http/                  # FastAPI server (HTTP ingress + admin)
│   │   ├── __init__.py
│   │   ├── server.py
│   │   └── routes/
│   │       ├── ingress.py     # POST /ingress/webhook/<name>, /api
│   │       ├── status.py      # GET /status
│   │       ├── metrics.py     # GET /metrics (Prometheus)
│   │       └── drafts.py      # GET/POST /norax/draft/<id>, approve/reject
│   │
│   ├── code/                  # Independent coding-session launcher (no harness)
│   │   ├── __init__.py
│   │   ├── session.py         # CodeSession class
│   │   └── worktree.py
│   │
│   ├── observability/         # logs, metrics, replay drills
│   │   ├── __init__.py
│   │   ├── log.py             # unified events.jsonl writer
│   │   ├── metrics.py         # in-memory metric histograms
│   │   └── tracing.py         # per-turn span ids
│   │
│   ├── config/                # All runtime config
│   │   ├── __init__.py
│   │   ├── loader.py          # loads YAML + env
│   │   └── schema.py          # pydantic models
│   │
│   └── version.py
│
├── config/                    # YAML configs (user-editable)
│   ├── runtime.yaml           # the master config
│   ├── tools.yaml             # 12 motor tools, tier, budget caps
│   ├── adapters.yaml          # adapter enable/disable + creds refs
│   ├── gateway.yaml           # gateway routing tiers, providers
│   └── policy.yaml            # callers, allowlists, dangerous patterns
│
├── memory/                    # The procedural / semantic / intel stores
│   ├── procedural/
│   │   ├── skills/            # ~10 seed skills land here in Phase 4
│   │   └── archive/           # skill_remove targets
│   ├── semantic/
│   ├── intel/
│   ├── episodic/              # daily MD logs
│   └── scratchpad.md
│
├── state/                     # runtime-derived; not committed
│   ├── events.jsonl           # append-only
│   ├── snapshots/
│   ├── budget.db              # sqlite
│   ├── kg.db                  # sqlite
│   ├── vectors.faiss
│   └── cache/
│       ├── exact.db
│       └── semantic.db
│
├── logs/                      # runtime logs
│   ├── runtime.jsonl
│   ├── gateway.jsonl
│   └── dispatch.jsonl
│
├── tests/
│   ├── unit/                  # one folder per package
│   ├── integration/
│   ├── golden/                # 1k-turn replay corpus for gateway
│   └── conftest.py
│
├── scripts/                   # ops scripts (not part of runtime)
│   ├── replay.py              # rebuild indexes from events.jsonl
│   ├── snapshot.py            # take a snapshot
│   └── healthcheck.py
│
├── ongoing.md                 # journal (already exists)
├── ongoing/                   # ongoing/* workspace docs
├── references/                # all 7 reference docs
├── research/                  # 2 research docs
└── architecture/              # this doc lives here
```

---

## 2. Boundaries — what each package owns

| Package | Owns | Talks to | Never talks to |
|---|---|---|---|
| `runtime` | Main loop, lifecycle, signals | every other package | nothing forbidden |
| `envelope` | Pure dataclasses; **zero I/O** | (imported by all) | filesystem, network |
| `adapter` | Translate channel format → `SensoryInput` | `envelope`, network/IO of its channel | `brain`, `dispatch`, `prompt` |
| `brain` | All 36 layers, layer firing decisions | `envelope`, `memory`, `safety` | `adapter`, `http`, network |
| `prompt` | Compose system prompt from blocks | `envelope`, `brain.context`, `dispatch.registry` (read-only), filesystem (skills dir) | `adapter`, `gateway_client` |
| `dispatch` | Registry of 12 tools, gate, budget, motor execution | `envelope`, `safety`, `memory` (for skill_* tools), per-tool resource | `brain` directly (only via runtime) |
| `gateway_client` | HTTP client to :4100; nothing else | network, `envelope` | filesystem, `brain`, `dispatch` |
| `memory` | Event store + derived indexes + replay | filesystem, sqlite, faiss | network |
| `safety` | SOUL load + RiskGate + injection classifier | filesystem (SOUL.md) | network |
| `http` | FastAPI server + routes | `runtime`, `dispatch` (drafts), `memory` (read) | direct brain mutation |
| `code` | Spawn coding CLI in worktree | filesystem, subprocess | brain — log read is on caller's schedule |
| `observability` | Log writes, metrics, traces | filesystem | brain mutation |
| `config` | Load + validate YAML/env | filesystem | nothing else |

**Imports allowed only downward in the table.** Cycle = bug. We add
a `tests/test_imports.py` that loads each module and asserts no
forbidden cross-import.

---

## 3. Wire shapes (the dataclasses every boundary speaks)

### 3a. `envelope/sensory.py`

```python
from dataclasses import dataclass, field
from typing import Literal, Optional, Any
from datetime import datetime

@dataclass(frozen=True)
class Principal:
    id: str                         # opaque id (discord uid, email, etc.)
    label: str                      # human-readable
    trust: bool                     # post-allowlist trust
    tier: Literal["owner","admin","user","guest"]

@dataclass
class SensoryInput:
    channel: Literal["chat","email","payment","schedule","heartbeat","node","http"]
    source: str                     # adapter name
    message_id: str                 # ulid
    timestamp: datetime
    sender: Principal
    body: str                       # human-readable text body
    attachments: list[dict] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    trusted: bool = False           # set by adapter; default False for safety
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 3b. `envelope/motor.py`

```python
@dataclass
class ToolCall:
    call_id: str                    # idempotency key
    name: str
    args: dict
    caller: Principal
    tier: Literal["T0","T1","T2","dynamic"]
    reason: Optional[str] = None    # required for T2
    parent_message_id: Optional[str] = None

@dataclass
class ToolResult:
    call_id: str
    ok: bool
    value: Any
    error: Optional[str] = None
    cost: dict = field(default_factory=dict)   # {"ms":int,"tokens":int,"$":float}
```

### 3c. `envelope/brain.py`

```python
@dataclass
class StateBlock:
    valence: float                  # L14
    confidence: float               # L26
    arousal: float                  # L27

@dataclass
class FocusBlock:
    summary: str                    # L4 active-focus
    suggested_skills: list[str] = field(default_factory=list)

@dataclass
class MemoryBlock:
    items: list[tuple[str,float]]   # (text, relevance)

@dataclass
class SkillsDirectoryBlock:
    entries: list[tuple[str,str,Literal["always","on-request","idle-learned","disabled"]]]
    # (name, scope, invocation)

@dataclass
class BrainContext:
    env: SensoryInput
    state: StateBlock
    focus: FocusBlock
    memory: MemoryBlock
    skills: SkillsDirectoryBlock
    allowed_tools: list[str]
    runtime_info: dict
    decision: Literal["emit_reply","silent","defer"]
    silent_notes: Optional[str] = None
```

---

## 4. Class skeletons (the ten objects we actually write)

These are the **public surface** of each package. Method bodies
empty in Phase 1; Phase 2–7 fill them in.

```python
# norax/runtime/core.py
class Runtime:
    def __init__(self, ingress, brain, dispatcher, gateway, prompt, memory, safety, log): ...
    async def run(self) -> None: ...
    async def shutdown(self, sig=None) -> None: ...

# norax/runtime/ingress_bus.py
class IngressBus:
    def __init__(self, adapters: list["Adapter"]): ...
    async def stream(self) -> AsyncIterator[SensoryInput]: ...

# norax/adapter/base.py
class Adapter(Protocol):
    name: str
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def events(self) -> AsyncIterator[SensoryInput]: ...

# norax/brain/pipeline.py
class BrainPipeline:
    def __init__(self, hot, post, in_turn_bg, sleep): ...
    async def hot_path(self, env: SensoryInput) -> BrainContext: ...
    async def post_hooks(self, call: ToolCall, result: ToolResult, ctx: BrainContext) -> None: ...

# norax/prompt/assembler.py
class SystemPromptAssembler:
    def __init__(self, blocks: list["Block"], cache: "PromptCache"): ...
    def compose(self, env: SensoryInput, ctx: BrainContext) -> str: ...

# norax/prompt/blocks/__init__.py — every block class implements:
class Block(Protocol):
    name: str
    def render(self, env: SensoryInput, ctx: BrainContext) -> Optional[str]: ...

# norax/dispatch/dispatcher.py
class Dispatcher:
    def __init__(self, registry, gate, budget, log): ...
    async def dispatch(self, call: ToolCall, ctx: BrainContext) -> ToolResult: ...

# norax/dispatch/registry.py
class ToolRegistry:
    def __init__(self, tools: list["Tool"]): ...
    def get(self, name: str) -> "Tool": ...
    def list(self, caller: Principal) -> list[str]: ...

# norax/dispatch/tools/base.py
class Tool(Protocol):
    name: str
    tier: Literal["T0","T1","T2","dynamic"]
    description: str
    schema: dict   # JSON schema for arguments
    async def invoke(self, args: dict, ctx: BrainContext) -> Any: ...

# norax/gateway_client/client.py
class GatewayClient:
    def __init__(self, base_url: str): ...
    async def call(self, system: str, messages: list[dict], route: str) -> dict: ...

# norax/memory/event_store.py
class EventStore:
    def __init__(self, path: Path): ...
    async def append(self, event: dict) -> None: ...
    async def replay(self) -> AsyncIterator[dict]: ...

# norax/safety/soul.py
class Soul:
    @classmethod
    def load(cls, path: Path) -> "Soul": ...
    def is_owner(self, principal: Principal) -> bool: ...
    def dangerous(self, command: str) -> bool: ...
```

---

## 5. The 12 motor tools (T1–T12) — final list

Reconciled across `norax_tools.md` and `system_prompt_and_skills.md`.
The tool count is exactly 12. Skill-management tools count as one
group (skill_open + skill_add + skill_edit + skill_remove are
sibling implementations under one logical surface).

| # | Tool | Tier | Used by |
|---|---|---|---|
| T1 | `read` | T0 | every skill |
| T2 | `write` | T1 | project-scaffold, memory-care, research |
| T3 | `edit` | T1 | memory-care, healthcheck |
| T4 | `exec` | T2 | healthcheck, github-ops, project-scaffold |
| T5 | `web_fetch` | T0 | research, github-ops |
| T6 | `web_search` | T0 | research |
| T7 | `tts` | T1 | voice-reply |
| T8 | `discord_send` | T1 | discord-ops |
| T9 | `skill_*` (open/add/edit/remove) | T0/T1 | brain itself + LLM |
| T10 | `code_launch` (+ read_log + list_diff + merge + discard) | T2 | coding-session |
| T11 | `nodes_invoke` | T2 | device-ops |
| T12 | `schedule` | T0 | reminders (creates an S4 envelope at fire-time) |

**Tier semantics:**
- **T0**: read-only or self-scoped writes (memory only). No gate.
- **T1**: side-effect on local FS or our own services. RiskGate
  scans args; budget counts.
- **T2**: side-effect on the world (exec, network outbound to
  third parties, paired-device control). Requires owner approval
  unless `auto_approve` flag is set in policy for that specific
  pattern.

---

## 6. Configuration files (YAML, ground-truth)

### 6a. `config/runtime.yaml`

```yaml
runtime:
  log_dir: ./logs
  state_dir: ./state
  event_log: ./state/events.jsonl
  snapshot_dir: ./state/snapshots
  shutdown_grace_seconds: 10
  heartbeat:
    enabled: true
    interval_seconds: 1800        # 30 min, the only timer
  prompt_cache:
    enabled: true
    invalidate_on: [skills_dir, memory, focus, state, runtime]

owner:
  id: null                         # set with NORAX_OWNER_ID
  label: "Owner"
  channels: []

paths:
  soul: ./SOUL.md
  identity: ./IDENTITY.md
  user: ./USER.md
  memory_root: ./memory
```

### 6b. `config/tools.yaml`

```yaml
tools:
  - { name: read,           tier: T0, impl: norax.dispatch.tools.read_tool:ReadTool }
  - { name: write,          tier: T1, impl: norax.dispatch.tools.write_tool:WriteTool }
  - { name: edit,           tier: T1, impl: norax.dispatch.tools.edit_tool:EditTool }
  - { name: exec,           tier: T2, impl: norax.dispatch.tools.exec_tool:ExecTool }
  - { name: web_fetch,      tier: T0, impl: norax.dispatch.tools.web_fetch_tool:WebFetchTool }
  - { name: web_search,     tier: T0, impl: norax.dispatch.tools.web_search_tool:WebSearchTool }
  - { name: tts,            tier: T1, impl: norax.dispatch.tools.tts_tool:TTSTool }
  - { name: discord_send,   tier: T1, impl: norax.dispatch.tools.discord_send_tool:DiscordSendTool }
  - { name: skill_open,     tier: T0, impl: norax.dispatch.tools.skill_open_tool:SkillOpenTool }
  - { name: skill_add,      tier: T1, impl: norax.dispatch.tools.skill_add_tool:SkillAddTool }
  - { name: skill_edit,     tier: T1, impl: norax.dispatch.tools.skill_edit_tool:SkillEditTool }
  - { name: skill_remove,   tier: T1, impl: norax.dispatch.tools.skill_remove_tool:SkillRemoveTool }
  - { name: code_launch,    tier: T2, impl: norax.dispatch.tools.code_launch_tool:CodeLaunchTool }
  - { name: nodes_invoke,   tier: T2, impl: norax.dispatch.tools.nodes_invoke_tool:NodesInvokeTool }
  - { name: schedule,       tier: T0, impl: norax.dispatch.tools.schedule_tool:ScheduleTool }

budget:
  default_caller:
    requests_per_day: 5000
    tokens_per_day: 2_000_000
    usd_per_month: 50.0
  owner:
    requests_per_day: 200000
    tokens_per_day: 200_000_000
    usd_per_month: 500.0
```

### 6c. `config/adapters.yaml`

```yaml
adapters:
  discord:
    enabled: true
    bot_token_env: DISCORD_BOT_TOKEN
    intents: ["messages","reactions","threads","voice_states"]
    allowlist_channels: []        # empty = all channels owner is in
  http:
    enabled: true
    bind: "127.0.0.1:4101"
    api_keys_env: NORAX_HTTP_KEYS
  email:
    enabled: true
    feed: "tcp://127.0.0.1:18797"   # gmail-monitor
  payment:
    enabled: true
    feed: "tcp://127.0.0.1:18799"   # cashapp-monitor
  schedule:
    enabled: true
    spool: "./state/schedule.jsonl"
  heartbeat:
    enabled: true
    interval_seconds: 1800
  node:
    enabled: false                  # turn on when paired devices needed
```

### 6d. `config/gateway.yaml`

Inherits from `master_proxy.md` — port the canonical 21-filter
pipeline configuration verbatim. Additionally, this file carries
the rich **per-model metadata** (alias, contextTokens, maxTokens,
cacheRetention=long) we rely on for routing and prompt-cache
reuse:

```yaml
models:
  ollama/llama4-sonnet:cloud:
    alias: sonnet-4.6
    contextTokens: 200000
    maxTokens: 8000
    cacheRetention: long
  ollama/llama4-opus:cloud:
    alias: opus-4.7
    contextTokens: 200000
    maxTokens: 8000
    cacheRetention: long
  ollama/qwen3-coder-next:cloud:
    alias: qwen3-coder
    contextTokens: 262144
    cacheRetention: short
  ollama/glm-5.1:cloud:
    alias: glm51
    contextTokens: 131072
    cacheRetention: short
  # ... additional model aliases follow the same schema
```

`cacheRetention: long` is **load-bearing** for Anthropic prompt
caching; without it Claude-route token costs roughly double.

### 6e. `config/policy.yaml`

```yaml
policy:
  dangerous_patterns:               # expanded by RiskGate against exec args
    - 'rm -rf'
    - 'DROP\s+(TABLE|DATABASE)'
    - '\bdd\b.*if=/dev'
    - 'chmod\s+777\s+/etc'
    - 'chmod\s+777\s+/boot'
  workspace_exempt_paths:           # patterns above don't fire in these roots
    - "~/norax/state"
    - "~/norax/logs"
    - "~/norax/code"
  auto_approve:                     # T2 patterns that skip owner approval
    - tool: exec
      args_match: '^uv (sync|run|pip)\b'
    - tool: exec
      args_match: '^git (status|diff|log|branch|fetch)\b'
  elevated:                         # owner-only T2 bypass envelope
    enabled: true
    allow_from:
      discord: []                   # configured privately per installation
    # Semantics: T2 calls from listed principals skip per-call owner
    # approval when the args do NOT match any dangerous_patterns.
  loop_guard:                       # dispatcher-level repeat detector
    enabled: true
    window: 40                      # recent tool calls tracked per caller
    warn: 10                        # emit loop_warn event
    critical: 20                    # force next turn to wait for fresh input
    breaker: 30                     # refuse identical calls until a non-call turn
    detectors: [genericRepeat, knownPollNoProgress, pingPong]
```

---

## 7. Phase 1 starting scaffold (the first commit)

Phase 1 produces the **skeleton runtime**: every package present,
every class declared, every method `raise NotImplementedError`,
and a `python -m norax` entrypoint that:

1. Loads config.
2. Initializes the runtime with **no-op brain, no-op dispatcher,
   no-op gateway client**.
3. Subscribes to a single test adapter (HTTP POST `/ingress/test`
   that creates a synthetic `SensoryInput`).
4. Writes every event to `state/events.jsonl`.
5. Logs to `logs/runtime.jsonl`.
6. Exits cleanly on SIGTERM with full event drain.

**Acceptance criteria for Phase 1 done:**

| Test | Passes? |
|---|---|
| `python -m norax` starts in < 500 ms | ✅ |
| `pytest tests/unit/` runs (every test exists, may be `pytest.skip`) | ✅ |
| Cycle-import test passes (no forbidden cross-imports) | ✅ |
| HTTP `POST /ingress/test {body:"hi"}` produces an event in `state/events.jsonl` | ✅ |
| `SIGTERM` drains and exits within 10 s | ✅ |
| `pytest tests/test_imports.py` passes (boundary check) | ✅ |
| `python -m norax.verify_event_chain --all-generations` validates event history | ✅ |

Nothing intelligent yet. Just the bones, breathing.

---

## 8. Phase 2–7 mapping (what fills each package)

| Phase | Package(s) filled | Acceptance |
|---|---|---|
| 2 | `gateway_client/`, plus the separate `norax-gateway` service (separate repo or sibling pkg) | 1k-turn golden replay byte-identical |
| 3 | `brain/` all 36 layers; `memory/` all 11 stores | Hot-path < 60ms p95 on synthetic envelopes; replay rebuilds clean |
| 4 | `dispatch/` with all 12 tools; `safety/` complete | 100 canned tool tests pass; gate trips on dangerous patterns |
| 5 | `adapter/` real adapters (Discord, HTTP, email, payment, schedule, heartbeat, node); `prompt/` assembler complete | Discord events arrive; HTTP serves status+metrics; assembler produces valid prompts |
| 6 | Owner QA in dedicated test channel | Owner approves quality, behavior, safety |
| 7 | Production cutover — point Discord token + noraxdev.org callbacks at norax-runtime | 14 days stable |

Phase 8 is workspace cleanup (archive the prior experiment), no
package work.

---

## 9. Dependency graph (top-level imports allowed)

```
runtime ─────────────────────────────────────────────────────┐
   ├─► adapter ─► envelope                                  │
   ├─► brain ─► envelope, memory, safety                    │
   ├─► prompt ─► envelope, brain.context (read), dispatch.registry (read), filesystem
   ├─► dispatch ─► envelope, safety, memory (for skill_*), per-tool deps
   ├─► gateway_client ─► envelope, httpx
   ├─► memory ─► (no internal deps; pure I/O)
   ├─► safety ─► envelope (Principal)
   ├─► http ─► runtime, dispatch (drafts only), memory (read), envelope
   ├─► code ─► subprocess, filesystem (worktree)
   ├─► observability ─► filesystem, prometheus_client
   └─► config ─► pydantic, yaml
```

Allowed test/utility outliers: `tests/` may import anything;
`scripts/` may import `memory` and `config` for replay/snapshot.

---

## 10. Python dependencies (pinned in `pyproject.toml`)

Minimal set. No framework where stdlib will do.

```toml
[project]
name = "norax"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.32",
  "httpx>=0.27",
  "pydantic>=2.9",
  "pyyaml>=6.0",
  "structlog>=24.4",
  "ulid-py>=1.1",
  "discord.py>=2.4",
  "faiss-cpu>=1.9",                # in-process vectors
  "prometheus-client>=0.21",
  "anyio>=4.6",
  "typer>=0.13",                   # CLI
]

[project.optional-dependencies]
dev = [
  "pytest>=8.3",
  "pytest-asyncio>=0.24",
  "pytest-cov>=5.0",
  "ruff>=0.7",
  "mypy>=1.13",
  "respx>=0.21",                   # httpx mocking
]
```

Embedded sqlite, hashlib, secrets, json, asyncio, pathlib — all
stdlib.

**No** langchain, **no** langgraph, **no** crew, **no** autogen,
**no** semantic-kernel, **no** llamaindex. We have a brain; we
don't need someone else's prompt-graph.

---

## 11. Naming conventions

- Modules: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE`.
- Layer files: `l{NN}_{name}.py`, e.g. `l14_amygdala.py`.
- Tool files: `{name}_tool.py`.
- Adapter files: `{channel}_in.py`.
- Async by default. Any sync function gets a `_sync` suffix and a
  comment explaining why.

---

## 12. Observability invariants (hard rules)

Every event written to `events.jsonl` has, at minimum:

```json
{
  "schema_version": "v1",
  "ts": "2026-04-22T23:45:01.234Z",
  "kind": "ingress|brain|dispatch|gateway|reply|skill_added|risk|budget|error|circuit|loop|task_crash|back_pressure_drop|subprocess_unhealthy",
  "trace_id": "01H...",                 // ULID per turn
  "span_id": "01H...",                  // ULID per layer/tool fire
  "parent_span_id": null|"01H...",
  "prev_hash": "sha256...",             // chain to previous event
  "hash": "sha256(prev_hash + payload)",// integrity check on replay
  "attrs": {
    "gen_ai.system": "anthropic",
    "gen_ai.request.model": "claude-sonnet-4.6",
    "gen_ai.usage.input_tokens": 1234,
    "gen_ai.usage.output_tokens": 567,
    "gen_ai.tool.name": "exec",
    "gen_ai.tool.call.id": "01H..."
  },
  "payload": {...}                       // kind-specific (post secret-redact)
}
```

`trace_id` lets us reconstruct an entire turn from the log. We
never log secrets in `payload`; the safety package's
`secrets.redact()` runs on every payload before write. Hash chain
is validated on every replay; mismatch emits a `chain_break`
event and aborts the rebuild.

Attribute keys follow **OpenTelemetry GenAI semantic
conventions** (`gen_ai.*`). Default exporter: JSONL only. If
`OTEL_EXPORTER_OTLP_ENDPOINT` env is set, also export to that
collector. No collector required to run.

---

## 13. Testing strategy

| Tier | What | Where |
|---|---|---|
| Unit | Each class in isolation, dependencies mocked | `tests/unit/<package>/` |
| Boundary | Import graph enforced (no cycles, no forbidden cross-imports) | `tests/boundary/` |
| Golden | Gateway pipeline byte-identical replay over 1,000 turns | `tests/golden/` |
| Behavioral | Synthetic SensoryInputs through the full pipeline; assert on motor calls and replies | `tests/integration/` |
| Safety | DANGEROUS regex matrix; injection-prompt corpus; budget overflow | `tests/safety/` |
| Property | hypothesis-driven shape checks on dataclasses | `tests/property/` |

Coverage target Phase 1: **just the boundary tests pass.**
Coverage target Phase 7: **≥ 85% line coverage on `dispatch/`,
`safety/`, `prompt/`, `memory/`, `brain/hot_path/`**.

---

## 14. The "next action" checklist

When this doc is approved, the **literal first commands** to run:

```bash
cd ~/norax
mkdir -p norax/{runtime,envelope,adapter,brain/{hot_path,post,in_turn_bg,sleep},prompt/blocks,dispatch/tools,gateway_client,memory/stores,safety,http/routes,code,observability,config}
mkdir -p config memory/{procedural/{skills,archive},semantic,intel,episodic} state/{snapshots,cache} logs tests/{unit,integration,golden,boundary,safety,property} scripts
touch pyproject.toml uv.lock .python-version README.md
echo "3.12" > .python-version
```

Then create `pyproject.toml` from §10, run `uv sync`, then write
the empty package files with `Protocol`/class skeletons from §4
and §5. Phase 1 done when §7 acceptance criteria pass.

---

## 15. Decisions locked by this doc

| Question | Decision |
|---|---|
| Language | Python 3.12+ |
| Async runtime | `asyncio` (anyio for compat shims) |
| HTTP server | FastAPI + uvicorn |
| HTTP client | httpx |
| Vector store | faiss-cpu in-process (no LanceDB subprocess) |
| KG store | sqlite |
| Event log | JSONL append-only |
| Discord client | `discord.py` directly |
| Process model | Single `norax-runtime` + sibling `norax-gateway` |
| Background tasks | **None** except heartbeat emitter |
| Sub-agents | **None.** Independent coding sessions only. |
| System-prompt assembly | 11-block deterministic, ours |
| Skills | ~10 seed, agent-authored growth via `skill_add` |
| Compaction | **Does not exist** in any module |
| Memory snapshots | Optimization only; events.jsonl is truth |
| Tier model | T0/T1/T2 + auto_approve patterns |
| Owner approval channel | HTTP draft routes + Discord interactive |
| Tests | pytest + pytest-asyncio + hypothesis (later) |
| Linting | ruff |
| Type checking | mypy strict on `envelope/`, `dispatch/`, `safety/`; pragmatic elsewhere |
| Secret handling | env vars referenced by `*_env` keys in YAML; never inlined |
| Versioning | semver on `norax/version.py`; bump per phase |

Anything not listed in this table is a Phase-2+ implementation
detail and may change.

---

## 16. What this doc does NOT specify (intentionally)

- Brain-layer implementations beyond their I/O contract — that's
  in `norax_brain_stack.md` and the layer-specific code.
- Gateway internals — that's `master_proxy.md`.
- Skill content — those get authored in Phase 4 and beyond.
- Specific reasoning prompts inside layers — emerge during build.
- Provider routing weights — config, not code; tune in production.

---

## 17. TL;DR

Single Python package `norax/` with 12 sub-packages
(`runtime`, `envelope`, `adapter`, `brain`, `prompt`, `dispatch`,
`gateway_client`, `memory`, `safety`, `http`, `code`,
`observability`). Strict downward-only imports. Pure-data
`envelope/` shared by all. 12 motor tools, 36 brain layers, 11
prompt blocks, 6 sensory adapters, 11 memory stores, 1 event log.
Zero background tasks except the heartbeat emitter. Zero
sub-agents. Zero compaction. Phase 1 = skeleton + boundary tests
pass. Phases 2–7 fill packages in order. Production cutover at
Phase 7 by pointing the Discord token and noraxdev.org callbacks
at `norax-runtime`. Phase 8 archives the prior experiment.

When you say go, the literal first commands are in §14.
r be
   hard-killed.
3. The IngressBus uses a bounded `asyncio.Queue(maxsize=256)`. If
   an adapter would block on enqueue, it MUST drop+log a
   `back_pressure_drop` event rather than block the producer.
4. Every Task in the runtime has an exception handler attached
   via `add_done_callback`; uncaught exceptions emit a
   `task_crash` event and trigger an orderly shutdown if the
   task was a critical-path one (runtime, brain.pipeline,
   ingress_bus).
5. All blocking I/O (sqlite writes, faiss writes, file I/O over
   1KB) goes through `asyncio.to_thread` so it doesn't stall the
   event loop.
6. No `time.sleep`. Ever. Always `asyncio.sleep`.
7. Tests assert the loop has no unawaited coroutines and no
   pending tasks at end-of-turn.

---

## 19. Reliability primitives (always-on)

- **Idempotency**: dispatcher dedupes by `(caller_id, call_id)`
  in a 60s LRU.
- **Circuit breakers**: gateway client + every outbound HTTP tool
  use a 5-fail/30s sliding-window breaker, 30s open, 1-probe
  half-open.
- **Retries with jitter**: exponential `100ms / 250ms / 500ms`
  ±50ms, max 3 attempts. Idempotent calls only (gateway, web
  fetch, discord send). Excluded: exec, write, edit.
- **Schema-versioned events**: every event carries
  `schema_version: "v1"`; bumps require a migration script.
- **Hash-chained event log**: each event includes
  `prev_hash` and `hash`; replay verifies the chain.
- **Secret scrubber**: `safety/secrets.py` regex set runs on
  every payload before any log write.
- **Path containment**: read/write/edit tools reject paths
  outside `/path/to/norax` + configured external dirs.
- **Task and process supervision**: runtime-owned tasks are observed by
  `runtime/core.py`, readiness is reported through the capability registry,
  and systemd owns process restart policy.
- **Untrusted-content fencing**: assembler renders any
  `SensoryInput.trusted == False` body inside an
  `<<<EXTERNAL_UNTRUSTED_CONTENT>>>` envelope in the user-role
  message, never the system prompt.

---

## 20. Operational artifacts (Phase 6 deliverable)

`operations/runbook.md`:
- Start/stop/restart for every `norax-*` systemd unit
- Where the logs are; how to grep them
- How to replay events.jsonl to recover state
- How to roll back to a snapshot
- How to bypass-disable a tool via `policy.yaml: deny: [...]`
- How to drain queues before a deploy
- How to extract a turn's full trace by `trace_id`
- What "healthy" looks like (reference metric thresholds)

This is the doc the owner reads at 2am when something is
wrong.
