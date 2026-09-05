# Norax Improvement Framework
## Comprehensive Implementation Map for Stack Upgrades

**Version:** 1.0  
**Date:** 2026-04-24  
**Status:** Planning  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Assessment](#2-current-state-assessment)
3. [Web Search & Fetch Improvements](#3-web-search--fetch-improvements)
4. [Harness & Runtime Improvements](#4-harness--runtime-improvements)
5. [Memory System Improvements](#5-memory-system-improvements)
6. [Coding Ability Improvements](#6-coding-ability-improvements)
7. [24/7 Uptime & Reliability](#7-247-uptime--reliability)
8. [Prompt & Context Management](#8-prompt--context-management)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [Risk Assessment & Safety](#10-risk-assessment--safety)

---

## 1. Executive Summary

This document maps every improvement we should make to Norax Gen5, organized by domain, with specific implementation details tied to our actual codebase. Each recommendation includes: current state, target state, specific files to modify, research-backed justification, and safety considerations.

**Philosophy:** No jank. Every change must be tested on staging before reaching production Norax. Every change must be reversible.

---

## 2. Current State Assessment

### 2.1 Architecture Overview

```
norax/
├── brain/           # Core reasoning (phase_orchestrator, risk_gate)
├── dispatch/        # Gateway, inbound routing, rate limiting  
├── envelope/        # Message formatting, Discord integration
├── memory/          # Episodic, semantic, procedural, state
├── prompt/          # System prompts, skill loading
├── skills/          # Loadable capability modules
├── tools/           # Function definitions (web_search, exec, etc.)
└── utils/           # Crypto, hashing, config
```

### 2.2 What We Do Well

- **Hash-chained event sourcing** — every turn is cryptographically linked
- **Hippocampal replay** — memory consolidation during sleep cycles
- **Phase orchestrator** — structured reasoning pipeline
- **Risk gate** — dangerous pattern detection before execution
- **Multi-store memory** — semantic, episodic, procedural, state
- **Gateway** — Discord event routing with event sourcing

### 2.3 Key Gaps

- No vector search index (semantic search is brute-force)
- Web search is basic (no reranking, no caching, no query expansion)
- No turn-level checkpointing (crash = lost context)
- No Docker sandbox for code execution
- No self-diagnosis/heartbeat beyond basic health
- Memory tools don't support agent-self-edit of procedural knowledge
- No degraded mode when upstream gateways fail
- Context trimming is crude (character-based, not importance-based)


---

## 3. Web Search & Fetch Improvements

### 3.1 Current Web Search Pipeline

**How it works now:**
1. `web_search(query)` → hits SearXNG instance
2. SearXNG fans out to Bing, Brave, DuckDuckGo, Google
3. Returns `{titles, urls, snippets}` — raw, unranked
4. No caching, no dedup, no query refinement
5. No reranking of results by relevance to current task

**Current `web_fetch`:**
1. `web_fetch(url)` → HTTP GET with `max_chars` truncation
2. Returns raw text, no parsing, no extraction
3. No JavaScript rendering
4. No retry on failure
5. No content-type awareness (PDF, HTML, JSON treated same)

### 3.2 Research: Best Practices from Production RAG Systems

**Source: LlamaIndex, LangChain, Anthropic RAG guides, Haystack docs**

| Best Practice | Our Status | Gap Level |
|---|---|---|
| Query decomposition (break complex queries into sub-queries) | ❌ Not implemented | High |
| Query expansion (add synonyms/context before searching) | ❌ Not implemented | High |
| Result reranking by relevance to task | ❌ Not implemented | High |
| Result deduplication (same content across engines) | ❌ Not implemented | Medium |
| Web content caching with TTL | ❌ Not implemented | Medium |
| Structured content extraction (not raw text) | ❌ Not implemented | High |
| Fallback chain (SearXNG→Serper→Direct API) | Partial (SearXNG→Ollama→Serper) | Low |
| Search result confidence scoring | ❌ Not implemented | Medium |
| Multi-hop search (follow links from results) | ❌ Not implemented | High |
| Rate limit awareness + backoff | Partial | Low |

### 3.3 Implementation Plan: Enhanced Web Search

#### Phase 3A: Query Intelligence Layer

**File:** `tools/web_search_enhanced.py` (new)

```python
class EnhancedWebSearch:
    """
    Drop-in replacement for raw web_search that adds:
    - Query decomposition for complex questions
    - Query expansion with domain context
    - Result reranking via embedding similarity
    - Deduplication by URL and content hash
    - Caching with TTL
    """
    
    def search(self, query: str, context: str = "", depth: int = 1):
        # 1. Decompose complex queries
        sub_queries = self._decompose(query) if self._is_complex(query) else [query]
        
        # 2. Expand each sub-query with context
        expanded = [self._expand(sq, context) for sq in sub_queries]
        
        # 3. Search with fallback chain
        raw_results = []
        for eq in expanded:
            cached = self.cache.get(eq)
            if cached:
                raw_results.extend(cached)
            else:
                results = self._search_with_fallback(eq)
                self.cache.set(eq, results, ttl=3600)
                raw_results.extend(results)
        
        # 4. Deduplicate
        deduped = self._deduplicate(raw_results)
        
        # 5. Rerank by relevance to original query
        ranked = self._rerank(deduped, query, context)
        
        return ranked
```

**Query Decomposition Logic:**
- If query contains "and", "vs", "compared to", "also", "while" → split
- If query has multiple question markers → split
- Otherwise, search as-is
- Use LLM call (small model via Ollama) for ambiguous cases

**Query Expansion Logic:**
- Append task context (current conversation topic)
- Add domain-specific synonyms from procedural memory
- Keep original query alongside expanded (search both, merge results)

#### Phase 3B: Enhanced Web Fetch

**File:** `tools/web_fetch_enhanced.py` (new)

```python
class EnhancedWebFetch:
    """
    Drop-in replacement for raw web_fetch that adds:
    - Content-type aware parsing (HTML→markdown, JSON→structured, PDF→text)
    - JavaScript rendering via headless browser (optional)
    - Smart truncation (extract relevant sections, not just first N chars)
    - Retry with exponential backoff
    - Rate limit detection
    """
    
    async def fetch(self, url: str, query: str = "", max_chars: int = 5000):
        # 1. Detect content type
        content_type = await self._detect_type(url)
        
        # 2. Fetch with retry
        raw = await self._fetch_with_retry(url, retries=3)
        
        # 3. Parse based on type
        if content_type == 'html':
            content = self._html_to_markdown(raw)
        elif content_type == 'json':
            content = self._json_to_text(raw)
        elif content_type == 'pdf':
            content = self._pdf_to_text(raw)
        else:
            content = raw[:max_chars]
        
        # 4. If query provided, extract relevant sections
        if query:
            content = self._extract_relevant(content, query, max_chars)
        
        return content
```

**HTML→Markdown conversion:**
- Use `markdownify` or `trafilatura` library
- Strip navigation, footers, ads via readability extraction
- Preserve tables, code blocks, lists

**Smart truncation:**
- If query is given, find paragraphs/sections most relevant
- Score each section by keyword overlap with query
- Return top-scoring sections up to max_chars
- Always include document title + first paragraph as context

#### Phase 3C: Search Result Reranking

**File:** `tools/reranker.py` (new)

Two options ranked by implementation complexity:

**Option A: Keyword-based reranking (no model needed)**
```python
def keyword_rerank(results, query):
    """Score results by query term frequency, position, and snippet length."""
    query_terms = set(query.lower().split())
    scored = []
    for r in results:
        score = 0
        text = (r.get('title','') + ' ' + r.get('snippet','')).lower()
        for term in query_terms:
            count = text.count(term)
            # Title matches worth 3x snippet matches
            score += count * (3 if term in r.get('title','').lower() else 1)
        scored.append((score, r))
    return [r for s, r in sorted(scored, reverse=True)]
```

**Option B: Embedding-based reranking (requires embedding model)**
```python
def embedding_rerank(results, query, model="nomic-embed-text"):
    """Score results by cosine similarity to query embedding."""
    query_emb = ollama.embeddings(model=model, prompt=query)
    scored = []
    for r in results:
        doc_text = r.get('title','') + '. ' + r.get('snippet','')
        doc_emb = ollama.embeddings(model=model, prompt=doc_text)
        sim = cosine_similarity(query_emb, doc_emb)
        scored.append((sim, r))
    return [r for s, r in sorted(scored, reverse=True)]
```

**Recommendation:** Start with Option A (zero dependencies), upgrade to Option B after embedding pipeline is set up in Memory Phase 5B.

#### Phase 3D: Multi-Hop Search

**File:** `tools/multi_hop_search.py` (new)

```python
async def multi_hop_search(query, max_hops=2):
    """
    For deep research tasks:
    1. Search for query
    2. Fetch top 2-3 results
    3. Extract key claims that need verification
    4. Search for verification of those claims
    5. Return consolidated, cited results
    """
    results = web_search(query)
    claims = []
    for url in results[:3]:
        content = web_fetch(url, query=query)
        claims.extend(extract_claims(content))
    
    # Verify top claims with second search
    verified = []
    for claim in claims[:5]:
        verification = web_search(f"verify: {claim}")
        if verification:
            verified.append({
                'claim': claim,
                'sources': [url] + [v['url'] for v in verification[:2]],
                'confidence': assess_confidence(claim, verification)
            })
    
    return verified
```

**Safety:** Multi-hop is expensive (3-10x API calls). Only invoke when:
- User explicitly asks for research/deep search
- Task requires factual verification
- Never for simple lookups

### 3.4 Web Search Safety Guardrails

- **Rate limiting:** Max 10 searches per minute, 100 per hour
- **Domain blocklist:** No .onion, no known malware domains
- **Content sanitization:** Strip all HTML/script tags from fetched content
- **PII detection:** Scan fetched content for emails/phones, mask before storing
- **Cache eviction:** 24h max TTL, auto-evict on memory pressure


---

## 4. Harness & Runtime Improvements

### 4.1 Current Harness Architecture

```
Discord → Gateway (dispatch/) → Brain (brain/) → Tools → Response
                ↓
          Event Store (hash-chained)
                ↓
          Memory System (memory/)
```

**Key files:**
- `brain/phase_orchestrator.py` — main reasoning loop
- `brain/risk_gate.py` — dangerous pattern detection
- `dispatch/gateway.py` — Discord event routing
- `dispatch/inbound.py` — message processing pipeline
- `dispatch/rate_limiter.py` — request throttling

### 4.2 Turn-Level Checkpointing

**Problem:** If Norax crashes mid-turn, all context for that turn is lost. The event store has the input but not the intermediate reasoning state.

**Solution:** Checkpoint after each phase in the orchestrator.

**File:** `brain/checkpoint.py` (new)

```python
import json, hashlib, os
from pathlib import Path

class TurnCheckpoint:
    """Save/restore turn state at each orchestrator phase."""
    
    def __init__(self, turn_id: str, store_dir: str = "/tmp/norax_checkpoints"):
        self.turn_id = turn_id
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True)
    
    def save(self, phase: str, state: dict):
        path = self.store_dir / f"{self.turn_id}_{phase}.json"
        payload = {
            'turn_id': self.turn_id,
            'phase': phase,
            'state': state,
            'hash': hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        }
        path.write_text(json.dumps(payload, indent=2))
        return str(path)
    
    def load(self, phase: str) -> dict | None:
        path = self.store_dir / f"{self.turn_id}_{phase}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        # Verify integrity
        expected = hashlib.sha256(json.dumps(payload['state'], sort_keys=True).encode()).hexdigest()
        if payload['hash'] != expected:
            raise ValueError(f"Checkpoint corrupted for {self.turn_id}_{phase}")
        return payload['state']
    
    def cleanup(self):
        """Remove checkpoints after turn completes successfully."""
        for f in self.store_dir.glob(f"{self.turn_id}_*.json"):
            f.unlink()
```

**Integration into `phase_orchestrator.py`:**

```python
# Before each phase:
checkpoint.save(phase_name, current_state)

# On crash recovery (gateway startup):
last_incomplete = find_incomplete_checkpoints()
if last_incomplete:
    state = checkpoint.load(last_incomplete.phase)
    resume_from_phase(last_incomplete.phase, state)
```

**Safety:** Checkpoints stored in `/tmp` (ephemeral). Auto-cleanup on successful completion. Max age 1 hour — stale checkpoints are discarded.

### 4.3 Inbound Message Queue

**Problem:** If the brain is busy processing a turn and a new message arrives, it either gets dropped or causes concurrent processing bugs.

**Solution:** Persistent inbound queue with strict FIFO processing.

**File:** `dispatch/message_queue.py` (new)

```python
import asyncio, json
from pathlib import Path
from collections import deque

class PersistentMessageQueue:
    """
    File-backed FIFO queue for inbound messages.
    Survives crashes — messages not acknowledged are reprocessed.
    """
    
    def __init__(self, queue_dir: str = "~/norax/data/queue"):
        self.queue_dir = Path(queue_dir).expanduser()
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._queue = deque()
        self._load_pending()
    
    def _load_pending(self):
        """On startup, reload any unacknowledged messages from disk."""
        for f in sorted(self.queue_dir.glob("msg_*.json")):
            self._queue.append(json.loads(f.read_text()))
    
    async def enqueue(self, message: dict):
        """Add message to queue and persist to disk."""
        msg_id = message.get('message_id', hashlib.md5(json.dumps(message).encode()).hexdigest())
        path = self.queue_dir / f"msg_{msg_id}.json"
        path.write_text(json.dumps(message))
        self._queue.append(message)
    
    async def dequeue(self) -> dict | None:
        """Get next message (non-blocking)."""
        if not self._queue:
            return None
        return self._queue.popleft()
    
    def acknowledge(self, message_id: str):
        """Remove persisted message after successful processing."""
        path = self.queue_dir / f"msg_{message_id}.json"
        if path.exists():
            path.unlink()
    
    @property
    def depth(self) -> int:
        return len(self._queue)
```

**Safeguards:**
- Max queue depth: 100 (new messages get rate-limited at gateway)
- Max age: 5 minutes (stale messages are discarded on startup)
- One-at-a-time processing enforced by brain lock

### 4.4 Graceful Degradation

**File:** `dispatch/degraded_mode.py` (new)

```python
class DegradedModeManager:
    """
    When upstream services fail, degrade gracefully instead of crashing.
    
    Levels:
    - NORMAL: All systems operational
    - DEGRADED_SEARCH: Web search down → use cached results + memory only
    - DEGRADED_MODEL: Primary model down → fallback to local Ollama model
    - DEGRADED_DISCORD: Discord API issues → queue messages, retry
    - MINIMAL: Only local processing, no external calls
    """
    
    LEVELS = ['NORMAL', 'DEGRADED_SEARCH', 'DEGRADED_MODEL', 'DEGRADED_DISCORD', 'MINIMAL']
    
    def __init__(self):
        self.level = 'NORMAL'
        self.failures = {}  # service -> (timestamp, count)
    
    def record_failure(self, service: str):
        ts, count = self.failures.get(service, (0, 0))
        self.failures[service] = (time.time(), count + 1)
        self._recalculate_level()
    
    def record_success(self, service: str):
        self.failures.pop(service, None)
        self._recalculate_level()
    
    def _recalculate_level(self):
        active_failures = {s for s, (ts, c) in self.failures.items() 
                          if time.time() - ts < 300 and c >= 3}
        if not active_failures:
            self.level = 'NORMAL'
        elif 'model' in active_failures:
            self.level = 'DEGRADED_MODEL'
        elif 'search' in active_failures:
            self.level = 'DEGRADED_SEARCH'
        elif 'discord' in active_failures:
            self.level = 'DEGRADED_DISCORD'
        else:
            self.level = 'MINIMAL'
```

### 4.5 Self-Diagnosis Heartbeat

**File:** `brain/health.py` (new)

```python
class SelfDiagnosis:
    """
    Run every 60 seconds. Only alert when something needs attention.
    
    Checks:
    - Memory usage (RSS, swap)
    - Disk space (data partition)
    - Queue depth
    - Last successful turn timestamp
    - Gateway connection status
    - Model response latency
    - Event store integrity (spot check last 10 hashes)
    """
    
    def check(self) -> dict:
        results = {
            'memory_rss_mb': self._check_memory(),
            'disk_free_gb': self._check_disk(),
            'queue_depth': self._check_queue(),
            'last_turn_age_s': self._check_last_turn(),
            'gateway_up': self._check_gateway(),
            'model_latency_ms': self._check_model(),
            'event_chain_ok': self._check_event_chain(),
        }
        alerts = [k for k, v in results.items() if self._is_alert(k, v)]
        return {'status': 'ALERT' if alerts else 'OK', 'alerts': alerts, 'details': results}
```

**Integration with heartbeat system:**
- If `status == 'OK'` → send `HEARTBEAT_OK`
- If `status == 'ALERT'` → send alert details to owner via DM


---

## 5. Memory System Improvements

### 5.1 Current Memory Architecture

```
memory/
├── episodic/     # Short-term conversation buffer
├── semantic/     # Knowledge store (JSON files)
├── procedural/   # Learned procedures and skills
├── state/        # Current agent state (events, scratchpad)
└── hippocampus.py # Replay/consolidation logic
```

**Current strengths:**
- Hippocampal replay consolidates episodic → semantic
- Hash-chained events prevent tampering
- Multi-store separation is clean

**Current weaknesses:**
- Semantic search is brute-force (scans all JSON files)
- No vector index for similarity search
- Agent can't edit its own procedural memory
- No importance scoring for memory eviction
- No cross-session continuity beyond event log

### 5.2 Vector Search Index (FAISS)

**Why:** Semantic search currently scans every file. With 1000+ memories, this becomes slow. FAISS enables sub-millisecond similarity search.

**File:** `memory/vector_index.py` (new)

```python
import faiss
import numpy as np
import json
from pathlib import Path

class VectorMemoryIndex:
    """
    FAISS-backed similarity search over semantic memories.
    
    Index structure:
    - Embedding model: nomic-embed-text (via Ollama, local)
    - Dimension: 768
    - Index type: IVF (inverted file) for >10k entries, flat for <10k
    - Metadata stored alongside: source, timestamp, channel, importance
    """
    
    def __init__(self, index_dir: str = "~/norax/data/vector_index"):
        self.index_dir = Path(index_dir).expanduser()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.dimension = 768
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        index_path = self.index_dir / "memory.faiss"
        meta_path = self.index_dir / "metadata.json"
        
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            self.metadata = json.loads(meta_path.read_text())
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
    
    def add(self, text: str, meta: dict):
        """Embed text and add to index."""
        embedding = self._embed(text)
        self.index.add(np.array([embedding], dtype='float32'))
        self.metadata.append(meta)
        self._save()
    
    def search(self, query: str, k: int = 5) -> list:
        """Find k most similar memories to query."""
        if self.index.ntotal == 0:
            return []
        q_emb = self._embed(query)
        distances, indices = self.index.search(np.array([q_emb], dtype='float32'), min(k, self.index.ntotal))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append({**self.metadata[idx], 'distance': float(dist)})
        return results
    
    def _embed(self, text: str) -> list:
        """Get embedding via Ollama."""
        import ollama
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response['embedding']
    
    def _save(self):
        faiss.write_index(self.index, str(self.index_dir / "memory.faiss"))
        Path(self.index_dir / "metadata.json").write_text(json.dumps(self.metadata))
```

**Integration with `search_memory` tool:**
- Replace brute-force scan with `VectorMemoryIndex.search(query, k)`
- Fall back to brute-force if FAISS index is empty or corrupted
- Auto-rebuild index on startup if metadata/FAISS mismatch detected

**Dependencies:** `pip install faiss-cpu` (~50MB). No GPU required for CPU index.

**Safety:**
- Index is read-only during normal operation (add only, no delete)
- Full rebuild requires explicit admin command
- Metadata includes source tracking (which conversation created this memory)

### 5.3 Agent Self-Edit Memory Tools

**Problem:** Norax can store new memories but can't refine, correct, or prune existing ones. If something was stored incorrectly, it persists forever.

**New tools to add:**

```python
# tools/memory_tools.py

def memory_edit(memory_id: str, field: str, new_value: str) -> str:
    """Edit a specific field of an existing semantic memory entry."""
    # Validates: memory_id exists, field is mutable, caller is owner/self
    # Logs: what changed, from what, to what (audit trail)

def memory_delete(memory_id: str, reason: str) -> str:
    """Delete a memory entry. Requires reason for audit."""
    # Validates: caller is owner, memory exists
    # Soft delete first (mark as deleted), hard delete after 24h
    # Also removes from vector index

def memory_consolidate(channel: str, scope: str = "recent") -> str:
    """
    Force hippocampal replay outside of sleep cycle.
    Useful after high-activity sessions.
    """
    # Runs consolidation on episodic memories for given channel
    # Merges duplicates, resolves conflicts, promotes to semantic
    # Reports: N consolidated, N duplicates merged, N conflicts resolved
```

**Safety guardrails:**
- `memory_delete` requires `reason` field (no silent deletion)
- `memory_edit` logs old + new value to event store (audit trail)
- `memory_consolidate` is idempotent (running twice = no-op for already-consolidated)
- These tools are **self-use only** — no external agent can trigger them

### 5.4 Importance-Based Memory Eviction

**Problem:** Memory grows unbounded. We need a principled way to prune.

**File:** `memory/eviction.py` (new)

```python
class MemoryEviction:
    """
    Score memories for retention importance.
    
    Factors (weighted):
    - Recency: How recently was this memory accessed? (0.3)
    - Relevance: How often is this memory retrieved? (0.25)
    - Source: Owner-stored memories > agent-learned > auto-captured (0.25)
    - Verification: Has this been cross-referenced/cited by later turns? (0.2)
    
    Threshold: Memories below 0.15 composite score after 7 days are candidates.
    """
    
    IMPORTANCE_WEIGHTS = {
        'recency': 0.30,
        'relevance': 0.25,
        'source_weight': 0.25,
        'verification': 0.20,
    }
    
    SOURCE_WEIGHTS = {
        'owner': 1.0,      # Never evict owner-stored memories without approval
        'agent_verified': 0.8,
        'agent_learned': 0.5,
        'auto_captured': 0.3,
    }
    
    def score(self, memory: dict) -> float:
        recency = self._recency_score(memory)
        relevance = self._relevance_score(memory)
        source = self.SOURCE_WEIGHTS.get(memory.get('source_type', 'auto_captured'), 0.3)
        verification = self._verification_score(memory)
        
        total = (
            self.IMPORTANCE_WEIGHTS['recency'] * recency +
            self.IMPORTANCE_WEIGHTS['relevance'] * relevance +
            self.IMPORTANCE_WEIGHTS['source_weight'] * source +
            self.IMPORTANCE_WEIGHTS['verification'] * verification
        )
        return total
    
    def evict_candidates(self, threshold: float = 0.15, min_age_days: int = 7) -> list:
        """Return memories eligible for eviction. Does NOT auto-evict."""
        candidates = []
        for mem in self.semantic_store.all():
            if self._age_days(mem) >= min_age_days and self.score(mem) < threshold:
                candidates.append(mem)
        return sorted(candidates, key=lambda m: self.score(m))
```

**Policy:**
- Never auto-evict. Generate eviction report for owner approval.
- Owner memories (source=owner) are never candidates.
- Run eviction analysis during sleep cycle only.

### 5.5 Cross-Session Continuity

**Problem:** Each session starts with memory search but doesn't carry forward active task context (what was I doing, what's pending).

**Solution:** Richer scratchpad with task tracking.

**File:** `memory/task_context.py` (new)

```python
class TaskContext:
    """
    Persistent across sessions via state/scratchpad.
    
    Tracks:
    - Active tasks (what the user asked me to do)
    - Pending actions (awaiting confirmation, waiting on external)
    - Recent outcomes (last 5 completed tasks + results)
    - User preferences learned this session
    - Environment notes (services up/down, quirks discovered)
    """
    
    FIELDS = ['active_tasks', 'pending_actions', 'recent_outcomes', 
              'learned_preferences', 'environment_notes']
    
    def update(self, field: str, entry: dict):
        """Add or update an entry in the specified field."""
        ...
    
    def snapshot(self) -> dict:
        """Full context snapshot for prompt injection."""
        return {
            'active': self.active_tasks,
            'pending': self.pending_actions,
            'recent': self.recent_outcomes[-5:],
            'preferences': self.learned_preferences,
            'environment': self.environment_notes,
        }
```

**Integration:** This snapshot gets injected into the system prompt on session start, replacing the current crude `STATE;valence=...` block with richer context.


---

## 6. Coding Ability Improvements

### 6.1 Current Coding Capabilities

Norax can currently:
- Read, write, edit files via tools
- Execute shell commands via `exec`
- Edit specific code sections via `edit` (find-and-replace)

Norax currently CANNOT:
- Run code in an isolated sandbox
- Automatically test code after writing it
- Iteratively fix failing tests
- Generate diffs for review before applying
- Understand project structure without manual exploration

### 6.2 Docker Sandbox for Code Execution

**Why:** Currently all `exec` calls run directly on the host. This is dangerous for:
- Running untrusted code from web searches
- Testing potentially destructive scripts
- Preventing accidental system damage

**File:** `tools/sandbox.py` (new)

```python
import docker, tempfile, json
from pathlib import Path

class CodeSandbox:
    """
    Execute code in an isolated Docker container.
    
    Features:
    - Ephemeral containers (destroyed after execution)
    - Resource limits (CPU, memory, time)
    - Network isolation (no internet by default)
    - File injection (pass code in, get results out)
    - Supported languages: Python, JavaScript, Bash
    """
    
    def __init__(self):
        self.client = docker.from_env()
        self.timeout = 30  # seconds
        self.memory_limit = '256m'
        self.cpu_period = 100000
        self.cpu_quota = 50000  # 0.5 CPU
    
    def execute(self, code: str, language: str = "python", 
                files: dict = None, network: bool = False) -> dict:
        """
        Run code in sandbox, return stdout/stderr/exit_code.
        
        files: {filename: content} to inject into /workspace/
        """
        # Create temp directory with code
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code file
            ext = {'python': '.py', 'javascript': '.js', 'bash': '.sh'}[language]
            code_path = Path(tmpdir) / f"main{ext}"
            code_path.write_text(code)
            
            # Write any extra files
            if files:
                for name, content in files.items():
                    (Path(tmpdir) / name).write_text(content)
            
            # Select image
            image = {
                'python': 'norax/sandbox-python:latest',
                'javascript': 'norax/sandbox-node:latest', 
                'bash': 'norax/sandbox-bash:latest',
            }[language]
            
            try:
                container = self.client.containers.run(
                    image=image,
                    command=f"main{ext}",
                    volumes={tmpdir: {'bind': '/workspace', 'mode': 'rw'}},
                    working_dir='/workspace',
                    mem_limit=self.memory_limit,
                    cpu_period=self.cpu_period,
                    cpu_quota=self.cpu_quota,
                    network_disabled=not network,
                    detach=True,
                )
                
                result = container.wait(timeout=self.timeout)
                stdout = container.logs(stdout=True, stderr=False).decode()
                stderr = container.logs(stdout=False, stderr=True).decode()
                exit_code = result.get('StatusCode', -1)
                
                container.remove()
                
                return {
                    'stdout': stdout[:5000],
                    'stderr': stderr[:5000],
                    'exit_code': exit_code,
                    'timed_out': False,
                }
                
            except Exception as e:
                try:
                    container.kill()
                    container.remove()
                except:
                    pass
                return {
                    'stdout': '',
                    'stderr': str(e),
                    'exit_code': -1,
                    'timed_out': 'timeout' in str(e).lower(),
                }
```

**Docker images to build (lightweight):**
- `norax/sandbox-python` — Python 3.12 slim + numpy, requests (~80MB)
- `norax/sandbox-node` — Node 20 alpine (~50MB)  
- `norax/sandbox-bash` — Alpine + bash (~8MB)

**Safety:**
- Container auto-destroyed after execution
- No network access by default (opt-in via `network=True`)
- Memory capped at 256MB, CPU at 0.5 cores
- 30-second timeout kills the container
- Output truncated to 5000 chars stdout/stderr
- Host filesystem NOT mounted (only temp workspace)

### 6.3 Test-Run-Iterate Loop

**Problem:** Norax writes code, but doesn't automatically verify it works. User has to say "test it" or "run it".

**Solution:** Automatic test cycle after code generation.

**File:** `brain/code_loop.py` (new)

```python
class CodeLoop:
    """
    Automatic write → test → fix cycle.
    
    Flow:
    1. Write code to file
    2. Run it in sandbox or locally (based on risk)
    3. If exit_code != 0, analyze stderr
    4. Generate fix patch
    5. Re-run (max 3 iterations)
    6. Report final result
    
    Only activated when:
    - Writing new files (not editing existing)
    - User asks to write/create/implement code
    - NOT for config edits, docs, or small patches
    """
    
    MAX_ITERATIONS = 3
    
    def run(self, filepath: str, language: str = "python") -> dict:
        for i in range(self.MAX_ITERATIONS):
            result = self._execute(filepath, language)
            if result['exit_code'] == 0:
                return {'success': True, 'iterations': i + 1, 'output': result['stdout']}
            
            # Analyze failure
            fix = self._generate_fix(filepath, result['stderr'])
            if not fix:
                return {'success': False, 'iterations': i + 1, 'error': result['stderr']}
            
            # Apply fix
            self._apply_fix(filepath, fix)
        
        return {'success': False, 'iterations': self.MAX_ITERATIONS, 'error': 'Max iterations reached'}
```

**Integration with `edit` tool:**
- After `edit` completes on a `.py`/`.js`/`.sh` file, optionally auto-run
- Only if `code_loop` config is enabled
- Never auto-run for configuration files, markdown, JSON, YAML

### 6.4 Diff-Based Edit Review

**Problem:** Current `edit` tool does blind find-and-replace. No preview, no undo beyond manual revert.

**Solution:** Generate and display diff before applying.

**File:** `tools/diff_edit.py` (new)

```python
class DiffEdit:
    """
    Preview + apply edits with diff display.
    
    Flow:
    1. Compute diff between old and new content
    2. Show diff to user (if interactive / important edit)
    3. Apply or abort based on confirmation
    
    For agent-internal edits (low risk): auto-apply, log diff
    For production config / system files: require confirmation
    """
    
    RISK_CATEGORIES = {
        'low': ['.py', '.js', '.ts', '.sh', '.md', '.txt'],      # Auto-apply
        'medium': ['.json', '.yaml', '.yml', '.toml', '.cfg'],  # Log + auto-apply
        'high': ['.env', '.service', '.conf', '/etc/*'],         # Require confirmation
    }
    
    def preview(self, filepath: str, old: str, new: str) -> str:
        """Generate unified diff."""
        import difflib
        diff = difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
        )
        return ''.join(diff)
    
    def apply(self, filepath: str, old: str, new: str, risk: str = None) -> dict:
        if risk is None:
            risk = self._classify_risk(filepath)
        
        diff = self.preview(filepath, old, new)
        
        if risk == 'high':
            return {'action': 'confirm_required', 'diff': diff, 'filepath': filepath}
        
        # Auto-apply for low/medium risk
        Path(filepath).write_text(new)
        return {'action': 'applied', 'diff': diff, 'filepath': filepath, 'risk': risk}
```


---

## 7. 24/7 Uptime & Reliability

### 7.1 Current Reliability

**Service:** `norax-ai.service` (systemd)

**Known failure modes:**
- Ollama model timeout → turn hangs, no fallback
- Discord gateway disconnect → requires service restart
- Memory pressure → no graceful handling
- Unhandled exception in orchestrator → silent turn failure
- No watchdog — if the service hangs (not crashed), systemd doesn't know

### 7.2 Watchdog Service

**File:** `dispatch/watchdog.py` (new)

```python
class NoraxWatchdog:
    """
    Separate process that monitors Norax health.
    
    Checks every 30 seconds:
    1. Can we reach the Discord gateway? (TCP check)
    2. Has a turn completed in the last 10 minutes? (activity check)
    3. Is the process memory usage reasonable? (RSS check)
    4. Is the event store still growing? (stall detection)
    
    Actions on failure:
    - Discord down: restart gateway connection only
    - No activity for 10min + queue has items: restart service
    - Memory > 2GB: trigger GC + memory cleanup
    - Event store stalled: log alert, continue (may be intentional idle)
    """
    
    CHECK_INTERVAL = 30  # seconds
    STALL_THRESHOLD = 600  # 10 minutes
    
    def run(self):
        while True:
            checks = {
                'discord': self._check_discord(),
                'activity': self._check_activity(),
                'memory': self._check_memory(),
                'event_store': self._check_events(),
            }
            self._handle_failures(checks)
            time.sleep(self.CHECK_INTERVAL)
```

**Deployment:** Run as a separate systemd service (`norax-watchdog.service`) watching the main `norax-ai.service`. If watchdog detects a hang, it runs `systemctl restart norax-ai`.

### 7.3 Auto-Recovery Procedures

**File:** `dispatch/recovery.py` (new)

```python
class AutoRecovery:
    """
    Structured recovery procedures for common failure modes.
    
    Each procedure is a step-by-step recovery that doesn't require
    a full service restart.
    """
    
    PROCEDURES = {
        'discord_disconnect': [
            'Wait 5 seconds',
            'Attempt WebSocket reconnect',
            'If fail: wait 30s, attempt again',
            'If fail: restart gateway process only',
            'Log recovery to event store',
        ],
        'model_timeout': [
            'Cancel pending request',
            'Switch to fallback model (local Ollama)',
            'Retry turn with fallback',
            'Notify user: "Primary model slow, using local fallback"',
            'Periodically try primary model again (every 5 min)',
        ],
        'memory_pressure': [
            'Force garbage collection',
            'Flush episodic buffer to semantic store',
            'Clear checkpoint temp files',
            'If still high: clear web fetch cache',
            'If still high: alert owner',
        ],
        'queue_overflow': [
            'Stop accepting new messages',
            'Process queue FIFO',
            'Discard messages older than 5 minutes',
            'Resume when queue < 10',
            'Alert owner with queue stats',
        ],
    }
```

### 7.4 Persistent Turn State Across Restarts

**Problem:** When the service restarts, all in-progress turns are lost. The event store has the inputs but not what was happening internally.

**Solution:** Checkpoint state to disk at each phase (see 4.2). On restart:
1. Scan `/tmp/norax_checkpoints/` for incomplete turns
2. Load the latest checkpoint for each
3. Resume processing from that phase
4. If checkpoint is >5 minutes old, discard (user has moved on)
5. If checkpoint is recent, resume and send: "Recovered from interruption, continuing..."

---

## 8. Prompt & Context Management

### 8.1 Current Context Strategy

Current context building:
- System prompt (identity, directives, rules)
- Memory search results (top-k by relevance)
- Recent event history (rolling window)
- Current turn input

**Problem:** Context window fills up, and we trim by character count — cutting off the least recent content regardless of importance. A critical instruction might get trimmed while a casual greeting stays.

### 8.2 Importance-Based Context Trimming

**File:** `prompt/context_builder.py` (enhanced)

```python
class ContextBuilder:
    """
    Build context window with importance-aware trimming.
    
    Priority levels (highest kept last when trimming):
    1. CRITICAL: System directives, authority, identity (never trim)
    2. HIGH: Active task context, owner's current request
    3. MEDIUM: Memory search results, recent outcomes
    4. LOW: Previous turn history, greetings, acknowledgments
    5. DISPOSABLE: Tool narration, routine logs
    
    When context overflows:
    - Remove DISPOSABLE first
    - Then trim LOW by removing oldest entries
    - Then compress MEDIUM (summarize 3 old turns → 1 summary)
    - NEVER remove CRITICAL or active HIGH
    """
    
    PRIORITY = {
        'system_directive': 'CRITICAL',
        'active_task': 'HIGH',
        'current_message': 'HIGH',
        'tool_result': 'MEDIUM',
        'memory_hit': 'MEDIUM',
        'turn_history': 'LOW',
        'greeting': 'LOW',
        'tool_narration': 'DISPOSABLE',
        'log_output': 'DISPOSABLE',
    }
    
    def build(self, components: list[dict], max_tokens: int = 8000) -> str:
        """Build context within token budget, trimming by priority."""
        # Sort: critical first, disposable last
        sorted_components = sorted(components, 
            key=lambda c: self._priority_rank(self.PRIORITY.get(c['type'], 'LOW')))
        
        context = ""
        for comp in sorted_components:
            addition = comp['content']
            if self._estimate_tokens(context + addition) > max_tokens:
                # Try to fit a compressed version
                if self.PRIORITY.get(comp['type']) in ('MEDIUM', 'LOW'):
                    compressed = self._compress(addition)
                    if self._estimate_tokens(context + compressed) <= max_tokens:
                        context += compressed
                        continue
                # Can't fit — stop adding
                break
            context += addition
        
        return context
```

### 8.3 Turn Summarization

**File:** `prompt/summarizer.py` (new)

```python
class TurnSummarizer:
    """
    Compress old turns into summaries.
    
    Instead of keeping full turn history:
    - Keep last 3 turns verbatim
    - Summarize turns 4-10 into a paragraph
    - Summarize turns 11+ into a sentence
    - Discard turns older than session scope
    
    Uses local Ollama model for summarization (no cloud call needed).
    """
    
    def summarize_turns(self, turns: list[dict], depth: str = 'paragraph') -> str:
        """Summarize a list of turns."""
        prompt = f"Summarize these conversation turns in 1-2 {'sentences' if depth == 'sentence' else 'paragraphs'}. Focus on key decisions, outcomes, and pending actions:\n\n"
        for t in turns:
            prompt += f"User: {t['input']}\nAgent: {t['output']}\n\n"
        
        # Use local model for speed
        response = ollama.chat(model='qwen2.5:1.5b', messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content']
```

### 8.4 Dynamic Skill Loading

**Problem:** All skills are loaded into the system prompt at startup, consuming context window space even when unused.

**Solution:** Load skills on-demand based on turn analysis.

**File:** `prompt/skill_loader.py` (enhanced)

```python
class DynamicSkillLoader:
    """
    Load skills based on turn intent, not all at once.
    
    Always loaded: core tools (read, write, exec, edit, search_memory)
    On-demand: web_search, web_fetch, docker, sandbox, etc.
    
    Detection:
    - "search for" / "look up" / "find" → load web_search
    - "run this code" / "test this" → load sandbox
    - "fetch this URL" → load web_fetch
    - "SSH into" / "deploy to" → load exec (already loaded)
    """
    
    CORE_ALWAYS = ['read', 'write', 'edit', 'exec', 'search_memory']
    
    INTENT_SKILL_MAP = {
        'web_search': ['search', 'look up', 'find online', 'what is', 'who is'],
        'web_fetch': ['fetch', 'url', 'website', 'page'],
        'sandbox': ['run code', 'test code', 'execute', 'benchmark'],
        'docker': ['container', 'deploy', 'docker'],
    }
    
    def load_for_turn(self, turn_input: str) -> list:
        """Determine which skills to load for this turn."""
        skills = list(self.CORE_ALWAYS)
        lower = turn_input.lower()
        for skill, triggers in self.INTENT_SKILL_MAP.items():
            if any(t in lower for t in triggers):
                skills.append(skill)
        return skills
```


---

## 9. Implementation Roadmap

### 9.1 Phase Dependency Map

Phase A: Foundation (Week 1-2)
  A1: Turn checkpoints -> required by A4, A5
  A2: Message queue -> required by A5
  A3: Self-diagnosis heartbeat -> required by A5
  A4: Graceful degradation -> independent

Phase B: Web Intelligence (Week 3-4)
  B1: Query decomposition -> required by B4
  B2: Enhanced web fetch -> required by B4
  B3: Result reranking (keyword) -> required by B4
  B4: Multi-hop search -> depends on B1, B2, B3

Phase C: Memory (Week 5-7)
  C1: FAISS vector index -> required by B3(upgrade), C4
  C2: Agent self-edit tools -> independent
  C3: Importance-based eviction -> independent
  C4: Cross-session continuity -> depends on C1
  C5: Context builder rewrite -> depends on C1, Phase A

Phase D: Code Intelligence (Week 8-10)
  D1: Docker sandbox images -> required by D2
  D2: Code sandbox -> depends on D1
  D3: Test-run-iterate loop -> depends on D2
  D4: Diff-based edit review -> independent

Phase E: Production Hardening (Week 11-12)
  E1: Watchdog service -> independent
  E2: Auto-recovery procedures -> depends on E1
  E3: Persistent turn state -> depends on A1
  E4: Full integration testing -> depends on all

### 9.2 Sprint Breakdown

#### Sprint 1 (Week 1-2): Foundation

| Task | Effort | Files | Staging Test |
|---|---|---|---|
| A1: Turn checkpoints | 1 day | brain/checkpoint.py, modify phase_orchestrator.py | Kill service mid-turn, verify resume |
| A2: Message queue | 1 day | dispatch/message_queue.py, modify gateway.py | Flood 10 messages, verify FIFO |
| A3: Self-diagnosis | 0.5 day | brain/health.py | Verify alerts on induced failures |
| A4: Graceful degradation | 1 day | dispatch/degraded_mode.py, modify tools/ | Block web search, verify fallback |

Sprint 1 Exit Criteria:
- Crash mid-turn -> resume within 5 seconds
- 10 messages in queue -> all processed in order
- Web search down -> agent responds with cached/memory results
- Memory leak simulated -> health alert fires

#### Sprint 2 (Week 3-4): Web Intelligence

| Task | Effort | Files | Staging Test |
|---|---|---|---|
| B1: Query decomposition | 1 day | tools/web_search_enhanced.py | Complex query -> verify sub-queries |
| B2: Enhanced web fetch | 2 days | tools/web_fetch_enhanced.py | Fetch HTML page, verify markdown output |
| B3: Keyword reranking | 0.5 day | tools/reranker.py | Search python async -> verify ordered by relevance |
| B4: Multi-hop search | 1.5 days | tools/multi_hop_search.py | Research query -> verify claim extraction |

Sprint 2 Exit Criteria:
- Complex compound query -> decomposes into sub-searches
- Fetch news article -> returns clean markdown, no nav/footer noise
- Web fetch on 404 -> graceful error, no crash
- Multi-hop returns claims with source citations

#### Sprint 3 (Week 5-7): Memory

| Task | Effort | Files | Staging Test |
|---|---|---|---|
| C1: FAISS vector index | 2 days | memory/vector_index.py, modify tools/ | Load 1000 memories, search <50ms |
| C2: Agent self-edit | 1 day | tools/memory_tools.py, modify prompt/ | Edit a memory, verify update + audit trail |
| C3: Eviction scoring | 1 day | memory/eviction.py | 30-day-old low-relevance -> appears in candidates |
| C4: Cross-session context | 1.5 days | memory/task_context.py, modify prompt/ | End task mid-session -> restart -> agent knows context |
| C5: Context builder rewrite | 2 days | prompt/context_builder.py, prompt/summarizer.py | Overflow test -> DISPOSABLE trimmed first |

Sprint 3 Exit Criteria:
- 1000 semantic memories -> search latency < 50ms
- Agent edits its own memory -> audit trail in event store
- 30-day idle memory -> appears in eviction report (not auto-deleted)
- Context overflow -> system directives survive, greetings trimmed

#### Sprint 4 (Week 8-10): Code Intelligence

| Task | Effort | Files | Staging Test |
|---|---|---|---|
| D1: Docker images | 1 day | docker/sandbox-{python,node,bash}/Dockerfile | Build + run hello world in each |
| D2: Code sandbox | 2 days | tools/sandbox.py | Run destructive command in sandbox -> host unaffected |
| D3: Test-run-iterate | 1.5 days | brain/code_loop.py | Write buggy code -> agent auto-fixes in 2 iterations |
| D4: Diff-based edit | 1 day | tools/diff_edit.py | Edit .env -> requires confirmation, edit .py -> auto |

Sprint 4 Exit Criteria:
- Destructive code in sandbox -> zero host impact
- Infinite loop in sandbox -> killed after 30s timeout
- Buggy Python file -> agent detects, fixes, re-tests
- Edit to .env -> user sees diff preview before apply

#### Sprint 5 (Week 11-12): Production Hardening

| Task | Effort | Files | Staging Test |
|---|---|---|---|
| E1: Watchdog service | 1.5 days | dispatch/watchdog.py, norax-watchdog.service | Kill gateway -> watchdog restarts within 60s |
| E2: Auto-recovery | 1.5 days | dispatch/recovery.py | Each failure mode -> verify correct procedure |
| E3: Persistent turn state | 1 day | modifies brain/checkpoint.py, dispatch/gateway.py | Hard crash -> restart -> agent continues |
| E4: Full integration test | 2 days | test suite | Run all sprint exit criteria in sequence |

Sprint 5 Exit Criteria:
- Watchdog detects hang -> restart within 60s
- Discord disconnect -> auto-reconnect without service restart
- All sprint 1-4 exit criteria still pass (regression suite)
- 24-hour stress test: no memory leak, no stuck turns, no dropped messages

### 9.3 Staging to Production Checklist

Before ANY change moves from staging to production Norax:

- Feature works on staging for 24+ hours without errors
- No memory leaks (RSS stable over 6+ hours)
- Existing tests still pass (regression)
- Risk gate still blocks dangerous patterns
- Event store integrity verified (hash chain valid)
- Owner explicitly approves the change
- Rollback plan documented (git revert hash)


---

## 10. Risk Assessment and Safety

### 10.1 Risk Categories by Phase

| Phase | Highest Risk | Mitigation |
|---|---|---|
| A: Foundation | Checkpoint data leaks sensitive turn content | Store in /tmp (ephemeral), auto-cleanup, no PII in checkpoints |
| B: Web Intelligence | Web fetch executes malicious content | HTML sanitization, no JS execution, content-type validation, domain blocklist |
| C: Memory | Self-edit allows unauthorized memory manipulation | Audit trail in event store, owner memories never auto-evicted |
| D: Code Intelligence | Sandbox escape | No network by default, resource limits, ephemeral containers, host FS not mounted |
| E: Hardening | Watchdog false-positive restarts kill active turns | Combine with checkpointing so turns resume after restart |

### 10.2 Safety Principles (Non-Negotiable)

1. Never auto-delete owner-stored data. Eviction reports are proposals, not actions.
2. Sandbox is maximum isolation. No host mounts, no network, resource-capped, time-limited.
3. Every memory edit is audited. Old value + new value + timestamp + reason stored in event chain.
4. Web content is untrusted. Strip all scripts, iframes, forms. Never execute fetched content.
5. Checkpoints are ephemeral. /tmp only, max 1 hour TTL, deleted on successful completion.
6. Degraded mode over crash. Better to return cached results than to fail silently.
7. Staging before production. Every change runs on staging first. No exceptions.
8. Event chain integrity. Any change to the runtime must preserve hash-chain validity.

### 10.3 Rollback Strategy

Every phase has a rollback plan:

| Phase | Rollback Method | Recovery Time |
|---|---|---|
| A | Remove checkpoint calls from orchestrator, restart service | Less than 1 min |
| B | Revert to raw web_search/web_fetch tools | Less than 1 min |
| C | Delete vector index, fall back to brute-force search | Less than 5 min |
| D | Remove sandbox tool, revert to direct exec | Less than 1 min |
| E | Disable watchdog service, manual monitoring | Less than 1 min |

All changes committed to git. Rollback = git revert plus service restart.

### 10.4 Monitoring During Implementation

Track these metrics continuously (write to event store):

- Turn success rate: completed turns / total turns (target: above 99%)
- Median turn latency: input to response (target: under 10s)
- Memory RSS: process memory usage (target: stable, under 1.5GB)
- Queue depth: pending messages (target: under 5 steady state)
- Search cache hit rate: cached results / total searches (target: above 30%)
- Context utilization: tokens used / tokens available (target: 60-80%)

Alert thresholds:
- Turn success rate below 95%: investigate
- Turn latency above 30s: investigate
- RSS above 2GB: restart + investigate
- Queue depth above 20: clear stale messages
- Search cache hit rate below 10%: check cache TTL

---

## Appendix A: File Inventory (New Files to Create)

| File | Purpose | Phase |
|---|---|---|
| brain/checkpoint.py | Turn-level state checkpoints | A |
| brain/health.py | Self-diagnosis heartbeat | A |
| brain/code_loop.py | Test-run-iterate auto-fix | D |
| dispatch/message_queue.py | Persistent inbound queue | A |
| dispatch/degraded_mode.py | Graceful degradation manager | A |
| dispatch/watchdog.py | External health monitor | E |
| dispatch/recovery.py | Auto-recovery procedures | E |
| memory/vector_index.py | FAISS similarity search | C |
| memory/eviction.py | Importance-based memory scoring | C |
| memory/task_context.py | Cross-session task tracking | C |
| tools/web_search_enhanced.py | Query decomposition + expansion | B |
| tools/web_fetch_enhanced.py | Content-type aware fetching | B |
| tools/reranker.py | Search result reranking | B |
| tools/multi_hop_search.py | Multi-hop research search | B |
| tools/sandbox.py | Docker code sandbox | D |
| tools/memory_tools.py | Agent self-edit memory | C |
| tools/diff_edit.py | Diff-based edit with review | D |
| prompt/context_builder.py | Importance-based context trimming | C |
| prompt/summarizer.py | Turn summarization | C |
| prompt/skill_loader.py | Dynamic skill loading | C |

## Appendix B: Dependencies to Install

| Package | Purpose | Size | Phase |
|---|---|---|---|
| faiss-cpu | Vector similarity search | ~50MB | C |
| markdownify | HTML to markdown conversion | ~1MB | B |
| trafilatura | Web content extraction | ~5MB | B |
| docker | Python Docker SDK | ~10MB | D |

## Appendix C: Current vs Target Comparison

| Capability | Current | Target | Phase |
|---|---|---|---|
| Semantic search | Brute-force file scan | FAISS vector index, sub-50ms | C |
| Web search | Raw SearXNG results | Decompose, expand, rerank, cache | B |
| Web fetch | Raw HTTP GET + truncate | Content-type aware, HTML to markdown, smart extraction | B |
| Code execution | Direct on host | Docker sandbox, isolated | D |
| Crash recovery | Lost context | Checkpoint + resume | A |
| Memory management | Grow forever | Importance scoring + eviction proposals | C |
| Context overflow | Character-based trim | Importance-based trim + summarization | C |
| Service reliability | Manual restart | Watchdog + auto-recovery | E |
| Message processing | Best-effort | Persistent queue + FIFO | A |
| Skill loading | All at once | Intent-based dynamic loading | C |

---

End of framework document. This is a living document - update as we learn from each sprint retrospective.
