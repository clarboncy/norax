"""Curiosity Engine — intrinsic motivation via novelty/surprise detection.

Detects when the current input is unusual relative to a bounded window of
recorded requests (low shingle similarity, rare entity mix, first-seen domain).
It can emit a conservative verification hint once enough baseline samples
exist; it does not claim semantic novelty from an empty history.

Zero LLM calls. Uses word-shingle similarity plus local entity/domain counts.

Usage:
    ce = CuriosityEngine(Path("~/norax/memory/curiosity.json"))
    ce.load()
    sig = ce.assess(user_text, entities=[...], domain="coding")
    # sig.novelty -> 0.0 (seen it all before) to 1.0 (completely new)
    # sig.explore_hint -> str injected into prompt when novelty is high
    ce.record(user_text, entities=[...], domain="coding")  # after the turn
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

from ..atomic import atomic_write_text, read_bounded_text

log = logging.getLogger("norax.brain.curiosity_engine")

WINDOW_SIZE = 200  # rolling window of recent turn fingerprints
SAVE_EVERY = 10  # persist every N records
MIN_BASELINE = 5  # novelty is unknown, not maximal, before this many examples
NOVELTY_HIGH = 0.65  # above this, inject exploration hint
NOVELTY_MEDIUM = 0.40
_MAX_STATE_BYTES = 2 * 1024 * 1024
_MAX_INPUT_CHARS = 32_768
_MAX_INPUT_WORDS = 2_048
_MAX_SHINGLES = 512
_MAX_STORED_SHINGLES = 60
_MAX_ENTITIES_PER_TURN = 20
_MAX_ENTITY_COUNTS = 500
_MAX_DOMAIN_COUNTS = 50
_MAX_LABEL_CHARS = 128
_MAX_COUNTER = 1_000_000_000


def _label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return " ".join(value.split())[:_MAX_LABEL_CHARS]


def _counter(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_COUNTER:
        return None
    return value


# Tokenizer: cheap word shingles, no external deps.
def _shingles(text: str, k: int = 3) -> set[str]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
        raise ValueError("k must be an integer between 1 and 10")
    words: list[str] = []
    for raw_word in text[:_MAX_INPUT_CHARS].split()[:_MAX_INPUT_WORDS]:
        word = raw_word.strip(".,!?;:()[]{}\"'`").lower()[:_MAX_LABEL_CHARS]
        if len(word) > 2:
            words.append(word)
    if len(words) < k:
        return set(words)
    return {" ".join(words[i : i + k]) for i in range(min(len(words) - k + 1, _MAX_SHINGLES))}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass
class TurnFingerprint:
    shingle_hash: str  # hashed so state file stays small
    shingles: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    domain: str = ""
    ts: float = 0.0


@dataclass
class CuriositySignal:
    novelty: float  # 0..1
    surprise: float  # distance from nearest recent fingerprint
    is_new_domain: bool
    is_new_entity_mix: bool
    explore_hint: str  # "" when nothing worth saying


class CuriosityEngine:
    """Rolling novelty detector over recent turn fingerprints."""

    def __init__(self, path: Path | str = "~/norax/memory/curiosity.json"):
        self.path = Path(path).expanduser()
        self.window: deque[TurnFingerprint] = deque(maxlen=WINDOW_SIZE)
        self.domain_counts: Counter[str] = Counter()
        self.entity_counts: Counter[str] = Counter()
        self.total_turns: int = 0
        self._since_save: int = 0

    # ------------------------------------------------------------------ io
    def load(self) -> None:
        try:
            data = json.loads(read_bounded_text(self.path, max_bytes=_MAX_STATE_BYTES))
            if not isinstance(data, dict):
                raise ValueError("curiosity state root must be an object")
            total_turns = _counter(data.get("total_turns", 0))
            raw_domains = data.get("domain_counts", {})
            raw_entities = data.get("entity_counts", {})
            raw_window = data.get("window", [])
            if (
                total_turns is None
                or not isinstance(raw_domains, dict)
                or not isinstance(raw_entities, dict)
                or not isinstance(raw_window, list)
            ):
                raise ValueError("curiosity state collections are malformed")
            if (
                len(raw_domains) > _MAX_DOMAIN_COUNTS
                or len(raw_entities) > _MAX_ENTITY_COUNTS
                or len(raw_window) > WINDOW_SIZE
            ):
                raise ValueError("curiosity state exceeds collection limits")

            domain_counts: Counter[str] = Counter()
            for raw_domain, raw_count in raw_domains.items():
                domain = _label(raw_domain)
                count = _counter(raw_count)
                if domain and count is not None:
                    domain_counts[domain] = count
            entity_counts: Counter[str] = Counter()
            for raw_entity, raw_count in raw_entities.items():
                entity = _label(raw_entity)
                count = _counter(raw_count)
                if entity and count is not None:
                    entity_counts[entity.lower()] = count

            window: deque[TurnFingerprint] = deque(maxlen=WINDOW_SIZE)
            for fp in raw_window:
                if not isinstance(fp, dict):
                    continue
                shingle_hash = _label(fp.get("shingle_hash", ""))
                domain = _label(fp.get("domain", ""))
                timestamp = fp.get("ts", 0.0)
                raw_shingles = fp.get("shingles", [])
                raw_fp_entities = fp.get("entities", [])
                if (
                    not shingle_hash
                    or domain is None
                    or isinstance(timestamp, bool)
                    or not isinstance(timestamp, (int, float))
                    or not math.isfinite(float(timestamp))
                    or timestamp < 0
                    or not isinstance(raw_shingles, list)
                    or len(raw_shingles) > _MAX_STORED_SHINGLES
                    or not isinstance(raw_fp_entities, list)
                    or len(raw_fp_entities) > _MAX_ENTITIES_PER_TURN
                ):
                    continue
                shingles = [_label(value) for value in raw_shingles]
                entities = [_label(value) for value in raw_fp_entities]
                if any(value is None for value in (*shingles, *entities)):
                    continue
                window.append(
                    TurnFingerprint(
                        shingle_hash=shingle_hash,
                        shingles=[value for value in shingles if value],
                        entities=[value.lower() for value in entities if value],
                        domain=domain,
                        ts=float(timestamp),
                    )
                )
            self.total_turns = total_turns
            self.domain_counts = domain_counts
            self.entity_counts = entity_counts
            self.window = window
            log.info(
                "curiosity_engine loaded: %d turns, window=%d, domains=%d",
                self.total_turns,
                len(self.window),
                len(self.domain_counts),
            )
        except FileNotFoundError:
            return
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as e:
            log.warning("curiosity_engine.load_failed: %r", e)

    def save(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "total_turns": self.total_turns,
                "domain_counts": dict(self.domain_counts.most_common(50)),
                "entity_counts": dict(self.entity_counts.most_common(500)),
                "window": [
                    {
                        "shingle_hash": fp.shingle_hash,
                        "shingles": fp.shingles[:60],
                        "entities": fp.entities[:20],
                        "domain": fp.domain,
                        "ts": fp.ts,
                    }
                    for fp in list(self.window)[-WINDOW_SIZE:]
                ],
                "saved_at": time.time(),
            }
            serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False)
            if len(serialized.encode("utf-8")) > _MAX_STATE_BYTES:
                raise ValueError(f"curiosity state exceeds {_MAX_STATE_BYTES} bytes")
            atomic_write_text(self.path, serialized, mode=0o600)
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("curiosity_engine.save_failed: %r", e)
            return False

    # ------------------------------------------------------------- assess
    def assess(
        self,
        user_text: str,
        entities: list[str] | None = None,
        domain: str = "",
    ) -> CuriositySignal:
        """Score how novel/surprising this input is vs. recent history."""
        if entities is not None and not isinstance(entities, list):
            raise TypeError("entities must be a list")
        normalized_domain = _label(domain)
        if normalized_domain is None:
            raise TypeError("domain must be a string")
        normalized_entities = [
            _label(entity) for entity in (entities or [])[:_MAX_ENTITIES_PER_TURN]
        ]
        entities = [entity.lower() for entity in normalized_entities if entity]
        shingles = _shingles(user_text)

        # Surprise: 1 - max jaccard vs. recent window
        max_sim = 0.0
        for fp in self.window:
            sim = _jaccard(shingles, set(fp.shingles))
            if sim > max_sim:
                max_sim = sim
                if max_sim >= 0.95:
                    break
        baseline_ready = len(self.window) >= MIN_BASELINE
        surprise = 1.0 - max_sim if self.window else 0.0

        # New domain: never seen, or very rare (<1% of turns)
        is_new_domain = bool(normalized_domain) and (
            normalized_domain not in self.domain_counts
            or (
                self.total_turns > 20
                and self.domain_counts[normalized_domain] / max(1, self.total_turns) < 0.01
            )
        )

        # New entity mix: >60% of entities never seen before
        new_ents = [e for e in entities if e not in self.entity_counts]
        is_new_entity_mix = bool(entities) and (len(new_ents) / len(entities)) > 0.6

        novelty = surprise
        if is_new_domain:
            novelty = min(1.0, novelty + 0.15)
        if is_new_entity_mix:
            novelty = min(1.0, novelty + 0.10)

        explore_hint = ""
        if baseline_ready and novelty >= NOVELTY_HIGH:
            domain_detail = f", new domain '{normalized_domain}'" if is_new_domain else ""
            entity_detail = ", unfamiliar entity mix" if is_new_entity_mix else ""
            explore_hint = (
                f"CURIOSITY: This input is novel territory (novelty={novelty:.2f}"
                f"{domain_detail}{entity_detail}). "
                "Existing patterns may not apply; verify unfamiliar specifics "
                "before relying on an analogy."
            )

        return CuriositySignal(
            novelty=round(novelty, 3),
            surprise=round(surprise, 3),
            is_new_domain=is_new_domain,
            is_new_entity_mix=is_new_entity_mix,
            explore_hint=explore_hint,
        )

    # ------------------------------------------------------------- record
    def record(
        self,
        user_text: str,
        entities: list[str] | None = None,
        domain: str = "",
    ) -> None:
        if entities is not None and not isinstance(entities, list):
            raise TypeError("entities must be a list")
        normalized_domain = _label(domain)
        if normalized_domain is None:
            raise TypeError("domain must be a string")
        shingles = _shingles(user_text)
        digest = hashlib.sha256(" ".join(sorted(shingles)).encode()).hexdigest()[:16]
        normalized_entities = [
            _label(entity) for entity in (entities or [])[:_MAX_ENTITIES_PER_TURN]
        ]
        ents = [entity.lower() for entity in normalized_entities if entity]
        self.window.append(
            TurnFingerprint(
                shingle_hash=digest,
                shingles=sorted(shingles)[:_MAX_STORED_SHINGLES],
                entities=ents,
                domain=normalized_domain,
                ts=time.time(),
            )
        )
        if normalized_domain:
            self._increment_bounded(
                self.domain_counts,
                normalized_domain,
                maximum_keys=_MAX_DOMAIN_COUNTS,
            )
        for e in ents:
            self._increment_bounded(
                self.entity_counts,
                e,
                maximum_keys=_MAX_ENTITY_COUNTS,
            )
        self.total_turns = min(_MAX_COUNTER, self.total_turns + 1)
        self._since_save += 1
        if self._since_save >= SAVE_EVERY:
            if self.save():
                self._since_save = 0

    @staticmethod
    def _increment_bounded(
        counter: Counter[str],
        key: str,
        *,
        maximum_keys: int,
    ) -> None:
        if key not in counter and len(counter) >= maximum_keys:
            least_common_key, _ = min(
                counter.items(),
                key=lambda item: (item[1], item[0]),
            )
            del counter[least_common_key]
        counter[key] = min(_MAX_COUNTER, counter[key] + 1)

    def stats(self) -> dict:
        return {
            "total_turns": self.total_turns,
            "window_size": len(self.window),
            "domains": dict(self.domain_counts.most_common(10)),
            "tracked_entities": len(self.entity_counts),
        }
