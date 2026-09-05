"""Hebbian learning — neurons that fire together wire together.

After each turn, we receive the set of neurons that were co-retrieved
(co-fired). We strengthen their weights by a small delta. Over many turns,
frequently useful memories rise to higher weights while unused ones decay.

Operates directly on the .md files (in-place weight tag updates).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..atomic import atomic_write_text, path_lock, read_bounded_text
from .store import Neuron

log = logging.getLogger("norax.memory.hebbian")

_WEIGHT_RX = re.compile(r"\|W([1-5])\b")

# Hebbian increment per co-fire event. Small — it takes ~10 co-fires to
# go from W1→W2, ~15 from W3→W4.  This is intentionally conservative.
_STRENGTHEN_DELTA = 0.06

# Decay lowers one discrete band every N co-firing turns. A fractional 0.02
# decrement was previously rounded back to the same W-tag and therefore never
# persisted at all.
_DECAY_EVERY_TURNS = 50
_MAX_MEMORY_FILE_BYTES = 16 * 1024 * 1024

# Weight bands: W1=0.4, W2=0.6, W3=0.8, W4=1.0, W5=1.3
_BAND = [(0.4, "1"), (0.6, "2"), (0.8, "3"), (1.0, "4"), (1.3, "5")]


def _weight_to_tag(w: float) -> str:
    """Map a float weight to the closest W-tag string."""
    best = "1"
    for threshold, tag in _BAND:
        if w >= threshold - 0.05:
            best = tag
    return f"|W{best}"


@dataclass
class HebbianLearner:
    """Track co-firing, strengthen/decay, persist to disk."""

    turn_count: int = 0
    _pending_strengthens: dict[str, float] = field(default_factory=dict)
    # entity_id → accumulated delta since last flush

    def record_cofiring(self, neurons: list[Neuron]) -> None:
        """Called once per turn with the neurons that were retrieved."""
        unique = {neuron.entity_id: neuron for neuron in neurons if neuron.entity_id}
        if len(unique) < 2:
            return  # no co-firing with a single neuron
        self.turn_count += 1
        for n in unique.values():
            cur = self._pending_strengthens.get(n.entity_id, 0.0)
            self._pending_strengthens[n.entity_id] = cur + _STRENGTHEN_DELTA

    def flush(self, memory_root: Path) -> int:
        """Write accumulated weight changes to disk. Returns count updated."""
        if not self._pending_strengthens:
            return 0

        updated = 0
        from .store import MemoryStore

        store = MemoryStore(root=memory_root)
        store.refresh()

        entity_to_neuron: dict[str, Neuron] = {}
        for n in store.iter_lines():
            if n.entity_id in self._pending_strengthens:
                entity_to_neuron[n.entity_id] = n

        changes_by_path: dict[Path, list[tuple[str, int, str | None, str]]] = {}
        applied: set[str] = set()
        for eid, delta in self._pending_strengthens.items():
            neuron = entity_to_neuron.get(eid)
            if neuron is None:
                applied.add(eid)
                continue
            new_weight = min(1.3, neuron.weight + delta)
            new_tag = _weight_to_tag(new_weight)
            old_tag_match = _WEIGHT_RX.search(neuron.text)

            if old_tag_match:
                old_tag = old_tag_match.group(0)
                if old_tag == new_tag:
                    applied.add(eid)
                    continue  # no change needed
            else:
                old_tag = None
            changes_by_path.setdefault(neuron.path, []).append(
                (eid, neuron.line - 1, old_tag, new_tag)
            )

        for path, changes in changes_by_path.items():
            try:
                with path_lock(path):
                    lines = read_bounded_text(
                        path,
                        max_bytes=_MAX_MEMORY_FILE_BYTES,
                    ).splitlines(keepends=True)
                    changed_ids: list[str] = []
                    for eid, idx, old_tag, new_tag in changes:
                        if not 0 <= idx < len(lines):
                            continue
                        line = lines[idx]
                        current = _WEIGHT_RX.search(line)
                        if old_tag:
                            if current is None or current.group(0) != old_tag:
                                continue
                            line = line.replace(current.group(0), new_tag, 1)
                        elif current is not None:
                            continue
                        else:
                            newline = "\n" if line.endswith("\n") else ""
                            line = line.rstrip("\n") + new_tag + newline
                        lines[idx] = line
                        changed_ids.append(eid)
                    if changed_ids:
                        atomic_write_text(path, "".join(lines))
                        updated += len(changed_ids)
                        applied.update(changed_ids)
            except Exception as e:
                log.warning("hebbian.flush.error path=%s: %r", path, e)

        self._pending_strengthens = {
            eid: delta for eid, delta in self._pending_strengthens.items() if eid not in applied
        }

        # Homeostatic decay every N turns
        if self.turn_count % _DECAY_EVERY_TURNS == 0 and self.turn_count > 0:
            updated += self._decay(memory_root)

        log.info("hebbian.flush: %d weights updated (turn %d)", updated, self.turn_count)
        return updated

    def _decay(self, memory_root: Path) -> int:
        """Apply small universal decay to prevent weight inflation."""
        from .store import MemoryStore

        store = MemoryStore(root=memory_root)
        store.refresh()

        decayed = 0
        changes_by_path: dict[Path, list[int]] = {}
        for n in store.all_canonical():
            if n.weight <= 0.4 or n.weight >= 1.3:
                continue  # W1 is the floor; W5 is permanent owner knowledge
            old_match = _WEIGHT_RX.search(n.text)
            if not old_match:
                continue  # don't add tags to untagged neurons during decay
            changes_by_path.setdefault(n.path, []).append(n.line - 1)

        for path, indexes in changes_by_path.items():
            try:
                with path_lock(path):
                    lines = read_bounded_text(
                        path,
                        max_bytes=_MAX_MEMORY_FILE_BYTES,
                    ).splitlines(keepends=True)
                    changed = 0
                    for idx in indexes:
                        if not 0 <= idx < len(lines):
                            continue
                        old_match = _WEIGHT_RX.search(lines[idx])
                        if old_match is None:
                            continue
                        old_band = int(old_match.group(1))
                        if old_band in {1, 5}:
                            continue
                        new_tag = f"|W{old_band - 1}"
                        lines[idx] = lines[idx].replace(old_match.group(0), new_tag, 1)
                        changed += 1
                    if changed:
                        atomic_write_text(path, "".join(lines))
                        decayed += changed
            except Exception as e:
                log.warning("hebbian.decay.error path=%s: %r", path, e)

        if decayed:
            log.info("hebbian.decay: %d neurons decayed at turn %d", decayed, self.turn_count)
        return decayed
