#!/usr/bin/env python3
"""Cross-source desktop element fusion.

Merges AT-SPI, CDP, and OCR observations only when independent sources have
compatible roles and spatial/text evidence. Same-source neighbors are never
deduplicated merely for being close.

Architecture:
  1. Collect elements from AT-SPI (structured, with roles + bounds)
  2. Collect elements from dmap OCR (text + approximate bounds)
  3. Collect visual hierarchy features (V1/V2/V4/IT)
  4. Spatial deduplication: IoU-based matching across modalities
  5. Confidence weighting: AT-SPI > CDP > OCR (structured > raw)
  6. Enrichment: OCR text fills gaps where AT-SPI names are empty
  7. Output: UnifiedElement list with merged metadata

Usage:
  from cv_fusion import CVFusion
  fusion = CVFusion()
  profile = fusion.fuse(atspi_state, dmap_output, visual_percept)
  print(profile.to_text())
"""

import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any, TypedDict

Bounds = tuple[int, int, int, int]


class RawElement(TypedDict):
    source: str
    role: str
    name: str
    value: str
    state: str
    bounds: Bounds
    actionable: bool
    app: str


class ParsedDmapElement(TypedDict):
    role: str
    name: str
    bounds: Bounds
    actionable: bool
    app: str


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class UnifiedElement:
    """A single UI element merged from multiple perception modalities."""

    ref: str  # p0, p1, p2...
    role: str  # button, text_field, label, link...
    name: str  # display text (best available)
    value: str = ""  # current value (inputs)
    state: str = ""  # accessibility state from a structured source
    bounds: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    sources: list[str] = field(default_factory=list)  # ['atspi', 'ocr', 'cv']
    confidence: float = 0.0  # source-agreement heuristic, not calibrated probability
    actionable: bool = False  # can be clicked/typed
    app: str = ""  # source application
    visual_hints: list[str] = field(default_factory=list)
    confidence_basis: str = "source_agreement_heuristic"

    @property
    def center(self) -> tuple[int, int]:
        """Center point (cx, cy) for clicking."""
        x, y, w, h = self.bounds
        return (x + w // 2, y + h // 2)

    @property
    def area(self) -> int:
        """Bounding box area."""
        x, y, w, h = self.bounds
        return w * h


@dataclass
class PerceptionProfile:
    """Unified perception output — the merged view of the screen."""

    timestamp: float = 0.0
    elements: list[UnifiedElement] = field(default_factory=list)
    app_type: str = "unknown"  # terminal, browser, editor, desktop
    app_confidence: float = 0.0  # compatibility name: non-calibrated heuristic score
    app_classification_method: str = "none"
    focused_app: str = ""
    screen_size: tuple[int, int] = (0, 0)
    fusion_stats: dict[str, int] = field(default_factory=dict)
    processing_ms: int = 0

    # Visual hierarchy summary
    edge_density: float = 0.0
    dominant_orientation: str = "unknown"
    layout_type: str = "unknown"
    has_dark_theme: bool = False
    contrast: float = 0.0
    rectangular_regions: int = 0
    text_blocks: int = 0
    button_candidates: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_text(self, max_elements: int = 60) -> str:
        """Compact text for LLM brain injection."""
        lines = []
        lines.append(
            f"FUSED PERCEPTION ({self.app_type}, {len(self.elements)} elements, {self.processing_ms}ms)"
        )

        if self.focused_app:
            lines.append(f"  Focus: {self.focused_app}")

        # Visual scene summary
        vis_parts = []
        if self.edge_density > 0:
            vis_parts.append(f"edges={self.edge_density:.2f}")
        if self.layout_type != "unknown":
            vis_parts.append(f"layout={self.layout_type}")
        if self.has_dark_theme:
            vis_parts.append("dark_theme")
        if self.contrast > 0:
            vis_parts.append(f"contrast={self.contrast:.2f}")
        if self.rectangular_regions:
            vis_parts.append(f"rects={self.rectangular_regions}")
        if self.text_blocks:
            vis_parts.append(f"text_blocks={self.text_blocks}")
        if self.button_candidates:
            vis_parts.append(f"btn_candidates={self.button_candidates}")
        if vis_parts:
            lines.append(f"  Visual: {', '.join(vis_parts)}")

        # Fusion stats
        stats = self.fusion_stats
        if stats:
            lines.append(f"  Sources: {', '.join(f'{k}={v}' for k, v in stats.items())}")

        # Elements: text-first, then interactive
        text_elems = [
            e
            for e in self.elements
            if e.role in ("label", "text", "paragraph", "heading") and e.name
        ]
        interactive = [e for e in self.elements if e.actionable]

        remaining = max(0, min(max_elements, len(self.elements)))
        visible_text = text_elems[: min(20, remaining)]
        remaining -= len(visible_text)
        visible_interactive = interactive[:remaining]

        if visible_text:
            lines.append("  Visible text:")
            for e in visible_text:
                lines.append(f"    {e.ref} {e.name[:80]}")

        if visible_interactive:
            lines.append("  Interactive:")
            for e in visible_interactive:
                extra = f' = "{e.value}"' if e.value else ""
                hints = f" [{', '.join(e.visual_hints)}]" if e.visual_hints else ""
                lines.append(f"    {e.ref} {e.role}: {e.name[:50]}{extra}{hints}")

        return "\n".join(lines)

    def summary(self) -> str:
        """One-line summary."""
        return (
            f"{self.app_type} | {len(self.elements)} elements | "
            f"{self.fusion_stats} | {self.processing_ms}ms"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────


def _iou(a: Bounds, b: Bounds) -> float:
    """Intersection-over-Union of two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _center_distance(a: Bounds, b: Bounds) -> float:
    """Euclidean distance between centers of two (x, y, w, h) boxes."""
    acx = a[0] + a[2] // 2
    acy = a[1] + a[3] // 2
    bcx = b[0] + b[2] // 2
    bcy = b[1] + b[3] // 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def _text_similarity(a: str, b: str) -> float:
    """Bounded normalized text similarity for cross-source matching."""
    if not a or not b:
        return 0.0
    a_lower = " ".join(re.findall(r"[a-z0-9]+", a.lower()))
    b_lower = " ".join(re.findall(r"[a-z0-9]+", b.lower()))
    if a_lower == b_lower:
        return 1.0
    if min(len(a_lower), len(b_lower)) >= 4 and (a_lower in b_lower or b_lower in a_lower):
        return 0.85
    return SequenceMatcher(None, a_lower, b_lower, autojunk=False).ratio()


def _role_family(role: str) -> str:
    normalized = role.lower().replace("-", "_").replace(" ", "_")
    families = {
        "button": {"button", "push_button", "toggle_button", "key", "control"},
        "text": {"text", "label", "paragraph", "heading", "static", "static_text"},
        "input": {"entry", "input", "text_field", "textbox", "searchbox", "combobox"},
        "link": {"link"},
        "choice": {"checkbox", "check_box", "radio", "radio_button", "switch"},
        "menu": {"menu", "menuitem", "menu_item", "tab", "page_tab"},
    }
    for family, roles in families.items():
        if normalized in roles:
            return family
    return normalized or "unknown"


def _roles_compatible(left: str, right: str) -> bool:
    left_family = _role_family(left)
    right_family = _role_family(right)
    if left_family == right_family:
        return True
    # OCR commonly observes the text printed inside an actionable control.
    return "text" in {left_family, right_family} and bool(
        {left_family, right_family} & {"button", "input", "link", "choice", "menu"}
    )


def _coerce_bounds(value: object) -> Bounds | None:
    """Return a usable x/y/width/height tuple without accepting malformed data."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(part, bool) or not isinstance(part, (int, float)) for part in value):
        return None
    x, y, width, height = (int(part) for part in value)
    if width <= 0 or height <= 0:
        return None
    return (x, y, width, height)


def _normalized_role(value: object) -> str:
    role = str(value or "text").strip().lower().replace("-", "_").replace(" ", "_")
    return role[:80] or "text"


def _is_actionable(role: str) -> bool:
    return _role_family(role) in {"button", "input", "link", "choice", "menu"}


# ─────────────────────────────────────────────────────────────────────────────
# CV FUSION ENGINE
# ─────────────────────────────────────────────────────────────────────────────


class CVFusion:
    """Cross-modal perception fusion engine.

    Merges:
      - AT-SPI elements (structured, with roles + bounds) → highest confidence
      - CDP elements (browser DOM, with roles + bounds) → high confidence
      - dmap OCR elements (text + approximate bounds) → medium confidence
      - Visual hierarchy features (V1/V2/V4/IT) → scene-level context
    """

    # Source priority (higher = more trusted)
    SOURCE_PRIORITY = {
        "atspi": 3,
        "cdp": 2,
        "ocr": 1,
        "cv": 0,  # visual hierarchy = scene-level, not per-element
    }

    # IoU threshold for spatial matching
    SPATIAL_MATCH_THRESHOLD = 0.3
    # Text similarity threshold for name matching
    TEXT_MATCH_THRESHOLD = 0.5
    # Center distance threshold (pixels) for proximity matching
    PROXIMITY_THRESHOLD = 50
    GRID_SIZE = 96
    MAX_RAW_ELEMENTS = 5_000
    MAX_UNIFIED_ELEMENTS = 2_000

    def __init__(self):
        self.ref_counter = 0

    def _next_ref(self) -> str:
        self.ref_counter += 1
        return f"p{self.ref_counter}"

    def fuse(
        self, atspi_state=None, cdp_state=None, dmap_output: str = "", visual_percept=None
    ) -> PerceptionProfile:
        """Fuse all perception modalities into a unified profile.

        Args:
            atspi_state: PerceptionState from perception.py (AT-SPI tier)
            cdp_state: PerceptionState from perception.py (CDP tier)
            dmap_output: Raw dmap.py OCR output string
            visual_percept: VisualPercept from visual_hierarchy.py

        Returns:
            PerceptionProfile with merged UnifiedElement list
        """
        t0 = time.monotonic()
        self.ref_counter = 0

        # ── 1. Collect raw elements from each modality ──
        raw_elements: list[RawElement] = []

        # AT-SPI elements
        atspi_count = 0
        if atspi_state and hasattr(atspi_state, "elements"):
            for e in atspi_state.elements:
                if len(raw_elements) >= self.MAX_RAW_ELEMENTS:
                    break
                raw_elements.append(
                    {
                        "source": "atspi",
                        "role": _normalized_role(e.role),
                        "name": str(e.name or "")[:4096],
                        "value": str(e.value or "")[:4096],
                        "state": str(getattr(e, "state", "") or "")[:1024],
                        "bounds": _coerce_bounds(e.bounds) or (0, 0, 0, 0),
                        "actionable": bool(e.actionable),
                        "app": str(e.app or atspi_state.focused_app or "")[:512],
                    }
                )
                atspi_count += 1

        # CDP elements
        cdp_count = 0
        if cdp_state and hasattr(cdp_state, "elements"):
            for e in cdp_state.elements:
                if len(raw_elements) >= self.MAX_RAW_ELEMENTS:
                    break
                raw_elements.append(
                    {
                        "source": "cdp",
                        "role": _normalized_role(e.role),
                        "name": str(e.name or "")[:4096],
                        "value": str(e.value or "")[:4096],
                        "state": str(getattr(e, "state", "") or "")[:1024],
                        "bounds": _coerce_bounds(e.bounds) or (0, 0, 0, 0),
                        "actionable": bool(e.actionable),
                        "app": str(e.app or cdp_state.focused_app or "")[:512],
                    }
                )
                cdp_count += 1

        # dmap OCR elements
        ocr_count = 0
        if dmap_output:
            ocr_elements = self._parse_dmap_output(dmap_output)
            for oe in ocr_elements:
                if len(raw_elements) >= self.MAX_RAW_ELEMENTS:
                    break
                raw_elements.append(
                    {
                        "source": "ocr",
                        "role": oe["role"],
                        "name": oe["name"],
                        "value": "",
                        "state": "",
                        "bounds": oe["bounds"],
                        "actionable": oe["actionable"],
                        "app": oe.get("app", ""),
                    }
                )
                ocr_count += 1

        # ── 2. Spatial deduplication and merging ──
        unified = self._merge_elements(raw_elements)

        # ── 3. Enrich with visual hierarchy features ──
        profile = PerceptionProfile(
            timestamp=time.time(),
            elements=unified,
            processing_ms=int((time.monotonic() - t0) * 1000),
            fusion_stats={
                "atspi": atspi_count,
                "cdp": cdp_count,
                "ocr": ocr_count,
                "merged": len(unified),
            },
        )

        # App type from visual percept IT stage
        if visual_percept:
            if visual_percept.it and visual_percept.it.available:
                profile.app_type = visual_percept.it.app_type
                profile.app_confidence = visual_percept.it.confidence
                profile.app_classification_method = getattr(
                    visual_percept.it, "method", "keyword_heuristic"
                )

            if visual_percept.v1 and visual_percept.v1.available:
                profile.edge_density = visual_percept.v1.edge_density
                profile.dominant_orientation = visual_percept.v1.dominant_orientation

            if visual_percept.v2 and visual_percept.v2.available:
                profile.layout_type = visual_percept.v2.layout_type
                profile.has_dark_theme = visual_percept.v2.has_dark_theme
                profile.contrast = visual_percept.v2.contrast

            if visual_percept.v4 and visual_percept.v4.available:
                profile.rectangular_regions = visual_percept.v4.rectangular_regions
                profile.text_blocks = visual_percept.v4.text_blocks
                profile.button_candidates = visual_percept.v4.button_candidates

        # Focused app
        if atspi_state and atspi_state.focused_app:
            profile.focused_app = atspi_state.focused_app
        elif cdp_state and cdp_state.focused_app:
            profile.focused_app = cdp_state.focused_app

        # Screen size
        if atspi_state and atspi_state.screen_size:
            profile.screen_size = atspi_state.screen_size

        # ── 4. Enrich elements with visual hints ──
        self._enrich_visual_hints(profile)

        return profile

    def _parse_dmap_output(self, output: str) -> list[ParsedDmapElement]:
        """Parse dmap.py OCR output into element dicts.

        Supports JSON from ``dmap read --json``, current compact output, and
        the legacy ``p1 Type:Name [x,y,w,h]`` representation.
        """
        elements: list[ParsedDmapElement] = []

        # Try JSON first
        try:
            data = json.loads(output)
            raw_json_elements = data.get("elements") if isinstance(data, dict) else data
            if isinstance(raw_json_elements, list):
                for raw in raw_json_elements[: self.MAX_RAW_ELEMENTS]:
                    if not isinstance(raw, dict):
                        continue
                    bounds = _coerce_bounds(raw.get("bbox") or raw.get("bounds"))
                    if bounds is None:
                        continue
                    role = _normalized_role(raw.get("type") or raw.get("role"))
                    name = str(raw.get("text") or raw.get("name") or "")[:4096]
                    elements.append(
                        {
                            "role": role,
                            "name": name,
                            "bounds": bounds,
                            "actionable": _is_actionable(role),
                            "app": str(raw.get("app") or "")[:512],
                        }
                    )
                return elements
        except (json.JSONDecodeError, TypeError):
            pass

        # Fall back to plain text parsing
        compact = re.compile(
            r"^\[(?P<ref>[A-Za-z]\w*)\]\s+(?P<role>[\w -]+)\s+"
            r"\((?P<x>-?\d+),(?P<y>-?\d+)\s+(?P<w>\d+)x(?P<h>\d+)\)"
            r'(?:\s+"(?P<name>.*)")?(?:\s+~\d+%)?$'
        )
        legacy_bounded = re.compile(
            r"^(?:p|s)\d+\s+([\w -]+):(.+?)\s+"
            r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]$",
            re.IGNORECASE,
        )
        legacy_unbounded = re.compile(r"^(?:p|s)\d+\s+([\w -]+):(.+)$", re.IGNORECASE)
        for line in output.strip().splitlines()[: self.MAX_RAW_ELEMENTS + 10]:
            line = line.strip()
            if not line or line.startswith(("DESKTOP:", "CLICKABLE:", "WINDOWS/PANELS:")):
                continue
            compact_match = compact.match(line)
            if compact_match:
                role = _normalized_role(compact_match.group("role"))
                bounds = _coerce_bounds(
                    tuple(int(compact_match.group(key)) for key in ("x", "y", "w", "h"))
                )
                if bounds is None:
                    continue
                elements.append(
                    {
                        "role": role,
                        "name": (compact_match.group("name") or "")[:4096],
                        "bounds": bounds,
                        "actionable": _is_actionable(role),
                        "app": "",
                    }
                )
                continue

            legacy_match = legacy_bounded.match(line)
            has_legacy_bounds = legacy_match is not None
            if legacy_match is None:
                legacy_match = legacy_unbounded.match(line)
            if not legacy_match:
                continue
            role = _normalized_role(legacy_match.group(1))
            raw_bounds: Bounds = (0, 0, 0, 0)
            if has_legacy_bounds:
                parsed_bounds = tuple(int(legacy_match.group(index)) for index in range(3, 7))
                raw_bounds = _coerce_bounds(parsed_bounds) or (0, 0, 0, 0)
            elements.append(
                {
                    "role": role,
                    "name": legacy_match.group(2).strip()[:4096],
                    "bounds": raw_bounds,
                    "actionable": _is_actionable(role),
                    "app": "",
                }
            )
        return elements

    def _merge_elements(self, raw: list[RawElement]) -> list[UnifiedElement]:
        """Merge compatible observations from different sources on a grid."""
        raw.sort(
            key=lambda element: self.SOURCE_PRIORITY.get(element.get("source", ""), 0),
            reverse=True,
        )
        unified: list[UnifiedElement] = []
        spatial_grid: dict[tuple[int, int], list[int]] = {}

        for element in raw[: self.MAX_RAW_ELEMENTS]:
            bounds = element["bounds"]
            has_bounds = bounds[2] > 0 and bounds[3] > 0
            if not has_bounds:
                bounds = (0, 0, 0, 0)

            candidate_indices: set[int] = set()
            grid_key: tuple[int, int] | None = None
            if has_bounds:
                center_x = bounds[0] + bounds[2] // 2
                center_y = bounds[1] + bounds[3] // 2
                grid_key = (center_x // self.GRID_SIZE, center_y // self.GRID_SIZE)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        candidate_indices.update(
                            spatial_grid.get((grid_key[0] + dx, grid_key[1] + dy), [])
                        )

            best_index: int | None = None
            best_evidence = 0.0
            source = element["source"]
            role = element["role"]
            name = element["name"]
            for candidate_index in candidate_indices:
                candidate = unified[candidate_index]
                if source in candidate.sources or not _roles_compatible(role, candidate.role):
                    continue
                overlap = _iou(bounds, candidate.bounds)
                distance = _center_distance(bounds, candidate.bounds)
                similarity = _text_similarity(name, candidate.name)
                strong_overlap = overlap >= 0.5
                corroborated_overlap = overlap >= 0.15 and similarity >= 0.65
                near_exact_text = distance <= 10 and similarity >= 0.85
                if not (strong_overlap or corroborated_overlap or near_exact_text):
                    continue
                evidence = max(overlap, similarity if distance <= self.PROXIMITY_THRESHOLD else 0)
                if evidence > best_evidence:
                    best_index = candidate_index
                    best_evidence = evidence

            if best_index is None:
                if len(unified) >= self.MAX_UNIFIED_ELEMENTS:
                    continue
                created = UnifiedElement(
                    ref=self._next_ref(),
                    role=role,
                    name=name,
                    value=element["value"],
                    state=element["state"],
                    bounds=bounds,
                    sources=[source],
                    confidence=self._source_confidence(source),
                    actionable=element["actionable"],
                    app=element["app"],
                )
                unified.append(created)
                if grid_key is not None:
                    spatial_grid.setdefault(grid_key, []).append(len(unified) - 1)
                continue

            merged = unified[best_index]
            if not merged.name and name:
                merged.name = name
            elif source == "ocr" and name and len(merged.name) < 5 < len(name):
                merged.name = name
            value = element["value"]
            if not merged.value and value:
                merged.value = value
            if not merged.state and element["state"]:
                merged.state = element["state"]
            merged.sources.append(source)
            merged.confidence = min(1.0, merged.confidence + 0.15)
            merged.actionable = merged.actionable or element["actionable"]
            if not merged.app and element["app"]:
                merged.app = str(element["app"])

        return unified

    def _source_confidence(self, source: str) -> float:
        """Base confidence by source."""
        return {
            "atspi": 0.9,
            "cdp": 0.8,
            "ocr": 0.6,
            "cv": 0.3,
        }.get(source, 0.5)

    def _enrich_visual_hints(self, profile: PerceptionProfile) -> None:
        """Add only element-local, directly derived visual hints."""
        for elem in profile.elements:
            # Add hints based on element properties
            if elem.role == "button" and elem.area > 0 and elem.area < 10000:
                elem.visual_hints.append("small_button")
            if elem.role in ("text", "label", "paragraph") and elem.area > 50000:
                elem.visual_hints.append("text_block")
            if elem.role in ("entry", "text_field"):
                elem.visual_hints.append("input_field")


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

_fusion = None


def get_fusion() -> CVFusion:
    global _fusion
    if _fusion is None:
        _fusion = CVFusion()
    return _fusion


def fuse(
    atspi_state=None, cdp_state=None, dmap_output="", visual_percept=None
) -> PerceptionProfile:
    return get_fusion().fuse(atspi_state, cdp_state, dmap_output, visual_percept)


def format_for_context(profile: PerceptionProfile) -> str:
    return profile.to_text()


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────


def _run_tests():
    passed = 0
    total = 0

    def test(name, cond):
        nonlocal passed, total
        total += 1
        if cond:
            passed += 1
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")

    print("=== CV Fusion Tests ===\n")

    # ── T1: IoU calculation ──
    test("T1: IoU identical boxes", abs(_iou((0, 0, 100, 100), (0, 0, 100, 100)) - 1.0) < 0.01)
    test("T2: IoU no overlap", _iou((0, 0, 50, 50), (100, 100, 50, 50)) == 0.0)
    test("T3: IoU partial overlap", 0 < _iou((0, 0, 100, 100), (50, 50, 100, 100)) < 0.5)

    # ── T4: Center distance ──
    test(
        "T4: Center distance", abs(_center_distance((0, 0, 10, 10), (100, 0, 10, 10)) - 100.0) < 1.0
    )

    # ── T5: Text similarity ──
    test("T5: Text identical", _text_similarity("Hello", "Hello") == 1.0)
    test("T6: Text substring", _text_similarity("Hello", "Hello World") >= 0.85)
    test("T7: Text different", _text_similarity("Hello", "Goodbye") < 0.5)

    # ── T8: dmap parsing ──
    fusion = CVFusion()
    parsed = fusion._parse_dmap_output("p1 Button:Submit [10,20,100,30]\np2 Label:Hello World")
    test("T8: dmap parses 2 elements", len(parsed) == 2)
    test("T9: dmap role extraction", parsed[0]["role"] == "button")
    test("T10: dmap bounds extraction", parsed[0]["bounds"] == (10, 20, 100, 30))
    test("T11: dmap actionable flag", parsed[0]["actionable"])

    # ── T12: Basic fusion (AT-SPI only) ──
    from perception import Element, PerceptionState

    atspi_state = PerceptionState(
        tier_used="atspi",
        focused_app="Firefox",
        elements=[
            Element(
                ref="a0",
                role="push button",
                name="Search",
                bounds=(100, 200, 80, 30),
                tier="atspi",
                actionable=True,
            ),
            Element(
                ref="a1",
                role="entry",
                name="Search box",
                bounds=(100, 150, 300, 30),
                tier="atspi",
                actionable=True,
            ),
            Element(
                ref="a2",
                role="label",
                name="Welcome to Wikipedia",
                bounds=(100, 50, 400, 40),
                tier="atspi",
            ),
        ],
    )
    profile = fusion.fuse(atspi_state=atspi_state)
    test("T12: Fusion from AT-SPI produces elements", len(profile.elements) == 3)
    test("T13: Fusion preserves names", any(e.name == "Search" for e in profile.elements))
    test("T14: Fusion preserves actionable", any(e.actionable for e in profile.elements))
    test("T15: Fusion assigns refs", all(e.ref.startswith("p") for e in profile.elements))
    test("T16: Fusion stats tracked", profile.fusion_stats.get("atspi") == 3)

    # ── T17: Cross-modal deduplication ──
    dmap_out = "p1 Button:Search [105, 205, 75, 25]\np3 Label:Welcome to Wikipedia"
    profile2 = fusion.fuse(atspi_state=atspi_state, dmap_output=dmap_out)
    # "Search" button should be merged (spatial overlap), not duplicated
    search_buttons = [
        e
        for e in profile2.elements
        if "search" in e.name.lower() and _role_family(e.role) == "button"
    ]
    search_button = search_buttons[0] if len(search_buttons) == 1 else None
    test("T17: Cross-modal dedup merges spatially overlapping elements", search_button is not None)
    test(
        "T18: Merged element has both sources",
        search_button is not None
        and "atspi" in search_button.sources
        and "ocr" in search_button.sources,
    )
    test(
        "T19: Merged confidence boosted",
        search_button is not None and search_button.confidence > 0.9,
    )

    # ── T20: Text output format ──
    text = profile.to_text()
    test("T20: Text output non-empty", len(text) > 0)
    test("T21: Text output contains app type", "FUSED PERCEPTION" in text)
    test("T22: Text output contains element names", "Search" in text)

    # ── T23: Empty input handling ──
    profile3 = fusion.fuse()
    test("T23: Empty input doesn't crash", len(profile3.elements) == 0)
    test(
        "T24: Empty input has stats",
        profile3.fusion_stats == {"atspi": 0, "cdp": 0, "ocr": 0, "merged": 0},
    )

    # ── T25: Visual percept enrichment ──
    class MockIT:
        available = True
        app_type = "browser"
        confidence = 0.8

    class MockV1:
        available = True
        edge_density = 0.15
        dominant_orientation = "horizontal"

    class MockV2:
        available = True
        layout_type = "columnar"
        has_dark_theme = True
        contrast = 0.7

    class MockV4:
        available = True
        rectangular_regions = 12
        text_blocks = 5
        button_candidates = 3

    class MockPercept:
        it = MockIT()
        v1 = MockV1()
        v2 = MockV2()
        v4 = MockV4()

    profile4 = fusion.fuse(atspi_state=atspi_state, visual_percept=MockPercept())
    test("T25: Visual percept enriches app_type", profile4.app_type == "browser")
    test("T26: Visual percept enriches edge_density", profile4.edge_density == 0.15)
    test("T27: Visual percept enriches layout", profile4.layout_type == "columnar")
    test("T28: Visual percept enriches dark theme", profile4.has_dark_theme)
    test("T29: Visual percept enriches rects", profile4.rectangular_regions == 12)

    # ── T30: UnifiedElement center calculation ──
    elem = UnifiedElement(ref="p1", role="button", name="Test", bounds=(100, 200, 50, 60))
    test("T30: Center calculation", elem.center == (125, 230))

    # ── T31: Performance ──
    import time as _time

    big_atspi = PerceptionState(
        tier_used="atspi",
        elements=[
            Element(
                ref=f"a{i}",
                role="label",
                name=f"Element {i}",
                bounds=(i * 10, i * 5, 80, 20),
                tier="atspi",
            )
            for i in range(200)
        ],
    )
    big_dmap = "\n".join(f"p{i} Label:Element {i} [{i * 10},{i * 5},80,20]" for i in range(200))
    t0 = _time.monotonic()
    profile5 = fusion.fuse(atspi_state=big_atspi, dmap_output=big_dmap)
    elapsed = (_time.monotonic() - t0) * 1000
    test("T31: 400 elements fuse <500ms", elapsed < 500)
    test("T32: Large fusion deduplicates", len(profile5.elements) <= 200)

    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} passed")
    print(f"\nFinal: {passed}/{total}")
    return passed == total


if __name__ == "__main__":
    success = _run_tests()
    sys.exit(0 if success else 1)
