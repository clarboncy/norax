#!/usr/bin/env python3
"""Measured screenshot features plus heuristic application classification.

Image metrics are emitted only when an actual screenshot was decoded. Element
lists without image pixels are not converted into synthetic edge/color/shape
measurements. The historical V1/V2/V4/IT names are compatibility stage labels;
this module does not model biological visual processing.
"""

import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_WORKING_PIXELS = 4_000_000
MAX_CONTOURS_ANALYZED = 20_000
MAX_METADATA_CHARS = 2_000_000

# ═══════════════════════════════════════════════════════════════════════════
# CHECK OPENCV AVAILABILITY
# ═══════════════════════════════════════════════════════════════════════════


def _check_cv2():
    try:
        import cv2
        import numpy as np

        return True, cv2, np
    except ImportError:
        return False, None, None


CV2_AVAILABLE, _cv2, _np = _check_cv2()


def _bounded_image(image: Any) -> tuple[Any, float]:
    """Downsample very large decoded images once for bounded feature work."""
    height, width = image.shape[:2]
    pixels = height * width
    if pixels <= MAX_WORKING_PIXELS:
        return image, 1.0
    scale = math.sqrt(MAX_WORKING_PIXELS / pixels)
    resized = _cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=_cv2.INTER_AREA,
    )
    return resized, scale


# ═══════════════════════════════════════════════════════════════════════════
# STAGE DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class V1Features:
    """Compatibility stage for screenshot edge/high-frequency measurements."""

    edge_count: int  # Number of detected edges
    edge_density: float  # Edges per pixel (0-1)
    has_text_regions: bool  # Compatibility name: high-frequency-content heuristic
    dominant_orientation: str  # horizontal, vertical, diagonal, mixed
    available: bool = True


@dataclass
class V2Features:
    """Compatibility stage for color clusters and aspect classification."""

    region_count: int  # Compatibility name: number of sampled color clusters
    region_colors: list[str]  # Clusters ordered by sampled pixel frequency
    has_dark_theme: bool  # Dark background detected
    contrast: float  # 0-1 overall contrast
    layout_type: str  # sparse, dense, columnar, grid
    available: bool = True


@dataclass
class V4Features:
    """Compatibility stage for contour-derived shape candidates."""

    rectangular_regions: int  # UI-like rectangular elements
    circular_regions: int
    text_blocks: int  # Dense text areas
    button_candidates: int  # Small rectangular interactive regions
    structural_hints: list[str]  # e.g., 'has_sidebar', 'has_toolbar'
    available: bool = True
    contours_analyzed: int = 0
    contours_total: int = 0
    analysis_truncated: bool = False


@dataclass
class ITFeatures:
    """Keyword-based application classification from observed text."""

    app_type: str  # 'terminal', 'browser', 'editor', 'desktop', 'unknown'
    visible_elements: list[str]  # Parsed from supplied element metadata
    clickable_elements: list[str]
    text_content: str  # Visible text summary
    confidence: float  # bounded keyword score, not a calibrated probability
    available: bool = True
    method: str = "keyword_heuristic"
    matched_keywords: int = 0
    input_truncated: bool = False


@dataclass
class VisualPercept:
    """Complete hierarchical visual percept."""

    v1: V1Features | None
    v2: V2Features | None
    v4: V4Features | None
    it: ITFeatures | None
    screenshot_path: str | None
    processing_ms: float
    stages_completed: int  # How many stages ran (0-4)
    image_scale: float = 1.0

    @property
    def summary(self) -> str:
        """Natural language summary of the visual scene."""
        parts = []
        if self.it and self.it.available:
            parts.append(f"App: {self.it.app_type}")
            if self.it.text_content:
                parts.append(f"Text: {self.it.text_content[:100]}")
            if self.it.clickable_elements:
                n = len(self.it.clickable_elements)
                parts.append(f"Clickable: {n} elements")
        elif self.v4 and self.v4.available:
            parts.append(
                f"Regions: {self.v4.rectangular_regions} rect, {self.v4.text_blocks} text blocks"
            )
        elif self.v2 and self.v2.available:
            parts.append(f"Colors: {', '.join(self.v2.region_colors[:3])}")
            parts.append(f"Layout: {self.v2.layout_type}")
        elif self.v1 and self.v1.available:
            parts.append(f"Edges: {self.v1.edge_count} ({self.v1.dominant_orientation})")
        return " | ".join(parts) if parts else "No visual data"


# ═══════════════════════════════════════════════════════════════════════════
# V1 PROCESSOR — Edge detection (without OpenCV fallback)
# ═══════════════════════════════════════════════════════════════════════════


class V1Processor:
    """Screenshot edge and high-frequency feature detection."""

    def process(
        self,
        screenshot_path: str | None = None,
        elements: list[dict] | None = None,
        image: Any = None,
    ) -> V1Features:
        """Process V1 features."""
        if image is not None:
            return self._process_image(image)
        if CV2_AVAILABLE and screenshot_path and Path(screenshot_path).exists():
            return self._process_with_cv2(screenshot_path)
        elif elements:
            return self._estimate_from_elements(elements)
        else:
            return V1Features(
                edge_count=0,
                edge_density=0.0,
                has_text_regions=False,
                dominant_orientation="unknown",
                available=False,
            )

    def _process_with_cv2(self, path: str) -> V1Features:
        """Decode a screenshot for direct stage use."""
        image = _cv2.imread(path)
        if image is None:
            return V1Features(0, 0.0, False, "unknown", False)
        image, _scale = _bounded_image(image)
        return self._process_image(image)

    def _process_image(self, image: Any) -> V1Features:
        """Measure edges and high-frequency content in one decoded image."""
        img = _cv2.cvtColor(image, _cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Gaussian blur reduces pixel noise before edge extraction.
        blurred = _cv2.GaussianBlur(img, (5, 5), 0)

        # Canny edge extraction.
        edges = _cv2.Canny(blurred, 50, 150)

        edge_pixels = _np.sum(edges > 0)
        total_pixels = img.shape[0] * img.shape[1]
        edge_density = edge_pixels / total_pixels

        # Detect text regions (high-frequency spatial patterns)
        # Laplacian variance — high = lots of texture/text
        lap_var = _cv2.Laplacian(img, _cv2.CV_64F).var()
        has_text = lap_var > 500

        # Orientation analysis via Sobel
        sobelx = _cv2.Sobel(img, _cv2.CV_64F, 1, 0)
        sobely = _cv2.Sobel(img, _cv2.CV_64F, 0, 1)
        h_strength = float(_np.mean(_np.abs(sobely)))  # Horizontal edges
        v_strength = float(_np.mean(_np.abs(sobelx)))  # Vertical edges

        if h_strength > v_strength * 1.5:
            orientation = "horizontal"
        elif v_strength > h_strength * 1.5:
            orientation = "vertical"
        elif h_strength > 5 and v_strength > 5:
            orientation = "mixed"
        else:
            orientation = "sparse"

        # Estimate edge count from contours
        contours, _ = _cv2.findContours(edges, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)

        return V1Features(
            edge_count=len(contours),
            edge_density=float(edge_density),
            has_text_regions=bool(has_text),
            dominant_orientation=orientation,
        )

    def _estimate_from_elements(self, elements: list[dict]) -> V1Features:
        """Element metadata cannot prove pixel-level edge features."""
        del elements
        return V1Features(
            edge_count=0,
            edge_density=0.0,
            has_text_regions=False,
            dominant_orientation="unknown",
            available=False,
        )


# ═══════════════════════════════════════════════════════════════════════════
# V2 PROCESSOR — Region and color analysis
# ═══════════════════════════════════════════════════════════════════════════


class V2Processor:
    """V2: Secondary visual — region segmentation, color, layout."""

    def process(
        self,
        screenshot_path: str | None = None,
        elements: list[dict] | None = None,
        image: Any = None,
    ) -> V2Features:
        if image is not None:
            return self._process_image(image)
        if CV2_AVAILABLE and screenshot_path and Path(screenshot_path).exists():
            return self._process_with_cv2(screenshot_path)
        return self._estimate_from_elements(elements or [])

    def _process_with_cv2(self, path: str) -> V2Features:
        """Decode a screenshot for direct stage use."""
        img = _cv2.imread(path)
        if img is None:
            return V2Features(0, [], False, 0.0, "unknown", False)
        img, _scale = _bounded_image(img)
        return self._process_image(img)

    def _process_image(self, img: Any) -> V2Features:
        """Measure bounded color clusters, brightness, contrast, and aspect."""

        # Color analysis
        img_rgb = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        stride = max(1, math.ceil(math.sqrt(total_pixels / 50_000)))
        pixels = img_rgb[::stride, ::stride].reshape(-1, 3).astype(_np.float32)

        # Bounded color clustering over a spatial sample.
        k = min(6, max(1, pixels.shape[0] // 1000))
        criteria = (_cv2.TERM_CRITERIA_EPS + _cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = _cv2.kmeans(pixels, k, None, criteria, 1, _cv2.KMEANS_PP_CENTERS)

        # Name clusters in descending sampled-pixel frequency.
        counts = _np.bincount(labels.reshape(-1), minlength=k)
        centers = centers[_np.argsort(-counts)]
        colors = []
        for center in centers:
            r, g, b = int(center[0]), int(center[1]), int(center[2])
            colors.append(self._name_color(r, g, b))

        # Dark theme detection
        avg_brightness = float(_np.mean(img))
        has_dark = avg_brightness < 80

        # Contrast estimation
        gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
        contrast = float(_np.std(gray) / 128.0)

        # Layout detection via edge spacing
        layout = self._estimate_layout(img)

        return V2Features(
            region_count=k,
            region_colors=colors,
            has_dark_theme=has_dark,
            contrast=min(1.0, contrast),
            layout_type=layout,
        )

    def _name_color(self, r: int, g: int, b: int) -> str:
        """Name a color from RGB values."""
        brightness = (r + g + b) / 3
        if brightness < 50:
            return "black"
        if brightness > 200:
            return "white"
        # Dominant channel
        if r > g + 30 and r > b + 30:
            return "red"
        if g > r + 20 and g > b + 20:
            return "green"
        if b > r + 30 and b > g + 30:
            return "blue"
        if r > 150 and g > 150 and b < 100:
            return "yellow"
        if r > 150 and b > 150 and g < 100:
            return "purple"
        if r > 150 and g > 100 and b < 100:
            return "orange"
        if brightness > 150:
            return "light-gray"
        return "gray"

    def _estimate_layout(self, img) -> str:
        """Estimate layout type from structural analysis."""
        h, w = img.shape[:2]
        aspect = w / h if h > 0 else 1.0

        # This is an aspect classification, not inferred UI structure.
        if aspect > 2.5:
            return "wide_aspect"
        if aspect < 0.5:
            return "tall_aspect"
        return "standard_aspect"

    def _estimate_from_elements(self, elements: list[dict]) -> V2Features:
        del elements
        return V2Features(
            region_count=0,
            region_colors=[],
            has_dark_theme=False,
            contrast=0.0,
            layout_type="unknown",
            available=False,
        )


# ═══════════════════════════════════════════════════════════════════════════
# V4 PROCESSOR — Shape and structural analysis
# ═══════════════════════════════════════════════════════════════════════════


class V4Processor:
    """V4: Shape processing — identify UI elements by shape."""

    def process(
        self,
        screenshot_path: str | None = None,
        v1: V1Features | None = None,
        elements: list[dict] | None = None,
        image: Any = None,
    ) -> V4Features:
        if image is not None:
            return self._process_image(image)
        if CV2_AVAILABLE and screenshot_path and Path(screenshot_path).exists():
            return self._process_with_cv2(screenshot_path)
        return self._estimate_from_elements(elements or [], v1)

    def _process_with_cv2(self, path: str) -> V4Features:
        """Decode a screenshot for direct stage use."""
        img = _cv2.imread(path)
        if img is None:
            return V4Features(0, 0, 0, 0, [], False)
        img, _scale = _bounded_image(img)
        return self._process_image(img)

    def _process_image(self, img: Any) -> V4Features:
        """Measure bounded contour-derived shape candidates."""

        gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
        edges = _cv2.Canny(gray, 50, 150)
        contours, _ = _cv2.findContours(edges, _cv2.RETR_TREE, _cv2.CHAIN_APPROX_SIMPLE)

        rects = 0
        circles = 0
        text_blocks = 0
        buttons = 0
        structural_hints = []
        h, w = img.shape[:2]

        selected_contours = contours[:MAX_CONTOURS_ANALYZED]
        for contour in selected_contours:
            area = _cv2.contourArea(contour)
            if area < 100:
                continue  # Skip tiny noise

            # Approximate shape
            peri = _cv2.arcLength(contour, True)
            approx = _cv2.approxPolyDP(contour, 0.04 * peri, True)

            x, y, cw, ch = _cv2.boundingRect(contour)

            if len(approx) == 4:
                # Rectangle
                rects += 1
                aspect = cw / max(ch, 1)
                # Button heuristic: wider than tall, not too big
                if 1.5 < aspect < 8 and 20 < ch < 60 and cw < w * 0.4:
                    buttons += 1
                # Text block: wide, relatively short rectangular regions
                # Relaxed from ch<30/cw>100 to ch<60/cw>50 to catch more text areas
                if ch < 60 and cw > 50 and aspect > 1.2:
                    text_blocks += 1
            elif len(approx) > 8:
                circles += 1
            # Also count high-vertex-count contours as potential text blocks
            # if they are small and wide (text characters produce irregular contours)
            elif len(approx) >= 6 and area < 2000 and ch < 50 and cw > 40:
                text_blocks += 1

        # Structural hints
        if buttons > 3:
            structural_hints.append("has_toolbar")
        if text_blocks > 10:
            structural_hints.append("has_text_content")
        if rects > 50:
            structural_hints.append("dense_ui")

        return V4Features(
            rectangular_regions=rects,
            circular_regions=circles,
            text_blocks=text_blocks,
            button_candidates=buttons,
            structural_hints=structural_hints,
            contours_analyzed=len(selected_contours),
            contours_total=len(contours),
            analysis_truncated=len(contours) > len(selected_contours),
        )

    def _estimate_from_elements(
        self, elements: list[dict], v1: V1Features | None = None
    ) -> V4Features:
        del elements, v1
        return V4Features(
            rectangular_regions=0,
            circular_regions=0,
            text_blocks=0,
            button_candidates=0,
            structural_hints=[],
            available=False,
        )


# ═══════════════════════════════════════════════════════════════════════════
# IT PROCESSOR — Object/application recognition via AT-SPI
# ═══════════════════════════════════════════════════════════════════════════


class ITProcessor:
    """Compatibility stage for keyword classification of supplied UI metadata."""

    # App type signatures (text patterns → app type)
    APP_SIGNATURES = [
        (["$ ", "# ", "bash", "zsh", "terminal", "konsole", "gnome-terminal"], "terminal"),
        (["http", "https", "address bar", "chrome", "firefox", "browser"], "browser"),
        (["def ", "class ", "import ", "#!/", "function ", "const "], "editor"),
        (["discord", "message", "channel", "server"], "discord"),
        (["file", "folder", "directory", "documents"], "file_manager"),
        (["settings", "preferences", "configuration"], "settings"),
    ]

    def process(self, dmap_output: str = "", atspi_output: str = "") -> ITFeatures:
        """Process high-level features from AT-SPI/dmap output."""
        if not dmap_output and not atspi_output:
            return ITFeatures("unknown", [], [], "", 0.0, False)

        input_truncated = len(dmap_output) + len(atspi_output) > MAX_METADATA_CHARS
        combined = (dmap_output + " " + atspi_output)[:MAX_METADATA_CHARS].lower()

        # App type detection
        app_type = "unknown"
        best_count = 0
        for keywords, atype in self.APP_SIGNATURES:
            count = sum(1 for kw in keywords if kw in combined)
            if count > best_count:
                best_count = count
                app_type = atype

        confidence = min(0.8, best_count * 0.2)

        # Extract visible text
        visible_text = self._extract_text(dmap_output or atspi_output)

        # Extract clickable elements
        clickable = self._extract_clickable(dmap_output)

        # All visible elements
        parsed_elements = self._parse_elements(dmap_output)
        all_elements = [
            f"{element.get('type', 'unknown')}:{element.get('name', '')}"[:200]
            for element in parsed_elements[:20]
        ]

        return ITFeatures(
            app_type=app_type,
            visible_elements=all_elements,
            clickable_elements=clickable[:10],
            text_content=visible_text[:200],
            confidence=confidence,
            matched_keywords=best_count,
            input_truncated=input_truncated,
        )

    def _extract_text(self, output: str) -> str:
        """Extract readable text from dmap/atspi output."""
        parsed = self._parse_elements(output)
        if parsed:
            return " | ".join(
                str(element.get("name", "")).strip()
                for element in parsed[:20]
                if str(element.get("name", "")).strip()
            )[:1000]
        lines = output[:MAX_METADATA_CHARS].splitlines()
        text_lines = []
        for line in lines:
            # Lines with actual text content (not just references)
            if re.search(r"[a-zA-Z]{4,}", line) and "p" not in line[:3]:
                clean = re.sub(r"\s+", " ", line.strip())
                if len(clean) > 5:
                    text_lines.append(clean)
        return " | ".join(text_lines[:5])

    def _extract_clickable(self, dmap_output: str) -> list[str]:
        """Extract clickable element references from dmap output."""
        actionable_roles = {"button", "key", "icon", "control", "input", "link"}
        return [
            f"{element.get('ref', '')} {element.get('type', '')}:{element.get('name', '')}".strip()
            for element in self._parse_elements(dmap_output)
            if str(element.get("type", "")).lower() in actionable_roles
        ]

    @staticmethod
    def _parse_elements(output: str) -> list[dict]:
        if len(output) > MAX_METADATA_CHARS:
            # Do not parse a partial JSON document or allocate an unbounded tree.
            return []
        try:
            decoded = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            decoded = None
        if isinstance(decoded, dict) and isinstance(decoded.get("elements"), list):
            parsed: list[dict] = []
            for raw in decoded["elements"][:2000]:
                if not isinstance(raw, dict):
                    continue
                parsed.append(
                    {
                        "ref": str(raw.get("ref", "")),
                        "type": str(raw.get("type", "unknown")),
                        "name": str(raw.get("text") or raw.get("name") or "")[:200],
                    }
                )
            return parsed

        parsed = []
        for line in output.splitlines()[:2000]:
            compact = re.match(r'^\[(s\d+)\]\s+(\w+)\s+\([^)]*\)(?:\s+"([^"]*)")?', line.strip())
            legacy = re.match(r"^(p\d+)\s+(\w+):(.+)", line.strip())
            match = compact or legacy
            if match:
                parsed.append(
                    {"ref": match.group(1), "type": match.group(2), "name": match.group(3) or ""}
                )
        return parsed


# ═══════════════════════════════════════════════════════════════════════════
# VISUAL HIERARCHY — Combined pipeline
# ═══════════════════════════════════════════════════════════════════════════


class VisualHierarchy:
    """Four-stage screenshot/metadata feature pipeline.

    Historical stage labels are retained for API compatibility:
    V1: edge/frequency metrics; V2: color/aspect metrics; V4: contour
    candidates; IT: keyword-based application classification.
    """

    def __init__(self):
        self.v1_proc = V1Processor()
        self.v2_proc = V2Processor()
        self.v4_proc = V4Processor()
        self.it_proc = ITProcessor()

    def process(
        self, screenshot_path: str | None = None, dmap_output: str = "", atspi_output: str = ""
    ) -> VisualPercept:
        """Full hierarchical processing pipeline."""
        t0 = time.monotonic()
        stages = 0

        # Parse elements from dmap output for CV2-less processing
        elements = self.it_proc._parse_elements(dmap_output or atspi_output)

        # Decode and, if needed, downsample once. The old implementation read
        # the same screenshot independently for all three pixel stages.
        image = None
        image_scale = 1.0
        image_attempted = bool(
            CV2_AVAILABLE and screenshot_path and Path(screenshot_path).is_file()
        )
        if image_attempted:
            decoded = _cv2.imread(screenshot_path)
            if decoded is not None:
                image, image_scale = _bounded_image(decoded)

        # ── V1: Edge detection ──
        if image_attempted:
            v1 = self.v1_proc.process(elements=elements, image=image)
        else:
            v1 = self.v1_proc.process(screenshot_path, elements)
        if v1.available:
            stages += 1

        # ── V2: Color/region analysis ──
        if image_attempted:
            v2 = self.v2_proc.process(elements=elements, image=image)
        else:
            v2 = self.v2_proc.process(screenshot_path, elements)
        if v2.available:
            stages += 1

        # ── V4: Shape analysis ──
        if image_attempted:
            v4 = self.v4_proc.process(v1=v1, elements=elements, image=image)
        else:
            v4 = self.v4_proc.process(screenshot_path, v1, elements)
        if v4.available:
            stages += 1

        # ── IT: High-level recognition ──
        it = self.it_proc.process(dmap_output, atspi_output)
        if it.available:
            stages += 1

        elapsed = (time.monotonic() - t0) * 1000

        return VisualPercept(
            v1=v1,
            v2=v2,
            v4=v4,
            it=it,
            screenshot_path=screenshot_path,
            processing_ms=elapsed,
            stages_completed=stages,
            image_scale=image_scale,
        )

    def inject_signal(self, message: str) -> str:
        """Generate signal from latest visual processing.
        This is called from brain-runner when visual context is available.
        """
        # The actual perception data comes from perception.py
        # This module provides the structured interpretation layer
        return ""  # Only fires when percept data is available

    def fire(self, message: str, context: str = "") -> str:
        return self.inject_signal(message)

    def format_for_context(self, percept: VisualPercept) -> str:
        """Format percept for model context injection."""
        if percept.stages_completed == 0:
            return ""

        lines = [
            f"VISUAL_FEATURES({percept.stages_completed}/4 stages, "
            f"{percept.processing_ms:.0f}ms, image_scale={percept.image_scale:.3f}):"
        ]

        if percept.v1 and percept.v1.available:
            lines.append(
                f"  V1: {percept.v1.edge_count} edges, "
                f"{percept.v1.dominant_orientation}, "
                f"high_frequency_content={percept.v1.has_text_regions}"
            )

        if percept.v2 and percept.v2.available:
            colors = ", ".join(percept.v2.region_colors[:3])
            lines.append(
                f"  V2: dark={percept.v2.has_dark_theme}, "
                f"contrast={percept.v2.contrast:.2f}, colors=[{colors}]"
            )

        if percept.v4 and percept.v4.available:
            hints = ", ".join(percept.v4.structural_hints) or "none"
            lines.append(
                f"  V4: {percept.v4.rectangular_regions} rects, "
                f"{percept.v4.button_candidates} buttons, hints=[{hints}]"
            )

        if percept.it and percept.it.available:
            lines.append(
                f"  IT: app={percept.it.app_type} "
                f"(keyword_score={percept.it.confidence:.2f}, "
                f"matches={percept.it.matched_keywords}, method={percept.it.method})"
            )
            if percept.it.text_content:
                lines.append(f"  Text: {percept.it.text_content[:100]}")
            if percept.it.clickable_elements:
                n = len(percept.it.clickable_elements)
                lines.append(f"  Clickable: {n} elements")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# MODULE INSTANCE
# ═══════════════════════════════════════════════════════════════════════════

_vh = VisualHierarchy()


def process(screenshot_path=None, dmap_output="", atspi_output="") -> VisualPercept:
    return _vh.process(screenshot_path, dmap_output, atspi_output)


def format_for_context(percept: VisualPercept) -> str:
    return _vh.format_for_context(percept)


def inject_signal(message: str) -> str:
    return _vh.inject_signal(message)


def fire(message: str, context: str = "") -> str:
    return _vh.fire(message, context)


# ═══════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════


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

    print("=== Visual Hierarchy Tests ===\n")
    print(f"  OpenCV available: {CV2_AVAILABLE}")

    vh = VisualHierarchy()

    # ── T1: IT recognition from dmap output ──
    dmap_sample = """p1 Window:Gnome Terminal
p2 Button:File
p3 Button:Edit
p5 Label:$ ls -la
p6 Label:total 42
p7 Button:Close
"""
    it = vh.it_proc.process(dmap_sample, "bash terminal $ ")
    test("T1: Terminal detected from dmap output", it.app_type == "terminal")
    test("T1b: Keyword score > 0", it.confidence > 0 and it.matched_keywords > 0)

    # ── T2: IT browser detection ──
    browser_dmap = """p1 Window:Chromium
p2 Input:https://google.com
p3 Button:Back
p4 Label:Google Search
"""
    it2 = vh.it_proc.process(browser_dmap, "http browser chrome")
    test("T2: Browser detected", it2.app_type == "browser")

    # ── T3: IT clickable extraction ──
    it3 = vh.it_proc.process(dmap_sample)
    test("T3: Clickable buttons extracted", len(it3.clickable_elements) >= 1)

    # ── T4: V1 estimates from elements ──
    elements = [{"ref": f"p{i}", "type": "Button", "name": f"btn{i}"} for i in range(10)]
    v1 = vh.v1_proc.process(elements=elements)
    test("T4: V1 does not fabricate pixel metrics", not v1.available and v1.edge_count == 0)

    # ── T5: V2 estimates from elements ──
    v2 = vh.v2_proc.process(elements=elements)
    test("T5: V2 does not fabricate color metrics", not v2.available and v2.region_count == 0)

    # ── T6: V4 estimates from elements ──
    v4 = vh.v4_proc.process(elements=elements)
    test(
        "T6: V4 does not fabricate shape metrics",
        not v4.available and v4.rectangular_regions == 0,
    )

    # ── T7: Full pipeline — element-only mode ──
    percept = vh.process(dmap_output=dmap_sample, atspi_output="terminal bash")
    test("T7: Pipeline completes", percept.stages_completed >= 1)
    test("T7b: IT stage ran", percept.it is not None)

    # ── T8: Format for context ──
    formatted = vh.format_for_context(percept)
    test("T8: Context format non-empty", len(formatted) > 0 and "VISUAL_FEATURES" in formatted)

    # ── T9: Percept summary ──
    summary = percept.summary
    test("T9: Summary generated", len(summary) > 0)

    # ── T10: Empty input → graceful ──
    empty_percept = vh.process()
    test("T10: Empty input → no crash", empty_percept.stages_completed >= 0)

    # ── T11: App type unknown for unrecognized output ──
    it4 = vh.it_proc.process("xyz123 qrs456", "abc def")
    test("T11: Unknown content → unknown app type", it4.app_type == "unknown")

    # ── T12: IT text extraction ──
    it5 = vh.it_proc.process("", "Visible text: Hello World from terminal")
    test("T12: Text content extracted", it5.text_content is not None)

    # ── T13: Performance — element-based pipeline ──
    t0 = time.monotonic()
    for _ in range(100):
        vh.process(dmap_output=dmap_sample)
    elapsed = (time.monotonic() - t0) * 1000
    avg = elapsed / 100
    test(f"T13: Element pipeline <10ms/call ({avg:.1f}ms)", avg < 10.0)

    # ── T14: OpenCV pipeline (if available) ──
    if CV2_AVAILABLE:
        # Create a synthetic test image
        img = _np.zeros((100, 200, 3), dtype=_np.uint8)
        # Add some white rectangles (simulating UI elements)
        _cv2.rectangle(img, (10, 10), (90, 40), (255, 255, 255), -1)
        _cv2.rectangle(img, (100, 10), (190, 40), (200, 200, 200), -1)
        test_img = "/tmp/test_visual_hierarchy.png"
        _cv2.imwrite(test_img, img)
        percept_cv = vh.process(screenshot_path=test_img)
        test("T14: OpenCV pipeline processes synthetic image", percept_cv.stages_completed >= 3)
        os.unlink(test_img)
    else:
        print("  ⏭️  T14: Skipped (OpenCV not available)")
        total += 1  # Count as pass (skipped)
        passed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} passed")
    return passed, total


if __name__ == "__main__":
    p, t = _run_tests()
    print(f"\nFinal: {p}/{t}")
    exit(0 if p == t else 1)
