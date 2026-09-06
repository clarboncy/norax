"""Stateful cognitive components and opt-in idle learning maintenance."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from ._mixin import RuntimeAccessMixin

log = logging.getLogger("norax.runtime.core")


class CognitionMixin(RuntimeAccessMixin):
    _active_inference: Any
    _analogy_engine: Any
    _best_of_n: Any
    _curiosity: Any
    _domain_transfer: Any
    _metacognitive: Any
    _output_verifier: Any
    _self_model: Any

    def _ensure_cognitive_components(self) -> None:
        """Lazily initialize and then reuse stateful cognitive components.

        Turn handlers are concurrent across channels, but initialization and
        each component's record/save methods are synchronous. They therefore
        execute without an asyncio scheduling point and safely share one
        in-process state snapshot.
        """
        if self._output_verifier is None:
            self._output_verifier = self.capabilities.try_init(
                "output_verifier",
                lambda: __import__(
                    "norax.brain.output_verifier", fromlist=["OutputVerifier"]
                ).OutputVerifier(),
            )

        experimental = bool(getattr(self, "_experimental_cognitive_signals", False))

        if experimental and self._self_model is None:
            try:
                from ..brain.self_model import SelfModel

                self._self_model = (
                    SelfModel(self._memory_root / "self_model.json")
                    if self._memory_root
                    else SelfModel()
                )
                self._self_model.load()
                self.capabilities.mark_ok("self_model")
            except Exception as exc:  # noqa: BLE001
                self._self_model = None
                self.capabilities.mark_failed("self_model", exc)

        if (
            experimental or bool(getattr(self, "_active_inference_enabled", False))
        ) and self._active_inference is None:
            try:
                from ..brain.active_inference import ActiveInference

                self._active_inference = (
                    ActiveInference(self._memory_root / "active_inference.json")
                    if self._memory_root
                    else ActiveInference()
                )
                self._active_inference.load()
                self.capabilities.mark_ok("active_inference")
            except Exception as exc:  # noqa: BLE001
                self._active_inference = None
                self.capabilities.mark_failed("active_inference", exc)

        if experimental and self._metacognitive is None:
            try:
                from ..brain.metacognitive import MetacognitiveCalibration

                self._metacognitive = (
                    MetacognitiveCalibration(self._memory_root / "metacog.json")
                    if self._memory_root
                    else MetacognitiveCalibration()
                )
                self._metacognitive.load()
                self.capabilities.mark_ok("metacognitive")
            except Exception as exc:  # noqa: BLE001
                self._metacognitive = None
                self.capabilities.mark_failed("metacognitive", exc)

        best_of_n_enabled = os.environ.get("NORAX_BEST_OF_N", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if best_of_n_enabled and self._best_of_n is None:
            try:
                from ..brain.best_of_n import BestOfN

                self._best_of_n = BestOfN(gateway=self.gateway)
                self.capabilities.mark_ok("best_of_n")
            except Exception as exc:  # noqa: BLE001
                self._best_of_n = None
                self.capabilities.mark_failed("best_of_n", exc)

        if not experimental:
            return

        if self._curiosity is None:
            try:
                from ..brain.curiosity_engine import CuriosityEngine

                self._curiosity = (
                    CuriosityEngine(self._memory_root / "curiosity.json")
                    if self._memory_root
                    else CuriosityEngine()
                )
                self._curiosity.load()
                self.capabilities.mark_ok("curiosity_engine")
            except Exception as exc:  # noqa: BLE001
                self._curiosity = None
                self.capabilities.mark_failed("curiosity_engine", exc)

        if self._domain_transfer is None:
            try:
                from ..brain.domain_transfer import DomainTransfer

                self._domain_transfer = DomainTransfer(
                    self_model=self._self_model,
                    memory_root=self._memory_root,
                )
                self.capabilities.mark_ok("domain_transfer")
            except Exception as exc:  # noqa: BLE001
                self._domain_transfer = None
                self.capabilities.mark_failed("domain_transfer", exc)
        elif self._self_model is not None:
            self._domain_transfer.self_model = self._self_model

        if self._analogy_engine is None:
            try:
                from ..brain.analogy_engine import AnalogyEngine

                self._analogy_engine = AnalogyEngine(
                    memory_root=self._memory_root,
                    episodic=self._episodic,
                )
                self.capabilities.mark_ok("analogy_engine")
            except Exception as exc:  # noqa: BLE001
                self._analogy_engine = None
                self.capabilities.mark_failed("analogy_engine", exc)

    async def _idle_sleep_loop(self) -> None:
        """Run sleep consolidation when idle for 5+ minutes."""
        IDLE_THRESHOLD = 300  # 5 minutes
        CHECK_INTERVAL = 120  # check every 2 minutes
        while True:
            try:
                await asyncio.sleep(CHECK_INTERVAL)
                idle_sec = time.time() - self._last_turn_time
                if idle_sec < IDLE_THRESHOLD:
                    continue
                if self._turn_work_count > 0 or any(
                    not task.done() for task in self._active_turn_tasks.values()
                ):
                    continue
                if self._memory_store is None:
                    continue
                # Check if there's anything in sleep/ to consolidate
                sleep_dir = self._memory_store.root / "sleep"
                buffers = list(sleep_dir.glob("buffer-*.md"))
                spills = list(sleep_dir.glob("spill-*.jsonl"))
                spill_mds = list(sleep_dir.glob("spill-*.md"))
                total_files = len(buffers) + len(spills) + len(spill_mds)
                if total_files and self._memory_coordinator is not None:
                    log.info(
                        "sleep_consolidation: idle %.0fs, %d files (buffers=%d spills=%d spill_mds=%d)",
                        idle_sec,
                        total_files,
                        len(buffers),
                        len(spills),
                        len(spill_mds),
                    )
                    try:
                        result = await self._memory_coordinator.consolidate_sleep()
                        if result["canonical_writes"]:
                            await self._memory_coordinator.sync_projections()
                        await self.events.append("sleep_consolidation", result)
                        log.info(
                            "sleep_consolidation: files=%d spills=%d candidates=%d writes=%d duplicates=%d",
                            result["files_processed"],
                            result["spills_processed"],
                            result["candidates_seen"],
                            result["canonical_writes"],
                            result["duplicates_skipped"],
                        )
                    except Exception as e:
                        log.warning("sleep_consolidation.error: %r", e)

                # Verified trajectory replay is an explicit learning feature.
                if self._idle_learning_enabled and self._episodic is not None:
                    try:
                        from ..brain.sleep.replay import HippocampalReplay

                        replay = HippocampalReplay(
                            episodic=self._episodic,
                            memory_root=self._memory_store.root,
                        )
                        replay_result = await asyncio.to_thread(replay.run)
                        if replay_result.procedural_patterns or replay_result.failure_patterns:
                            if self._memory_coordinator is not None:
                                self._memory_coordinator.canonical_changed("hippocampal_replay")
                            await self.events.append(
                                "hippocampal_replay",
                                {
                                    "patterns": len(replay_result.procedural_patterns),
                                    "failures": len(replay_result.failure_patterns),
                                    "coactivation_pairs_observed": (
                                        replay_result.coactivation_pairs_observed
                                    ),
                                    "high_surprise_episodes_seen": (
                                        replay_result.high_surprise_episodes_seen
                                    ),
                                },
                            )
                            log.info(
                                "hippocampal_replay: %d patterns, %d failures",
                                len(replay_result.procedural_patterns),
                                len(replay_result.failure_patterns),
                            )
                            # Feed replay patterns back into ToolExperienceMemory
                            # so future tool retrieval can surface verified
                            # sequences and failure recovery guidance.
                            try:
                                from ..memory.tool_experience import ToolExperienceMemory

                                tem = ToolExperienceMemory(self._memory_store.root)
                                ingested = await asyncio.to_thread(
                                    tem.ingest_replay_patterns,
                                    self._memory_store.root / "procedural",
                                )
                                if ingested:
                                    log.info(
                                        "hippocampal_replay: %d patterns ingested into tool_experience",
                                        ingested,
                                    )
                            except Exception as e:
                                log.debug("tool_experience.ingest_replay.error: %r", e)
                    except Exception as e:
                        log.debug("hippocampal_replay.error: %r", e)

                # Sprint D: Skill acquisition — mine trajectories for patterns
                if self._skill_learner is not None:
                    try:
                        from ..brain.harness_optimizer import load_trajectories

                        event_log = getattr(self.events, "path", None)
                        if event_log and isinstance(event_log, Path) and event_log.exists():
                            trajectories = await asyncio.to_thread(
                                load_trajectories, event_log, limit=200
                            )
                            if trajectories:
                                patterns = await asyncio.to_thread(
                                    self._skill_learner.mine, trajectories
                                )
                                if patterns:
                                    sl_result = await asyncio.to_thread(
                                        self._skill_learner.generate, patterns
                                    )
                                    if sl_result.skills_created or sl_result.skills_updated:
                                        if self._memory_coordinator is not None:
                                            self._memory_coordinator.canonical_changed(
                                                "skill_learning"
                                            )
                                        await self.events.append(
                                            "skill_learning",
                                            {
                                                "created": sl_result.skills_created,
                                                "updated": sl_result.skills_updated,
                                                "skills": sl_result.skills,
                                            },
                                        )
                                        log.info(
                                            "skill_learning: created=%d updated=%d skills=%s",
                                            sl_result.skills_created,
                                            sl_result.skills_updated,
                                            sl_result.skills,
                                        )
                    except Exception as e:
                        log.debug("skill_learning.error: %r", e)

                if self._episodic is not None:
                    try:
                        await asyncio.to_thread(self._episodic.prune_old)
                    except Exception as e:
                        log.debug("episodic.prune.error: %r", e)

                # Incrementally sync only when an upstream component reported
                # a real canonical mutation.  Unconditionally marking this
                # dirty caused a full projection pass every idle poll.
                try:
                    if self._memory_coordinator is not None:
                        await self._memory_coordinator.sync_projections()
                except Exception as e:
                    log.debug("memory_projection_sync.error: %r", e)

                # The opt-in calibration component is already loaded and
                # updated on evidence-bearing turns.  Reuse it; never create a
                # second default-on telemetry path during idle maintenance.
                try:
                    if self._metacognitive is not None:
                        report = self._metacognitive.get_report()
                    else:
                        report = None
                    if report is not None and report.is_reliable and report.is_overconfident:
                        log.warning(
                            "metacognitive: OVERCONFIDENT bias=%.2f brier=%.3f — reducing confidence",
                            report.bias_magnitude,
                            report.brier_score,
                        )
                        await self.events.append(
                            "metacognitive_alert",
                            {
                                "bias": report.bias.value,
                                "magnitude": round(report.bias_magnitude, 3),
                                "brier": round(report.brier_score, 3),
                                "trend": report.trend,
                            },
                        )
                except Exception as e:
                    log.debug("metacognitive.idle.error: %r", e)

                # Retrospective harness analysis: mine live trajectories into
                # a reviewable report. This path never calls an LLM or edits
                # source code.
                try:
                    rho_last = getattr(self, "_rho_last_run", 0.0)
                    if self._harness_analysis_enabled and time.time() - rho_last > 1800:  # 30 min
                        from ..brain.harness_optimizer import analyze_harness

                        event_log_path = getattr(self.events, "path", None)
                        if (
                            event_log_path
                            and isinstance(event_log_path, Path)
                            and event_log_path.exists()
                        ):
                            event_mtime = event_log_path.stat().st_mtime_ns
                            if event_mtime != getattr(self, "_rho_last_event_mtime_ns", None):
                                out_dir = self._memory_store.root / "state" / "harness_optimizer"
                                rho_result = await asyncio.to_thread(
                                    analyze_harness,
                                    event_log_path,
                                    out_dir,
                                    k=10,
                                    limit=200,
                                )
                                await self.events.append(
                                    "rho_analysis",
                                    {
                                        "status": rho_result.get("status"),
                                        "source_mutations": 0,
                                        "proposal_count": rho_result.get("proposal_count", 0),
                                        "report_path": rho_result.get("report_path"),
                                    },
                                )
                                log.info(
                                    "rho_analysis: status=%s proposals=%s",
                                    rho_result.get("status"),
                                    rho_result.get("proposal_count", 0),
                                )
                                self._rho_last_event_mtime_ns = event_log_path.stat().st_mtime_ns
                        self._rho_last_run = time.time()
                except Exception as e:
                    log.debug("rho_analysis.error: %r", e)

            except asyncio.CancelledError:
                break
            except Exception:
                log.debug("idle_sleep_loop.error", exc_info=True)
                await asyncio.sleep(60)
