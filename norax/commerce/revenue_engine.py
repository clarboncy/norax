"""Evidence-backed accounting and discovery status for commerce features.

Only receipts with an external evidence identifier count as earned revenue.
Opportunity discovery is reported separately and never converted into dollar
projections or autonomous financial action.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ..atomic import atomic_write_text, path_lock, read_bounded_text
from .airdrop_hunter import AirdropHunter
from .payment_gate import (
    SessionPersistenceError,
    commerce_state_dir,
    get_session_store,
    get_wallet_info,
)

log = logging.getLogger("norax.commerce.engine")

_SOURCE_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_EVM_TRANSACTION_RE = re.compile(r"0x[a-fA-F0-9]{64}")
_REVENUE_MAX_BYTES = 128 * 1024 * 1024
_REVENUE_MAX_ENTRIES = 250_000
_EVIDENCE_MAX_CHARS = 512
_DESCRIPTION_MAX_CHARS = 2_000


def _default_revenue_log() -> Path:
    configured = os.environ.get("NORAX_REVENUE_LOG")
    if configured:
        return Path(configured).expanduser()
    return commerce_state_dir(create=True) / "revenue_log.json"


class RevenuePersistenceError(RuntimeError):
    """Revenue ledger could not be read or committed safely."""


@dataclass(frozen=True)
class RevenueEntry:
    timestamp: float
    source: str
    amount_usd: float
    description: str
    tx_hash: str = ""  # transaction hash or another durable external receipt id

    @property
    def verified(self) -> bool:
        return bool(self.tx_hash.strip())

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "amount_usd": self.amount_usd,
            "description": self.description,
            "tx_hash": self.tx_hash,
            "verified": self.verified,
        }


class RevenueEngine:
    """Maintains a durable, idempotent ledger of externally evidenced receipts."""

    def __init__(
        self,
        persist_path: Path | None = None,
        *,
        airdrop: AirdropHunter | None = None,
    ) -> None:
        self.persist_path = persist_path or _default_revenue_log()
        self.airdrop = airdrop or AirdropHunter()
        self.entries: list[RevenueEntry] = []
        self._load_error = ""
        self._loaded_signature: tuple[int, int, int, int, int] | None = None
        self._had_persisted_state = False
        self._load()

    def _file_signature(self) -> tuple[int, int, int, int, int] | None:
        try:
            stat_result = self.persist_path.lstat()
        except FileNotFoundError:
            return None
        return (
            stat_result.st_dev,
            stat_result.st_ino,
            stat_result.st_size,
            stat_result.st_mtime_ns,
            stat_result.st_ctime_ns,
        )

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.persist_path.with_name(f".{self.persist_path.name}.lock")
        with path_lock(self.persist_path):
            flags = (
                os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            fd = os.open(lock_path, flags, 0o600)
            try:
                lock_stat = os.fstat(fd)
                if not stat.S_ISREG(lock_stat.st_mode):
                    raise RevenuePersistenceError("revenue ledger lock is not a regular file")
                os.fchmod(fd, 0o600)
                try:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_EX)
                except ImportError:  # pragma: no cover - non-POSIX fallback
                    pass
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
                except ImportError:  # pragma: no cover - non-POSIX fallback
                    pass
                os.close(fd)

    @staticmethod
    def _entry_from_dict(row: dict) -> RevenueEntry:
        raw_timestamp = row.get("timestamp")
        raw_amount = row.get("amount_usd")
        if (
            isinstance(raw_timestamp, bool)
            or not isinstance(raw_timestamp, int | float)
            or isinstance(raw_amount, bool)
            or not isinstance(raw_amount, int | float)
        ):
            raise ValueError("invalid revenue numeric value")
        timestamp = float(raw_timestamp)
        amount = float(raw_amount)
        source = row.get("source")
        description = row.get("description", "")
        evidence_id = row.get("tx_hash", "")
        if (
            not isinstance(source, str)
            or not isinstance(description, str)
            or not isinstance(evidence_id, str)
        ):
            raise ValueError("invalid revenue metadata types")
        evidence_id = evidence_id.strip()
        if not math.isfinite(timestamp) or timestamp <= 0 or timestamp > time.time() + 300:
            raise ValueError("invalid revenue timestamp")
        if not math.isfinite(amount) or amount <= 0 or amount > 1_000_000_000:
            raise ValueError("invalid revenue amount")
        if not _SOURCE_RE.fullmatch(source):
            raise ValueError("invalid revenue source")
        if (
            not description.strip()
            or len(description) > _DESCRIPTION_MAX_CHARS
            or len(evidence_id) > _EVIDENCE_MAX_CHARS
            or "\x00" in description
            or any(ord(character) < 0x20 for character in evidence_id)
        ):
            raise ValueError("invalid revenue metadata")
        if (
            source == "api_payment"
            and evidence_id
            and not _EVM_TRANSACTION_RE.fullmatch(evidence_id)
        ):
            raise ValueError("api_payment evidence must be an EVM transaction hash")
        return RevenueEntry(timestamp, source, amount, description, evidence_id)

    def _load(self, *, force: bool = False) -> None:
        signature = self._file_signature()
        if not force and signature == self._loaded_signature and not self._load_error:
            return
        if signature is None:
            if self._had_persisted_state:
                self.entries = []
                self._load_error = "revenue ledger disappeared"
                return
            self.entries = []
            self._loaded_signature = None
            self._load_error = ""
            return
        try:
            payload = json.loads(read_bounded_text(self.persist_path, max_bytes=_REVENUE_MAX_BYTES))
            rows = payload.get("entries") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                raise ValueError("revenue ledger must contain an entries list")
            raw_version = payload.get("version", 1)
            if (
                isinstance(raw_version, bool)
                or not isinstance(raw_version, int)
                or raw_version not in {1, 2}
            ):
                raise ValueError("unsupported revenue ledger version")
            if len(rows) > _REVENUE_MAX_ENTRIES:
                raise ValueError("revenue ledger exceeds its entry limit")
            if not all(isinstance(row, dict) for row in rows):
                raise ValueError("revenue ledger contains a non-object entry")
            loaded_entries = [self._entry_from_dict(row) for row in rows]
            evidence_ids = [entry.tx_hash.casefold() for entry in loaded_entries if entry.tx_hash]
            if len(evidence_ids) != len(set(evidence_ids)):
                raise ValueError("revenue ledger contains duplicate evidence ids")
            self.entries = loaded_entries
            self._loaded_signature = signature
            self._had_persisted_state = True
            self._load_error = ""
        except Exception as exc:  # noqa: BLE001
            self.entries = []
            self._loaded_signature = signature
            self._load_error = str(exc)
            log.warning("revenue_engine.log_load_failed error=%r", exc)

    def _require_healthy(self) -> None:
        if self._load_error:
            raise RevenuePersistenceError(
                f"revenue ledger could not be restored safely: {self._load_error}"
            )

    def _persist_unlocked(self) -> None:
        if len(self.entries) > _REVENUE_MAX_ENTRIES:
            raise RevenuePersistenceError("revenue ledger exceeds its entry limit")
        payload = {
            "version": 2,
            "entries": [entry.to_dict() for entry in self.entries],
            "verified_total_earned_usd": self.total_earned,
            "updated": time.time(),
        }
        serialized = (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
        if len(serialized.encode("utf-8")) > _REVENUE_MAX_BYTES:
            raise RevenuePersistenceError("revenue ledger exceeds its byte limit")
        atomic_write_text(
            self.persist_path,
            serialized,
            durable=True,
            mode=0o600,
        )
        self._loaded_signature = self._file_signature()
        self._had_persisted_state = True

    @property
    def total_earned(self) -> float:
        """Total externally evidenced revenue; legacy unverified rows are excluded."""
        return math.fsum(entry.amount_usd for entry in self.entries if entry.verified)

    @property
    def unverified_total(self) -> float:
        return math.fsum(entry.amount_usd for entry in self.entries if not entry.verified)

    def record_earning(
        self, source: str, amount_usd: float, description: str, tx_hash: str = ""
    ) -> RevenueEntry:
        """Idempotently record a receipt backed by a transaction/evidence id."""
        candidate = self._entry_from_dict(
            {
                "timestamp": time.time(),
                "source": source,
                "amount_usd": amount_usd,
                "description": description,
                "tx_hash": tx_hash,
            }
        )
        if not candidate.verified:
            raise ValueError("an external transaction or evidence id is required")

        with self._transaction():
            self._load(force=True)
            self._require_healthy()
            normalized_evidence = candidate.tx_hash.casefold()
            for existing in self.entries:
                if existing.tx_hash.casefold() != normalized_evidence:
                    continue
                if (
                    existing.source == candidate.source
                    and existing.amount_usd == candidate.amount_usd
                    and existing.description == candidate.description
                ):
                    return existing
                raise ValueError("evidence id is already recorded with different receipt data")
            self.entries.append(candidate)
            try:
                self._persist_unlocked()
            except Exception as exc:  # noqa: BLE001
                self.entries.pop()
                raise RevenuePersistenceError("could not durably record revenue receipt") from exc
        log.info("revenue receipt: +$%.2f from %s", candidate.amount_usd, candidate.source)
        return candidate

    def status(self) -> dict:
        """Return measured commerce status without speculative projections."""
        self._load()
        try:
            active_sessions = get_session_store().active_count()
            sessions_ok = True
        except SessionPersistenceError:
            active_sessions = 0
            sessions_ok = False
        wallet = get_wallet_info()
        wallet_configured = bool(wallet.get("evm_address"))
        return {
            "verified_total_earned_usd": self.total_earned,
            # Compatibility alias; semantics are explicitly verified only.
            "total_earned_usd": self.total_earned,
            "unverified_legacy_total_usd": self.unverified_total,
            "verified_receipt_count": sum(entry.verified for entry in self.entries),
            "ledger_ok": not self._load_error,
            "ledger_error": self._load_error or None,
            "active_api_sessions": active_sessions,
            "session_store_ok": sessions_ok,
            "airdrop_campaigns": self.airdrop.status_report(),
            "wallet": wallet,
            "revenue_paths": {
                "api_service": {
                    "status": (
                        "active_sessions"
                        if active_sessions
                        else "configured"
                        if wallet_configured
                        else "not_configured"
                    ),
                    "active_sessions": active_sessions,
                    "observed_revenue_usd": self.total_earned,
                },
                "opportunity_discovery": {
                    "status": "discovery_only",
                    "financial_actions_enabled": False,
                    "unverified_leads_are_revenue": False,
                },
            },
        }

    async def idle_scan(self) -> dict:
        """Discover unverified leads; never participate or assign dollar value."""
        log.info("revenue: running discovery-only idle scan")
        try:
            leads = await self.airdrop.scan_for_new_campaigns()
            error = None
        except Exception as exc:  # noqa: BLE001
            log.warning("opportunity discovery failed: %s", exc)
            leads = []
            error = type(exc).__name__
        return {
            "unverified_leads": leads[:5],
            "discovery_error": error,
            "financial_actions_taken": 0,
            "verified_total_earned_usd": self.total_earned,
        }


_engine: RevenueEngine | None = None


def get_revenue_engine() -> RevenueEngine:
    global _engine
    if _engine is None:
        _engine = RevenueEngine()
    return _engine
