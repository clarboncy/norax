"""Discovery-only tracker for potential zero-capital reward campaigns.

Financial campaigns change quickly and search snippets are not proof that a
campaign is legitimate, current, free, or safe. This module therefore treats
web results only as unverified leads. It never connects a wallet, signs a
message, sends a transaction, or labels a lead actionable without a recent
manual verification and approval recorded in the tracker.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from ..atomic import atomic_write_text, read_bounded_text
from .payment_gate import commerce_state_dir

log = logging.getLogger("norax.commerce.airdrop")

_VERIFICATION_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
_TRACKER_MAX_BYTES = 16 * 1024 * 1024
_TRACKER_MAX_CAMPAIGNS = 20_000
_CAMPAIGN_STATUSES = frozenset({"researching", "approved", "rejected", "completed"})


def _clean_text(
    value: object,
    *,
    field: str,
    max_chars: int,
    allow_empty: bool = True,
    multiline: bool = False,
) -> str:
    if not isinstance(value, str) or len(value) > max_chars or "\x00" in value:
        raise ValueError(f"invalid campaign {field}")
    if multiline:
        if any(ord(char) < 0x20 and char not in "\n\r\t" for char in value):
            raise ValueError(f"invalid campaign {field}")
        cleaned = value.strip()
    else:
        if any(ord(char) < 0x20 for char in value):
            raise ValueError(f"invalid campaign {field}")
        cleaned = " ".join(value.split())
    if not allow_empty and not cleaned:
        raise ValueError(f"campaign {field} is required")
    return cleaned


def _http_url(value: object, *, field: str, allow_empty: bool = False) -> str:
    url = _clean_text(value, field=field, max_chars=2_000, allow_empty=allow_empty)
    if not url and allow_empty:
        return ""
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(f"invalid campaign {field}")
    return url


def _default_tracker_path() -> Path:
    configured = os.environ.get("NORAX_AIRDROP_TRACKER")
    if configured:
        return Path(configured).expanduser()
    return commerce_state_dir(create=True) / "airdrop_tracker.json"


@dataclass
class AirdropCampaign:
    """Operator-reviewed campaign metadata; never an authorization receipt."""

    name: str
    url: str
    chain: str = ""
    funding: str = ""
    status: str = "researching"  # researching|approved|rejected|completed
    tasks_total: int = 0
    tasks_done: int = 0
    joined_at: float = 0.0
    notes: str = ""
    estimated_value: str = ""  # legacy display-only field, never aggregated
    zero_capital: bool | None = None
    requires_transaction: bool | None = None
    verified_at: float = 0.0
    source_url: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "url": self.url,
            "chain": self.chain,
            "funding": self.funding,
            "status": self.status,
            "tasks_total": self.tasks_total,
            "tasks_done": self.tasks_done,
            "joined_at": self.joined_at,
            "notes": self.notes,
            "estimated_value": self.estimated_value,
            "zero_capital": self.zero_capital,
            "requires_transaction": self.requires_transaction,
            "verified_at": self.verified_at,
            "source_url": self.source_url,
        }


class AirdropHunter:
    """Tracks reviewed campaigns and discovers unverified public leads."""

    def __init__(self, tracker_path: Path | None = None) -> None:
        self.tracker_path = tracker_path or _default_tracker_path()
        self.campaigns: list[AirdropCampaign] = []
        self.load_error = ""
        self._load()

    @staticmethod
    def _campaign_from_dict(row: dict) -> AirdropCampaign:
        name = _clean_text(row.get("name"), field="name", max_chars=300, allow_empty=False)
        url = _http_url(row.get("url"), field="url")
        chain = _clean_text(row.get("chain", ""), field="chain", max_chars=100)
        funding = _clean_text(row.get("funding", ""), field="funding", max_chars=500)
        status = _clean_text(row.get("status", "researching"), field="status", max_chars=32)
        notes = _clean_text(
            row.get("notes", ""),
            field="notes",
            max_chars=10_000,
            multiline=True,
        )
        estimated_value = _clean_text(
            row.get("estimated_value", ""),
            field="estimated_value",
            max_chars=200,
        )
        source_url = _http_url(row.get("source_url", ""), field="source_url", allow_empty=True)
        if status not in _CAMPAIGN_STATUSES:
            raise ValueError("invalid campaign status")

        tasks_total = row.get("tasks_total", 0)
        tasks_done = row.get("tasks_done", 0)
        if (
            isinstance(tasks_total, bool)
            or not isinstance(tasks_total, int)
            or isinstance(tasks_done, bool)
            or not isinstance(tasks_done, int)
            or not 0 <= tasks_done <= tasks_total <= 1_000_000
        ):
            raise ValueError("invalid campaign task counters")

        now = time.time()
        timestamps: list[float] = []
        for field in ("joined_at", "verified_at"):
            raw_value = row.get(field, 0.0)
            if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
                raise ValueError(f"invalid campaign {field}")
            value = float(raw_value)
            if not math.isfinite(value) or value < 0 or value > now + 300:
                raise ValueError(f"invalid campaign {field}")
            timestamps.append(value)

        zero_capital = row.get("zero_capital")
        requires_transaction = row.get("requires_transaction")
        if zero_capital is not None and not isinstance(zero_capital, bool):
            raise ValueError("invalid campaign zero_capital flag")
        if requires_transaction is not None and not isinstance(requires_transaction, bool):
            raise ValueError("invalid campaign requires_transaction flag")

        return AirdropCampaign(
            name=name,
            url=url,
            chain=chain,
            funding=funding,
            status=status,
            tasks_total=tasks_total,
            tasks_done=tasks_done,
            joined_at=timestamps[0],
            notes=notes,
            estimated_value=estimated_value,
            zero_capital=zero_capital,
            requires_transaction=requires_transaction,
            verified_at=timestamps[1],
            source_url=source_url,
        )

    def _load(self) -> None:
        if not self.tracker_path.exists():
            return
        try:
            payload = json.loads(read_bounded_text(self.tracker_path, max_bytes=_TRACKER_MAX_BYTES))
            rows = payload.get("campaigns", []) if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                raise ValueError("campaign tracker must contain a campaigns list")
            raw_version = payload.get("version", 1)
            if (
                isinstance(raw_version, bool)
                or not isinstance(raw_version, int)
                or raw_version not in {1, 2}
            ):
                raise ValueError("unsupported campaign tracker version")
            if len(rows) > _TRACKER_MAX_CAMPAIGNS:
                raise ValueError("campaign tracker exceeds its campaign limit")
            campaigns: list[AirdropCampaign] = []
            seen_urls: set[str] = set()
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("campaign tracker contains a non-object campaign")
                campaign = self._campaign_from_dict(row)
                if campaign.url in seen_urls:
                    raise ValueError("campaign tracker contains duplicate campaign URLs")
                seen_urls.add(campaign.url)
                campaigns.append(campaign)
            self.campaigns = campaigns
            self.load_error = ""
        except Exception as exc:  # noqa: BLE001
            self.campaigns = []
            self.load_error = str(exc)
            log.warning("airdrop.tracker_load_failed error=%r", exc)

    def _persist(self) -> None:
        if self.load_error:
            raise RuntimeError(f"campaign tracker is unavailable: {self.load_error}")
        if len(self.campaigns) > _TRACKER_MAX_CAMPAIGNS:
            raise ValueError("campaign tracker exceeds its campaign limit")
        canonical = [self._campaign_from_dict(campaign.to_dict()) for campaign in self.campaigns]
        urls = [campaign.url for campaign in canonical]
        if len(urls) != len(set(urls)):
            raise ValueError("campaign tracker contains duplicate campaign URLs")
        payload = {
            "version": 2,
            "campaigns": [campaign.to_dict() for campaign in canonical],
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
        if len(serialized.encode("utf-8")) > _TRACKER_MAX_BYTES:
            raise ValueError("campaign tracker exceeds its byte limit")
        atomic_write_text(
            self.tracker_path,
            serialized,
            durable=True,
            mode=0o600,
        )
        self.campaigns = canonical

    @staticmethod
    def _recently_verified(campaign: AirdropCampaign) -> bool:
        verified_at = campaign.verified_at
        if (
            isinstance(verified_at, bool)
            or not isinstance(verified_at, int | float)
            or not math.isfinite(float(verified_at))
        ):
            return False
        age = time.time() - float(verified_at)
        return 0 <= age <= _VERIFICATION_MAX_AGE_SECONDS

    def get_zero_capital_campaigns(self) -> list[dict]:
        """Return recently reviewed campaigns explicitly marked zero-capital."""
        return [
            campaign.to_dict()
            for campaign in self.campaigns
            if campaign.zero_capital is True and self._recently_verified(campaign)
        ]

    def get_actionable_campaigns(self) -> list[dict]:
        """Return manually approved, recent, non-transaction campaigns only."""
        return [
            campaign.to_dict()
            for campaign in self.campaigns
            if campaign.status == "approved"
            and campaign.zero_capital is True
            and campaign.requires_transaction is False
            and self._recently_verified(campaign)
        ]

    async def scan_for_new_campaigns(self) -> list[dict]:
        """Search for unverified leads without taking any financial action."""
        from ..dispatch.tools import t_web_search

        month = time.strftime("%B %Y")
        result = await t_web_search(
            query=f"potential zero capital testnet reward campaigns {month}",
            count=10,
        )
        items = result.get("items", []) if isinstance(result, dict) else []
        leads: list[dict] = []
        tracked_urls = {campaign.url for campaign in self.campaigns}
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                title = _clean_text(
                    item.get("title") or "",
                    field="lead title",
                    max_chars=300,
                    allow_empty=False,
                )
                url = _http_url(item.get("url") or "", field="lead url")
                snippet = _clean_text(
                    item.get("snippet") or "",
                    field="lead snippet",
                    max_chars=2_000,
                )
            except ValueError:
                continue
            if url in tracked_urls:
                continue
            text = f"{title} {snippet}".lower()
            if not any(term in text for term in ("testnet", "faucet", "zero capital", "free")):
                continue
            leads.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet[:500],
                    "status": "unverified_lead",
                    "discovered_at": time.time(),
                }
            )
        log.info("airdrop discovery: found %d unverified leads", len(leads))
        return leads[:10]

    async def participate_pharos(self) -> dict:
        """Compatibility shim: autonomous financial participation is disabled."""
        raise PermissionError(
            "autonomous wallet connection and campaign participation are disabled; "
            "discovery results require current manual verification and explicit approval"
        )

    def status_report(self) -> dict:
        """Return measured tracker state without projected campaign values."""
        return {
            "tracker_ok": not self.load_error,
            "tracker_error": self.load_error or None,
            "total_tracked": len(self.campaigns),
            "recently_verified_zero_capital": len(self.get_zero_capital_campaigns()),
            "manually_approved_actionable": len(self.get_actionable_campaigns()),
            "financial_actions_enabled": False,
            "campaigns": [campaign.to_dict() for campaign in self.campaigns],
        }
