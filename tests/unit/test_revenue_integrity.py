from __future__ import annotations

import json
import stat
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from norax.commerce.airdrop_hunter import (
    AirdropCampaign,
    AirdropHunter,
    _default_tracker_path,
)
from norax.commerce.revenue_engine import (
    RevenueEngine,
    RevenuePersistenceError,
    _default_revenue_log,
)

TX_1 = "0x" + "1" * 64
TX_2 = "0x" + "2" * 64


def _engine(tmp_path):
    hunter = AirdropHunter(tmp_path / "airdrop.json")
    return RevenueEngine(tmp_path / "revenue.json", airdrop=hunter)


def test_default_commerce_ledgers_share_external_state_root(tmp_path, monkeypatch):
    state_root = tmp_path / "state"
    monkeypatch.setenv("NORAX_STATE_DIR", str(state_root))
    monkeypatch.delenv("NORAX_COMMERCE_STATE_DIR", raising=False)
    monkeypatch.delenv("NORAX_REVENUE_LOG", raising=False)
    monkeypatch.delenv("NORAX_AIRDROP_TRACKER", raising=False)

    commerce_root = (state_root / "commerce").resolve()
    assert _default_revenue_log() == commerce_root / "revenue_log.json"
    assert _default_tracker_path() == commerce_root / "airdrop_tracker.json"


def test_revenue_requires_external_evidence_and_counts_only_verified_rows(tmp_path):
    engine = _engine(tmp_path)

    with pytest.raises(ValueError, match="evidence"):
        engine.record_earning("api_payment", 1.0, "payment")

    entry = engine.record_earning("api_payment", 1.0, "payment", TX_1)
    assert entry.verified is True
    assert engine.total_earned == 1.0
    assert stat.S_IMODE((tmp_path / "revenue.json").stat().st_mode) == 0o600


def test_revenue_receipts_are_cross_worker_idempotent(tmp_path):
    first = _engine(tmp_path)
    second = _engine(tmp_path)

    def record(engine):
        return engine.record_earning("api_payment", 2.5, "service", TX_1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(record, (first, second)))

    restored = _engine(tmp_path)
    assert len(results) == 2
    assert len(restored.entries) == 1
    assert restored.total_earned == 2.5


def test_corrupt_or_disappearing_revenue_ledger_fails_closed(tmp_path):
    engine = _engine(tmp_path)
    engine.record_earning("api_payment", 1.0, "first", TX_1)
    (tmp_path / "revenue.json").unlink()

    with pytest.raises(RevenuePersistenceError):
        engine.record_earning("api_payment", 1.0, "second", TX_2)

    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("not-json", encoding="utf-8")
    corrupt = RevenueEngine(corrupt_path, airdrop=AirdropHunter(tmp_path / "tracker.json"))
    with pytest.raises(RevenuePersistenceError):
        corrupt.record_earning("api_payment", 1.0, "payment", TX_1)
    assert corrupt_path.read_text(encoding="utf-8") == "not-json"


def test_legacy_rows_without_receipts_are_reported_but_not_counted_as_earned(tmp_path):
    path = tmp_path / "revenue.json"
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "timestamp": time.time(),
                        "source": "content",
                        "amount_usd": 12.5,
                        "description": "legacy manual estimate",
                        "tx_hash": "",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    engine = RevenueEngine(path, airdrop=AirdropHunter(tmp_path / "tracker.json"))
    assert engine.total_earned == 0
    assert engine.unverified_total == 12.5


def test_airdrop_tracker_has_no_hard_coded_available_campaigns(tmp_path):
    hunter = AirdropHunter(tmp_path / "tracker.json")

    assert hunter.get_zero_capital_campaigns() == []
    assert hunter.get_actionable_campaigns() == []
    assert hunter.status_report()["financial_actions_enabled"] is False


def test_only_recent_manually_approved_non_transaction_campaign_is_actionable(tmp_path):
    hunter = AirdropHunter(tmp_path / "tracker.json")
    hunter.campaigns = [
        AirdropCampaign(
            name="reviewed",
            url="https://example.test/campaign",
            status="approved",
            zero_capital=True,
            requires_transaction=False,
            verified_at=time.time(),
        ),
        AirdropCampaign(
            name="transactional",
            url="https://example.test/tx",
            status="approved",
            zero_capital=True,
            requires_transaction=True,
            verified_at=time.time(),
        ),
    ]

    assert [item["name"] for item in hunter.get_zero_capital_campaigns()] == [
        "reviewed",
        "transactional",
    ]
    assert [item["name"] for item in hunter.get_actionable_campaigns()] == ["reviewed"]


@pytest.mark.asyncio
async def test_financial_participation_compatibility_method_is_fail_closed(tmp_path):
    hunter = AirdropHunter(tmp_path / "tracker.json")

    with pytest.raises(PermissionError, match="explicit approval"):
        await hunter.participate_pharos()


def test_revenue_ledger_rejects_symlink_and_coerced_numbers(tmp_path):
    target = tmp_path / "controlled.json"
    target.write_text(json.dumps({"entries": []}), encoding="utf-8")
    link = tmp_path / "revenue.json"
    link.symlink_to(target)
    linked = RevenueEngine(link, airdrop=AirdropHunter(tmp_path / "tracker.json"))
    with pytest.raises(RevenuePersistenceError):
        linked.record_earning("api_payment", 1.0, "payment", TX_1)

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text(
        json.dumps(
            {
                "version": 2,
                "entries": [
                    {
                        "timestamp": time.time(),
                        "source": "api_payment",
                        "amount_usd": "1.0",
                        "description": "payment",
                        "tx_hash": TX_1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    malformed = RevenueEngine(
        malformed_path,
        airdrop=AirdropHunter(tmp_path / "tracker-2.json"),
    )
    assert malformed.status()["ledger_ok"] is False


def test_airdrop_tracker_rejects_truthy_flags_and_symlinks(tmp_path):
    malformed_path = tmp_path / "malformed-airdrop.json"
    malformed_path.write_text(
        json.dumps(
            {
                "version": 2,
                "campaigns": [
                    {
                        "name": "untrusted",
                        "url": "https://example.test/campaign",
                        "zero_capital": "false",
                        "requires_transaction": "false",
                        "verified_at": time.time(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    malformed = AirdropHunter(malformed_path)
    assert malformed.load_error
    assert malformed.get_actionable_campaigns() == []

    target = tmp_path / "controlled-airdrop.json"
    target.write_text(json.dumps({"campaigns": []}), encoding="utf-8")
    link = tmp_path / "airdrop-link.json"
    link.symlink_to(target)
    linked = AirdropHunter(link)
    assert linked.load_error


def test_future_campaign_verification_is_not_recent(tmp_path):
    hunter = AirdropHunter(tmp_path / "tracker.json")
    hunter.campaigns = [
        AirdropCampaign(
            name="future",
            url="https://example.test/future",
            status="approved",
            zero_capital=True,
            requires_transaction=False,
            verified_at=time.time() + 60,
        )
    ]

    assert hunter.get_zero_capital_campaigns() == []
    assert hunter.get_actionable_campaigns() == []
