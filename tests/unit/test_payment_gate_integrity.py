from __future__ import annotations

import asyncio
import json
import stat
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from norax.commerce import payment_gate
from norax.commerce.api_endpoints import _paid_service
from norax.commerce.payment_gate import (
    PaymentAlreadyRedeemedError,
    SessionPersistenceError,
    SessionStore,
)

VALID_WALLET = "0x1111111111111111111111111111111111111111"
VALID_SENDER = "0x2222222222222222222222222222222222222222"
VALID_TX = "0x" + "a" * 64


@pytest.fixture(autouse=True)
def isolated_secret(tmp_path, monkeypatch):
    monkeypatch.setattr(payment_gate, "_SECRET_PATH", tmp_path / "commerce-secret.key")


def test_session_credits_keep_micro_unit_precision_across_restart(tmp_path):
    state_path = tmp_path / "sessions.json"
    first = SessionStore(state_path)
    session = first.create(VALID_SENDER, "agent_task", 1_000_000, VALID_TX)
    assert first.consume(session.token, 250_000)

    restored = SessionStore(state_path)
    restored_session = restored.get(session.token)
    assert restored_session is not None
    assert restored_session.credits_remaining == 750_000
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(payment_gate._SECRET_PATH.stat().st_mode) == 0o600


def test_default_commerce_state_is_external_and_honors_state_root(tmp_path, monkeypatch):
    state_root = tmp_path / "state"
    monkeypatch.setenv("NORAX_STATE_DIR", str(state_root))
    monkeypatch.delenv("NORAX_COMMERCE_STATE_DIR", raising=False)

    store = SessionStore()

    expected = (state_root / "commerce").resolve()
    assert payment_gate.commerce_state_dir() == expected
    assert store._persist_path == expected / "sessions.json"
    assert store._lock_path == expected / ".sessions.json.lock"
    assert store._lock_path.is_file()
    assert stat.S_IMODE(expected.stat().st_mode) == 0o700


def test_default_commerce_state_rejects_relative_environment_path(monkeypatch):
    monkeypatch.setenv("NORAX_COMMERCE_STATE_DIR", "relative-commerce-state")

    with pytest.raises(ValueError, match="absolute"):
        SessionStore()


def test_redeeming_same_active_transaction_is_idempotent_not_a_refill(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    session = store.create(VALID_SENDER, "agent_task", 1_000_000, VALID_TX)
    assert store.consume(session.token, 400_000)

    same = store.create(VALID_SENDER, "agent_task", 1_000_000, VALID_TX.upper())
    assert same.token == session.token
    assert same.credits_remaining == 600_000


def test_independent_workers_cannot_spend_stale_credit_snapshot(tmp_path):
    state_path = tmp_path / "sessions.json"
    first = SessionStore(state_path)
    session = first.create(VALID_SENDER, "agent_task", 1_000_000, VALID_TX)
    second = SessionStore(state_path)

    assert first.consume(session.token, 750_000) is True
    assert second.consume(session.token, 750_000) is False

    current = second.get(session.token)
    assert current is not None
    assert current.credits_remaining == 250_000


def test_expired_or_changed_transaction_redemption_is_rejected(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    session = store.create(VALID_SENDER, "agent_task", 1_000_000, VALID_TX)
    session.expires_at = 0
    store._persist()

    restored = SessionStore(tmp_path / "sessions.json")
    with pytest.raises(PaymentAlreadyRedeemedError):
        restored.create(VALID_SENDER, "agent_task", 1_000_000, VALID_TX)


def test_credit_consumption_rejects_nonpositive_values(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    session = store.create(VALID_SENDER, "agent_task", 1_000_000, VALID_TX)
    assert not store.consume(session.token, 0)
    assert not store.consume(session.token, -100)
    assert session.credits_remaining == 1_000_000


def test_session_creation_fails_closed_when_persistence_fails(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions.json")
    monkeypatch.setattr(
        payment_gate,
        "atomic_write_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(SessionPersistenceError):
        store.create(VALID_SENDER, "agent_task", 1_000_000, VALID_TX)

    assert store._sessions == {}
    assert store._redeemed_tx_hashes == {}


def test_credit_consume_rolls_back_when_persistence_fails(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions.json")
    session = store.create(VALID_SENDER, "agent_task", 1_000_000, VALID_TX)
    monkeypatch.setattr(
        payment_gate,
        "atomic_write_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(SessionPersistenceError):
        store.consume(session.token, 250_000)

    assert session.credits_remaining == 1_000_000


def test_corrupt_session_store_fails_closed_against_replay(tmp_path):
    state_path = tmp_path / "sessions.json"
    state_path.write_text("{not-json")
    store = SessionStore(state_path)

    with pytest.raises(SessionPersistenceError):
        store.create(VALID_SENDER, "agent_task", 1_000_000, VALID_TX)


def test_disappearing_session_store_fails_closed(tmp_path):
    state_path = tmp_path / "sessions.json"
    store = SessionStore(state_path)
    session = store.create(VALID_SENDER, "agent_task", 1_000_000, VALID_TX)
    state_path.unlink()

    with pytest.raises(SessionPersistenceError, match="disappeared"):
        store.get(session.token)


def test_commerce_secret_rejects_symlink(tmp_path, monkeypatch):
    target = tmp_path / "controlled-secret"
    target.write_bytes(b"x" * 32)
    link = tmp_path / "commerce-secret.key"
    link.symlink_to(target)
    monkeypatch.setattr(payment_gate, "_SECRET_PATH", link)

    with pytest.raises(RuntimeError, match="non-symlink"):
        payment_gate._get_secret()


def test_session_restore_rejects_credit_inflation_and_forged_replay_map(tmp_path):
    state_path = tmp_path / "sessions.json"
    store = SessionStore(state_path)
    session = store.create(VALID_SENDER, "agent_task", 1_000_000, VALID_TX)

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["sessions"][session.token]["credits_remaining"] = 2_000_000
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SessionPersistenceError, match="signed token"):
        SessionStore(state_path).get(session.token)

    payload["sessions"][session.token]["credits_remaining"] = 1_000_000
    payload["redeemed_tx_hashes"][VALID_TX] = "nx_not-signed"
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SessionPersistenceError, match="invalid signed token"):
        SessionStore(state_path).get(session.token)


def test_session_restore_rejects_symlink_and_nonliteral_numeric_state(tmp_path):
    target = tmp_path / "controlled.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "sessions.json"
    link.symlink_to(target)
    with pytest.raises(SessionPersistenceError):
        SessionStore(link).active_count()

    store = SessionStore(tmp_path / "valid.json")
    session = store.create(VALID_SENDER, "agent_task", 1_000_000, VALID_TX)
    payload = json.loads((tmp_path / "valid.json").read_text(encoding="utf-8"))
    payload["sessions"][session.token]["credits_remaining"] = "1000000"
    (tmp_path / "valid.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SessionPersistenceError, match="credits"):
        SessionStore(tmp_path / "valid.json").get(session.token)


def test_refund_cannot_exceed_signed_credit_limit(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    session = store.create(VALID_SENDER, "agent_task", 1_000_000, VALID_TX)

    assert store.refund(session.token, 1) is False
    assert session.credits_remaining == 1_000_000


@pytest.mark.asyncio
async def test_paid_session_cannot_be_used_for_a_different_service(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    session = store.create(VALID_SENDER, "agent_task", 1_000_000, VALID_TX)

    async def handler(_request):
        raise AssertionError("wrong-service handler must not execute")

    with pytest.raises(HTTPException) as exc_info:
        await _paid_service("summarize", session.token, store, handler, object())

    assert exc_info.value.status_code == 403
    assert session.credits_remaining == 1_000_000


@pytest.mark.asyncio
async def test_failed_paid_service_refund_is_durable(tmp_path):
    state_path = tmp_path / "sessions.json"
    store = SessionStore(state_path)
    session = store.create(VALID_SENDER, "agent_task", 1_000_000, VALID_TX)

    async def handler(_request):
        raise RuntimeError("upstream unavailable")

    with pytest.raises(HTTPException) as exc_info:
        await _paid_service("agent_task", session.token, store, handler, object())

    assert exc_info.value.status_code == 500
    assert session.credits_remaining == 1_000_000
    restored = SessionStore(state_path).get(session.token)
    assert restored is not None
    assert restored.credits_remaining == 1_000_000


@pytest.mark.asyncio
async def test_cancelled_paid_service_refunds_durably(tmp_path):
    state_path = tmp_path / "sessions.json"
    store = SessionStore(state_path)
    session = store.create(VALID_SENDER, "agent_task", 1_000_000, VALID_TX)
    started = asyncio.Event()

    async def handler(_request):
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(_paid_service("agent_task", session.token, store, handler, object()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    restored = SessionStore(state_path).get(session.token)
    assert restored is not None
    assert restored.credits_remaining == 1_000_000


class _FakeEth:
    chain_id = 8453
    block_number = 103

    def __init__(self, *, receipt_status: int = 1, direct_eth: bool = False):
        recipient_word = "0" * 24 + VALID_WALLET[2:]
        amount_word = f"{1_000_000:064x}"
        self._tx = {
            "to": VALID_WALLET if direct_eth else payment_gate.USDC_CONTRACT,
            "from": VALID_SENDER,
            "value": 10**18,
            "input": "0xa9059cbb" + recipient_word + amount_word,
        }
        self._receipt = SimpleNamespace(status=receipt_status, blockNumber=100)

    def get_transaction(self, _tx_hash):
        return self._tx

    def get_transaction_receipt(self, _tx_hash):
        return self._receipt


class _FakeWeb3:
    def __init__(self, **kwargs):
        self.eth = _FakeEth(**kwargs)

    @staticmethod
    def is_connected():
        return True


def test_reverted_transaction_cannot_purchase_credits(monkeypatch):
    monkeypatch.setattr(payment_gate, "NORAX_EVM", VALID_WALLET)
    monkeypatch.setattr(payment_gate, "_get_web3", lambda: _FakeWeb3(receipt_status=0))

    result = payment_gate.verify_payment(VALID_TX, "agent_task")
    assert not result.valid
    assert result.error == "Transaction reverted"


def test_successful_confirmed_usdc_transfer_is_accepted(monkeypatch):
    monkeypatch.setattr(payment_gate, "NORAX_EVM", VALID_WALLET)
    monkeypatch.setattr(payment_gate, "_get_web3", _FakeWeb3)

    result = payment_gate.verify_payment(VALID_TX, "agent_task")
    assert result.valid
    assert result.amount_usdc == 1_000_000
    assert result.sender == VALID_SENDER


def test_eth_has_no_fabricated_default_exchange_rate(monkeypatch):
    monkeypatch.setattr(payment_gate, "NORAX_EVM", VALID_WALLET)
    monkeypatch.setattr(payment_gate, "_get_web3", lambda: _FakeWeb3(direct_eth=True))
    monkeypatch.delenv("NORAX_ETH_USDC_RATE", raising=False)

    result = payment_gate.verify_payment(VALID_TX, "agent_task")
    assert not result.valid
    assert "ETH payments are disabled" in (result.error or "")
