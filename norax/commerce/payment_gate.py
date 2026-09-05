"""Payment verification for Norax commerce.

Verifies that a user has paid before serving premium API requests.
Uses confirmed on-chain USDC transfers to the Norax wallet. ETH is accepted
only when the operator explicitly configures a trusted conversion rate; no
private keys are needed.

Payment flow:
1. User sends USDC (or explicitly enabled ETH) to Norax's EVM wallet on Base
2. User calls /api/pay with their tx hash + desired service
3. Norax verifies the on-chain transfer (amount, token, recipient)
4. If valid, Norax issues a signed session token (JWT-like)
5. User uses session token for API calls until credits run out

No KYC, no Stripe, no bank account. Pure on-chain verification.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import stat
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ..atomic import atomic_create_bytes, atomic_write_text, read_bounded_text

log = logging.getLogger("norax.commerce.payment")

# ── Constants ──────────────────────────────────────────────────────

NORAX_EVM = os.environ.get("NORAX_EVM_ADDRESS", "")
BASE_RPC = "https://mainnet.base.org"

# USDC on Base (native USDC, 6 decimals)
USDC_CONTRACT = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

# Minimum payments per service (in USDC units, 6 decimals)
SERVICE_PRICES = {
    "code_review": 50_0000,  # $0.50
    "research": 25_0000,  # $0.25
    "agent_task": 100_0000,  # $1.00
    "debug": 75_0000,  # $0.75
    "summarize": 15_0000,  # $0.15
}

# Session token TTL (seconds)
SESSION_TTL = 3600 * 24  # 24 hours

_SESSION_STORE_MAX_BYTES = 128 * 1024 * 1024
_SESSION_STORE_MAX_RECORDS = 250_000
_SESSION_TOKEN_MAX_CHARS = 512
_MAX_USDC_MICRO_UNITS = 1_000_000_000_000_000


def commerce_state_dir(*, create: bool = False) -> Path:
    """Return the private commerce state root without writing into source."""
    explicit = os.environ.get("NORAX_COMMERCE_STATE_DIR", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
    else:
        state_root = os.environ.get("NORAX_STATE_DIR", "").strip()
        if state_root:
            path = Path(state_root).expanduser() / "commerce"
        else:
            xdg_state = os.environ.get("XDG_STATE_HOME", "").strip()
            root = Path(xdg_state).expanduser() if xdg_state else Path.home() / ".local/state"
            path = root / "norax/commerce"
    if not path.is_absolute():
        raise ValueError("commerce state directory must be an absolute path")
    resolved = path.resolve()
    if create:
        resolved.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            resolved.chmod(0o700)
    return resolved


# Tests and explicit embedders may replace this private override. Normal
# deployments resolve the path lazily so environment isolation is honored.
_SECRET_PATH: Path | None = None


# ── Data classes ───────────────────────────────────────────────────


@dataclass
class PaymentVerification:
    """Result of verifying an on-chain payment."""

    valid: bool
    amount_usdc: int  # USDC micro-units
    service: str | None = None
    error: str | None = None
    tx_hash: str = ""
    block_number: int = 0
    sender: str = ""


@dataclass
class Session:
    """Active paid session."""

    token: str
    service: str
    credits_remaining: int  # in USDC micro-units
    created_at: float
    expires_at: float
    tx_hash: str = ""
    sender: str = ""

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def has_credits(self) -> bool:
        return self.credits_remaining > 0 and not self.expired

    def consume(self, amount: int) -> bool:
        """Consume credits for a request. Returns True if enough credits."""
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            return False
        if not self.has_credits:
            return False
        if self.credits_remaining < amount:
            return False
        self.credits_remaining -= amount
        return True

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "service": self.service,
            "credits_remaining": self.credits_remaining / 1e6,
            "expires_at": self.expires_at,
            "tx_hash": self.tx_hash,
        }


# ── Secret management ───────────────────────────────────────────────


def _get_secret() -> bytes:
    """Get or create the HMAC secret for signing session tokens."""
    p = _SECRET_PATH or (commerce_state_dir(create=True) / "session-signing.key")
    created = atomic_create_bytes(p, os.urandom(32), mode=0o600)
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(p, flags)
    except OSError as exc:
        raise RuntimeError(
            f"could not securely open regular non-symlink commerce signing secret: {p}"
        ) from exc
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise RuntimeError(
                f"commerce signing secret must be a regular, singly linked file: {p}"
            )
        if os.name == "posix" and file_stat.st_uid != os.getuid():
            raise RuntimeError(f"commerce signing secret must be owned by this user: {p}")
        try:
            os.fchmod(fd, 0o600)
        except OSError as exc:
            raise RuntimeError(
                f"could not restrict commerce signing secret permissions: {p}"
            ) from exc
        secret = os.read(fd, 33)
    finally:
        os.close(fd)
    if len(secret) != 32:
        raise RuntimeError(f"invalid commerce signing secret length at {p}")
    if created:
        log.info("commerce: generated new session signing secret")
    return secret


def _sign(payload: str) -> str:
    """Sign a payload with HMAC-SHA256."""
    secret = _get_secret()
    return hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()


def _make_token(
    sender: str,
    service: str,
    credits: int,
    tx_hash: str,
    *,
    expires: int | None = None,
) -> str:
    """Create a signed session token."""
    if not _EVM_ADDRESS_RE.fullmatch(sender):
        raise ValueError("sender must be a valid EVM address")
    if service not in SERVICE_PRICES:
        raise ValueError("service is not recognized")
    if (
        isinstance(credits, bool)
        or not isinstance(credits, int)
        or not 0 < credits <= _MAX_USDC_MICRO_UNITS
    ):
        raise ValueError("credits are outside the supported range")
    if not _TX_HASH_RE.fullmatch(tx_hash):
        raise ValueError("tx_hash must be a valid transaction hash")
    if expires is None:
        normalized_expiry = int(time.time()) + SESSION_TTL
    elif isinstance(expires, bool) or not isinstance(expires, int) or expires <= 0:
        raise ValueError("expires must be a positive integer timestamp")
    else:
        normalized_expiry = expires
    payload = f"{sender}:{service}:{credits}:{normalized_expiry}:{tx_hash}"
    sig = _sign(payload)
    return f"nx_{payload}:{sig}"


def _verify_token(
    token: str,
    *,
    allow_expired: bool = False,
    secret: bytes | None = None,
    now: float | None = None,
) -> Session | None:
    """Verify a session token and return the session if valid."""
    if (
        not isinstance(token, str)
        or not 1 <= len(token) <= _SESSION_TOKEN_MAX_CHARS
        or not token.startswith("nx_")
    ):
        return None
    try:
        body = token[3:]
        parts = body.rsplit(":", 1)
        if len(parts) != 2:
            return None
        payload, sig = parts
        if not re.fullmatch(r"[a-f0-9]{64}", sig):
            return None
        signing_secret = _get_secret() if secret is None else secret
        expected_sig = hmac.new(signing_secret, payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        # Parse payload: sender:service:credits:expires:tx_hash
        fields = payload.split(":")
        if len(fields) != 5:
            return None
        sender, service, raw_credits, raw_expires, tx_hash = fields
        if not raw_credits.isascii() or not raw_credits.isdecimal():
            return None
        if not raw_expires.isascii() or not raw_expires.isdecimal():
            return None
        credits = int(raw_credits)
        expires = int(raw_expires)
        checked_at = time.time() if now is None else now
        if (
            service not in SERVICE_PRICES
            or not 0 < credits <= _MAX_USDC_MICRO_UNITS
            or not _EVM_ADDRESS_RE.fullmatch(sender)
            or not _TX_HASH_RE.fullmatch(tx_hash)
            or expires <= 0
            or (not allow_expired and checked_at > expires)
        ):
            return None
        return Session(
            token=token,
            service=service,
            credits_remaining=credits,
            created_at=checked_at,
            expires_at=expires,
            tx_hash=tx_hash,
            sender=sender,
        )
    except Exception:
        return None


# ── On-chain verification ───────────────────────────────────────────


def _get_web3():
    """Get a Web3 instance for Base."""
    from web3 import Web3

    return Web3(Web3.HTTPProvider(BASE_RPC, request_kwargs={"timeout": 15}))


_EVM_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
_TX_HASH_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")


def _eth_usdc_rate() -> Decimal | None:
    """Return the operator-supplied USD/USDC value of one ETH, if enabled."""
    raw = os.environ.get("NORAX_ETH_USDC_RATE", "").strip()
    if not raw:
        return None
    try:
        rate = Decimal(raw)
    except InvalidOperation:
        return None
    return rate if rate.is_finite() and rate > 0 else None


def _receipt_value(receipt: object, key: str, default: Any = None) -> Any:
    if isinstance(receipt, dict):
        return receipt.get(key, default)
    return getattr(receipt, key, default)


def verify_payment(tx_hash: str, expected_service: str | None = None) -> PaymentVerification:
    """Verify an on-chain payment on Base.

    Checks that the transaction succeeded on Base, targets the configured
    wallet, contains enough value for a known service, and has at least three
    blocks of confirmation depth. Reuse is prevented by ``SessionStore``.
    """
    tx_hash = str(tx_hash or "").strip()
    if expected_service not in SERVICE_PRICES:
        return PaymentVerification(valid=False, amount_usdc=0, error="Unknown service")
    if not _EVM_ADDRESS_RE.fullmatch(NORAX_EVM):
        return PaymentVerification(
            valid=False,
            amount_usdc=0,
            error="NORAX_EVM_ADDRESS is not configured with a valid EVM address",
        )
    if not _TX_HASH_RE.fullmatch(tx_hash):
        return PaymentVerification(valid=False, amount_usdc=0, error="Invalid transaction hash")
    try:
        w3 = _get_web3()
        if not w3.is_connected():
            return PaymentVerification(
                valid=False, amount_usdc=0, error="Cannot connect to Base RPC"
            )

        if int(w3.eth.chain_id) != 8453:
            return PaymentVerification(valid=False, amount_usdc=0, error="RPC is not Base mainnet")

        tx = w3.eth.get_transaction(tx_hash)
        if tx is None:
            return PaymentVerification(valid=False, amount_usdc=0, error="Transaction not found")

        receipt = w3.eth.get_transaction_receipt(tx_hash)
        if receipt is None:
            return PaymentVerification(
                valid=False, amount_usdc=0, error="Transaction not yet mined"
            )

        if int(_receipt_value(receipt, "status", 0) or 0) != 1:
            return PaymentVerification(valid=False, amount_usdc=0, error="Transaction reverted")

        # Check confirmation depth
        current_block = w3.eth.block_number
        block_number = int(_receipt_value(receipt, "blockNumber", 0) or 0)
        if block_number <= 0 or current_block - block_number < 3:
            return PaymentVerification(
                valid=False, amount_usdc=0, error="Transaction needs 3+ confirmations"
            )

        # Check recipient
        to_addr = tx["to"].lower() if tx["to"] else ""
        norax_addr = NORAX_EVM.lower()

        # Case 1: Direct ETH transfer
        if to_addr == norax_addr:
            rate = _eth_usdc_rate()
            if rate is None:
                return PaymentVerification(
                    valid=False,
                    amount_usdc=0,
                    error="ETH payments are disabled until NORAX_ETH_USDC_RATE is configured",
                )
            eth_amount = int(tx["value"])
            usdc_equivalent = int(Decimal(eth_amount) * rate * Decimal(1_000_000) / Decimal(10**18))
            required = SERVICE_PRICES[expected_service]

            if usdc_equivalent >= required:
                return PaymentVerification(
                    valid=True,
                    amount_usdc=usdc_equivalent,
                    service=expected_service,
                    tx_hash=tx_hash,
                    block_number=block_number,
                    sender=tx["from"],
                )
            else:
                return PaymentVerification(
                    valid=False,
                    amount_usdc=usdc_equivalent,
                    error=f"Insufficient: ${usdc_equivalent / 1e6:.2f} sent, ${required / 1e6:.2f} required",
                )

        # Case 2: USDC ERC20 transfer
        if to_addr == USDC_CONTRACT.lower():
            # Decode transfer() call: 0xa9059cbb + address + amount
            raw_input = tx["input"]
            input_data = raw_input if isinstance(raw_input, str) else raw_input.hex()
            if not input_data.startswith("0x"):
                input_data = "0x" + input_data
            if not input_data.startswith("0xa9059cbb"):
                return PaymentVerification(
                    valid=False, amount_usdc=0, error="Not a transfer() call"
                )

            # Parse recipient (bytes 4-36) and amount (bytes 36-68)
            recipient = "0x" + input_data[34:74]
            amount_hex = input_data[74:138]
            if len(input_data) < 138:
                return PaymentVerification(
                    valid=False, amount_usdc=0, error="Malformed transfer() calldata"
                )
            amount = int(amount_hex, 16)

            if recipient.lower() != norax_addr:
                return PaymentVerification(
                    valid=False, amount_usdc=0, error="USDC sent to wrong address"
                )

            usdc_amount = amount  # already in micro-units (6 decimals)

            required = SERVICE_PRICES[expected_service]

            if usdc_amount >= required:
                return PaymentVerification(
                    valid=True,
                    amount_usdc=usdc_amount,
                    service=expected_service,
                    tx_hash=tx_hash,
                    block_number=block_number,
                    sender=tx["from"],
                )
            else:
                return PaymentVerification(
                    valid=False,
                    amount_usdc=usdc_amount,
                    error=f"Insufficient: ${usdc_amount / 1e6:.2f} sent, ${required / 1e6:.2f} required",
                )

        return PaymentVerification(
            valid=False,
            amount_usdc=0,
            error=f"Transaction recipient {to_addr} is not Norax wallet or USDC contract",
        )

    except Exception as e:
        log.error("payment verification failed: %s", e)
        return PaymentVerification(valid=False, amount_usdc=0, error=str(e))


# ── Session store ───────────────────────────────────────────────────


class PaymentAlreadyRedeemedError(ValueError):
    """Raised when an on-chain transaction is submitted more than once."""


class SessionPersistenceError(RuntimeError):
    """Raised when paid-session state cannot be loaded or committed safely."""


class SessionStore:
    """In-memory session store with disk persistence."""

    def __init__(self, persist_path: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}
        self._redeemed_tx_hashes: dict[str, str] = {}
        self._load_error: str | None = None
        self._persist_path = persist_path or (commerce_state_dir(create=True) / "sessions.json")
        self._lock_path = self._persist_path.with_name(f".{self._persist_path.name}.lock")
        self._loaded_signature: tuple[int, int, int, int, int] | None = None
        self._had_persisted_state = False
        with self._transaction():
            self._load(force=True)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Serialize paid-credit transactions across threads and workers."""
        with self._lock:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            flags = (
                os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            fd = os.open(self._lock_path, flags, 0o600)
            try:
                lock_stat = os.fstat(fd)
                if not stat.S_ISREG(lock_stat.st_mode):
                    raise SessionPersistenceError("paid-session lock is not a regular file")
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

    def _file_signature(self) -> tuple[int, int, int, int, int] | None:
        try:
            current = self._persist_path.lstat()
        except FileNotFoundError:
            return None
        return (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )

    @staticmethod
    def _session_from_dict(
        token: str,
        row: dict,
        *,
        version: int,
        secret: bytes,
        now: float,
    ) -> tuple[Session, Session]:
        if not isinstance(token, str) or len(token) > _SESSION_TOKEN_MAX_CHARS:
            raise ValueError("invalid paid-session token")
        signed = _verify_token(token, allow_expired=True, secret=secret, now=now)
        if signed is None:
            raise ValueError("paid-session token signature is invalid")

        stored_credits = row.get("credits_remaining")
        if version == 1:
            if isinstance(stored_credits, bool) or not isinstance(stored_credits, int | float):
                raise ValueError("invalid legacy paid-session credits")
            legacy_credits = Decimal(str(stored_credits))
            if not legacy_credits.is_finite():
                raise ValueError("invalid legacy paid-session credits")
            credits = int((legacy_credits * Decimal(1_000_000)).to_integral_value())
        else:
            if isinstance(stored_credits, bool) or not isinstance(stored_credits, int):
                raise ValueError("invalid paid-session credits")
            credits = stored_credits

        created_value = row.get("created_at")
        expires_value = row.get("expires_at")
        if (
            isinstance(created_value, bool)
            or not isinstance(created_value, int | float)
            or isinstance(expires_value, bool)
            or not isinstance(expires_value, int | float)
        ):
            raise ValueError("invalid paid-session timestamps")
        created_at = float(created_value)
        expires_at = float(expires_value)
        if (
            not math.isfinite(created_at)
            or not math.isfinite(expires_at)
            or created_at <= 0
            or not expires_at.is_integer()
            or expires_at < created_at
            or expires_at > created_at + SESSION_TTL + 1
        ):
            raise ValueError("invalid paid-session timestamps")

        service = row.get("service")
        tx_hash = row.get("tx_hash", "")
        sender = row.get("sender", "")
        if (
            not isinstance(service, str)
            or not isinstance(tx_hash, str)
            or not isinstance(sender, str)
            or service != signed.service
            or tx_hash.lower() != signed.tx_hash.lower()
            or sender.lower() != signed.sender.lower()
            or int(expires_at) != int(signed.expires_at)
            or not 0 <= credits <= signed.credits_remaining
        ):
            raise ValueError("paid-session state does not match its signed token")

        return (
            Session(
                token=token,
                service=service,
                credits_remaining=credits,
                created_at=created_at,
                expires_at=expires_at,
                tx_hash=tx_hash.lower(),
                sender=sender.lower(),
            ),
            signed,
        )

    def _load(self, *, force: bool = False) -> None:
        signature = self._file_signature()
        if not force and signature == self._loaded_signature and self._load_error is None:
            return
        if signature is None:
            if self._had_persisted_state:
                self._load_error = "paid-session state disappeared"
                return
            self._sessions = {}
            self._redeemed_tx_hashes = {}
            self._loaded_signature = None
            self._load_error = None
            return
        try:
            data = json.loads(
                read_bounded_text(self._persist_path, max_bytes=_SESSION_STORE_MAX_BYTES)
            )
            if not isinstance(data, dict):
                raise ValueError("commerce session store must be an object")
            raw_version = data.get("version")
            if raw_version is None:
                version = 1
            elif isinstance(raw_version, bool) or not isinstance(raw_version, int):
                raise ValueError("invalid commerce session store version")
            else:
                version = raw_version
            if version not in {1, 2}:
                raise ValueError("unsupported commerce session store version")
            stored_sessions = data.get("sessions", {}) if version == 2 else data
            redeemed = data.get("redeemed_tx_hashes", {}) if version == 2 else {}
            if not isinstance(stored_sessions, dict) or not isinstance(redeemed, dict):
                raise ValueError("invalid commerce session store")
            if (
                len(stored_sessions) > _SESSION_STORE_MAX_RECORDS
                or len(redeemed) > _SESSION_STORE_MAX_RECORDS
            ):
                raise ValueError("commerce session store exceeds its record limit")

            now = time.time()
            secret = _get_secret() if stored_sessions or redeemed else b""
            verified_tokens: dict[str, Session] = {}
            loaded_redeemed: dict[str, str] = {}
            for tx_hash, token in redeemed.items():
                if (
                    not isinstance(tx_hash, str)
                    or not isinstance(token, str)
                    or not _TX_HASH_RE.fullmatch(tx_hash)
                ):
                    raise ValueError("invalid redeemed transaction record")
                signed = verified_tokens.get(token)
                if signed is None:
                    signed = _verify_token(
                        token,
                        allow_expired=True,
                        secret=secret,
                        now=now,
                    )
                    if signed is None:
                        raise ValueError("redeemed transaction has an invalid signed token")
                    verified_tokens[token] = signed
                normalized_tx = tx_hash.lower()
                if signed.tx_hash.lower() != normalized_tx:
                    raise ValueError("redeemed transaction does not match its signed token")
                loaded_redeemed[normalized_tx] = token

            loaded_sessions: dict[str, Session] = {}
            for token, s in stored_sessions.items():
                if not isinstance(s, dict):
                    raise ValueError("commerce session store contains a non-object session")
                sess, signed = self._session_from_dict(
                    token,
                    s,
                    version=version,
                    secret=secret,
                    now=now,
                )
                if now <= sess.expires_at:
                    loaded_sessions[token] = sess
                normalized_tx = signed.tx_hash.lower()
                existing_token = loaded_redeemed.setdefault(normalized_tx, token)
                if existing_token != token:
                    raise ValueError("transaction is associated with conflicting session tokens")
            self._sessions = loaded_sessions
            self._redeemed_tx_hashes = loaded_redeemed
            self._loaded_signature = signature
            self._had_persisted_state = True
            self._load_error = None
        except Exception as e:
            self._sessions.clear()
            self._redeemed_tx_hashes.clear()
            self._loaded_signature = signature
            self._load_error = str(e)
            log.warning(
                "payment_gate.session_restore_failed path=%s error=%r", self._persist_path, e
            )

    def _persist(self) -> bool:
        with self._transaction():
            self._load()
            self._require_healthy()
            return self._persist_unlocked()

    def _persist_unlocked(self) -> bool:
        try:
            if (
                len(self._sessions) > _SESSION_STORE_MAX_RECORDS
                or len(self._redeemed_tx_hashes) > _SESSION_STORE_MAX_RECORDS
            ):
                raise ValueError("commerce session store exceeds its record limit")
            sessions: dict[str, dict] = {}
            for token, s in self._sessions.items():
                if not s.expired:
                    sessions[token] = {
                        "service": s.service,
                        "credits_remaining": s.credits_remaining,
                        "created_at": s.created_at,
                        "expires_at": s.expires_at,
                        "tx_hash": s.tx_hash,
                        "sender": s.sender,
                    }
            data = {
                "version": 2,
                "sessions": sessions,
                "redeemed_tx_hashes": self._redeemed_tx_hashes,
            }
            serialized = json.dumps(
                data,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            if len(serialized.encode("utf-8")) > _SESSION_STORE_MAX_BYTES:
                raise ValueError("commerce session store exceeds its byte limit")
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                self._persist_path,
                serialized,
                durable=True,
                mode=0o600,
            )
            self._loaded_signature = self._file_signature()
            self._had_persisted_state = True
            return True
        except Exception as e:
            log.warning("session persist failed: %s", e)
            return False

    def _require_healthy(self) -> None:
        if self._load_error is not None:
            raise SessionPersistenceError(
                f"session store could not be restored safely: {self._load_error}"
            )

    def create(self, sender: str, service: str, credits: int, tx_hash: str) -> Session:
        """Create a new session after payment verification."""
        if service not in SERVICE_PRICES:
            raise ValueError(f"unknown service: {service}")
        if (
            isinstance(credits, bool)
            or not isinstance(credits, int)
            or not 0 < credits <= _MAX_USDC_MICRO_UNITS
        ):
            raise ValueError("credits must be a positive integer number of micro-units")
        if not isinstance(sender, str) or not isinstance(tx_hash, str):
            raise ValueError("sender and tx_hash must be strings")
        normalized_sender = sender.lower()
        normalized_tx = tx_hash.lower()
        if not _EVM_ADDRESS_RE.fullmatch(normalized_sender):
            raise ValueError("sender must be a valid EVM address")
        if not _TX_HASH_RE.fullmatch(normalized_tx):
            raise ValueError("tx_hash must be a valid transaction hash")
        with self._transaction():
            self._load()
            self._require_healthy()
            redeemed_token = self._redeemed_tx_hashes.get(normalized_tx)
            if redeemed_token:
                existing = self._sessions.get(redeemed_token)
                if (
                    existing is not None
                    and not existing.expired
                    and existing.sender.lower() == normalized_sender
                    and existing.service == service
                ):
                    return existing
                raise PaymentAlreadyRedeemedError("transaction has already been redeemed")
            now = time.time()
            expires = int(now) + SESSION_TTL
            token = _make_token(
                normalized_sender,
                service,
                credits,
                normalized_tx,
                expires=expires,
            )
            sess = Session(
                token=token,
                service=service,
                credits_remaining=credits,
                created_at=now,
                expires_at=float(expires),
                tx_hash=normalized_tx,
                sender=normalized_sender,
            )
            self._sessions[token] = sess
            self._redeemed_tx_hashes[normalized_tx] = token
            if not self._persist_unlocked():
                self._sessions.pop(token, None)
                if self._redeemed_tx_hashes.get(normalized_tx) == token:
                    self._redeemed_tx_hashes.pop(normalized_tx, None)
                raise SessionPersistenceError("could not durably create paid session")
            return sess

    def get(self, token: str) -> Session | None:
        """Get a session by token. Returns None if invalid/expired."""
        if not isinstance(token, str) or len(token) > _SESSION_TOKEN_MAX_CHARS:
            return None
        with self._transaction():
            self._load()
            self._require_healthy()
            sess = self._sessions.get(token)
            if sess is not None and sess.expired:
                del self._sessions[token]
                if not self._persist_unlocked():
                    raise SessionPersistenceError("could not durably expire paid session")
                return None
            return sess

    def consume(self, token: str, amount: int) -> bool:
        """Consume credits from a session."""
        if not isinstance(token, str) or len(token) > _SESSION_TOKEN_MAX_CHARS:
            return False
        with self._transaction():
            self._load()
            self._require_healthy()
            sess = self._sessions.get(token)
            if sess is not None and sess.expired:
                del self._sessions[token]
                if not self._persist_unlocked():
                    raise SessionPersistenceError("could not durably expire paid session")
                return False
            if sess is None:
                return False
            before = sess.credits_remaining
            if not sess.consume(amount):
                return False
            if not self._persist_unlocked():
                sess.credits_remaining = before
                raise SessionPersistenceError("could not durably consume session credits")
            return True

    def refund(self, token: str, amount: int) -> bool:
        """Refund positive micro-unit credits and persist before reporting success."""
        if (
            not isinstance(token, str)
            or len(token) > _SESSION_TOKEN_MAX_CHARS
            or isinstance(amount, bool)
            or not isinstance(amount, int)
            or amount <= 0
        ):
            return False
        with self._transaction():
            self._load()
            self._require_healthy()
            sess = self._sessions.get(token)
            if sess is not None and sess.expired:
                del self._sessions[token]
                if not self._persist_unlocked():
                    raise SessionPersistenceError("could not durably expire paid session")
                return False
            if sess is None:
                return False
            before = sess.credits_remaining
            signed = _verify_token(token)
            if signed is None or before + amount > signed.credits_remaining:
                return False
            sess.credits_remaining += amount
            if not self._persist_unlocked():
                sess.credits_remaining = before
                raise SessionPersistenceError("could not durably refund session credits")
            return True

    def active_count(self) -> int:
        with self._transaction():
            self._load()
            self._require_healthy()
            return sum(1 for s in self._sessions.values() if not s.expired)


# ── Convenience ─────────────────────────────────────────────────────

# Singleton session store
_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store


def get_wallet_info() -> dict:
    """Get Norax wallet info for display on payment page."""
    return {
        "evm_address": NORAX_EVM,
        "network": "Base",
        "usdc_contract": USDC_CONTRACT,
        "accepted_tokens": ["USDC"] + (["ETH"] if _eth_usdc_rate() is not None else []),
        "services": {k: v / 1e6 for k, v in SERVICE_PRICES.items()},
        "chain_id": 8453,
    }
