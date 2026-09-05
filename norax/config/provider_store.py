"""Private runtime storage for user-configured model providers."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..atomic import atomic_write_text

log = logging.getLogger("norax.config.provider_store")

_PROVIDER_NAME = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")
_ALLOWED_KINDS = {"openai", "openrouter", "ollama"}
_MAX_STORE_BYTES = 2 * 1024 * 1024
_MAX_PROVIDERS = 256


def _provider_bool(value: Any, *, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return default
    raise ValueError("enabled must be a boolean")


def validate_provider_api_key(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 65_536:
        raise ValueError("api_key must be a string of at most 65536 characters")
    if value and (not value.isascii() or any(not char.isprintable() for char in value)):
        raise ValueError("api_key must contain printable ASCII characters only")
    return value


def provider_config_dir() -> Path:
    configured = os.environ.get("NORAX_CONFIG_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    xdg = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return (xdg / "norax").resolve()


def validate_provider_spec(spec: dict[str, Any]) -> dict[str, Any]:
    name = str(spec.get("name") or "").strip().lower()
    if not _PROVIDER_NAME.fullmatch(name):
        raise ValueError("provider name must match ^[a-z][a-z0-9_-]{1,31}$")
    base_url = str(spec.get("base_url") or "").strip().rstrip("/")
    if len(base_url) > 2_048:
        raise ValueError("base_url must not exceed 2048 characters")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain credentials, a query, or a fragment")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("base_url contains an invalid port") from exc
    if any(not character.isprintable() or character.isspace() for character in base_url):
        raise ValueError("base_url must not contain whitespace or control characters")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        allow_http = os.environ.get("NORAX_ALLOW_INSECURE_PROVIDER_HTTP", "").lower()
        if allow_http not in {"1", "true", "yes", "on"}:
            raise ValueError("non-loopback providers must use HTTPS")
    kind = str(spec.get("provider_kind") or "openai").strip().lower()
    if kind not in _ALLOWED_KINDS:
        raise ValueError(f"provider_kind must be one of {sorted(_ALLOWED_KINDS)}")
    raw_models = spec.get("models") or []
    if not isinstance(raw_models, (list, tuple)):
        raise ValueError("models must be a list")
    if len(raw_models) > 100:
        raise ValueError("models must contain at most 100 entries")
    models: list[str] = []
    for item in raw_models:
        if not isinstance(item, str):
            raise ValueError("each model id must be a string")
        model = item.strip()
        if len(model) > 512:
            raise ValueError("model ids must not exceed 512 characters")
        if model and not all(
            character.isprintable() and not character.isspace() for character in model
        ):
            raise ValueError("model ids must not contain whitespace or control characters")
        if model and model not in models:
            models.append(model)
    return {
        "name": name,
        "base_url": base_url,
        "provider_kind": kind,
        "models": models,
        "enabled": _provider_bool(spec.get("enabled", True)),
    }


class ProviderStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or provider_config_dir()).expanduser().resolve()
        self.providers_path = self.root / "providers.json"
        self.secrets_path = self.root / "provider-secrets.json"
        self.settings_path = self.root / "runtime-settings.json"
        self.lock_path = self.root / ".provider-store.lock"
        self._lock = threading.RLock()

    @contextmanager
    def _store_lock(self, *, exclusive: bool) -> Iterator[None]:
        """Serialize complete multi-file snapshots across threads/processes."""
        import fcntl

        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.root, 0o700)
            flags = (
                os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            fd = os.open(self.lock_path, flags, 0o600)
            try:
                os.fchmod(fd, 0o600)
                fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
                try:
                    yield
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def runtime_settings(self) -> dict[str, Any]:
        with self._store_lock(exclusive=False):
            settings = self._load(self.settings_path).get("settings") or {}
        if not isinstance(settings, dict):
            log.warning("provider_store.invalid_settings path=%s", self.settings_path)
            return {}
        if len(settings) > 256:
            log.warning("provider_store.too_many_settings path=%s", self.settings_path)
            return {}
        return dict(settings)

    def set_runtime_setting(self, name: str, value: Any) -> None:
        if not isinstance(name, str) or not name.strip() or len(name) > 128:
            raise ValueError("runtime setting name must contain 1-128 characters")
        with self._store_lock(exclusive=True):
            document = self._load(self.settings_path, strict=True)
            raw_settings = document.get("settings") or {}
            if not isinstance(raw_settings, dict):
                raise ValueError("runtime settings file contains an invalid settings object")
            settings = dict(raw_settings)
            if name not in settings and len(settings) >= 256:
                raise ValueError("runtime settings may contain at most 256 entries")
            settings[name] = value
            self._write(self.settings_path, {"version": 1, "settings": settings})

    def list_public(self) -> list[dict[str, Any]]:
        with self._store_lock(exclusive=False):
            return self._list_public_unlocked()

    def _list_public_unlocked(self) -> list[dict[str, Any]]:
        providers = self._load(self.providers_path).get("providers") or {}
        if not isinstance(providers, dict):
            log.warning("provider_store.invalid_providers path=%s", self.providers_path)
            return []
        if len(providers) > _MAX_PROVIDERS:
            log.warning("provider_store.too_many_providers path=%s", self.providers_path)
            return []
        rows: list[dict[str, Any]] = []
        for name, value in sorted(providers.items()):
            if not isinstance(value, dict):
                log.warning("provider_store.invalid_provider name=%s", name)
                continue
            try:
                rows.append(validate_provider_spec({**value, "name": name}))
            except ValueError as exc:
                log.warning("provider_store.invalid_provider name=%s error=%s", name, exc)
        return rows

    def list_runtime(self) -> list[dict[str, Any]]:
        with self._store_lock(exclusive=False):
            secrets = self._load(self.secrets_path).get("providers") or {}
            if not isinstance(secrets, dict):
                log.warning("provider_store.invalid_secrets path=%s", self.secrets_path)
                secrets = {}
            rows = []
            for public in self._list_public_unlocked():
                row = dict(public)
                secret = secrets.get(row["name"]) or {}
                if not isinstance(secret, dict):
                    secret = {}
                try:
                    row["api_key"] = validate_provider_api_key(secret.get("api_key") or "")
                except ValueError as exc:
                    log.warning("provider_store.invalid_api_key name=%s error=%s", row["name"], exc)
                    row["api_key"] = ""
                rows.append(row)
            return rows

    def upsert(self, spec: dict[str, Any], api_key: str | None = None) -> dict[str, Any]:
        clean = validate_provider_spec(spec)
        if api_key is not None:
            api_key = validate_provider_api_key(api_key)
        with self._store_lock(exclusive=True):
            providers_doc = self._load(self.providers_path, strict=True)
            raw_providers = providers_doc.get("providers") or {}
            if not isinstance(raw_providers, dict):
                raise ValueError("provider metadata file contains an invalid providers object")
            providers = dict(raw_providers)
            if clean["name"] not in providers and len(providers) >= _MAX_PROVIDERS:
                raise ValueError(f"provider store may contain at most {_MAX_PROVIDERS} providers")
            provider_secrets: dict[str, Any] | None = None
            secrets_doc: dict[str, Any] | None = None
            if api_key is not None:
                secrets_doc = self._load(self.secrets_path, strict=True)
                raw_secrets = secrets_doc.get("providers") or {}
                if not isinstance(raw_secrets, dict):
                    raise ValueError("provider secrets file contains an invalid providers object")
                provider_secrets = dict(raw_secrets)
            providers[clean["name"]] = {key: value for key, value in clean.items() if key != "name"}
            if provider_secrets is not None:
                if api_key:
                    provider_secrets[clean["name"]] = {"api_key": api_key}
                else:
                    provider_secrets.pop(clean["name"], None)
                # Commit credentials before publishing provider metadata.  A
                # crash can then leave only an unreachable orphan secret, not
                # a visible provider with missing credentials.  Roll back the
                # credential document if the metadata commit itself fails.
                self._write(self.secrets_path, {"version": 1, "providers": provider_secrets})
                try:
                    self._write(self.providers_path, {"version": 1, "providers": providers})
                except Exception:
                    assert secrets_doc is not None
                    try:
                        self._write(self.secrets_path, secrets_doc)
                    except Exception:  # noqa: BLE001
                        log.critical(
                            "provider_store.secret_rollback_failed provider=%s",
                            clean["name"],
                            exc_info=True,
                        )
                    raise
            else:
                self._write(self.providers_path, {"version": 1, "providers": providers})
        return clean

    def remove(self, name: str) -> bool:
        clean_name = str(name).strip().lower()
        if not _PROVIDER_NAME.fullmatch(clean_name):
            return False
        with self._store_lock(exclusive=True):
            providers_doc = self._load(self.providers_path, strict=True)
            secrets_doc = self._load(self.secrets_path, strict=True)
            raw_providers = providers_doc.get("providers") or {}
            raw_secrets = secrets_doc.get("providers") or {}
            if not isinstance(raw_providers, dict) or not isinstance(raw_secrets, dict):
                raise ValueError("provider store contains an invalid providers object")
            providers = dict(raw_providers)
            provider_secrets = dict(raw_secrets)
            existed = providers.pop(clean_name, None) is not None
            provider_secrets.pop(clean_name, None)
            if existed:
                original_providers_doc = providers_doc
                # Remove public metadata first.  If the process dies between
                # the two atomic replaces, a secret may be orphaned but can no
                # longer make a provider live or visible.
                self._write(self.providers_path, {"version": 1, "providers": providers})
                try:
                    self._write(
                        self.secrets_path,
                        {"version": 1, "providers": provider_secrets},
                    )
                except Exception:
                    try:
                        self._write(self.providers_path, original_providers_doc)
                    except Exception:  # noqa: BLE001
                        log.critical(
                            "provider_store.metadata_rollback_failed provider=%s",
                            clean_name,
                            exc_info=True,
                        )
                    raise
        return existed

    @staticmethod
    def _load(path: Path, *, strict: bool = False) -> dict[str, Any]:
        try:
            # Do not follow a replaced state-file symlink, and never allocate
            # an unbounded JSON document supplied through local state.
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            with os.fdopen(fd, "rb") as handle:
                raw = handle.read(_MAX_STORE_BYTES + 1)
            if len(raw) > _MAX_STORE_BYTES:
                raise ValueError(f"provider store file exceeds {_MAX_STORE_BYTES} bytes")
            value = json.loads(raw.decode("utf-8"))
        except FileNotFoundError:
            return {"version": 1, "providers": {}}
        except (UnicodeError, ValueError, OSError) as exc:
            if strict:
                raise ValueError(
                    f"cannot safely update invalid provider store file: {path}"
                ) from exc
            log.warning("provider_store.load_failed path=%s error=%s", path, exc)
            return {"version": 1, "providers": {}}
        if isinstance(value, dict):
            return value
        if strict:
            raise ValueError(f"cannot safely update invalid provider store file: {path}")
        log.warning("provider_store.invalid_document path=%s", path)
        return {"version": 1, "providers": {}}

    def _write(self, path: Path, value: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        atomic_write_text(
            path,
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            durable=True,
            mode=0o600,
        )
