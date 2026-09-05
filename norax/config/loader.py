"""JSONC config loader (JSON + // line comments + /* block */ + trailing commas).

Implemented against stdlib only — no `json5` / `pyyaml` dependency.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

log = logging.getLogger("norax.config")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_CONFIG = PROJECT_ROOT / "config" / "runtime.jsonc"
_PACKAGED_DEFAULTS = Path(__file__).resolve().parents[1] / "defaults"
DEFAULT_CONFIG = _SOURCE_CONFIG if _SOURCE_CONFIG.exists() else _PACKAGED_DEFAULTS / "runtime.jsonc"
DEFAULT_PROJECT_ROOT = PROJECT_ROOT if _SOURCE_CONFIG.exists() else _PACKAGED_DEFAULTS
_MAX_CONFIG_BYTES = 2 * 1024 * 1024
_MAX_CRON_JOBS = 256
_MAX_DISCORD_GUILDS = 256
_MAX_DISCORD_CHANNELS_PER_GUILD = 1_000
_MAX_DISCORD_USERS_PER_GUILD = 10_000
_MAX_DISCORD_ALLOW_FROM = 10_000


def _env_config_path() -> Path | None:
    p = os.environ.get("NORAX_CONFIG")
    return Path(p).resolve() if p else None


def _env_project_root() -> Path | None:
    p = os.environ.get("NORAX_PROJECT_ROOT")
    return Path(p).resolve() if p else None


# Preprocessor: strip // ... EOL and /* ... */ and trailing commas.
# Regex is comment-aware so that // and /* inside strings are preserved.
_STRIP_RX = re.compile(
    r"""
    (?P<str>  "(?:\\.|[^"\\])*" )          |
    (?P<line> //[^\n]* )                    |
    (?P<block> /\*.*?\*/ )
    """,
    re.VERBOSE | re.DOTALL,
)
_TRAILING_COMMA_RX = re.compile(r",(\s*[}\]])")
_GATEWAY_NAME_RX = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_HEADER_NAME_RX = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_GATEWAY_PROVIDER_KEYS = {
    "api_key",
    "base_url",
    "chat_path",
    "extra_headers",
    "model_prefix",
    "provider_kind",
    "stream_required",
}
_GATEWAY_KEYS = {
    *_GATEWAY_PROVIDER_KEYS,
    "default_model",
    "default_provider",
    "failover_models",
    "providers",
    "routes",
    "timeout_seconds",
}
_COGNITION_KEYS = {"experimental_signals", "idle_learning", "harness_analysis"}
_VISION_KEYS = {"enabled", "model"}
_TOP_LEVEL_KEYS = {
    "runtime",
    "http",
    "gateway",
    "channels",
    "cognition",
    "cron",
    "owner",
    "agent_os",
    "paths",
    "connectors",
    "mcp",
    "a2a",
    "commerce",
    "vision",
}
_SECTION_KEYS: dict[str, set[str]] = {
    "runtime": {
        "shutdown_grace_seconds",
        "max_tool_rounds",
        "max_concurrent_turns",
        "max_pending_turns",
        "thinking_effort",
        "reasoning_output",
        "planning_mode",
        "memory_depth",
        "weak_model_boost",
        "stream_replies",
        "response_length",
        "tool_activity",
    },
    "http": {"bind"},
    "channels": {"discord"},
    "owner": {"id", "label"},
    "agent_os": {"chat_token"},
    "paths": {"state_dir", "log_dir", "soul", "identity", "user", "memory_root"},
    "cron": {"enabled", "resolution_sec", "jobs"},
}
_CONNECTOR_KEYS: dict[str, set[str]] = {
    "planner": {"enabled"},
    "multi_agent": {"enabled", "max_concurrent"},
    "mcp_client": {"enabled", "required", "servers"},
    "a2a_server": {
        "enabled",
        "required",
        "host",
        "port",
        "auth_token",
        "max_tasks",
        "max_concurrent",
        "base_url",
    },
    "commerce": {"enabled", "required"},
    "active_inference": {"enabled"},
}
_DISCORD_KEYS = {
    "enabled",
    "token",
    "allowBots",
    "dmPolicy",
    "groupPolicy",
    "allowFrom",
    "guilds",
}
_DISCORD_GUILD_KEYS = {"requireMention", "users", "channels"}
_DISCORD_CHANNEL_KEYS = {"requireMention", "enabled"}
_CRON_JOB_KEYS = {"name", "schedule", "kind", "jitter_seconds", "enabled"}


def _bounded_config_text(value: Any, *, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    text = value.strip()
    if not text or len(text) > maximum or any(not char.isprintable() for char in text):
        raise ValueError(f"{label} must contain 1-{maximum} printable characters")
    return text


def _strip_jsonc(src: str) -> str:
    def repl(m: re.Match) -> str:
        if m.group("str") is not None:
            return m.group("str")
        return ""  # drop comments

    no_comments = _STRIP_RX.sub(repl, src)
    return _TRAILING_COMMA_RX.sub(r"\1", no_comments)


def load_jsonc(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(_MAX_CONFIG_BYTES + 1)
    if len(raw) > _MAX_CONFIG_BYTES:
        raise ValueError(f"configuration exceeds {_MAX_CONFIG_BYTES} bytes: {path}")
    src = raw.decode("utf-8")
    value = json.loads(_strip_jsonc(src))
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a JSON object: {path}")
    return value


def _config_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    label: str,
    maximum: int | None = None,
) -> int:
    try:
        if isinstance(value, bool):
            raise ValueError
        parsed = int(value)
        if parsed < minimum or (maximum is not None and parsed > maximum):
            raise ValueError
        return parsed
    except (TypeError, ValueError, OverflowError):
        log.warning("config.invalid_integer key=%s value=%r default=%d", label, value, default)
        return default


def _config_float(
    value: Any,
    *,
    default: float,
    minimum: float,
    label: str,
    maximum: float | None = None,
) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError
        parsed = float(value)
        if (
            not math.isfinite(parsed)
            or parsed < minimum
            or (maximum is not None and parsed > maximum)
        ):
            raise ValueError
        return parsed
    except (TypeError, ValueError, OverflowError):
        log.warning("config.invalid_number key=%s value=%r default=%s", label, value, default)
        return default


def _config_bool(value: Any, *, default: bool, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    log.warning("config.invalid_boolean key=%s value=%r default=%s", label, value, default)
    return default


def _config_choice(
    value: Any,
    *,
    default: str,
    choices: set[str],
    label: str,
) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in choices:
            return normalized
    log.warning("config.invalid_choice key=%s value=%r default=%s", label, value, default)
    return default


def _gateway_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    normalized = value.strip().lower()
    if not _GATEWAY_NAME_RX.fullmatch(normalized):
        raise ValueError(f"{label} must match {_GATEWAY_NAME_RX.pattern}")
    return normalized


def _model_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    model = value.strip()
    if not model or len(model) > 512:
        raise ValueError(f"{label} must contain 1-512 characters")
    if any(not character.isprintable() or character.isspace() for character in model):
        raise ValueError(f"{label} must not contain whitespace or control characters")
    return model


def _gateway_url(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    url = value.strip().rstrip("/")
    if (
        not url
        or len(url) > 2_048
        or any(not character.isprintable() or character.isspace() for character in url)
    ):
        raise ValueError(f"{label} must be a valid HTTP(S) URL of at most 2048 characters")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{label} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{label} must not contain credentials, a query, or a fragment")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} contains an invalid port") from exc
    return url


def _gateway_headers(value: Any, *, label: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if len(value) > 64:
        raise ValueError(f"{label} must contain at most 64 headers")
    headers: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        if not isinstance(raw_name, str) or not _HEADER_NAME_RX.fullmatch(raw_name):
            raise ValueError(f"{label} contains an invalid header name")
        if (
            not isinstance(raw_value, str)
            or len(raw_value) > 8_192
            or not raw_value.isascii()
            or any(not char.isprintable() and char != "\t" for char in raw_value)
        ):
            raise ValueError(f"{label}.{raw_name} must contain only printable ASCII")
        headers[raw_name] = raw_value
    return headers


def _gateway_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{label} must be a boolean")


def _gateway_provider_config(
    value: Any,
    *,
    label: str,
    require_url: bool,
    default_kind: str = "ollama",
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(value) - _GATEWAY_PROVIDER_KEYS
    if unknown:
        raise ValueError(f"{label} contains unsupported settings: {sorted(unknown)}")
    out = dict(value)
    base_url = value.get("base_url")
    if require_url and base_url in (None, ""):
        raise ValueError(f"{label}.base_url is required")
    out["base_url"] = _gateway_url(
        base_url or "http://127.0.0.1:11434/v1",
        label=f"{label}.base_url",
    )
    out["provider_kind"] = _gateway_name(
        value.get("provider_kind") or default_kind,
        label=f"{label}.provider_kind",
    )
    api_key = value.get("api_key")
    if api_key is not None and (
        not isinstance(api_key, str)
        or len(api_key) > 65_536
        or not api_key.isascii()
        or any(not char.isprintable() for char in api_key)
    ):
        raise ValueError(f"{label}.api_key must contain at most 65536 printable ASCII characters")
    out["api_key"] = api_key
    out["extra_headers"] = _gateway_headers(
        value.get("extra_headers"), label=f"{label}.extra_headers"
    )
    out["stream_required"] = _gateway_bool(
        value.get("stream_required", False), label=f"{label}.stream_required"
    )
    model_prefix = value.get("model_prefix")
    if model_prefix is not None:
        model_prefix = _gateway_name(model_prefix, label=f"{label}.model_prefix")
    out["model_prefix"] = model_prefix
    chat_path = value.get("chat_path", "/chat/completions")
    if (
        not isinstance(chat_path, str)
        or not chat_path.startswith("/")
        or len(chat_path) > 512
        or "?" in chat_path
        or "#" in chat_path
        or any(not character.isprintable() or character.isspace() for character in chat_path)
    ):
        raise ValueError(f"{label}.chat_path must be an absolute URL path")
    out["chat_path"] = chat_path
    return out


def _http_host(value: Any, *, label: str) -> str:
    host = str(value or "").strip()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if not host or len(host) > 253 or any(char.isspace() for char in host):
        raise ValueError(f"{label} must be a non-empty IP address or hostname")
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    labels = host.rstrip(".").split(".")
    if any(
        not part
        or len(part) > 63
        or part.startswith("-")
        or part.endswith("-")
        or not all(char.isalnum() or char == "-" for char in part)
        for part in labels
    ):
        raise ValueError(f"{label} must be a valid IP address or hostname")
    return host


def _http_port(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer between 0 and 65535")
    raw = str(value).strip()
    if not raw.isdigit():
        raise ValueError(f"{label} must be an integer between 0 and 65535")
    port = int(raw)
    if not 0 <= port <= 65_535:
        raise ValueError(f"{label} must be an integer between 0 and 65535")
    return port


def _parse_http_bind(value: Any) -> tuple[str, int]:
    raw = str(value or "").strip()
    if raw.startswith("["):
        closing = raw.find("]")
        if closing < 0 or raw[closing + 1 : closing + 2] != ":":
            raise ValueError("http.bind must use [IPv6]:port syntax")
        host_raw = raw[1:closing]
        port_raw = raw[closing + 2 :]
    else:
        if raw.count(":") != 1:
            raise ValueError("http.bind must use host:port or [IPv6]:port syntax")
        host_raw, port_raw = raw.rsplit(":", 1)
    return _http_host(host_raw, label="http.bind host"), _http_port(
        port_raw,
        label="http.bind port",
    )


@dataclass
class CronJobConfig:
    name: str
    schedule: dict[str, Any] = field(default_factory=dict)
    kind: str = "tick"
    jitter_seconds: float = 0.0
    enabled: bool = True


@dataclass
class CronConfig:
    enabled: bool = False
    resolution_sec: float = 5.0
    jobs: list[CronJobConfig] = field(default_factory=list)

    def to_raw(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "resolution_sec": self.resolution_sec,
            "jobs": [
                {
                    "name": j.name,
                    "schedule": j.schedule,
                    "kind": j.kind,
                    "jitter_seconds": j.jitter_seconds,
                    "enabled": j.enabled,
                }
                for j in self.jobs
            ],
        }


def _parse_cron(raw: dict[str, Any]) -> CronConfig:
    if not raw:
        return CronConfig()
    unknown = set(raw) - _SECTION_KEYS["cron"]
    if unknown:
        raise ValueError(f"config.cron contains unsupported settings: {sorted(unknown)}")
    jobs: list[CronJobConfig] = []
    raw_jobs = raw.get("jobs") or []
    if not isinstance(raw_jobs, list):
        raise ValueError("cron.jobs must be a list")
    if len(raw_jobs) > _MAX_CRON_JOBS:
        raise ValueError(f"cron.jobs must contain at most {_MAX_CRON_JOBS} jobs")
    seen_names: set[str] = set()
    for index, j in enumerate(raw_jobs):
        if not isinstance(j, dict):
            raise ValueError(f"cron.jobs.{index} must be an object")
        unknown = set(j) - _CRON_JOB_KEYS
        if unknown:
            raise ValueError(
                f"config.cron.jobs.{index} contains unsupported settings: {sorted(unknown)}"
            )
        schedule = j.get("schedule") or {}
        if not isinstance(schedule, dict):
            raise ValueError(f"cron.jobs.{index}.schedule must be an object")
        name = _bounded_config_text(
            j.get("name") or f"job-{index}",
            label=f"cron.jobs.{index}.name",
            maximum=128,
        )
        if name in seen_names:
            raise ValueError(f"cron.jobs contains duplicate job name {name!r}")
        seen_names.add(name)
        kind = _bounded_config_text(
            j.get("kind") or "tick",
            label=f"cron.jobs.{index}.kind",
            maximum=64,
        )
        # The adapter owns schedule semantics. Validate against that same
        # implementation during config load so startup cannot accept ingress
        # and then fail when the scheduler computes its first fire.
        from ..adapter.cron_in import CronJob

        try:
            CronJob(name=name, schedule=dict(schedule)).compute_next(
                now=datetime.now(UTC).astimezone()
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"invalid cron schedule for job {name!r}") from error
        jobs.append(
            CronJobConfig(
                name=name,
                schedule=dict(schedule),
                kind=kind,
                jitter_seconds=_config_float(
                    j.get("jitter_seconds", 0.0),
                    default=0.0,
                    minimum=0.0,
                    maximum=86_400.0,
                    label=f"cron.jobs.{index}.jitter_seconds",
                ),
                enabled=_config_bool(
                    j.get("enabled", True),
                    default=True,
                    label=f"cron.jobs.{index}.enabled",
                ),
            )
        )
    return CronConfig(
        enabled=_config_bool(
            raw.get("enabled", False),
            default=False,
            label="cron.enabled",
        ),
        resolution_sec=_config_float(
            raw.get("resolution_sec", 5.0),
            default=5.0,
            minimum=0.001,
            maximum=3_600.0,
            label="cron.resolution_sec",
        ),
        jobs=jobs,
    )


@dataclass
class DiscordChannelPolicy:
    require_mention: bool = False
    enabled: bool = True


@dataclass
class DiscordGuildPolicy:
    require_mention: bool = True
    users: set[str] = field(default_factory=set)
    channels: dict[str, DiscordChannelPolicy] = field(default_factory=dict)


@dataclass
class DiscordConfig:
    enabled: bool = False
    token: str | None = None
    allow_bots: bool = False
    dm_policy: str = "deny"  # "allowlist" | "deny"
    group_policy: str = "deny"  # "allowlist" | "deny"
    guilds: dict[str, DiscordGuildPolicy] = field(default_factory=dict)
    allow_from: set[str] = field(default_factory=set)

    @property
    def all_allowed_users(self) -> set[str]:
        """Union of allowFrom and every guild's users allowlist."""
        out: set[str] = set(self.allow_from)
        for g in self.guilds.values():
            out |= set(g.users)
        return out


def _parse_discord(raw: dict[str, Any]) -> DiscordConfig:
    if not isinstance(raw, dict):
        raise ValueError("channels.discord must be an object")
    raw = dict(raw)
    env_config = os.environ.get("NORAX_DISCORD_CONFIG_JSON")
    if env_config:
        parsed = json.loads(env_config)
        if not isinstance(parsed, dict):
            raise ValueError("NORAX_DISCORD_CONFIG_JSON must be a JSON object")
        raw.update(parsed)
    if not raw:
        return DiscordConfig()
    unknown = set(raw) - _DISCORD_KEYS
    if unknown:
        raise ValueError(f"channels.discord contains unsupported settings: {sorted(unknown)}")
    guilds_raw = raw.get("guilds") or {}
    if not isinstance(guilds_raw, dict):
        raise ValueError("channels.discord.guilds must be an object")
    if len(guilds_raw) > _MAX_DISCORD_GUILDS:
        raise ValueError(
            f"channels.discord.guilds must contain at most {_MAX_DISCORD_GUILDS} guilds"
        )
    guilds: dict[str, DiscordGuildPolicy] = {}
    for gid, gval in guilds_raw.items():
        guild_id = _bounded_config_text(str(gid), label="Discord guild id", maximum=128)
        if guild_id in guilds:
            raise ValueError(f"duplicate normalized Discord guild id: {guild_id!r}")
        if not isinstance(gval, dict):
            raise ValueError(f"channels.discord.guilds.{guild_id} must be an object")
        chans_raw = gval.get("channels") or {}
        if not isinstance(chans_raw, dict):
            raise ValueError(f"channels.discord.guilds.{guild_id}.channels must be an object")
        if len(chans_raw) > _MAX_DISCORD_CHANNELS_PER_GUILD:
            raise ValueError(
                f"channels.discord.guilds.{guild_id}.channels must contain at most "
                f"{_MAX_DISCORD_CHANNELS_PER_GUILD} channels"
            )
        unknown = set(gval) - _DISCORD_GUILD_KEYS
        if unknown:
            raise ValueError(
                f"channels.discord.guilds.{guild_id} contains unsupported settings: "
                f"{sorted(unknown)}"
            )
        chans: dict[str, DiscordChannelPolicy] = {}
        for cid, cval in chans_raw.items():
            channel_id = _bounded_config_text(
                str(cid), label=f"channels.discord.guilds.{guild_id} channel id", maximum=128
            )
            if channel_id in chans:
                raise ValueError(
                    f"duplicate normalized Discord channel id in guild {guild_id}: {channel_id!r}"
                )
            if not isinstance(cval, dict):
                raise ValueError(
                    f"channels.discord.guilds.{guild_id}.channels.{channel_id} must be an object"
                )
            unknown = set(cval) - _DISCORD_CHANNEL_KEYS
            if unknown:
                raise ValueError(
                    f"channels.discord.guilds.{guild_id}.channels.{channel_id} contains unsupported "
                    f"settings: {sorted(unknown)}"
                )
            chans[channel_id] = DiscordChannelPolicy(
                require_mention=_config_bool(
                    cval.get("requireMention", False),
                    default=False,
                    label=(
                        f"channels.discord.guilds.{guild_id}.channels.{channel_id}.requireMention"
                    ),
                ),
                enabled=_config_bool(
                    cval.get("enabled", True),
                    default=True,
                    label=f"channels.discord.guilds.{guild_id}.channels.{channel_id}.enabled",
                ),
            )
        users_list = gval.get("users") or []
        if not isinstance(users_list, (list, tuple, set)):
            log.warning(
                "config.invalid_list key=channels.discord.guilds.%s.users value=%r",
                gid,
                users_list,
            )
            users_list = []
        if len(users_list) > _MAX_DISCORD_USERS_PER_GUILD:
            raise ValueError(
                f"channels.discord.guilds.{guild_id}.users must contain at most "
                f"{_MAX_DISCORD_USERS_PER_GUILD} users"
            )
        normalized_users = {
            _bounded_config_text(
                str(user),
                label=f"channels.discord.guilds.{guild_id} user id",
                maximum=128,
            )
            for user in users_list
            if str(user).strip()
        }
        guilds[guild_id] = DiscordGuildPolicy(
            require_mention=_config_bool(
                gval.get("requireMention", True),
                default=True,
                label=f"channels.discord.guilds.{guild_id}.requireMention",
            ),
            users=normalized_users,
            channels=chans,
        )
    # Token: inline value OR env NORAX_DISCORD_TOKEN. Env takes priority.
    env_token = os.environ.get("NORAX_DISCORD_TOKEN")
    token = env_token or raw.get("token")
    if token is not None:
        token = str(token).strip() or None
    if token is not None and (
        len(token) > 4_096
        or not token.isascii()
        or any(not character.isprintable() for character in token)
    ):
        raise ValueError("Discord token must contain at most 4096 printable ASCII characters")
    enabled_env = os.environ.get("NORAX_DISCORD_ENABLED")
    file_enabled = _config_bool(
        raw.get("enabled", False),
        default=False,
        label="channels.discord.enabled",
    )
    enabled = (
        _config_bool(
            enabled_env,
            default=file_enabled,
            label="NORAX_DISCORD_ENABLED",
        )
        if enabled_env is not None
        else file_enabled
    )
    dm_policy = str(raw.get("dmPolicy", "deny")).strip().lower()
    if dm_policy not in {"allowlist", "deny"}:
        log.warning("config.invalid_choice key=channels.discord.dmPolicy value=%r", dm_policy)
        dm_policy = "deny"
    group_policy = str(raw.get("groupPolicy", "deny")).strip().lower()
    if group_policy not in {"allowlist", "deny"}:
        log.warning("config.invalid_choice key=channels.discord.groupPolicy value=%r", group_policy)
        group_policy = "deny"
    allow_from = raw.get("allowFrom") or []
    if not isinstance(allow_from, (list, tuple, set)):
        log.warning("config.invalid_list key=channels.discord.allowFrom value=%r", allow_from)
        allow_from = []
    if len(allow_from) > _MAX_DISCORD_ALLOW_FROM:
        raise ValueError(
            f"channels.discord.allowFrom must contain at most {_MAX_DISCORD_ALLOW_FROM} users"
        )
    normalized_allow_from = {
        _bounded_config_text(str(user), label="Discord allowFrom user id", maximum=128)
        for user in allow_from
        if str(user).strip()
    }
    return DiscordConfig(
        enabled=enabled,
        token=token,
        allow_bots=_config_bool(
            raw.get("allowBots", False),
            default=False,
            label="channels.discord.allowBots",
        ),
        dm_policy=dm_policy,
        group_policy=group_policy,
        guilds=guilds,
        allow_from=normalized_allow_from,
    )


@dataclass
class Config:
    raw: dict[str, Any]
    project_root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.raw, dict):
            raise ValueError("configuration root must be a JSON object")
        unknown = set(self.raw) - _TOP_LEVEL_KEYS
        if unknown:
            raise ValueError(f"configuration contains unsupported sections: {sorted(unknown)}")

    def _section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"config.{name} must be an object")
        allowed = _SECTION_KEYS.get(name)
        if allowed is not None:
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(f"config.{name} contains unsupported settings: {sorted(unknown)}")
        return value

    def validate_schema(self) -> None:
        """Eagerly validate every active config surface before side effects."""
        for name in _SECTION_KEYS:
            self._section(name)
        # These properties perform their own deeper validation and
        # normalization. Evaluating all of them here prevents a typo in a
        # lazily accessed optional connector from masquerading as accepted
        # configuration until hours later.
        _ = self.gateway
        _ = self.connectors
        _ = self.cognition
        _ = self.vision
        _ = self.discord
        _ = self.cron
        _ = self.http_bind

    @property
    def connectors(self) -> dict[str, Any]:
        """Return effective configuration for implemented runtime connectors.

        The newer ``connectors`` section takes precedence over legacy
        top-level sections.  Only connectors with a production call site are
        accepted: advertising a switch that nothing consumes makes both
        configuration and readiness output misleading.

        ``enabled`` is an availability gate.  A connector can still require a
        turn-level activation mode (for example, planner mode or automatic
        multi-agent decomposition) before it is used.
        """
        defaults: dict[str, dict[str, Any]] = {
            "planner": {"enabled": True},
            "multi_agent": {"enabled": True, "max_concurrent": 4},
            "mcp_client": {"enabled": False, "required": False},
            "a2a_server": {"enabled": False, "required": False},
            "commerce": {"enabled": False, "required": False},
            "active_inference": {"enabled": False},
        }

        # Preserve the documented legacy layout while giving an explicit
        # connectors entry final authority.  Copy the complete section so
        # connector-specific fields such as MCP servers and A2A bind settings
        # reach the implementation that consumes them.
        for connector, legacy_key in (
            ("mcp_client", "mcp"),
            ("a2a_server", "a2a"),
            ("commerce", "commerce"),
        ):
            legacy = self.raw.get(legacy_key)
            if legacy is not None:
                if not isinstance(legacy, dict):
                    raise ValueError(f"config.{legacy_key} must be an object")
                defaults[connector].update(legacy)

        conns = self._section("connectors")
        for key, val in conns.items():
            if key not in defaults:
                raise ValueError(f"unsupported connector: {key}")
            if not isinstance(val, dict):
                raise ValueError(f"config.connectors.{key} must be an object")
            defaults[key].update(val)

        for key, value in defaults.items():
            unknown = set(value) - _CONNECTOR_KEYS[key]
            if unknown:
                raise ValueError(
                    f"config connector {key} contains unsupported settings: {sorted(unknown)}"
                )
            value["enabled"] = _config_bool(
                value.get("enabled", False),
                default=False,
                label=f"connectors.{key}.enabled",
            )
            if "required" in _CONNECTOR_KEYS[key]:
                value["required"] = _config_bool(
                    value.get("required", False),
                    default=False,
                    label=f"connectors.{key}.required",
                )
        defaults["multi_agent"]["max_concurrent"] = min(
            32,
            _config_int(
                defaults["multi_agent"].get("max_concurrent", 4),
                default=4,
                minimum=1,
                label="connectors.multi_agent.max_concurrent",
            ),
        )
        servers = defaults["mcp_client"].get("servers")
        if servers is not None:
            if not isinstance(servers, list) or len(servers) > 64:
                raise ValueError("connectors.mcp_client.servers must be a list of at most 64")
            normalized_servers: list[dict[str, Any]] = []
            for index, server in enumerate(servers):
                if not isinstance(server, dict):
                    raise ValueError(f"connectors.mcp_client.servers.{index} must be an object")
                unknown = set(server) - {"name", "transport", "command", "args"}
                if unknown:
                    raise ValueError(
                        f"connectors.mcp_client.servers.{index} contains unsupported settings: "
                        f"{sorted(unknown)}"
                    )
                transport = server.get("transport", "stdio")
                if transport != "stdio":
                    raise ValueError(
                        f"connectors.mcp_client.servers.{index}.transport must be stdio"
                    )
                command = server.get("command")
                if not isinstance(command, str) or not command.strip() or len(command) > 4_096:
                    raise ValueError(f"connectors.mcp_client.servers.{index}.command is required")
                args = server.get("args") or []
                if (
                    not isinstance(args, list)
                    or len(args) > 256
                    or any(not isinstance(arg, str) or len(arg) > 4_096 for arg in args)
                ):
                    raise ValueError(
                        f"connectors.mcp_client.servers.{index}.args must be a string list"
                    )
                name = server.get("name") or "mcp-server"
                if not isinstance(name, str) or not name.strip() or len(name) > 128:
                    raise ValueError(
                        f"connectors.mcp_client.servers.{index}.name must contain 1-128 characters"
                    )
                normalized_servers.append(
                    {
                        "name": name.strip(),
                        "transport": "stdio",
                        "command": command.strip(),
                        "args": list(args),
                    }
                )
            defaults["mcp_client"]["servers"] = normalized_servers
        return {key: dict(value) for key, value in defaults.items()}

    @property
    def gateway(self) -> dict[str, Any]:
        """Return the validated gateway settings consumed by ``Runtime.build``."""
        raw = self._section("gateway")
        unknown = set(raw) - _GATEWAY_KEYS
        if unknown:
            raise ValueError(f"config.gateway contains unsupported settings: {sorted(unknown)}")

        default_model = _model_id(
            raw.get("default_model", "qwen3.8-27b-fast:latest"),
            label="gateway.default_model",
        )
        failover_raw = raw.get("failover_models") or []
        if not isinstance(failover_raw, list) or len(failover_raw) > 32:
            raise ValueError("gateway.failover_models must be a list of at most 32 models")
        failover_models = [
            _model_id(item, label=f"gateway.failover_models.{index}")
            for index, item in enumerate(failover_raw)
        ]

        providers_raw = raw.get("providers")
        providers: dict[str, dict[str, Any]] = {}
        if providers_raw is not None:
            if not isinstance(providers_raw, dict):
                raise ValueError("gateway.providers must be an object")
            if len(providers_raw) > 64:
                raise ValueError("gateway.providers must contain at most 64 providers")
            for raw_name, provider in providers_raw.items():
                name = _gateway_name(raw_name, label="gateway provider name")
                if name in providers:
                    raise ValueError(f"duplicate normalized gateway provider name: {name}")
                providers[name] = _gateway_provider_config(
                    provider,
                    label=f"gateway.providers.{name}",
                    require_url=True,
                    default_kind=name,
                )

        default_provider = _gateway_name(
            raw.get("default_provider") or (next(iter(providers), "ollama")),
            label="gateway.default_provider",
        )
        if providers and default_provider not in providers:
            raise ValueError(
                f"gateway.default_provider {default_provider!r} is not a configured provider"
            )

        routes_raw = raw.get("routes") or []
        if not isinstance(routes_raw, list) or len(routes_raw) > 512:
            raise ValueError("gateway.routes must be a list of at most 512 routes")
        routes: list[list[str]] = []
        for index, route in enumerate(routes_raw):
            if not isinstance(route, (list, tuple)) or len(route) != 2:
                raise ValueError(f"gateway.routes.{index} must contain [pattern, provider]")
            pattern, raw_provider = route
            if (
                not isinstance(pattern, str)
                or not pattern
                or len(pattern) > 512
                or any(not character.isprintable() or character.isspace() for character in pattern)
            ):
                raise ValueError(f"gateway.routes.{index} contains an invalid model pattern")
            provider = _gateway_name(raw_provider, label=f"gateway.routes.{index} provider")
            if providers and provider not in providers:
                raise ValueError(f"gateway.routes.{index} references unknown provider {provider!r}")
            routes.append([pattern, provider])

        out: dict[str, Any] = {
            "default_model": default_model,
            "default_provider": default_provider,
            "failover_models": failover_models,
            "timeout_seconds": _config_float(
                raw.get("timeout_seconds", 600.0),
                default=600.0,
                minimum=0.001,
                maximum=3_600.0,
                label="gateway.timeout_seconds",
            ),
            "routes": routes,
        }
        if providers_raw is not None:
            out["providers"] = providers
        else:
            out.update(
                _gateway_provider_config(
                    {key: raw[key] for key in _GATEWAY_PROVIDER_KEYS if key in raw},
                    label="gateway",
                    require_url=False,
                )
            )
        return out

    @property
    def cognition(self) -> dict[str, bool]:
        raw = self._section("cognition")
        unknown = set(raw) - _COGNITION_KEYS
        if unknown:
            raise ValueError(f"config.cognition contains unsupported settings: {sorted(unknown)}")
        result: dict[str, bool] = {}
        for key in sorted(_COGNITION_KEYS):
            value = raw.get(key, False)
            # Accept the earlier ``{"enabled": ...}`` spelling while making
            # every other nested setting fail instead of becoming dead config.
            if isinstance(value, dict):
                if set(value) != {"enabled"}:
                    raise ValueError(f"config.cognition.{key} only supports enabled")
                value = value["enabled"]
            result[key] = _config_bool(
                value,
                default=False,
                label=f"cognition.{key}",
            )
        return result

    @property
    def vision(self) -> dict[str, Any]:
        raw = self._section("vision")
        unknown = set(raw) - _VISION_KEYS
        if unknown:
            raise ValueError(f"config.vision contains unsupported settings: {sorted(unknown)}")
        enabled = _config_bool(
            raw.get("enabled", False),
            default=False,
            label="vision.enabled",
        )
        raw_model = raw.get("model")
        model = _model_id(raw_model, label="vision.model") if raw_model not in (None, "") else ""
        if enabled and not model:
            raise ValueError("vision.model is required when vision is enabled")
        return {"enabled": enabled, "model": model}

    @property
    def state_dir(self) -> Path:
        configured = os.environ.get("NORAX_STATE_DIR")
        if configured:
            return Path(configured).expanduser().resolve()
        paths = self._section("paths")
        custom = paths.get("state_dir")
        if custom:
            path = Path(str(custom)).expanduser()
            return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()
        return self.project_root / "state"

    @property
    def log_dir(self) -> Path:
        configured = os.environ.get("NORAX_LOG_DIR")
        if configured:
            return Path(configured).expanduser().resolve()
        paths = self._section("paths")
        custom = paths.get("log_dir")
        if custom:
            path = Path(str(custom)).expanduser()
            return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()
        return self.project_root / "logs"

    @property
    def soul_path(self) -> Path:
        paths = self._section("paths")
        custom = paths.get("soul")
        if custom:
            return (self.project_root / str(custom)).resolve()
        return self.project_root / "soul" / "SOUL.md"

    @property
    def identity_path(self) -> Path:
        paths = self._section("paths")
        custom = paths.get("identity")
        if custom:
            return (self.project_root / str(custom)).resolve()
        return self.project_root / "soul" / "IDENTITY.md"

    @property
    def user_path(self) -> Path:
        paths = self._section("paths")
        custom = paths.get("user")
        if custom:
            return (self.project_root / str(custom)).resolve()
        return self.project_root / "soul" / "USER.md"

    @property
    def memory_root(self) -> Path:
        configured = os.environ.get("NORAX_MEMORY_ROOT")
        if configured:
            return Path(configured).expanduser().resolve()
        paths = self._section("paths")
        custom = paths.get("memory_root")
        if custom:
            path = Path(str(custom)).expanduser()
            return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()
        return self.project_root / "memory"

    def validate_required_paths(self) -> list[str]:
        """Return list of missing required path descriptions; empty if all OK."""
        missing: list[str] = []
        for label, path in (
            ("soul", self.soul_path),
            ("identity", self.identity_path),
            ("user", self.user_path),
        ):
            if not path.exists():
                missing.append(f"{label}: {path}")
        return missing

    @property
    def event_log(self) -> Path:
        return self.state_dir / "events.jsonl"

    @property
    def http_bind(self) -> tuple[str, int]:
        # Environment overrides take priority. Fail clearly on malformed
        # values rather than binding an unintended interface or port.
        env_host = os.environ.get("NORAX_HTTP_BIND")
        env_port = os.environ.get("NORAX_HTTP_PORT")
        http = self._section("http")
        host, port = _parse_http_bind(http.get("bind", "127.0.0.1:4101"))
        if env_host is not None:
            host = _http_host(env_host, label="NORAX_HTTP_BIND")
        if env_port is not None:
            port = _http_port(env_port, label="NORAX_HTTP_PORT")
        return host, port

    @property
    def shutdown_grace_seconds(self) -> float:
        runtime = self._section("runtime")
        return _config_float(
            runtime.get("shutdown_grace_seconds", 10),
            default=10.0,
            minimum=0.0,
            label="runtime.shutdown_grace_seconds",
        )

    @property
    def max_tool_rounds(self) -> int:
        configured = os.environ.get("NORAX_MAX_TOOL_ROUNDS")
        runtime = self._section("runtime")
        file_value = _config_int(
            runtime.get("max_tool_rounds", 8),
            default=8,
            minimum=0,
            label="runtime.max_tool_rounds",
        )
        if configured in (None, ""):
            return file_value
        return _config_int(
            configured,
            default=file_value,
            minimum=0,
            label="NORAX_MAX_TOOL_ROUNDS",
        )

    @property
    def max_concurrent_turns(self) -> int:
        runtime = self._section("runtime")
        return min(
            64,
            _config_int(
                runtime.get("max_concurrent_turns", 8),
                default=8,
                minimum=1,
                label="runtime.max_concurrent_turns",
            ),
        )

    @property
    def max_pending_turns(self) -> int:
        runtime = self._section("runtime")
        return min(
            10_000,
            _config_int(
                runtime.get("max_pending_turns", 256),
                default=256,
                minimum=1,
                label="runtime.max_pending_turns",
            ),
        )

    @property
    def thinking_effort(self) -> str:
        runtime = self._section("runtime")
        configured = os.environ.get("NORAX_THINKING_EFFORT")
        return _config_choice(
            configured
            if configured not in (None, "")
            else runtime.get("thinking_effort", "medium"),
            default="medium",
            choices={"off", "low", "medium", "high", "xhigh", "max", "ultra"},
            label="NORAX_THINKING_EFFORT"
            if configured not in (None, "")
            else "runtime.thinking_effort",
        )

    @property
    def reasoning_output(self) -> bool:
        runtime = self._section("runtime")
        return _config_bool(
            runtime.get("reasoning_output", False),
            default=False,
            label="runtime.reasoning_output",
        )

    @property
    def planning_mode(self) -> str:
        return _config_choice(
            self._section("runtime").get("planning_mode", "direct"),
            default="direct",
            choices={"direct", "orchestrator"},
            label="runtime.planning_mode",
        )

    @property
    def memory_depth(self) -> str:
        return _config_choice(
            self._section("runtime").get("memory_depth", "auto"),
            default="auto",
            choices={"auto", "light", "balanced", "deep"},
            label="runtime.memory_depth",
        )

    @property
    def weak_model_boost(self) -> str:
        return _config_choice(
            self._section("runtime").get("weak_model_boost", "auto"),
            default="auto",
            choices={"auto", "on", "off"},
            label="runtime.weak_model_boost",
        )

    @property
    def stream_replies(self) -> bool:
        return _config_bool(
            self._section("runtime").get("stream_replies", True),
            default=True,
            label="runtime.stream_replies",
        )

    @property
    def response_length(self) -> str:
        return _config_choice(
            self._section("runtime").get("response_length", "balanced"),
            default="balanced",
            choices={"concise", "balanced", "detailed"},
            label="runtime.response_length",
        )

    @property
    def tool_activity(self) -> str:
        return _config_choice(
            self._section("runtime").get("tool_activity", "normal"),
            default="normal",
            choices={"minimal", "normal", "verbose"},
            label="runtime.tool_activity",
        )

    @property
    def owner_id(self) -> str | None:
        oid = os.environ.get("NORAX_OWNER_ID") or self._section("owner").get("id")
        return str(oid) if oid is not None else None

    @property
    def owner_label(self) -> str:
        return str(
            os.environ.get("NORAX_OWNER_LABEL") or self._section("owner").get("label") or "Owner"
        )

    @property
    def agent_os_chat_token(self) -> str:
        """Shared secret authenticating the Agent OS dashboard chat bridge."""
        return str(
            os.environ.get("NORAX_AGENT_OS_CHAT_TOKEN")
            or self._section("agent_os").get("chat_token")
            or ""
        )

    @property
    def discord(self) -> DiscordConfig:
        return _parse_discord(self._section("channels").get("discord") or {})

    @property
    def cron(self) -> CronConfig:
        return _parse_cron(self._section("cron"))


def load_config(path: Path | None = None) -> Config:
    cfg_path = path or _env_config_path() or DEFAULT_CONFIG
    root = _env_project_root() or DEFAULT_PROJECT_ROOT
    raw = load_jsonc(cfg_path) if cfg_path.exists() else {}
    cfg = Config(raw=raw, project_root=root)
    cfg.validate_schema()
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    # Fail fast on missing required prompt files — silent degradation hides
    # deployment mistakes and makes behavior CWD-dependent.
    missing = cfg.validate_required_paths()
    if missing:
        raise FileNotFoundError("Required prompt files not found: " + "; ".join(missing))
    return cfg
