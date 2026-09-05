"""Discord adapter — inbound + outbound.

Lifecycle
---------
- `start()` spawns a background task that runs `discord.Client.start(token)`.
- `stop()` calls `client.close()` and awaits the task.
- `events()` yields `SensoryInput` envelopes as messages arrive.

Inbound policy (gate, in order):
  1. Drop bot messages unless `allow_bots`.
  2. DM: allowed iff `allow_dms` AND author.id is in `all_allowed_users`.
  3. Guild: guild_id must be in `guild_allowlist`.
     - author.id must be in guild.users.
     - channel.id — if listed in guild.channels, that channel's
       `enabled` + `require_mention` wins; otherwise guild default applies.
     - `require_mention`: if True, message must mention the bot user
       (or include @<bot_id>). Raw `@everyone` does not count.

Principal assignment:
  - `tier = "owner"` if author.id == owner_id, else `"user"`.
  - `trust = True` iff author.id in `all_allowed_users`.

Outbound send():
  - Strips leading reply tag (`[[reply_to_current]]` / `[[reply_to:<id>]]`).
  - Uses `discord.MessageReference(..., fail_if_not_exists=False)` for
    reply threading; falls back to plain send on invalid ref.
  - Optional emoji reaction when `react_to` + `emoji` are provided.

Testability:
  - `client_factory` defaults to `discord.Client`; tests pass a fake class.
  - No live network calls happen in `__init__`. `start()` is the only
    place a real gateway connection is attempted.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ..config.loader import DiscordConfig
from ..envelope import Principal, SensoryInput, ThreadBinding
from .reply_tag import parse_reply_tag

log = logging.getLogger("norax.adapter.discord")


# Discord hard-caps regular text messages at 2000 chars. We keep 1900 to
# leave headroom for reply prefixes / zero-width padding Discord adds.
_DISCORD_MSG_LIMIT = 1900


def _split_discord_message(text: str, *, limit: int = _DISCORD_MSG_LIMIT) -> list[str]:
    """Split `text` into <= `limit`-char chunks at natural break points.

    Preference order: paragraph break (\\n\\n), line break (\\n), space, hard.
    Also respects fenced code blocks — if a chunk starts inside a ``` block
    we re-open/close the fence so syntax highlighting survives the split.
    """
    text = text or ""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    open_fence: str | None = None  # language tag of an unclosed ```

    while len(remaining) > limit:
        window = remaining[:limit]
        # Prefer paragraph, then line, then space.
        for sep in ("\n\n", "\n", " "):
            idx = window.rfind(sep)
            if idx > limit // 2:
                cut = idx + len(sep)
                break
        else:
            cut = limit

        piece = remaining[:cut]
        remaining = remaining[cut:]

        # Handle code fences: count unmatched ``` in this piece.
        fence_count = piece.count("```")
        prefix = ""
        suffix = ""
        if open_fence is not None:
            prefix = f"```{open_fence}\n"
        if (fence_count + (1 if open_fence else 0)) % 2 == 1:
            # We're leaving this piece inside a code block — close it,
            # remember the language, and reopen next piece.
            # Detect latest language in this piece; default "" (plain).
            last_open = piece.rfind("```")
            lang_line = piece[last_open + 3 :].split("\n", 1)[0].strip()
            open_fence = lang_line if lang_line else ""
            suffix = "\n```"
        else:
            open_fence = None

        chunks.append(prefix + piece + suffix)

    if remaining:
        prefix = f"```{open_fence}\n" if open_fence is not None else ""
        chunks.append(prefix + remaining)

    return chunks


class TypingKeepAlive:
    """Keeps Discord's "X is typing…" indicator alive on a channel.

    Discord's `typing` trigger fires the indicator for ~10s. We
    re-trigger every 7s on a background task. `stop()` cancels it.
    """

    def __init__(self, channel: Any, *, interval: float = 7.0) -> None:
        self._channel = channel
        self._interval = interval
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def _trigger_once(self) -> None:
        """Fire a single typing indicator, handling discord.py API variants.

        discord.py 2.x removed the public `trigger_typing()` coroutine;
        the supported API is `channel.typing()` as an async context
        manager. Underneath it still hits POST /channels/{id}/typing.
        We prefer the new API and fall back for older builds.
        """
        # Prefer the low-level HTTP call (1 POST, no context manager
        # shenanigans — fire-and-forget typing pulse).
        try:
            state = getattr(self._channel, "_state", None)
            http = getattr(state, "http", None) if state else None
            chan_id = getattr(self._channel, "id", None)
            if http is not None and chan_id is not None:
                await asyncio.wait_for(http.send_typing(int(chan_id)), timeout=3.0)
                return
        except (TimeoutError, Exception):  # noqa: BLE001
            pass
        # Public API fallback: `async with channel.typing()` — we enter
        # and immediately exit; discord.py sends one typing pulse.
        try:
            typing_cm = self._channel.typing()
            await asyncio.wait_for(typing_cm.__aenter__(), timeout=3.0)
            try:
                await asyncio.wait_for(typing_cm.__aexit__(None, None, None), timeout=3.0)
            except (TimeoutError, Exception):  # noqa: BLE001
                pass
            return
        except (TimeoutError, Exception):  # noqa: BLE001
            pass
        # Legacy discord.py <2.0
        trigger = getattr(self._channel, "trigger_typing", None)
        if trigger is not None:
            try:
                await asyncio.wait_for(trigger(), timeout=3.0)
            except (TimeoutError, Exception):  # noqa: BLE001
                pass

    async def start(self) -> None:
        try:
            await asyncio.wait_for(self._trigger_once(), timeout=5.0)
        except Exception:  # noqa: BLE001
            log.debug("discord.typing.initial_failed", exc_info=True)
        self._task = asyncio.create_task(self._loop(), name="discord-typing")

    async def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
                    return
                except TimeoutError:
                    pass
                try:
                    await asyncio.wait_for(self._trigger_once(), timeout=5.0)
                except (TimeoutError, Exception):  # noqa: BLE001
                    log.debug("discord.typing.refresh_failed", exc_info=True)
        except asyncio.CancelledError:
            return

    async def stop(self, *, timeout: float = 2.0) -> None:
        self._stop.set()
        t = self._task
        if t is not None:
            t.cancel()
            try:
                await asyncio.wait_for(t, timeout=timeout)
            except (TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
                log.debug("discord.typing.stop_timeout", exc_info=True)


class StreamingMessage:
    """Discord-side handle for an in-place streaming reply.

    Usage:
        sm = await adapter.begin_streaming_message(target, reply_to=mid)
        async for delta in stream:
            await sm.append(delta)
        await sm.finalize(final_text)

    Implementation:
      - Edits a single Discord message in-place.
      - Edits are rate-limited to one every `min_edit_interval` seconds
        (default 1.2s) to stay well under Discord's per-channel cap of
        5 edits / 5s.
      - If `first_message` is None, the initial message is deferred until
        the first `_push()` or `finalize()` — useful when you want only the
        typing indicator visible during early generation.
      - Long live output is streamed across ordered Discord messages, each
        below `_DISCORD_MSG_LIMIT`; no temporary "response continues" stub is
        left behind if the response spans multiple messages.
    """

    def __init__(
        self,
        *,
        channel: Any,
        first_message: Any | None = None,
        reference: Any | None = None,
        min_edit_interval: float = 1.2,
        edit_threshold_chars: int = 60,
    ) -> None:
        self._channel = channel
        self._msg = first_message
        self._messages: list[Any] = [first_message] if first_message is not None else []
        self._reference = reference
        self._buf: str = ""
        self._last_pushed: str = ""
        self._pushed_chunks: list[str] = []
        self._lock = asyncio.Lock()
        self._min_interval = min_edit_interval
        self._edit_threshold = edit_threshold_chars
        self._last_edit_at: float = 0.0
        self._closed = False
        self._sent_ids: list[str] = []
        if first_message is not None:
            self._sent_ids.append(str(getattr(first_message, "id", "")))

    @property
    def message_ids(self) -> list[str]:
        return list(self._sent_ids)

    async def append(self, delta: str) -> None:
        if not delta or self._closed:
            return
        now = asyncio.get_running_loop().time()
        if self._last_edit_at == 0.0:
            # First-ever append: anchor the rate-limit clock so the
            # initial burst doesn't fire an immediate edit.
            self._last_edit_at = now
        self._buf += delta
        new_chars = len(self._buf) - len(self._last_pushed)
        if new_chars >= self._edit_threshold and (now - self._last_edit_at) >= self._min_interval:
            await self._push()

    async def _push(self) -> None:
        async with self._lock:
            text = self._buf
            if text == self._last_pushed:
                return
            # Use stable fixed-width live chunks. Once a chunk is full it never
            # changes, so each push edits only the current tail and creates at
            # most one new message. finalize() later replaces these with the
            # naturally split final chunks.
            chunks = [
                text[i : i + _DISCORD_MSG_LIMIT] for i in range(0, len(text), _DISCORD_MSG_LIMIT)
            ]
            delivered = True
            for idx, chunk in enumerate(chunks):
                if idx < len(self._pushed_chunks) and self._pushed_chunks[idx] == chunk:
                    continue
                try:
                    if idx < len(self._messages):
                        await self._messages[idx].edit(content=chunk or "…")
                    else:
                        kwargs: dict[str, Any] = {}
                        if idx == 0 and self._reference is not None:
                            kwargs["reference"] = self._reference
                        msg = await self._channel.send(chunk, **kwargs)
                        self._messages.append(msg)
                        self._sent_ids.append(str(getattr(msg, "id", "")))
                        if self._msg is None:
                            self._msg = msg
                except Exception:  # noqa: BLE001
                    delivered = False
                    log.debug("discord.stream.push_failed", exc_info=True)
            if delivered:
                self._pushed_chunks = chunks
                self._last_pushed = text
                self._last_edit_at = asyncio.get_running_loop().time()

    async def finalize(self, final_text: str | None = None) -> dict:
        async with self._lock:
            if self._closed:
                return {
                    "ok": True,
                    "message_id": self._sent_ids[0] if self._sent_ids else "",
                    "message_ids": list(self._sent_ids),
                    "chunks": len(self._sent_ids),
                }
            self._closed = True
            text = (final_text if final_text is not None else self._buf) or ""

            # Nothing to show and no message ever created → silently succeed.
            if not text and not self._messages:
                return {"ok": True, "message_id": "", "message_ids": [], "chunks": 0}

            chunks = _split_discord_message(text, limit=_DISCORD_MSG_LIMIT) if text else []
            if not chunks:
                chunks = ["⚠️ Empty Discord stream finalize: no text/chunks available"]

            delivered = True
            for idx, chunk in enumerate(chunks):
                try:
                    if idx < len(self._messages):
                        await self._messages[idx].edit(content=chunk)
                    else:
                        kwargs: dict[str, Any] = {}
                        if idx == 0 and self._reference is not None:
                            kwargs["reference"] = self._reference
                        msg = await self._channel.send(chunk, **kwargs)
                        self._messages.append(msg)
                        self._sent_ids.append(str(getattr(msg, "id", "")))
                        if self._msg is None:
                            self._msg = msg
                except Exception:  # noqa: BLE001
                    delivered = False
                    log.exception("discord.stream.finalize_delivery_failed chunk=%d", idx)

            # A recovered/failover response may be shorter than its streamed
            # preview. Remove stale preview chunks so Discord shows exactly the
            # completed response and nothing else.
            stale_messages = self._messages[len(chunks) :]
            for stale in stale_messages:
                try:
                    await stale.delete()
                except Exception:  # noqa: BLE001
                    delivered = False
                    log.exception("discord.stream.finalize_stale_delete_failed")
            if stale_messages:
                self._messages = self._messages[: len(chunks)]
                self._sent_ids = [str(getattr(message, "id", "")) for message in self._messages]
                self._msg = self._messages[0] if self._messages else None

            self._pushed_chunks = chunks
            self._last_pushed = text
            self._last_edit_at = asyncio.get_running_loop().time()
            return {
                "ok": delivered,
                "message_id": self._sent_ids[0] if self._sent_ids else "",
                "message_ids": list(self._sent_ids),
                "chunks": len(chunks),
            }

    async def delete(self) -> None:
        """Delete the Discord message(s) created by this streaming message.

        Used when the model outputs NO_REPLY / HEARTBEAT_OK and we want
        to suppress the message entirely instead of showing literal text.
        """
        self._closed = True
        for mid in self._sent_ids:
            try:
                msg = await self._channel.fetch_message(int(mid))
                await msg.delete()
            except Exception:  # noqa: BLE001
                log.debug("discord.stream.delete_failed mid=%s", mid)
        self._sent_ids.clear()
        self._msg = None


def _import_discord():
    """Lazy import so tests can patch without loading real discord.py."""
    import discord  # type: ignore

    return discord


class DiscordInAdapter:
    name = "discord"
    failure_is_fatal = False

    def __init__(
        self,
        *,
        config: DiscordConfig,
        owner_id: str | None = None,
        client_factory: Callable[..., Any] | None = None,
        intents: Any | None = None,
    ) -> None:
        self.config = config
        self.owner_id = str(owner_id) if owner_id else None
        self._client_factory = client_factory
        self._intents = intents
        self._queue: asyncio.Queue[SensoryInput] = asyncio.Queue(maxsize=512)
        self._client: Any | None = None
        self._tree: Any | None = None  # discord.app_commands.CommandTree
        self._task: asyncio.Task | None = None
        self._started = False
        self._stopped = False
        self._allowed_users = config.all_allowed_users
        # DM channel cache: user_id -> channel_id for outbound DM resolution.
        # Populated from incoming messages so we never need fetch_user/create_dm.
        self._dm_channel_cache: dict[int, int] = {}
        # Callback: (name: str, args: dict, principal: Principal, interaction_ctx: dict) -> awaitable[str]
        self._slash_handler: (
            Callable[[str, dict, Any, dict], Awaitable[dict | str | None]] | None
        ) = None
        # Connection state tracking for readiness/degradation
        self._ws_connected: bool = False
        self._last_ready_time: float = 0.0
        self._reconnect_count: int = 0
        self._last_disconnect_time: float = 0.0
        self._last_error: str = ""

    @property
    def connection_state(self) -> dict[str, Any]:
        """Return Discord websocket connection state for readiness checks."""
        import time as _time

        return {
            "connected": self._ws_connected,
            "last_ready": self._last_ready_time,
            "reconnect_count": self._reconnect_count,
            "last_disconnect": self._last_disconnect_time,
            "last_error": self._last_error,
            "uptime_sec": (_time.monotonic() - self._last_ready_time)
            if self._ws_connected and self._last_ready_time
            else 0,
        }

    def set_slash_handler(
        self, handler: Callable[[str, dict, Any, dict], Awaitable[dict | str | None]]
    ) -> None:
        """Wire a callback invoked when a Discord slash command fires.

        The callback receives (command_name, args_dict, principal, ctx_dict)
        and returns the reply text (str). The adapter handles the
        interaction response lifecycle (defer + followup on long calls).
        """
        self._slash_handler = handler

    # ------------------------------------------------------------------
    # Gate
    # ------------------------------------------------------------------
    def _is_allowed(self, msg: Any) -> tuple[bool, str]:
        """Return (ok, reason). reason is a short tag for logging on drop."""
        author = getattr(msg, "author", None)
        if author is None:
            return False, "no_author"
        if getattr(author, "bot", False) and not self.config.allow_bots:
            return False, "bot_message"
        author_id = str(getattr(author, "id", ""))
        # Echo-loop guard: never process our own messages, even when
        # allow_bots is True (which is for *other* bots, not self).
        bot_user = self._client_user() if self._client else None
        bot_id = str(getattr(bot_user, "id", "")) if bot_user else ""
        if bot_id and author_id == bot_id:
            return False, "self_message"
        guild = getattr(msg, "guild", None)

        if guild is None:
            # DM
            if self.config.dm_policy != "allowlist":
                return False, "dm_policy_deny"
            if author_id not in self._allowed_users:
                return False, "dm_not_allowlisted"
            return True, "dm_ok"

        # Guild
        if self.config.group_policy != "allowlist":
            return False, "group_policy_deny"
        guild_id = str(getattr(guild, "id", ""))
        gpolicy = self.config.guilds.get(guild_id)
        if gpolicy is None:
            return False, "guild_not_allowlisted"
        if author_id not in gpolicy.users:
            return False, "user_not_allowlisted"

        channel = getattr(msg, "channel", None)
        channel_id = str(getattr(channel, "id", ""))
        require_mention = gpolicy.require_mention
        chan_policy = gpolicy.channels.get(channel_id)
        if chan_policy is not None:
            if not chan_policy.enabled:
                return False, "channel_disabled"
            require_mention = chan_policy.require_mention

        if require_mention:
            if not self._mentions_bot(msg):
                return False, "missing_mention"

        return True, "guild_ok"

    def _mentions_bot(self, msg: Any) -> bool:
        bot_user = self._client_user() if self._client else None
        bot_id = str(getattr(bot_user, "id", "")) if bot_user else ""
        mentions = getattr(msg, "mentions", None) or []
        for m in mentions:
            if bot_id and str(getattr(m, "id", "")) == bot_id:
                return True
        # Fallback: raw content check
        content = getattr(msg, "content", "") or ""
        if bot_id and (f"<@{bot_id}>" in content or f"<@!{bot_id}>" in content):
            return True
        return False

    def _client_user(self) -> Any | None:
        if self._client is None:
            return None
        return getattr(self._client, "user", None)

    # ------------------------------------------------------------------
    # Envelope build
    # ------------------------------------------------------------------
    def _envelope_from(self, msg: Any) -> SensoryInput:
        author = msg.author
        author_id = str(author.id)
        label = getattr(author, "display_name", None) or getattr(author, "name", "") or author_id
        tier: Literal["owner", "admin", "user", "guest"] = (
            "owner" if self.owner_id and author_id == self.owner_id else "user"
        )
        trust = author_id in self._allowed_users
        sender = Principal(id=author_id, label=str(label), trust=trust, tier=tier)

        channel = getattr(msg, "channel", None)
        channel_id = str(getattr(channel, "id", "")) if channel else ""
        guild = getattr(msg, "guild", None)
        guild_id = str(getattr(guild, "id", "")) if guild else None

        # Thread detection: channel.type may be discord.ChannelType.public_thread etc.
        thread_binding: ThreadBinding | None = None
        chan_type = getattr(channel, "type", None)
        chan_type_name = getattr(chan_type, "name", "") if chan_type is not None else ""
        if chan_type_name in {"public_thread", "private_thread", "news_thread"}:
            thread_binding = ThreadBinding(
                channel="discord",
                thread_id=channel_id,
                kind="custom",
            )

        body = getattr(msg, "content", "") or ""
        ts = getattr(msg, "created_at", None) or datetime.now(UTC)

        attachments = []
        for a in getattr(msg, "attachments", []) or []:
            attachments.append(
                {
                    "id": str(getattr(a, "id", "")),
                    "filename": getattr(a, "filename", ""),
                    "url": getattr(a, "url", ""),
                    "content_type": getattr(a, "content_type", None),
                    "size": getattr(a, "size", 0),
                }
            )

        return SensoryInput(
            channel="chat",
            source=self.name,
            message_id=str(msg.id),
            timestamp=ts,
            sender=sender,
            body=body,
            attachments=attachments,
            raw={
                "channel_id": channel_id,
                "guild_id": guild_id,
                "discord_message_id": str(msg.id),
            },
            trusted=trust,
            thread_binding=thread_binding,
            metadata={},
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._started:
            raise RuntimeError("DiscordInAdapter.start() may only be called once")
        if self._stopped:
            raise RuntimeError("DiscordInAdapter cannot restart after stop()")
        if not self.config.enabled:
            log.info("discord.start skipped: disabled")
            return
        if not self.config.token:
            log.warning("discord.start skipped: no token configured")
            return

        client = self._build_client()
        self._started = True
        self._task = asyncio.create_task(self._run_client_supervised(client), name="discord-client")
        log.info("discord adapter started")

    def _build_client(self) -> Any:
        """Construct and fully wire a fresh Discord client.

        Factored out of start() so the supervisor can build a brand-new client
        on each reconnect — a discord.py Client cannot be restarted once closed.
        """
        if self._client_factory is None:
            discord = _import_discord()
            intents = self._intents
            if intents is None:
                intents = discord.Intents.default()
                intents.message_content = True
                intents.guilds = True
                intents.messages = True
            client = discord.Client(intents=intents)
        else:
            client = self._client_factory()

        self._client = client

        # Build the CommandTree (for native Discord slash commands). We
        # register commands unconditionally; invocation is gated by the
        # same sender-allowlist as regular messages.
        try:
            discord_mod = _import_discord()
            tree = discord_mod.app_commands.CommandTree(client)
            self._tree = tree
            self._register_slash_commands(tree, discord_mod)
        except Exception:  # noqa: BLE001
            log.exception("discord.tree setup failed — slash commands disabled")
            self._tree = None

        @client.event
        async def on_ready():  # noqa: D401
            import time as _time

            self._ws_connected = True
            self._last_ready_time = _time.monotonic()
            if self._reconnect_count > 0:
                log.info(
                    "discord.reconnected count=%d outage_sec=%.0f",
                    self._reconnect_count,
                    _time.monotonic() - self._last_disconnect_time,
                )
            log.info("discord.ready user=%s", getattr(client.user, "id", "?"))
            # Pre-seed DM channel cache for the owner so outbound
            # message_send works immediately without needing a prior
            # incoming DM to seed the cache.
            if self.owner_id:
                try:
                    owner_uid = int(self.owner_id)
                    # Check cache first.
                    if owner_uid not in self._dm_channel_cache:
                        try:
                            user = await client.fetch_user(owner_uid)
                            if user is not None:
                                dm = getattr(user, "dm_channel", None)
                                if dm is None:
                                    dm = await user.create_dm()
                                if dm is not None:
                                    self._dm_channel_cache[owner_uid] = dm.id
                                    log.info(
                                        "discord.dm_cache_seeded owner=%s channel=%s",
                                        owner_uid,
                                        dm.id,
                                    )
                        except Exception:  # noqa: BLE001
                            log.debug(
                                "discord.dm_cache_seed_failed owner=%s (non-fatal)", owner_uid
                            )
                except Exception:  # noqa: BLE001
                    pass
            # Sync the command tree so Discord knows about our commands.
            # Guild-scoped sync is instant; global sync can take minutes.
            if self._tree is not None:
                try:
                    await self._sync_slash_commands(discord_mod)
                except Exception:  # noqa: BLE001
                    log.exception("discord.tree.sync failed")

        @client.event
        async def on_message(message):  # noqa: D401
            try:
                # Cache DM channel IDs from incoming messages so outbound
                # message_send can resolve user_id → DM channel without
                # needing fetch_user / create_dm (which require privileged
                # intents the bot may not have).
                try:
                    ch = getattr(message, "channel", None)
                    if ch is not None:
                        ch_type = getattr(ch, "type", None)
                        # DM channels: type 1 in discord.py enum
                        is_dm = getattr(ch_type, "value", ch_type) == 1
                        if is_dm:
                            author_id = getattr(getattr(message, "author", None), "id", None)
                            channel_id = getattr(ch, "id", None)
                            if author_id and channel_id:
                                self._dm_channel_cache[int(author_id)] = int(channel_id)
                                log.debug(
                                    "discord.dm_cache author=%s channel=%s", author_id, channel_id
                                )
                except Exception:  # noqa: BLE001
                    pass  # caching must never break message handling

                ok, reason = self._is_allowed(message)
                if not ok:
                    log.debug(
                        "discord.drop reason=%s author=%s",
                        reason,
                        getattr(getattr(message, "author", None), "id", "?"),
                    )
                    return
                env = self._envelope_from(message)
                try:
                    self._queue.put_nowait(env)
                except asyncio.QueueFull:
                    log.warning("discord.back_pressure_drop message_id=%s", env.message_id)
            except Exception:
                log.exception("discord.on_message crashed")

        return client

    async def _run_client_supervised(self, client: Any) -> None:
        """Run the Discord client, surfacing failures and auto-reconnecting.

        `discord.Client.start()` has its own internal reconnect loop for
        transient gateway drops, but it *returns or raises* on fatal errors
        (bad token, missing privileged intents, or an unexpected exception).
        Previously the start() task was fire-and-forget, so such a failure
        left the bot silently offline with nothing in the log. This supervisor
        logs every failure and retries with capped exponential backoff —
        except for unrecoverable auth/intent errors, which it logs loudly and
        stops on (retrying those would hot-loop and risk a Discord ban).
        """
        try:
            discord_mod = _import_discord()
            fatal_errors: tuple[type[BaseException], ...] = tuple(
                e
                for e in (
                    getattr(discord_mod, "LoginFailure", None),
                    getattr(discord_mod, "PrivilegedIntentsRequired", None),
                )
                if e is not None and isinstance(e, type)
            )
        except Exception:  # noqa: BLE001
            fatal_errors = ()

        backoff = 5.0
        backoff_max = 300.0
        while not self._stopped:
            import time as _time

            if client is None:
                try:
                    client = self._build_client()
                except Exception as e:  # noqa: BLE001
                    self._last_error = f"{type(e).__name__}: {e}"
                    log.exception("discord.client rebuild failed — retrying in %.0fs", backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff_max, backoff * 2)
                    self._reconnect_count += 1
                    continue

            try:
                await client.start(self.config.token)
                # Returned without raising → client was closed intentionally.
                if self._stopped:
                    return
                log.warning("discord.client exited cleanly; reconnecting in %.0fs", backoff)
            except asyncio.CancelledError:
                raise
            except fatal_errors as e:  # type: ignore[misc]
                self._ws_connected = False
                self._last_error = f"{type(e).__name__}: {e}"
                log.error(
                    "discord.client FATAL (%s): %s — not retrying. "
                    "Check NORAX_DISCORD_TOKEN and that Message Content / "
                    "Server Members privileged intents are enabled in the "
                    "Discord developer portal.",
                    type(e).__name__,
                    e,
                )
                raise
            except Exception as e:  # noqa: BLE001
                self._ws_connected = False
                self._last_error = f"{type(e).__name__}: {e}"
                log.exception("discord.client crashed: %r — reconnecting in %.0fs", e, backoff)

            # Track disconnect
            if self._ws_connected:
                self._ws_connected = False
                self._last_disconnect_time = _time.monotonic()
                backoff = 5.0

            if self._stopped:
                return
            # A closed discord.py client cannot be restarted, so tear it down
            # and build a fresh, fully-wired one before the next attempt.
            try:
                if not getattr(client, "is_closed", lambda: True)():
                    await client.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise
            backoff = min(backoff_max, backoff * 2)
            self._reconnect_count += 1
            client = None

    async def stop(self) -> None:
        self._stopped = True
        self._ws_connected = False
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:  # noqa: BLE001
                log.exception("discord.client.close failed")
        task = self._task
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            except asyncio.CancelledError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise
            except Exception:  # noqa: BLE001
                log.exception("discord.task await failed")
        self._task = None
        self._client = None
        self._tree = None

    async def events(self) -> AsyncIterator[SensoryInput]:
        while not self._stopped:
            client_task = self._task
            if client_task is None:
                return
            queued = asyncio.create_task(self._queue.get())
            try:
                done, _pending = await asyncio.wait(
                    {queued, client_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if queued in done:
                    yield queued.result()
                    continue
                if self._stopped:
                    return
                if client_task.cancelled():
                    raise RuntimeError("Discord client was cancelled unexpectedly")
                error = client_task.exception()
                if error is not None:
                    raise RuntimeError("Discord client failed") from error
                raise RuntimeError("Discord client stopped unexpectedly")
            finally:
                if not queued.done():
                    queued.cancel()
                    await asyncio.gather(queued, return_exceptions=True)

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------
    async def send(
        self,
        target: str,
        text: str,
        *,
        reply_to: str | None = None,
        react_to: str | None = None,
        emoji: str | None = None,
        files: list[str] | None = None,
        **_: Any,
    ) -> dict:
        """Send a message and optional file attachments to Discord.

        `target` is a channel id or user id (for DM). `reply_to` is a message
        id to reply to. Reply-tag parsing happens first: a leading
        [[reply_to_current]] implies the caller should have supplied `reply_to`.
        """
        parsed = parse_reply_tag(text, current_message_id=reply_to)
        resolved_reply = parsed.reply_to if parsed.had_tag else reply_to
        body = parsed.text

        if self._client is None:
            return {"ok": False, "error": "client_not_started", "target": target}

        try:
            channel = await self._fetch_target(int(target))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"target_lookup_failed:{e!r}", "target": target}
        if channel is None:
            return {"ok": False, "error": "target_not_found", "target": target}

        # Reactions path takes priority (no text sent)
        if react_to and emoji:
            try:
                msg = await channel.fetch_message(int(react_to))
                await msg.add_reaction(emoji)
                return {"ok": True, "reacted": react_to, "emoji": emoji}
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "error": f"react_failed:{e!r}"}

        send_kwargs: dict[str, Any] = {}
        if resolved_reply:
            try:
                discord = _import_discord()
                send_kwargs["reference"] = discord.MessageReference(
                    message_id=int(resolved_reply),
                    channel_id=int(target),
                    fail_if_not_exists=False,
                )
            except Exception:  # noqa: BLE001
                # Reference build failed — fall back to plain send.
                log.debug(
                    "discord.reply_ref_failed id=%s; sending without reference", resolved_reply
                )

        if not body.strip() and not files:
            return {"ok": False, "error": "empty_body_after_tag_strip"}

        discord_files: list[Any] = []
        try:
            if files:
                discord = _import_discord()
                for raw_path in files:
                    p = Path(raw_path).expanduser().resolve()
                    if not p.is_file():
                        return {"ok": False, "error": f"file_not_found:{p}", "target": target}
                    if p.stat().st_size > 24 * 1024 * 1024:
                        return {"ok": False, "error": f"file_too_large:{p}", "target": target}
                    discord_files.append(discord.File(str(p), filename=p.name))

            chunks = _split_discord_message(body, limit=1900) if body.strip() else [""]
            first_sent = None
            sent_ids: list[str] = []
            for i, chunk in enumerate(chunks):
                kwargs = dict(send_kwargs) if i == 0 else {}
                if i == 0 and discord_files:
                    kwargs["files"] = discord_files
                sent = await channel.send(chunk, **kwargs)
                sid = str(getattr(sent, "id", ""))
                sent_ids.append(sid)
                if first_sent is None:
                    first_sent = sent
            return {
                "ok": True,
                "message_id": str(getattr(first_sent, "id", "") or ""),
                "message_ids": sent_ids,
                "chunks": len(chunks),
                "target": target,
            }
        except Exception as e:  # noqa: BLE001
            log.exception("discord.send failed target=%s", target)
            return {"ok": False, "error": repr(e), "target": target}

    async def _fetch_target(self, target_id: int) -> Any | None:
        """Resolve a Discord outbound target.

        Target may be either a channel id or a user id. This keeps
        `message_send(channel=\"discord\", target=<owner_id>, ...)` usable for
        DMs instead of requiring callers to know Discord's hidden DM channel id.
        """
        if self._client is None:
            return None

        # 1. Check DM cache first — zero API calls, always works even
        #    without privileged intents.
        cached_ch = self._dm_channel_cache.get(target_id)
        if cached_ch is not None:
            channel = self._client.get_channel(cached_ch)
            if channel is not None:
                return channel
            # Cache stale; fall through to full resolution.

        # 2. Try as a direct channel/guild channel ID.
        try:
            channel = await self._fetch_channel(target_id)
            if channel is not None:
                return channel
        except Exception:  # noqa: BLE001
            pass  # Not a channel ID, try user lookup

        # 3. Try as a user ID → DM channel.
        try:
            user = None
            getter = getattr(self._client, "get_user", None)
            if getter is not None:
                user = getter(target_id)
            if user is None:
                fetch_user = getattr(self._client, "fetch_user", None)
                if fetch_user is not None:
                    user = await fetch_user(target_id)
            if user is None:
                return None

            dm = getattr(user, "dm_channel", None)
            if dm is not None:
                # Cache it for future outbound.
                self._dm_channel_cache[target_id] = dm.id
                return dm
            create_dm = getattr(user, "create_dm", None)
            if create_dm is None:
                return None
            dm_channel = await create_dm()
            # Cache it.
            if dm_channel is not None:
                self._dm_channel_cache[target_id] = dm_channel.id
            return dm_channel
        except Exception:  # noqa: BLE001
            return None

    async def _fetch_channel(self, channel_id: int) -> Any | None:
        if self._client is None:
            return None
        channel = self._client.get_channel(channel_id)
        if channel is not None:
            return channel
        # Fallback to REST fetch
        fetcher = getattr(self._client, "fetch_channel", None)
        if fetcher is None:
            return None
        return await fetcher(channel_id)

    # ------------------------------------------------------------------
    # Typing indicator
    # ------------------------------------------------------------------
    async def start_typing(self, target: str) -> TypingKeepAlive | None:
        """Begin the "Norax is typing…" indicator on a channel.

        Returns a keep-alive handle whose `.stop()` ends the indicator.
        Discord's typing trigger expires after ~10s, so the helper task
        re-triggers on a 7s interval.
        """
        if self._client is None:
            return None
        try:
            channel = await self._fetch_channel(int(target))
        except Exception:  # noqa: BLE001
            return None
        if channel is None:
            return None
        keep = TypingKeepAlive(channel)
        await keep.start()
        return keep

    # ------------------------------------------------------------------
    # Streaming message
    # ------------------------------------------------------------------
    async def begin_streaming_message(
        self,
        target: str,
        *,
        reply_to: str | None = None,
        defer_initial: bool = True,
    ) -> StreamingMessage | None:
        """Return a handle that supports rate-limited edits + finalize().

        If `defer_initial` is True (default), no placeholder message is
        created — the first visible message appears on the first `_push()`
        or `finalize()` call.  This pairs with `start_typing()` so that
        only the "X is typing" indicator shows during early generation.

        Returns None if the channel isn't reachable; callers should fall
        back to plain `.send()` in that case.
        """
        if self._client is None:
            return None
        try:
            channel = await self._fetch_channel(int(target))
        except Exception:  # noqa: BLE001
            return None
        if channel is None:
            return None

        reference = None
        if reply_to:
            try:
                discord_mod = _import_discord()
                reference = discord_mod.MessageReference(
                    message_id=int(reply_to),
                    channel_id=int(target),
                    fail_if_not_exists=False,
                )
            except Exception:  # noqa: BLE001
                pass

        if defer_initial:
            return StreamingMessage(channel=channel, reference=reference)

        # Legacy: send a placeholder immediately.
        try:
            msg = await channel.send("…", reference=reference)
        except Exception:  # noqa: BLE001
            log.exception("discord.begin_streaming_message: initial send failed")
            return None
        return StreamingMessage(channel=channel, first_message=msg)

    # ------------------------------------------------------------------
    # Slash-command registration
    # ------------------------------------------------------------------
    _EXPECTED_SLASH_COMMANDS = frozenset(
        {
            "status",
            "models",
            "model",
            "settings",
            "think",
            "reasoning",
            "new",
            "dump",
            "approvals",
            "approve",
            "reject",
            "stop",
            "restart",
            "help",
        }
    )

    @staticmethod
    def _slash_command_names(commands: Any) -> str:
        names = sorted(getattr(cmd, "name", "?") for cmd in commands)
        return ", ".join(names)

    def _verify_slash_sync(self, scope: str, synced: Any) -> None:
        got = {getattr(cmd, "name", "?") for cmd in synced}
        missing = self._EXPECTED_SLASH_COMMANDS - got
        if missing:
            log.error(
                "discord.tree.sync INCOMPLETE scope=%s missing=[%s] got=[%s]",
                scope,
                ", ".join(sorted(missing)),
                self._slash_command_names(synced),
            )
            return
        log.info(
            "discord.tree.synced count=%d scope=%s names=[%s]",
            len(synced),
            scope,
            self._slash_command_names(synced),
        )

    async def _sync_slash_commands(self, discord_mod: Any) -> None:
        """Push the local command tree to Discord and verify names.

        Guild sync runs first (instant in configured servers).  Each guild
        pass clears stale command IDs from earlier renames, copies the
        current global tree, then pushes.  Global sync follows so DMs and
        other servers see the same set — failures there must not block guild
        restore.
        """
        if self._tree is None:
            return

        guild_ids = list(self.config.guilds.keys())

        for gid in guild_ids:
            try:
                g_obj = discord_mod.Object(id=int(gid))
                self._tree.clear_commands(guild=g_obj)
                self._tree.copy_global_to(guild=g_obj)
                gsynced = await self._tree.sync(guild=g_obj)
                self._verify_slash_sync(f"guild:{gid}", gsynced)
            except Exception:  # noqa: BLE001
                log.exception("discord.tree.sync guild=%s failed", gid)

        try:
            synced = await self._tree.sync()
            self._verify_slash_sync("global", synced)
        except Exception:  # noqa: BLE001
            log.exception("discord.tree.sync global failed")
            if guild_ids:
                log.warning(
                    "discord.tree.sync global failed but guild sync ran — "
                    "slash commands should still work in configured guilds"
                )

    def _register_slash_commands(self, tree: Any, discord_mod: Any) -> None:
        """Register native Discord slash commands on the CommandTree.

        Each command builds a synthetic call into the slash-handler
        callback (wired by the runtime via `set_slash_handler`). The
        callback returns reply text which we forward via interaction
        response / followup.
        """
        from ..envelope import Principal  # local import: avoid cycles

        adapter = self  # closure

        async def _run(
            interaction: Any,
            name: str,
            args: dict,
            *,
            thinking: bool = False,
        ) -> None:
            user = interaction.user
            uid = str(getattr(user, "id", ""))
            trusted = uid in adapter._allowed_users
            tier: Literal["owner", "admin", "user", "guest"] = (
                "owner" if (adapter.owner_id and uid == adapter.owner_id) else "user"
            )

            # Reject unauthorized users early with an ephemeral reply.
            if not trusted:
                try:
                    await interaction.response.send_message("Not authorized.", ephemeral=True)
                except Exception:  # noqa: BLE001
                    log.exception("slash.auth_reject_failed")
                return

            if adapter._slash_handler is None:
                await interaction.response.send_message(
                    "Command handler not ready.", ephemeral=True
                )
                return

            # Defer so we have up to 15 min to respond (most commands
            # are instant, but /model, /new, /credits, etc. may touch state).
            deferred = False
            try:
                await interaction.response.defer(thinking=thinking, ephemeral=False)
                deferred = True
            except Exception:  # noqa: BLE001
                log.exception("slash.defer_failed cmd=/%s", name)
                if not interaction.response.is_done():
                    try:
                        await interaction.response.send_message(
                            "Command failed to start — try again in a few seconds.",
                            ephemeral=True,
                        )
                    except Exception:  # noqa: BLE001
                        log.exception("slash.defer_fallback_failed cmd=/%s", name)
                return

            principal = Principal(
                id=uid,
                label=getattr(user, "display_name", None) or getattr(user, "name", "") or uid,
                trust=trusted,
                tier=tier,
            )
            ctx = {
                "channel_id": str(getattr(getattr(interaction, "channel", None), "id", "") or ""),
                "guild_id": str(getattr(getattr(interaction, "guild", None), "id", "") or "")
                or None,
                "interaction_id": str(getattr(interaction, "id", "")),
            }
            try:
                result = await adapter._slash_handler(name, args, principal, ctx)
            except Exception as e:  # noqa: BLE001
                log.exception("slash.handler crashed: /%s", name)
                result = {"reply": f"Command `/{name}` failed: {e}"}

            reply = (result or {}).get("reply") if isinstance(result, dict) else str(result or "")
            post_send = (result or {}).get("post_send") if isinstance(result, dict) else None
            reply = (reply or "").strip() or "(no output)"
            chunks = _split_discord_message(reply)

            delivered = False
            try:
                for i, chunk in enumerate(chunks):
                    if deferred or interaction.response.is_done():
                        await interaction.followup.send(chunk)
                    elif i == 0:
                        await interaction.response.send_message(chunk)
                    else:
                        await interaction.followup.send(chunk)
                delivered = True
            except Exception:  # noqa: BLE001
                log.exception("slash.reply_failed cmd=/%s", name)

            # Run post-send effects (e.g. shutdown, restart) AFTER reply.
            if post_send is not None and delivered:
                try:
                    await post_send()
                except Exception:  # noqa: BLE001
                    log.exception("slash.post_send failed")
            elif post_send is not None:
                log.error("slash.post_send suppressed after failed reply cmd=/%s", name)

        # ---- command definitions ----
        # NOTE: discord.py's @tree.command decorator uses the function
        # signature to define options. We keep signatures aligned with
        # the text-mode commands in norax/commands.

        @tree.command(name="status", description="Show Norax runtime status.")
        async def _status(interaction: Any) -> None:
            await _run(interaction, "status", {})

        async def _interaction_auth(interaction: Any) -> tuple[Any, str, bool, bool] | None:
            """Return (user, uid, trusted, is_owner) or None if rejected."""
            user = interaction.user
            uid = str(getattr(user, "id", ""))
            trusted = uid in adapter._allowed_users
            if not trusted:
                try:
                    await interaction.response.send_message("Not authorized.", ephemeral=True)
                except Exception:  # noqa: BLE001
                    pass
                return None
            if adapter._slash_handler is None:
                await interaction.response.send_message(
                    "Command handler not ready.", ephemeral=True
                )
                return None
            is_owner = bool(adapter.owner_id and uid == adapter.owner_id)
            return user, uid, trusted, is_owner

        def _principal_ctx(user: Any, uid: str, trusted: bool, is_owner: bool, interaction: Any):
            return (
                Principal(
                    id=uid,
                    label=(getattr(user, "display_name", None) or getattr(user, "name", "") or uid),
                    trust=trusted,
                    tier="owner" if is_owner else "user",
                ),
                {
                    "channel_id": str(
                        getattr(getattr(interaction, "channel", None), "id", "") or ""
                    ),
                    "guild_id": (
                        str(getattr(getattr(interaction, "guild", None), "id", "") or "") or None
                    ),
                    "interaction_id": str(getattr(interaction, "id", "")),
                },
            )

        async def _current_model_from_handler(principal: Any, ctx: dict) -> str:
            if adapter._slash_handler is None:
                return ""
            try:
                result = await adapter._slash_handler("model", {"args": []}, principal, ctx)
            except Exception:  # noqa: BLE001
                return ""
            if isinstance(result, dict):
                data = result.get("data")
                if isinstance(data, dict) and data.get("model"):
                    return str(data["model"])
                import re as _re

                m = _re.search(r"`([^`]+)`", (result.get("reply") or ""))
                if m:
                    return m.group(1)
            return ""

        async def _model_state_from_handler(principal: Any, ctx: dict) -> tuple[str, dict]:
            if adapter._slash_handler is not None:
                try:
                    result = await adapter._slash_handler("settings", {"args": []}, principal, ctx)
                    if isinstance(result, dict) and isinstance(result.get("data"), dict):
                        data = result["data"]
                        return str(data.get("model") or ""), data.get("model_catalog") or {}
                except Exception:  # noqa: BLE001
                    log.debug("model_catalog.fetch_failed", exc_info=True)
            return await _current_model_from_handler(principal, ctx), {}

        @tree.command(name="models", description="Show model selector dropdown.")
        async def _models(interaction: Any) -> None:
            auth = await _interaction_auth(interaction)
            if auth is None:
                return
            user, uid, trusted, is_owner = auth
            if adapter._slash_handler is None:
                await interaction.response.send_message(
                    "Command handler not ready.", ephemeral=True
                )
                return

            try:
                await interaction.response.defer(ephemeral=False, thinking=True)
            except Exception:  # noqa: BLE001
                log.debug("models.defer_failed", exc_info=True)

            principal, ctx = _principal_ctx(user, uid, trusted, is_owner, interaction)
            current_model, model_catalog = await _model_state_from_handler(principal, ctx)

            from .discord_settings import build_model_selector_view

            content, view = build_model_selector_view(
                discord_mod,
                handler=adapter._slash_handler,
                principal=principal,
                ctx=ctx,
                current_model=current_model,
                is_owner=is_owner,
                model_catalog=model_catalog,
            )
            try:
                await interaction.followup.send(content, view=view)
            except Exception:  # noqa: BLE001
                log.exception("models_selector.send failed")

        @tree.command(name="model", description="Show or set the default model.")
        @discord_mod.app_commands.describe(
            id="Model id to switch to (omit for dropdown selector). Owner only."
        )
        async def _model(interaction: Any, id: str | None = None) -> None:
            if id:
                await _run(interaction, "model", {"args": [id]})
                return

            auth = await _interaction_auth(interaction)
            if auth is None:
                return
            user, uid, trusted, is_owner = auth
            if adapter._slash_handler is None:
                await interaction.response.send_message(
                    "Command handler not ready.", ephemeral=True
                )
                return

            # Defer immediately — building the selector view can exceed
            # Discord's 3s ack window and trigger 10062 Unknown interaction.
            try:
                await interaction.response.defer(ephemeral=False, thinking=True)
            except Exception:  # noqa: BLE001
                log.debug("model.defer_failed", exc_info=True)

            principal, ctx = _principal_ctx(user, uid, trusted, is_owner, interaction)
            current_model, model_catalog = await _model_state_from_handler(principal, ctx)

            from .discord_settings import build_model_selector_view

            content, view = build_model_selector_view(
                discord_mod,
                handler=adapter._slash_handler,
                principal=principal,
                ctx=ctx,
                current_model=current_model,
                is_owner=is_owner,
                model_catalog=model_catalog,
            )
            try:
                await interaction.followup.send(content, view=view)
            except Exception:  # noqa: BLE001
                log.exception("model_selector.send failed")

        @tree.command(
            name="settings",
            description="Norax control panel — model, performance, and Discord output.",
        )
        async def _settings(interaction: Any) -> None:
            auth = await _interaction_auth(interaction)
            if auth is None:
                return
            user, uid, trusted, is_owner = auth
            if adapter._slash_handler is None:
                await interaction.response.send_message(
                    "Command handler not ready.", ephemeral=True
                )
                return

            try:
                await interaction.response.defer(ephemeral=False, thinking=True)
            except Exception:  # noqa: BLE001
                log.debug("settings.defer_failed", exc_info=True)

            principal, ctx = _principal_ctx(user, uid, trusted, is_owner, interaction)

            from .discord_settings import (
                build_settings_view,
                fetch_settings_snapshot,
                settings_payload,
            )

            snap = await fetch_settings_snapshot(adapter._slash_handler, principal, ctx)
            view = build_settings_view(
                discord_mod,
                handler=adapter._slash_handler,
                principal=principal,
                ctx=ctx,
                snap=snap,
                is_owner=is_owner,
            )
            payload = settings_payload(snap, page="main", discord_mod=discord_mod)
            try:
                await interaction.followup.send(
                    content=payload.get("content"),
                    embed=payload.get("embed"),
                    view=view,
                )
            except Exception:  # noqa: BLE001
                log.exception("settings_panel.send failed")

        @tree.command(name="think", description="Show or set reasoning effort. (Owner only to set)")
        @discord_mod.app_commands.describe(level="off, low, medium, high, or xhigh")
        async def _think(interaction: Any, level: str | None = None) -> None:
            await _run(interaction, "think", {"args": [level] if level else []})

        @tree.command(
            name="reasoning",
            description="Show or toggle visible reasoning output. (Owner only to set)",
        )
        @discord_mod.app_commands.describe(state="on or off")
        async def _reasoning(interaction: Any, state: str | None = None) -> None:
            await _run(interaction, "reasoning", {"args": [state] if state else []})

        @tree.command(name="new", description="Reset this channel's conversation context.")
        async def _new(interaction: Any) -> None:
            await _run(interaction, "new", {})

        @tree.command(
            name="dump", description="Dump context to sleep. Default full; use half for older 50%."
        )
        @discord_mod.app_commands.describe(mode="full or half")
        async def _dump(interaction: Any, mode: str | None = None) -> None:
            await _run(interaction, "dump", {"args": [mode] if mode else []})

        @tree.command(name="approvals", description="List pending tool approvals. (Owner only)")
        async def _approvals(interaction: Any) -> None:
            await _run(interaction, "approvals", {})

        @tree.command(name="approve", description="Approve one exact tool call. (Owner only)")
        @discord_mod.app_commands.describe(id="Interrupt ID shown by the agent")
        async def _approve(interaction: Any, id: str) -> None:
            await _run(interaction, "approve", {"args": [id]})

        @tree.command(name="reject", description="Reject one pending tool call. (Owner only)")
        @discord_mod.app_commands.describe(id="Interrupt ID shown by the agent")
        async def _reject(interaction: Any, id: str) -> None:
            await _run(interaction, "reject", {"args": [id]})

        @tree.command(
            name="stop", description="Cancel the current stream / active turn. (Owner only)"
        )
        async def _stop(interaction: Any) -> None:
            await _run(interaction, "stop", {})

        @tree.command(name="restart", description="Gracefully restart Norax. (Owner only)")
        async def _restart(interaction: Any) -> None:
            await _run(interaction, "restart", {})

        @tree.command(name="help", description="List Norax commands.")
        async def _help(interaction: Any) -> None:
            await _run(interaction, "help", {})

        log.info("discord.slash commands registered: %d", len(self._EXPECTED_SLASH_COMMANDS))
