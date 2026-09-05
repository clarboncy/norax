"""Phase 8 — Discord channel adapter, OutboundRegistry, reply-tag parser.

No live network: `DiscordInAdapter` is constructed without ever calling
`start()`. Gate tests call `_is_allowed`/`_envelope_from` with fake
objects that look like `discord.Message`.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from norax.adapter.discord_in import DiscordInAdapter
from norax.adapter.reply_tag import parse_reply_tag
from norax.config.loader import (
    DiscordChannelPolicy,
    DiscordConfig,
    DiscordGuildPolicy,
    _parse_discord,
)
from norax.dispatch import tools as dispatch_tools
from norax.runtime.outbound import OutboundRegistry


# ---------------------------------------------------------------------------
# Reply-tag parser
# ---------------------------------------------------------------------------
class TestReplyTag:
    def test_no_tag_passthrough(self):
        r = parse_reply_tag("hello world", current_message_id="123")
        assert r.text == "hello world"
        assert r.reply_to is None
        assert r.had_tag is False

    def test_reply_to_current_resolves_to_current_id(self):
        r = parse_reply_tag("[[reply_to_current]] hi", current_message_id="abc")
        assert r.text == "hi"
        assert r.reply_to == "abc"
        assert r.had_tag is True

    def test_reply_to_current_without_id(self):
        r = parse_reply_tag("[[reply_to_current]] hi", current_message_id=None)
        assert r.text == "hi"
        assert r.reply_to is None
        assert r.had_tag is True

    def test_reply_to_explicit_id(self):
        r = parse_reply_tag("[[reply_to:12345]] hello", current_message_id="abc")
        assert r.text == "hello"
        assert r.reply_to == "12345"
        assert r.had_tag is True

    def test_reply_to_explicit_with_whitespace(self):
        r = parse_reply_tag("[[ reply_to : 12345 ]] hello")
        assert r.text == "hello"
        assert r.reply_to == "12345"
        assert r.had_tag is True

    def test_tag_mid_string_ignored(self):
        r = parse_reply_tag("prefix [[reply_to_current]] body", current_message_id="x")
        assert r.had_tag is False
        assert r.text == "prefix [[reply_to_current]] body"

    def test_empty_string(self):
        r = parse_reply_tag("", current_message_id="x")
        assert r.had_tag is False
        assert r.text == ""


# ---------------------------------------------------------------------------
# Slash command registration
# ---------------------------------------------------------------------------
class TestDiscordSlashCommands:
    def test_think_and_reasoning_registered_native_discord_commands(self):
        registered: list[tuple[str, str]] = []

        class _FakeTree:
            def command(self, *, name, description):
                def deco(fn):
                    registered.append((name, description))
                    return fn

                return deco

        class _FakeAppCommands:
            @staticmethod
            def describe(**_kwargs):
                def deco(fn):
                    return fn

                return deco

        fake_discord = SimpleNamespace(app_commands=_FakeAppCommands())
        cfg = DiscordConfig(enabled=True, token="fake")
        adapter = DiscordInAdapter(config=cfg, owner_id="111")

        adapter._register_slash_commands(_FakeTree(), fake_discord)

        names = {name for name, _desc in registered}
        assert "think" in names
        assert "reasoning" in names
        assert "settings" in names
        assert {"approvals", "approve", "reject"} <= names
        assert "restart" in names
        assert names == DiscordInAdapter._EXPECTED_SLASH_COMMANDS


class TestDiscordSlashSync:
    @staticmethod
    def _fake_synced_commands():
        return [SimpleNamespace(name=n) for n in DiscordInAdapter._EXPECTED_SLASH_COMMANDS]

    async def test_guild_sync_copies_global_before_push(self):
        calls: list[str] = []

        class _FakeTree:
            def clear_commands(self, guild):
                calls.append(f"clear:{getattr(guild, 'id', guild)}")

            def copy_global_to(self, guild):
                calls.append(f"copy:{getattr(guild, 'id', guild)}")

            async def sync(self, *, guild=None):
                if guild is not None:
                    calls.append(f"sync:guild:{getattr(guild, 'id', guild)}")
                else:
                    calls.append("sync:global")
                return TestDiscordSlashSync._fake_synced_commands()

        fake_discord = SimpleNamespace(Object=lambda id: SimpleNamespace(id=id))
        cfg = DiscordConfig(
            enabled=True,
            token="fake",
            guilds={"123456789012345678": DiscordGuildPolicy(require_mention=True)},
        )
        adapter = DiscordInAdapter(config=cfg, owner_id="111")
        adapter._tree = _FakeTree()

        await adapter._sync_slash_commands(fake_discord)

        assert calls == [
            "clear:123456789012345678",
            "copy:123456789012345678",
            "sync:guild:123456789012345678",
            "sync:global",
        ]

    async def test_global_sync_when_no_guilds(self):
        calls: list[str] = []

        class _FakeTree:
            async def sync(self, *, guild=None):
                assert guild is None
                calls.append("sync:global")
                return TestDiscordSlashSync._fake_synced_commands()

        fake_discord = SimpleNamespace(Object=lambda id: SimpleNamespace(id=id))
        cfg = DiscordConfig(enabled=True, token="fake")
        adapter = DiscordInAdapter(config=cfg, owner_id="111")
        adapter._tree = _FakeTree()

        await adapter._sync_slash_commands(fake_discord)

        assert calls == ["sync:global"]


# ---------------------------------------------------------------------------
# Config parser
# ---------------------------------------------------------------------------
class TestDiscordConfigParser:
    def test_empty_defaults(self, monkeypatch):
        monkeypatch.delenv("NORAX_DISCORD_CONFIG_JSON", raising=False)
        monkeypatch.delenv("NORAX_DISCORD_TOKEN", raising=False)
        monkeypatch.delenv("NORAX_DISCORD_ENABLED", raising=False)
        c = _parse_discord({})
        assert c.enabled is False
        assert c.token is None
        assert c.allow_bots is False
        assert c.dm_policy == "deny"
        assert c.group_policy == "deny"
        assert c.guilds == {}

    def test_full_parse(self, monkeypatch):
        monkeypatch.delenv("NORAX_DISCORD_CONFIG_JSON", raising=False)
        monkeypatch.delenv("NORAX_DISCORD_TOKEN", raising=False)
        monkeypatch.delenv("NORAX_DISCORD_ENABLED", raising=False)
        raw = {
            "enabled": True,
            "token": "abc",
            "allowBots": True,
            "dmPolicy": "allowlist",
            "groupPolicy": "allowlist",
            "allowFrom": ["111"],
            "guilds": {
                "999": {
                    "requireMention": True,
                    "users": ["111", "222"],
                    "channels": {
                        "42": {"requireMention": False, "enabled": True},
                        "43": {"requireMention": True, "enabled": False},
                    },
                }
            },
        }
        c = _parse_discord(raw)
        assert c.enabled is True
        assert c.token == "abc"
        assert c.dm_policy == "allowlist"
        assert "999" in c.guilds
        g = c.guilds["999"]
        assert g.require_mention is True
        assert g.users == {"111", "222"}
        assert g.channels["42"].enabled is True
        assert g.channels["42"].require_mention is False
        assert g.channels["43"].enabled is False
        assert c.all_allowed_users == {"111", "222"}

    def test_env_token_overrides_inline(self, monkeypatch):
        monkeypatch.setenv("NORAX_DISCORD_TOKEN", "env-token")
        c = _parse_discord({"token": "inline"})
        assert c.token == "env-token"

    def test_blank_token_becomes_none(self, monkeypatch):
        monkeypatch.delenv("NORAX_DISCORD_TOKEN", raising=False)
        c = _parse_discord({"token": "   "})
        assert c.token is None

    def test_string_false_values_do_not_bypass_boolean_gates(self, monkeypatch):
        monkeypatch.delenv("NORAX_DISCORD_ENABLED", raising=False)
        c = _parse_discord(
            {
                "enabled": "false",
                "allowBots": "false",
                "guilds": {
                    "999": {
                        "requireMention": "false",
                        "users": ["111"],
                        "channels": {"42": {"enabled": "false", "requireMention": "false"}},
                    }
                },
            }
        )

        assert c.enabled is False
        assert c.allow_bots is False
        assert c.guilds["999"].require_mention is False
        assert c.guilds["999"].channels["42"].enabled is False
        assert c.guilds["999"].channels["42"].require_mention is False

    def test_invalid_policy_and_string_allowlists_fail_closed(self, monkeypatch):
        monkeypatch.delenv("NORAX_DISCORD_ENABLED", raising=False)
        c = _parse_discord(
            {
                "dmPolicy": "anything-goes",
                "groupPolicy": "public",
                "allowFrom": "123",
                "guilds": {"999": {"users": "111"}},
            }
        )

        assert c.dm_policy == "deny"
        assert c.group_policy == "deny"
        assert c.allow_from == set()
        assert c.guilds["999"].users == set()

    def test_unbounded_or_malformed_policy_maps_are_rejected(self, monkeypatch):
        monkeypatch.delenv("NORAX_DISCORD_CONFIG_JSON", raising=False)
        with pytest.raises(ValueError, match="at most 256 guilds"):
            _parse_discord({"guilds": {str(index): {} for index in range(257)}})
        with pytest.raises(ValueError, match="must be an object"):
            _parse_discord({"guilds": {"999": {"channels": {"42": True}}}})

    def test_discord_token_is_bounded_and_printable(self, monkeypatch):
        monkeypatch.delenv("NORAX_DISCORD_TOKEN", raising=False)
        with pytest.raises(ValueError, match="Discord token"):
            _parse_discord({"token": "x" * 4_097})
        with pytest.raises(ValueError, match="Discord token"):
            _parse_discord({"token": "valid-prefix\nsecret"})


class TestDiscordLifecycle:
    @pytest.mark.asyncio
    async def test_terminal_client_failure_reaches_event_stream(self):
        async def fail() -> None:
            await asyncio.sleep(0)
            raise ConnectionError("gateway task died")

        adapter = DiscordInAdapter(config=_cfg_basic())
        adapter._task = asyncio.create_task(fail())

        with pytest.raises(RuntimeError, match="Discord client failed") as caught:
            await anext(adapter.events())

        assert isinstance(caught.value.__cause__, ConnectionError)

    @pytest.mark.asyncio
    async def test_fatal_login_failure_is_not_reported_as_clean_exit(self, monkeypatch):
        class LoginFailure(Exception):
            pass

        class Client:
            async def start(self, _token):
                raise LoginFailure("bad token")

        monkeypatch.setattr(
            "norax.adapter.discord_in._import_discord",
            lambda: SimpleNamespace(
                LoginFailure=LoginFailure,
                PrivilegedIntentsRequired=type("PrivilegedIntentsRequired", (Exception,), {}),
            ),
        )
        adapter = DiscordInAdapter(config=_cfg_basic())

        with pytest.raises(LoginFailure, match="bad token"):
            await adapter._run_client_supervised(Client())

        assert adapter.connection_state["connected"] is False
        assert "LoginFailure: bad token" in adapter.connection_state["last_error"]

    @pytest.mark.asyncio
    async def test_stop_settles_supervisor_and_clears_owned_client(self, monkeypatch):
        closed = asyncio.Event()

        class Client:
            user = None

            async def start(self, _token):
                await closed.wait()

            async def close(self):
                closed.set()

        adapter = DiscordInAdapter(config=_cfg_basic())
        client = Client()
        monkeypatch.setattr(adapter, "_build_client", lambda: client)

        await adapter.start()
        with pytest.raises(RuntimeError, match="only be called once"):
            await adapter.start()
        await adapter.stop()

        assert adapter._task is None
        assert adapter._client is None
        assert adapter._tree is None


# ---------------------------------------------------------------------------
# Discord inbound gate
# ---------------------------------------------------------------------------
def _fake_message(
    *,
    author_id="111",
    author_bot=False,
    guild_id="999",
    channel_id="42",
    channel_type="text",
    content="hi",
    message_id="55",
    mentions=None,
):
    author = SimpleNamespace(id=author_id, name="u", display_name="User", bot=author_bot)
    channel = SimpleNamespace(
        id=channel_id,
        type=SimpleNamespace(name=channel_type),
    )
    guild = SimpleNamespace(id=guild_id) if guild_id else None
    return SimpleNamespace(
        author=author,
        channel=channel,
        guild=guild,
        mentions=mentions or [],
        content=content,
        created_at=None,
        attachments=[],
        id=message_id,
    )


def _cfg_basic(**overrides) -> DiscordConfig:
    base = DiscordConfig(
        enabled=True,
        token="t",
        allow_bots=False,
        dm_policy="allowlist",
        group_policy="allowlist",
        allow_from=set(),
        guilds={
            "999": DiscordGuildPolicy(
                require_mention=False,
                users={"111"},
                channels={"42": DiscordChannelPolicy(require_mention=False, enabled=True)},
            ),
        },
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


class TestGate:
    def test_allow_guild_listed(self):
        a = DiscordInAdapter(config=_cfg_basic(), owner_id="111")
        ok, reason = a._is_allowed(_fake_message())
        assert ok, reason

    def test_reject_bot(self):
        a = DiscordInAdapter(config=_cfg_basic())
        ok, reason = a._is_allowed(_fake_message(author_bot=True))
        assert not ok and reason == "bot_message"

    def test_allow_bot_when_configured(self):
        cfg = _cfg_basic()
        cfg.allow_bots = True
        a = DiscordInAdapter(config=cfg)
        ok, _ = a._is_allowed(_fake_message(author_bot=True))
        assert ok

    def test_reject_wrong_guild(self):
        a = DiscordInAdapter(config=_cfg_basic())
        ok, reason = a._is_allowed(_fake_message(guild_id="777"))
        assert not ok and reason == "guild_not_allowlisted"

    def test_reject_wrong_user(self):
        a = DiscordInAdapter(config=_cfg_basic())
        ok, reason = a._is_allowed(_fake_message(author_id="222"))
        assert not ok and reason == "user_not_allowlisted"

    def test_require_mention_without_mention_rejected(self):
        cfg = _cfg_basic()
        cfg.guilds["999"].require_mention = True
        cfg.guilds["999"].channels = {}  # fall back to guild default
        a = DiscordInAdapter(config=cfg)
        a._client = SimpleNamespace(user=SimpleNamespace(id="bot123"))
        ok, reason = a._is_allowed(_fake_message())
        assert not ok and reason == "missing_mention"

    def test_require_mention_with_mention_accepted(self):
        cfg = _cfg_basic()
        cfg.guilds["999"].require_mention = True
        cfg.guilds["999"].channels = {}
        a = DiscordInAdapter(config=cfg)
        a._client = SimpleNamespace(user=SimpleNamespace(id="bot123"))
        msg = _fake_message(mentions=[SimpleNamespace(id="bot123")])
        ok, _ = a._is_allowed(msg)
        assert ok

    def test_require_mention_via_content_fallback(self):
        cfg = _cfg_basic()
        cfg.guilds["999"].require_mention = True
        cfg.guilds["999"].channels = {}
        a = DiscordInAdapter(config=cfg)
        a._client = SimpleNamespace(user=SimpleNamespace(id="bot123"))
        msg = _fake_message(content="<@bot123> hi")
        ok, _ = a._is_allowed(msg)
        assert ok

    def test_channel_level_require_mention_overrides_guild(self):
        cfg = _cfg_basic()
        cfg.guilds["999"].require_mention = True  # guild says yes
        cfg.guilds["999"].channels = {
            "42": DiscordChannelPolicy(require_mention=False, enabled=True)
        }
        a = DiscordInAdapter(config=cfg)
        ok, _ = a._is_allowed(_fake_message())
        assert ok

    def test_disabled_channel_rejected(self):
        cfg = _cfg_basic()
        cfg.guilds["999"].channels = {
            "42": DiscordChannelPolicy(require_mention=False, enabled=False)
        }
        a = DiscordInAdapter(config=cfg)
        ok, reason = a._is_allowed(_fake_message())
        assert not ok and reason == "channel_disabled"

    def test_dm_allowlist_accept(self):
        cfg = _cfg_basic()
        a = DiscordInAdapter(config=cfg)
        ok, _ = a._is_allowed(_fake_message(guild_id=None))
        assert ok

    def test_dm_deny_policy(self):
        cfg = _cfg_basic()
        cfg.dm_policy = "deny"
        a = DiscordInAdapter(config=cfg)
        ok, reason = a._is_allowed(_fake_message(guild_id=None))
        assert not ok and reason == "dm_policy_deny"

    def test_dm_not_in_allowlist(self):
        cfg = _cfg_basic()
        a = DiscordInAdapter(config=cfg)
        ok, reason = a._is_allowed(_fake_message(guild_id=None, author_id="222"))
        assert not ok and reason == "dm_not_allowlisted"


# ---------------------------------------------------------------------------
# Envelope construction
# ---------------------------------------------------------------------------
class TestEnvelope:
    def test_owner_tier_assigned(self):
        a = DiscordInAdapter(config=_cfg_basic(), owner_id="111")
        env = a._envelope_from(_fake_message(author_id="111"))
        assert env.sender.tier == "owner"
        assert env.sender.trust is True
        assert env.trusted is True
        assert env.channel == "chat"
        assert env.source == "discord"
        assert env.raw["channel_id"] == "42"
        assert env.raw["guild_id"] == "999"

    def test_non_owner_user_tier(self):
        cfg = _cfg_basic()
        cfg.guilds["999"].users.add("222")
        a = DiscordInAdapter(config=cfg, owner_id="111")
        env = a._envelope_from(_fake_message(author_id="222"))
        assert env.sender.tier == "user"
        assert env.sender.trust is True  # in allowlist

    def test_untrusted_when_not_in_allowlist(self):
        cfg = _cfg_basic()
        a = DiscordInAdapter(config=cfg, owner_id="111")
        # gate would reject this, but envelope build still should work
        env = a._envelope_from(_fake_message(author_id="999"))
        assert env.sender.trust is False
        assert env.trusted is False

    def test_thread_binding_on_thread(self):
        a = DiscordInAdapter(config=_cfg_basic())
        env = a._envelope_from(_fake_message(channel_type="public_thread"))
        assert env.thread_binding is not None
        assert env.thread_binding.channel == "discord"
        assert env.thread_binding.thread_id == "42"

    def test_no_thread_binding_on_text_channel(self):
        a = DiscordInAdapter(config=_cfg_basic())
        env = a._envelope_from(_fake_message(channel_type="text"))
        assert env.thread_binding is None


# ---------------------------------------------------------------------------
# OutboundRegistry + live t_message_send
# ---------------------------------------------------------------------------
class _FakeOut:
    def __init__(self):
        self.calls = []

    async def send(self, target, text, *, reply_to=None, files=None, **_):
        self.calls.append({"target": target, "text": text, "reply_to": reply_to, "files": files})
        return {"ok": True, "message_id": "new-id", "target": target}


class TestOutbound:
    def test_unbound_tool_does_not_claim_delivery(self):
        out = asyncio.run(dispatch_tools.t_message_send(channel="discord", target="42", text="hi"))
        assert out["ok"] is False
        assert out["queued"] is False
        assert out["error"] == "outbound_not_bound"

    def test_register_and_get(self):
        reg = OutboundRegistry()
        a = _FakeOut()
        reg.register("discord", a)
        assert reg.get("discord") is a
        assert reg.has("discord")
        assert "discord" in reg.channels()

    def test_send_unregistered(self):
        reg = OutboundRegistry()
        result = asyncio.run(reg.send("nowhere", "t", "hi"))
        assert result["ok"] is False
        assert result["error"] == "channel_not_registered"

    def test_send_dispatches(self):
        reg = OutboundRegistry()
        a = _FakeOut()
        reg.register("discord", a)
        result = asyncio.run(reg.send("discord", "42", "hi", reply_to="5"))
        assert result["ok"] is True
        assert a.calls == [{"target": "42", "text": "hi", "reply_to": "5", "files": None}]

    def test_send_wraps_exceptions(self):
        class Boom:
            async def send(self, *a, **k):
                raise RuntimeError("boom")

        reg = OutboundRegistry()
        reg.register("discord", Boom())
        result = asyncio.run(reg.send("discord", "42", "hi"))
        assert result["ok"] is False
        assert "boom" in result["error"]

    def test_tool_message_send_registered(self):
        reg = OutboundRegistry()
        reg.register("discord", _FakeOut())
        dispatch_tools.bind_outbound(reg)
        fn = dispatch_tools.REGISTRY["message_send"].fn
        out = asyncio.run(fn(channel="discord", target="42", text="hi"))
        assert out["ok"] is True
        # reset to stub
        dispatch_tools.bind_outbound(OutboundRegistry())

    def test_tool_message_send_unregistered(self):
        dispatch_tools.bind_outbound(OutboundRegistry())
        fn = dispatch_tools.REGISTRY["message_send"].fn
        out = asyncio.run(fn(channel="discord", target="42", text="hi"))
        assert out["ok"] is False
        assert out["error"] == "channel_not_registered"

    def test_tool_message_send_file_list_registered(self):
        reg = OutboundRegistry()
        fake = _FakeOut()
        reg.register("discord", fake)
        dispatch_tools.bind_outbound(reg)
        fn = dispatch_tools.REGISTRY["message_send"].fn
        out = asyncio.run(fn(channel="discord", target="42", text="file", files=["/tmp/x.zip"]))
        assert out["ok"] is True
        assert fake.calls[0]["files"] == ["/tmp/x.zip"]
        dispatch_tools.bind_outbound(OutboundRegistry())


# ---------------------------------------------------------------------------
# DiscordInAdapter.send — outbound path (mocked client)
# ---------------------------------------------------------------------------
class _FakeSentMessage:
    def __init__(self, mid="sent-1"):
        self.id = mid


class _FakeChannel:
    def __init__(self, channel_id=42):
        self.id = channel_id
        self.sent = []
        self.reactions = []
        self._messages: dict[int, _FakeFetchedMessage] = {}

    async def send(self, text, **kwargs):
        self.sent.append({"text": text, "kwargs": kwargs})
        return _FakeSentMessage(mid=f"sent-{len(self.sent)}")

    async def fetch_message(self, mid):
        if mid not in self._messages:
            self._messages[mid] = _FakeFetchedMessage(mid, self)
        return self._messages[mid]


class _FakeFetchedMessage:
    def __init__(self, mid, channel):
        self.id = mid
        self._channel = channel

    async def add_reaction(self, emoji):
        self._channel.reactions.append((self.id, emoji))


class _FakeUser:
    def __init__(self, user_id, dm_channel):
        self.id = user_id
        self.dm_channel = dm_channel

    async def create_dm(self):
        return self.dm_channel


class _FakeClient:
    def __init__(self, channel: _FakeChannel, user: _FakeUser | None = None):
        self.user = SimpleNamespace(id="bot-1")
        self._channel = channel
        self._dm_user = user

    def get_channel(self, cid):
        return self._channel if cid == self._channel.id else None

    def get_user(self, uid):
        return self._dm_user if self._dm_user and uid == self._dm_user.id else None


class _FakeDiscordForReply:
    class MessageReference:
        def __init__(self, message_id, channel_id, fail_if_not_exists=False):
            self.message_id = message_id
            self.channel_id = channel_id
            self.fail_if_not_exists = fail_if_not_exists


class TestDiscordSend:
    def _adapter(self, *, with_client=True):
        a = DiscordInAdapter(config=_cfg_basic())
        ch = _FakeChannel(channel_id=42)
        if with_client:
            a._client = _FakeClient(ch)
        return a, ch

    def test_send_no_client_errors(self):
        a = DiscordInAdapter(config=_cfg_basic())
        result = asyncio.run(a.send("42", "hi"))
        assert result["ok"] is False
        assert result["error"] == "client_not_started"

    def test_send_plain(self):
        a, ch = self._adapter()
        result = asyncio.run(a.send("42", "hello"))
        assert result["ok"] is True
        assert ch.sent == [{"text": "hello", "kwargs": {}}]

    def test_send_strips_reply_tag(self, monkeypatch):
        monkeypatch.setattr(
            "norax.adapter.discord_in._import_discord",
            lambda: _FakeDiscordForReply,
        )
        a, ch = self._adapter()
        result = asyncio.run(a.send("42", "[[reply_to:99]] hi there", reply_to=None))
        assert result["ok"] is True
        assert ch.sent[0]["text"] == "hi there"
        ref = ch.sent[0]["kwargs"].get("reference")
        # Either a real MessageReference or dict-like — we inspect by repr.
        assert ref is not None

    def test_send_reply_to_current_uses_reply_to_arg(self, monkeypatch):
        monkeypatch.setattr(
            "norax.adapter.discord_in._import_discord",
            lambda: _FakeDiscordForReply,
        )
        a, ch = self._adapter()
        result = asyncio.run(a.send("42", "[[reply_to_current]] hi", reply_to="55"))
        assert result["ok"] is True
        assert ch.sent[0]["text"] == "hi"
        # Reference was built with message_id=55? Check by presence.
        assert ch.sent[0]["kwargs"].get("reference") is not None

    def test_send_empty_after_strip(self):
        a, _ = self._adapter()
        result = asyncio.run(a.send("42", "[[reply_to_current]]", reply_to="1"))
        assert result["ok"] is False
        assert result["error"] == "empty_body_after_tag_strip"

    def test_send_reaction_path(self):
        a, ch = self._adapter()
        result = asyncio.run(a.send("42", "", react_to="100", emoji="👍"))
        assert result["ok"] is True
        assert result["reacted"] == "100"
        assert (100, "👍") in ch.reactions

    def test_send_target_not_found(self):
        a = DiscordInAdapter(config=_cfg_basic())
        a._client = _FakeClient(_FakeChannel(channel_id=1))
        result = asyncio.run(a.send("9999", "hi"))
        assert result["ok"] is False
        assert result["error"] == "target_not_found"

    def test_send_with_file_attachment(self, tmp_path, monkeypatch):
        class FakeDiscord:
            class File:
                def __init__(self, path, filename=None):
                    self.path = path
                    self.filename = filename

        monkeypatch.setattr("norax.adapter.discord_in._import_discord", lambda: FakeDiscord)
        f = tmp_path / "bundle.zip"
        f.write_bytes(b"zip")
        a, ch = self._adapter()
        result = asyncio.run(a.send("42", "attached", files=[str(f)]))
        assert result["ok"] is True
        sent_file = ch.sent[0]["kwargs"]["files"][0]
        assert sent_file.filename == "bundle.zip"

    def test_send_file_only_attachment(self, tmp_path, monkeypatch):
        class FakeDiscord:
            class File:
                def __init__(self, path, filename=None):
                    self.path = path
                    self.filename = filename

        monkeypatch.setattr("norax.adapter.discord_in._import_discord", lambda: FakeDiscord)
        f = tmp_path / "bundle.zip"
        f.write_bytes(b"zip")
        a, ch = self._adapter()
        result = asyncio.run(a.send("42", "", files=[str(f)]))
        assert result["ok"] is True
        assert ch.sent[0]["text"] == ""
        assert ch.sent[0]["kwargs"]["files"][0].filename == "bundle.zip"


# ---------------------------------------------------------------------------
# Runtime.build integration (no live Discord connection)
# ---------------------------------------------------------------------------
class TestRuntimeBuildDiscord:
    def test_build_with_discord_disabled(self, tmp_path, monkeypatch):
        from norax.config.loader import Config
        from norax.runtime.core import Runtime

        monkeypatch.delenv("NORAX_DISCORD_TOKEN", raising=False)
        raw = {
            "http": {"bind": "127.0.0.1:0"},
            "owner": {"id": "111", "label": "O"},
            "channels": {"discord": {"enabled": False}},
        }
        cfg = Config(raw=raw, project_root=tmp_path)
        (tmp_path / "state").mkdir()
        (tmp_path / "logs").mkdir()
        rt = Runtime.build(cfg)
        assert rt.outbound.has("discord") is False

    def test_build_with_discord_enabled_no_token_skips(self, tmp_path, monkeypatch, caplog):
        from norax.config.loader import Config
        from norax.runtime.core import Runtime

        monkeypatch.delenv("NORAX_DISCORD_TOKEN", raising=False)
        raw = {
            "http": {"bind": "127.0.0.1:0"},
            "owner": {"id": "111", "label": "O"},
            "channels": {
                "discord": {
                    "enabled": True,
                    "token": None,
                    "groupPolicy": "allowlist",
                    "guilds": {},
                }
            },
        }
        cfg = Config(raw=raw, project_root=tmp_path)
        (tmp_path / "state").mkdir()
        (tmp_path / "logs").mkdir()
        with caplog.at_level("WARNING"):
            rt = Runtime.build(cfg)
        assert rt.outbound.has("discord") is False
        assert any("no token configured" in r.message for r in caplog.records)

    def test_build_with_discord_enabled_and_token_registers(self, tmp_path, monkeypatch):
        from norax.config.loader import Config
        from norax.runtime.core import Runtime

        monkeypatch.setenv("NORAX_DISCORD_TOKEN", "fake-token-for-wiring")
        raw = {
            "http": {"bind": "127.0.0.1:0"},
            "owner": {"id": "111", "label": "O"},
            "channels": {
                "discord": {
                    "enabled": True,
                    "groupPolicy": "allowlist",
                    "guilds": {"999": {"users": ["111"], "requireMention": False}},
                }
            },
        }
        cfg = Config(raw=raw, project_root=tmp_path)
        (tmp_path / "state").mkdir()
        (tmp_path / "logs").mkdir()
        rt = Runtime.build(cfg)
        assert rt.outbound.has("discord") is True
        # Importing DiscordInAdapter must NOT have opened a connection.
        disc = rt.outbound.get("discord")
        assert disc._client is None
        assert disc._task is None
