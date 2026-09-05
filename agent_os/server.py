"""Norax Agent OS — web dashboard + control plane.

Standalone FastAPI service. Serves a single-file UI and proxies to the
real Norax runtime (memory/tools/personality) via WebSocket + HTTP ingress.

Runs locally (127.0.0.1) or externally (bind 0.0.0.0 + token auth).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pty
import secrets
import signal
import smtplib
import sqlite3
import stat
import struct
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from email.message import EmailMessage
from pathlib import Path

import httpx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static" / "index.html"
PROJECT_ROOT = Path(os.environ.get("NORAX_PROJECT_ROOT", BASE.parent)).expanduser().resolve()

TOKEN = os.environ.get("NORAX_OS_TOKEN", "")  # empty = no auth (local mode)
BIND = os.environ.get("NORAX_OS_BIND", "127.0.0.1")
PORT = int(os.environ.get("NORAX_OS_PORT", "8822"))

# Runtime (Norax AI) — the real agent with memory/tools/personality.
# Use the runtime's explicitly configured port unless NORAX_RUNTIME overrides it.
_default_runtime_port = os.environ.get("NORAX_HTTP_PORT", "4101")
RUNTIME_BASE = os.environ.get("NORAX_RUNTIME", f"http://127.0.0.1:{_default_runtime_port}")
RUNTIME_CHAT_TOKEN = os.environ.get("NORAX_RUNTIME_CHAT_TOKEN") or os.environ.get(
    "NORAX_AGENT_OS_CHAT_TOKEN", ""
)

logging.basicConfig(
    level=os.environ.get("NORAX_OS_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("norax.agent_os")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Self-heal the schema before serving; see init_db().
    init_db()
    log.info(
        "agent_os ready bind=%s port=%s auth=%s runtime=%s",
        BIND,
        PORT,
        "token" if TOKEN else "open(local)",
        RUNTIME_BASE,
    )
    yield


app = FastAPI(title="Norax Agent OS", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


def agent_os_state_dir(*, create: bool = False) -> Path:
    """Return the private dashboard state root without polluting source."""
    explicit = os.environ.get("NORAX_AGENT_OS_STATE_DIR", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
    else:
        state_root = os.environ.get("NORAX_STATE_DIR", "").strip()
        if state_root:
            path = Path(state_root).expanduser() / "agent-os"
        else:
            xdg_state = os.environ.get("XDG_STATE_HOME", "").strip()
            root = Path(xdg_state).expanduser() if xdg_state else Path.home() / ".local/state"
            path = root / "norax/agent-os"
    if not path.is_absolute():
        raise ValueError("Agent OS state directory must be an absolute path")
    resolved = path.resolve()
    if create:
        resolved.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            resolved.chmod(0o700)
    return resolved


DB_PATH = agent_os_state_dir() / "agent_os.db"


def _prepare_db_path() -> None:
    """Create or validate the dashboard database as private regular state."""
    parent = DB_PATH.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        parent.chmod(0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(DB_PATH, flags, 0o600)
    except FileExistsError:
        pass
    else:
        os.close(fd)
    file_stat = DB_PATH.lstat()
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise RuntimeError("Agent OS database must be a regular non-symlink file")
    if os.name == "posix":
        if file_stat.st_uid != os.getuid():
            raise RuntimeError("Agent OS database must be owned by the service user")
        DB_PATH.chmod(0o600)


def _connect() -> sqlite3.Connection:
    """Open a hardened connection. WAL + busy_timeout are per-connection state."""
    _prepare_db_path()
    db = sqlite3.connect(DB_PATH, timeout=10.0)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA busy_timeout=5000")
    return db


def init_db() -> None:
    """Create the schema if absent so a lost/moved DB self-heals on boot.

    Previously the schema existed only because it had been created
    out-of-band; losing the file broke every chat write with
    ``no such table: messages``.
    """
    with _connect() as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS chats("
            "  id TEXT PRIMARY KEY, backend TEXT, model TEXT,"
            "  title TEXT, created REAL)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS messages("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT,"
            "  role TEXT, content TEXT, ts REAL)"
        )
        # The dedupe probe filters on (chat_id, ts); without this index it
        # degrades to a full table scan on every persisted message.
        db.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages(chat_id, ts)")


def persist_message(chat_id: str, role: str, content: str, ts: float | None = None) -> bool:
    """Persist one transcript event, suppressing fan-out/reconnect duplicates."""
    now = ts or time.time()
    try:
        with _connect() as db:
            duplicate = db.execute(
                "SELECT 1 FROM messages WHERE chat_id=? AND ts>? AND role=? AND content=? LIMIT 1",
                (chat_id, now - 30, role, content),
            ).fetchone()
            if duplicate:
                return False
            db.execute(
                "INSERT INTO messages(chat_id,role,content,ts) VALUES(?,?,?,?)",
                (chat_id, role, content, now),
            )
        return True
    except sqlite3.Error:
        # Transcript persistence is a projection, never a delivery dependency:
        # a storage fault must not break live chat.
        log.exception("persist_message failed chat_id=%s role=%s", chat_id, role)
        return False


# ---------------------------------------------------------------- auth
def _trusted_peer(host: str | None) -> bool:
    """Trust loopback and Tailscale CGNAT peers; require TOKEN elsewhere."""
    if not host:
        return False
    if host in {"127.0.0.1", "::1"}:
        return True
    try:
        first, second, *_ = (int(part) for part in host.split("."))
        return first == 100 and 64 <= second <= 127
    except (ValueError, TypeError):
        return False


def _token_ok(candidate: str | None) -> bool:
    """Constant-time token comparison (avoids leaking the token byte-by-byte)."""
    if not candidate or not TOKEN:
        return False
    return secrets.compare_digest(candidate, TOKEN)


def auth(req: Request) -> None:
    if not TOKEN or _trusted_peer(req.client.host if req.client else None):
        return
    tok = req.headers.get("x-os-token") or req.query_params.get("token")
    if not _token_ok(tok):
        raise HTTPException(401, "bad token")


# ---------------------------------------------------------------- helpers
async def probe(client: httpx.AsyncClient, url: str, path: str = "") -> dict:
    t0 = time.monotonic()
    try:
        r = await client.get(url + path, timeout=4.0)
        return {
            "ok": r.status_code < 500,
            "code": r.status_code,
            "ms": round((time.monotonic() - t0) * 1000),
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "code": 0, "ms": 0, "err": type(e).__name__}


def sh(cmd: list[str], timeout: int = 8) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return (out.stdout or out.stderr or "").strip()
    except Exception as e:  # noqa: BLE001
        return f"err:{type(e).__name__}"


async def sh_async(cmd: list[str], timeout: int = 8) -> str:
    """Run a subprocess without stalling the event loop.

    ``sh`` blocks for up to ``timeout`` seconds. Calling it directly from an
    async handler freezes every other request -- including the terminal and
    chat WebSockets -- whenever a probe hangs (a wedged GPU makes
    ``nvidia-smi`` do exactly that).
    """
    return await asyncio.to_thread(sh, cmd, timeout)


SERVICES = [
    name.strip()
    for name in os.environ.get(
        "NORAX_OS_SERVICES",
        "norax-ai,norax-remote-relay,norax-reranker,ollama-embeddings",
    ).split(",")
    if name.strip()
]


# ---------------------------------------------------------------- routes
@app.get("/")
async def index():
    return FileResponse(STATIC, headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/api/health")
async def health(_=Depends(auth)):
    async with httpx.AsyncClient() as c:
        checks = await asyncio.gather(
            probe(c, "http://127.0.0.1:4146", "/v1/models"),
            probe(c, "http://127.0.0.1:11434", "/api/tags"),
            probe(c, "http://127.0.0.1:11435", "/v1/models"),
            probe(c, "http://127.0.0.1:11436", "/api/tags"),
            probe(c, "http://127.0.0.1:8811", "/health"),
        )
    names = [
        "codex:4146",
        "ollama-cloud:11434",
        "ollama-local:11435",
        "embed:11436",
        "reranker:8811",
    ]
    return dict(zip(names, checks, strict=False))


@app.get("/api/system")
async def system(_=Depends(auth)):
    gpu, df_out, mem_out, uptime = await asyncio.gather(
        sh_async(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ),
        sh_async(["df", "-h", "/", "--output=avail,pcent"]),
        sh_async(["free", "-m"]),
        sh_async(["uptime", "-p"]),
    )
    gpus = []
    for line in gpu.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 4:
            gpus.append({"name": p[0], "util": p[1], "mem_used": p[2], "mem_total": p[3]})
    df = df_out.splitlines()
    disk = df[-1].split() if len(df) > 1 else ["?", "?"]
    mem = mem_out.splitlines()
    memline = mem[1].split() if len(mem) > 1 else []
    load = Path("/proc/loadavg").read_text().split()[:3]
    return {
        "gpus": gpus,
        "disk_avail": disk[0],
        "disk_pct": disk[1],
        "mem_used_mb": memline[2] if len(memline) > 2 else "?",
        "mem_total_mb": memline[1] if len(memline) > 1 else "?",
        "uptime": uptime,
        "load": load,
        "time": time.time(),
    }


@app.get("/api/services")
async def services(_=Depends(auth)):
    states = await asyncio.gather(
        *(sh_async(["systemctl", "--user", "is-active", f"{s}.service"]) for s in SERVICES)
    )
    return dict(zip(SERVICES, states, strict=False))


@app.post("/api/services/{name}/{action}")
async def svc_action(name: str, action: str, _=Depends(auth)):
    if name not in SERVICES or action not in ("start", "stop", "restart"):
        raise HTTPException(400, "bad request")
    log.info("service action name=%s action=%s", name, action)
    r = await sh_async(["systemctl", "--user", action, f"{name}.service"], timeout=20)
    return {"ok": True, "out": r}


# ---------------------------------------------------------------- terminal
@app.websocket("/ws/terminal")
async def terminal_ws(ws: WebSocket):
    """Expose a real persistent PTY login shell, matching the local terminal."""
    peer = ws.client.host if ws.client else None
    if TOKEN and not _trusted_peer(peer) and not _token_ok(ws.query_params.get("token", "")):
        await ws.close(code=4401)
        return
    await ws.accept()
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(Path.home())
        env = os.environ.copy()
        env.update({"TERM": "xterm-256color", "COLORTERM": "truecolor"})
        os.execvpe("bash", ["bash", "-l"], env)
    loop = asyncio.get_running_loop()

    async def output():
        while True:
            data = await loop.run_in_executor(None, os.read, fd, 65536)
            if not data:
                break
            await ws.send_bytes(data)

    async def input_():
        while True:
            message = await ws.receive()
            if message.get("bytes") is not None:
                os.write(fd, message["bytes"])
                continue
            text = message.get("text")
            if text is None:
                break
            try:
                event = json.loads(text)
            except ValueError:
                os.write(fd, text.encode())
                continue
            if event.get("type") == "resize":
                rows, cols = int(event.get("rows", 24)), int(event.get("cols", 80))
                import fcntl
                import termios

                fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
            elif event.get("type") == "input":
                os.write(fd, event.get("data", "").encode())

    tasks = [asyncio.create_task(output()), asyncio.create_task(input_())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.kill(pid, signal.SIGHUP)
        except ProcessLookupError:
            pass
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------- files
ALLOWED_ROOTS = [
    Path(item).expanduser().resolve()
    for item in os.environ.get("NORAX_OS_ALLOWED_ROOTS", str(PROJECT_ROOT)).split(os.pathsep)
    if item.strip()
]


def _safe(path: str) -> Path:
    """Resolve a request path and confine it to ALLOWED_ROOTS.

    Uses component-wise containment (``is_relative_to``) rather than string
    prefix matching. A plain ``startswith`` can accept sibling directories
    that merely share a prefix, exposing files outside the intended roots.
    """
    p = Path(path).resolve()
    if not any(p == root or p.is_relative_to(root) for root in ALLOWED_ROOTS):
        raise HTTPException(403, "path outside allowed roots")
    return p


@app.get("/api/files")
async def files(path: str | None = None, _=Depends(auth)):
    p = _safe(path or str(PROJECT_ROOT))
    if not p.is_dir():
        raise HTTPException(400, "not a dir")
    items = []
    for child in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name)):
        try:
            items.append({"name": child.name, "dir": child.is_dir(), "size": child.stat().st_size})
        except OSError:
            continue
    return {"path": str(p), "items": items[:500]}


@app.get("/api/file")
async def file_get(path: str, _=Depends(auth)):
    p = _safe(path)
    if p.stat().st_size > 400_000:
        raise HTTPException(400, "file too large")
    return {"path": str(p), "content": p.read_text(errors="replace")}


@app.post("/api/file")
async def file_put(body: dict, _=Depends(auth)):
    p = _safe(body["path"])
    p.write_text(body["content"])
    return {"ok": True, "size": p.stat().st_size}


# ---------------------------------------------------------------- memory
MEM_ROOT = (
    Path(os.environ.get("NORAX_MEMORY_ROOT", Path.home() / ".local" / "share" / "norax" / "memory"))
    .expanduser()
    .resolve()
)


@app.get("/api/memory")
async def memory(_=Depends(auth)):
    def info(p: Path):
        return {
            "name": p.name,
            "size": p.stat().st_size if p.is_file() else None,
            "dir": p.is_dir(),
            "mtime": p.stat().st_mtime,
        }

    items = []
    if MEM_ROOT.exists():
        for child in sorted(MEM_ROOT.iterdir()):
            try:
                items.append(info(child))
            except OSError:
                continue
    return {"root": str(MEM_ROOT), "items": items}


@app.post("/api/memory/search")
async def mem_search(body: dict, _=Depends(auth)):
    q = body.get("q", "").lower()
    hits = []
    for f in [MEM_ROOT / "semantic.md", MEM_ROOT / "scratchpad.md"]:
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text(errors="replace").splitlines()):
            if q in line.lower():
                hits.append({"file": f.name, "line": i + 1, "text": line[:400]})
            if len(hits) >= 50:
                break
    return {"hits": hits}


# ---------------------------------------------------------------- email
@app.post("/api/email/test")
async def email_test(body: dict, _=Depends(auth)):
    """Send a test alert email via Gmail SMTP (app password)."""
    pw = os.environ.get("NORAX_SMTP_PASS", body.get("smtp_pass", ""))
    smtp_user = os.environ.get("NORAX_SMTP_USER", body.get("smtp_user", ""))
    to = body.get("to", smtp_user)
    if not pw or not smtp_user:
        raise HTTPException(400, "NORAX_SMTP_PASS and NORAX_SMTP_USER not set")
    msg = EmailMessage()
    msg["From"] = smtp_user
    msg["To"] = to
    msg["Subject"] = "[Norax OS] alert channel test"
    msg.set_content("Email backup channel is live. — Norax Agent OS")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as s:
            s.login(smtp_user, pw)
            s.send_message(msg)
        return {"ok": True, "to": to}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e


# ---------------------------------------------------------------- norax live
@app.get("/api/norax/commands")
async def norax_commands(_=Depends(auth)):
    """Return the runtime command catalog used by the web chat command picker."""
    project_root = str(BASE.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from norax.commands import (  # noqa: PLC0415
            COMMANDS,
            MODEL_PROVIDERS,
            PROVIDER_LABELS,
            THINK_LABELS,
            fetch_ollama_models,
            merged_ollama_local_options,
            merged_ollama_options,
            model_options_for_provider,
        )

        descriptions = {
            "settings": "Open the Norax control panel",
            "model": "Show or change the active model",
            "models": "List available models",
            "think": "Show or set reasoning effort",
            "reasoning": "Show or hide reasoning output",
            "planning": "Choose direct or orchestrated planning",
            "rounds": "Set the maximum tool rounds",
            "memory": "Set memory retrieval depth",
            "boost": "Configure weak-model scaffolding",
            "stream": "Toggle live reply streaming",
            "length": "Set preferred response length",
            "activity": "Set tool activity narration",
            "status": "Show runtime status",
            "new": "Start a fresh conversation",
            "stop": "Stop the active run",
            "restart": "Restart the Norax runtime",
            "help": "List all commands",
        }
        providers = []
        live_ollama = await fetch_ollama_models()
        for provider in MODEL_PROVIDERS:
            if provider == "ollama":
                model_opts = merged_ollama_options(live_ollama)
            elif provider == "ollama_local":
                model_opts = merged_ollama_local_options(live_ollama)
            else:
                model_opts = model_options_for_provider(provider)
            providers.append(
                {
                    "id": provider,
                    "label": PROVIDER_LABELS.get(provider, provider.replace("_", " ").title()),
                    "models": [{"id": model_id, "label": label} for model_id, label in model_opts],
                }
            )
        return {
            "commands": [
                {"name": name, "description": descriptions.get(name, f"Run /{name}")}
                for name in sorted(COMMANDS)
            ],
            "providers": providers,
            "thinking": [{"id": key, "label": value} for key, value in THINK_LABELS.items()],
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"command catalog unavailable: {exc}") from exc


@app.post("/api/norax/send")
async def norax_send(body: dict, _=Depends(auth)):
    """Proxy a message to the real Norax runtime via /ingress/agent_os."""
    if not RUNTIME_CHAT_TOKEN:
        raise HTTPException(503, "NORAX_RUNTIME_CHAT_TOKEN not set")
    msg = body.get("message", "").strip()
    thread_id = body.get("thread_id", "agent-os-main")
    if not msg:
        raise HTTPException(400, "empty message")
    # Persist before dispatch so history/reconnects cannot lose the optimistic user turn.
    persist_message(thread_id, "user", msg)
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(
            f"{RUNTIME_BASE}/ingress/agent_os",
            json={"body": msg, "thread_id": thread_id},
            headers={"Authorization": f"Bearer {RUNTIME_CHAT_TOKEN}"},
        )
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    return r.json()


@app.get("/api/norax/history")
async def norax_history(_=Depends(auth)):
    """Proxy history request to the runtime."""
    if not RUNTIME_CHAT_TOKEN:
        raise HTTPException(503, "NORAX_RUNTIME_CHAT_TOKEN not set")
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(
            f"{RUNTIME_BASE}/history/agent_os?channel_id=chat",
            headers={"Authorization": f"Bearer {RUNTIME_CHAT_TOKEN}"},
        )
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    live = r.json()
    with _connect() as db:
        rows = db.execute(
            "SELECT id,role,content,ts FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT 200",
            ("agent-os-main",),
        ).fetchall()[::-1]
    durable = [
        {"id": row_id, "role": role, "text": content, "ts": ts}
        for row_id, role, content, ts in rows
    ]
    messages = durable or live.get("messages", [])
    pending = bool(messages and messages[-1].get("role") == "user")
    return {"ok": True, "messages": messages, "pending": pending}


@app.websocket("/ws/norax")
async def norax_ws(ws: WebSocket):
    """Bridge dashboard WebSocket to the runtime's /ws/agent_os.

    The dashboard connects here with its own token (if set). We then open
    a backend connection to the runtime's WebSocket and bidirectionally
    pipe messages. This avoids exposing the runtime chat token to the
    browser — the server holds it.
    """
    # Auth: trusted local/Tailscale peers connect directly; other peers need TOKEN.
    peer = ws.client.host if ws.client else None
    if TOKEN and not _trusted_peer(peer):
        if not _token_ok(ws.query_params.get("token", "")):
            await ws.close(code=4401)
            return
    if not RUNTIME_CHAT_TOKEN:
        await ws.close(code=4503, reason="runtime chat token not configured")
        return
    await ws.accept()
    # Connect to runtime WebSocket
    import websockets

    runtime_url = f"{RUNTIME_BASE.replace('http', 'ws')}/ws/agent_os?token={RUNTIME_CHAT_TOKEN}"
    try:
        async with websockets.connect(runtime_url) as upstream:

            async def pipe_up():
                try:
                    while True:
                        data = await ws.receive_text()
                        await upstream.send(data)
                except WebSocketDisconnect:
                    pass

            async def pipe_down():
                try:
                    async for msg in upstream:
                        # Live delivery is a projection; mirror visible assistant replies
                        # into our canonical dashboard transcript before broadcasting.
                        try:
                            event = json.loads(msg)
                            if event.get("type") in {"reply", "message"}:
                                text = event.get("text") or event.get("content") or ""
                                role = event.get("role", "assistant")
                                if text and role in {"user", "assistant"}:
                                    if role == "user" and event.get("sender"):
                                        text = f"{event['sender']}: {text}"
                                    persist_message("agent-os-main", role, text)
                        except (ValueError, TypeError):
                            pass
                        if isinstance(msg, str):
                            await ws.send_text(msg)
                        else:
                            await ws.send_bytes(msg)
                except Exception:  # noqa: BLE001
                    pass

            # Whichever side ends first tears the pair down. ``gather`` waited
            # for BOTH, so a browser refresh left ``pipe_down`` parked on the
            # upstream iterator forever: the runtime connection was never
            # closed and every reconnect leaked another socket (and another
            # zombie entry in the runtime's broadcast set).
            tasks = [asyncio.create_task(pipe_up()), asyncio.create_task(pipe_down())]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.warning("norax_ws bridge closed", exc_info=True)


@app.exception_handler(Exception)
async def on_err(request: Request, exc: Exception):
    # Log the detail; return an opaque body. Echoing ``str(exc)`` leaked
    # absolute filesystem paths and internal state to the client.
    log.exception("unhandled error path=%s", request.url.path)
    return JSONResponse({"error": "internal error"}, status_code=500)


def run_server() -> None:
    """Serve Agent OS only on its explicitly configured endpoint.

    Uvicorn's bind failure is deliberate: silently selecting another port
    leaves the dashboard, health checks, and operator looking at different
    endpoints.  The lifespan initializes the database after the socket binds.
    """
    uvicorn.run(app, host=BIND, port=PORT, log_level="warning")


if __name__ == "__main__":
    run_server()
