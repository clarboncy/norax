"""Verify headless browser navigation, typing, clicking, and DOM readback.

Uses only a temporary loopback page and a process-owned browser context.
Does not attach to the user's browser or change deployment configuration.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["NORAX_BROWSER_HEADED"] = "0"


async def main() -> int:
    from norax.dispatch.browser import shutdown_browser, t_browser

    body = (
        b'<input id="entry"><button id="go" onclick="document.getElementById(\'result\').textContent='
        b'document.getElementById(\'entry\').value">Go</button><p id="result"></p>'
    )

    async def serve(reader, writer):
        try:
            await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    try:
        async with server, asyncio.timeout(60):
            port = server.sockets[0].getsockname()[1]
            results = [
                await t_browser(
                    action="navigate", url=f"http://127.0.0.1:{port}", session_id="acceptance"
                ),
                await t_browser(
                    action="type",
                    selector="#entry",
                    text="BROWSER_VERIFIED",
                    session_id="acceptance",
                ),
                await t_browser(action="click", selector="#go", session_id="acceptance"),
                await t_browser(action="extract", selector="#result", session_id="acceptance"),
            ]
            ok = (
                all(row.get("ok") is True for row in results)
                and results[-1].get("content") == "BROWSER_VERIFIED"
            )
            print(
                json.dumps(
                    {
                        "ok": ok,
                        "actions": [
                            {
                                "action": row.get("action"),
                                "ok": row.get("ok"),
                                "error": row.get("error"),
                            }
                            for row in results
                        ],
                        "dom_readback_verified": results[-1].get("content") == "BROWSER_VERIFIED",
                    },
                    indent=2,
                )
            )
            return 0 if ok else 1
    finally:
        await shutdown_browser()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
