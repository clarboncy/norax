"""Tool schema retrieval helpers.

Keeps the always-on prompt small while allowing the planner to surface the
right specialized tool schemas from the live registry when a turn implies them.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..dispatch.tools import REGISTRY
from .policy import ALL_OWNER_TOOLS, REMOTE_TOOLS

_ALIAS_PATTERNS: dict[str, list[str]] = {
    "read": [
        r"\bread\b",
        r"\breading\b",
        r"\bopen file\b",
        r"\bview\b",
        r"\bcat\b",
        r"\bshow file\b",
        r"\bexamine\b",
        r"\binspect\b",
        r"\bcheck file\b",
        r"\blook at\b",
    ],
    "write": [
        r"\bwrite\b",
        r"\bwriting\b",
        r"\bcreate file\b",
        r"\bsave\b",
        r"\bnew file\b",
        r"\boverwrite\b",
        r"\bdump\b",
    ],
    "write_chunk": [
        r"\bwrite_chunk\b",
        r"\bchunk\b",
        r"\blarge file\b",
        r"\bbig file\b",
        r"\bappend chunk\b",
        r"\bsplit write\b",
    ],
    "edit": [
        r"\bedit\b",
        r"\bediting\b",
        r"\bmodify\b",
        r"\bchange\b",
        r"\breplace\b",
        r"\bupdate\b",
        r"\bpatch\b",
        r"\bfix\b",
        r"\binsert\b",
        r"\bdelete line\b",
        r"\bremove line\b",
    ],
    "append_memory": [
        r"\bappend\b",
        r"\badd memory\b",
        r"\bremember\b",
        r"\bnote\b",
        r"\bmemorize\b",
        r"\bstore fact\b",
    ],
    "exec": [
        r"\bexec\b",
        r"\bexecute\b",
        r"\brun\b",
        r"\bcommand\b",
        r"\bshell\b",
        r"\bbash\b",
        r"\bterminal\b",
        r"\bcli\b",
        r"\bpip\b",
        r"\bnpm\b",
        r"\bgit\b",
        r"\bsystemctl\b",
        r"\binstall\b",
        r"\bdeploy\b",
        r"\bbuild\b",
        r"\bcompile\b",
        r"\btest\b",
        r"\bpytest\b",
        r"\bmake\b",
        r"\bdocker\b",
        r"\bpython\b",
        r"\bscript\b",
    ],
    "shell": [
        r"\bshell\b",
        r"\bsession\b",
        r"\bpersist\b",
        r"\bmulti-step\b",
        r"\bstateful\b",
        r"\bcwd\b",
        r"\bworking dir\b",
    ],
    "list_dir": [
        r"\blist\b",
        r"\bls\b",
        r"\bdirectory\b",
        r"\bdir\b",
        r"\bfolder\b",
        r"\bfiles?\b",
        r"\bcontents?\b",
        r"\btree\b",
        r"\bfind file\b",
        r"\bbrowse dir\b",
    ],
    "search_memory": [
        r"\bsearch memory\b",
        r"\brecall\b",
        r"\bremember when\b",
        r"\bmemory\b",
        r"\bsemantic\b",
        r"\bprocedural\b",
        r"\bintel\b",
        r"\bepisodic\b",
        r"\bpast\b",
        r"\bhistory\b",
        r"\bforget\b",
    ],
    "status": [
        r"\bstatus\b",
        r"\bhealth\b",
        r"\buptime\b",
        r"\bversion\b",
        r"\bruntime\b",
        r"\bcheck system\b",
        r"\bdiagnostics?\b",
    ],
    "browser": [
        r"\bbrowser\b",
        r"\bnavigate\b",
        r"\bclick\b",
        r"\bscreenshot\b",
        r"\bscreen shot\b",
        r"\bplaywright\b",
        r"\bweb page\b",
        r"\bscrape\b",
        r"\bform\b",
        r"\bfill out\b",
        r"\blogin\b",
        r"\bwebdriver\b",
    ],
    "computer_use": [
        r"\bcomputer_use\b",
        r"\bdesktop\b",
        r"\bscreen\b",
        r"\bscreenshot\b",
        r"\bmouse\b",
        r"\bkeyboard\b",
        r"\bxdotool\b",
        r"\bydotool\b",
        r"\bgrim\b",
        r"\bscot\b",
        r"\bwayland\b",
        r"\bx11\b",
        r"\bgui\b",
    ],
    "sandbox_exec": [
        r"\bsandbox\b",
        r"\bcontainer\b",
        r"\bdocker\b",
        r"\bpodman\b",
        r"\bisolate\b",
        r"\bisolated\b",
        r"\bunsafe code\b",
        r"\buntrusted code\b",
    ],
    "repo_explore": [
        r"\brepo\b",
        r"\brepository\b",
        r"\bexplore\b",
        r"\bcodebase\b",
        r"\bfastcontext\b",
        r"\bfind in repo\b",
        r"\bsearch code\b",
        r"\bgrep repo\b",
    ],
    "message_send": [
        r"\bsend\b",
        r"\bsent\b",
        r"\bdm\b",
        r"\bmessage\b",
        r"\breply\b",
        r"\bdiscord\b",
        r"\bemail\b",
        r"\bzip\b",
        r"\battachment\b",
        r"\battach\b",
        r"\bupload\b",
        r"\bshare\b",
        r"\bdeliver\b",
    ],
    "web_search": [
        r"\bweb\b",
        r"\bwebsearch\b",
        r"\bsearch web\b",
        r"\bgoogle\b",
        r"\bduckduckgo\b",
        r"\bresearch\b",
        r"\bsources?\b",
    ],
    "web_fetch": [
        r"\bfetch\b",
        r"\burl\b",
        r"https?://",
        r"\bpage\b",
        r"\bsite\b",
        r"\bsources?\b",
    ],
    "gateway_config_patch": [
        r"\bgateway\b",
        r"\bprovider\b",
        r"\bmodel\b",
        r"\brouter\b",
        r"\bollama\b",
        r"\bcodex\b",
        r"\bgpt\b",
        r"\bopus\b",
        r"\bconfig\b",
        r"\bpatch config\b",
    ],
    "schedule_reminder": [
        r"\bremind\b",
        r"\breminder\b",
        r"\bschedule\b",
        r"\btimer\b",
        r"\btomorrow\b",
        r"\blater\b",
        r"\bcron\b",
        r"\bat \d",
    ],
    "remote_enroll": [
        r"\benroll\b",
        r"\bremote\b",
        r"\bnode\b",
        r"\bmachine\b",
        r"\blaptop\b",
        r"\bdesktop\b",
        r"\bworkstation\b",
        r"\bssh\b",
    ],
    "remote_list_nodes": [r"\bremote\b", r"\bnodes?\b", r"\bmachines?\b", r"\blist nodes?\b"],
    "remote_exec": [r"\bremote\b", r"\bssh\b", r"\brun .*\b(on|remote)\b", r"\bexec .*\bremote\b"],
    "remote_read": [r"\bremote\b", r"\bread .*\b(on|remote)\b"],
    "remote_list": [r"\bremote\b", r"\blist .*\b(on|remote)\b", r"\bls .*\bremote\b"],
    "remote_write": [r"\bremote\b", r"\bwrite .*\b(on|remote)\b", r"\bcopy .*\b(to|remote)\b"],
}


def infer_tools_from_text(text: str, *, candidates: Iterable[str] | None = None) -> list[str]:
    """Infer specialized tools from request/retrieved context text.

    This is intentionally additive: it never removes core tools, and it only
    returns registered tool names so stale memory cannot expose phantom tools.
    """
    hay = (text or "").lower()
    allowed = set(candidates or ALL_OWNER_TOOLS)
    hits: list[str] = []
    for name, patterns in _ALIAS_PATTERNS.items():
        if name not in allowed or name not in REGISTRY:
            continue
        if any(re.search(p, hay) for p in patterns):
            hits.append(name)
    if any(t in hits for t in REMOTE_TOOLS):
        hits.extend([t for t in REMOTE_TOOLS if t in allowed and t in REGISTRY])
    return list(dict.fromkeys(hits))


def infer_tools_from_memory_items(items: Iterable[object]) -> list[str]:
    """Infer tools from retrieved memory/procedural snippets."""
    chunks: list[str] = []
    for item in items or []:
        if isinstance(item, tuple):
            first = item[0] if item else ""
            chunks.append(str(getattr(first, "text", first)))
        else:
            chunks.append(str(getattr(item, "text", item)))
    return infer_tools_from_text("\n".join(chunks))


def available_tool_manifest() -> str:
    """Compact registry manifest for memory/tool retriever diagnostics."""
    return "\n".join(f"- {name}: {spec.description}" for name, spec in sorted(REGISTRY.items()))
