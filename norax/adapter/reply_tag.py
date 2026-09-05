"""Reply-tag parser. Norax Discord reply-tag parser: [[reply_to_current]] / [[reply_to:<id>]].

Tag MUST be the very first token of the message (optionally preceded by
whitespace stripping). We strip the tag and return the resolved reply target.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TAG_RX = re.compile(
    r"^\s*\[\[\s*(?:"
    r"(?P<current>reply_to_current)"
    r"|"
    r"reply_to\s*:\s*(?P<id>[A-Za-z0-9_\-]+)"
    r")\s*\]\]\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReplyTag:
    text: str  # text with tag stripped
    reply_to: str | None  # resolved reply target (message id) or None
    had_tag: bool  # True if a tag was parsed


def parse_reply_tag(text: str, *, current_message_id: str | None = None) -> ReplyTag:
    """Parse a leading [[reply_to_*]] tag.

    - `[[reply_to_current]]`: reply_to = current_message_id (if provided)
    - `[[reply_to:<id>]]`: reply_to = <id>
    - no tag: reply_to = None
    """
    if not text:
        return ReplyTag(text=text or "", reply_to=None, had_tag=False)
    m = _TAG_RX.match(text)
    if not m:
        return ReplyTag(text=text, reply_to=None, had_tag=False)
    stripped = text[m.end() :]
    if m.group("current"):
        return ReplyTag(text=stripped, reply_to=current_message_id, had_tag=True)
    return ReplyTag(text=stripped, reply_to=m.group("id"), had_tag=True)
