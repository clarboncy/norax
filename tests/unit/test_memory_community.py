"""Regression coverage for entity-community projection generation."""

from __future__ import annotations

import json

from norax.memory.community import detect_communities


def test_detect_communities_writes_projection_and_elapsed_time(tmp_path):
    (tmp_path / "entity_graph.json").write_text(
        json.dumps(
            {
                "entities": {"Norax": ["a", "b"], "Colby": ["c"]},
                "links": [["Norax", "Colby"]],
            }
        ),
        encoding="utf-8",
    )

    result = detect_communities(tmp_path)

    assert result["entity_count"] == 2
    assert result["community_count"] >= 1
    assert result["elapsed_sec"] >= 0
    written = json.loads((tmp_path / "index" / "communities.json").read_text(encoding="utf-8"))
    assert written["entity_count"] == 2
    assert written["elapsed_sec"] >= 0
