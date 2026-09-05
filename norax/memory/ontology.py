"""Ontology Support — structured entity types, relation types, property schemas.

Provides type-safe knowledge representation:
  - Entity types: person, project, tool, code, error, concept, etc.
  - Relation types: created_by, depends_on, part_of, instance_of, etc.
  - Property schemas: each entity type has expected properties
  - Auto-classification: classify new memories against ontology

Usage:
    onto = Ontology()
    onto.classify("Alice owns Norax AI")
    # → {entities: [{name: "Alice", type: "person"}, {name: "Norax AI", type: "project"}],
    #    relations: [{source: "Alice", target: "Norax AI", type: "owns"}]}
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger("norax.memory.ontology")


@dataclass
class EntityType:
    """Definition of an entity type."""

    name: str
    description: str
    properties: dict[str, str] = field(default_factory=dict)  # property_name → type
    patterns: list[str] = field(default_factory=list)  # regex patterns for detection


@dataclass
class RelationType:
    """Definition of a relation type."""

    name: str
    description: str
    source_types: list[str] = field(default_factory=list)  # valid source entity types
    target_types: list[str] = field(default_factory=list)  # valid target entity types
    patterns: list[str] = field(default_factory=list)  # regex patterns for detection


@dataclass
class ClassificationResult:
    """Result of classifying text against the ontology."""

    entities: list[dict] = field(default_factory=list)  # [{name, type, confidence}]
    relations: list[dict] = field(default_factory=list)  # [{source, target, type}]


# ── Pre-built ontologies ─────────────────────────────────────────────────

DEFAULT_ENTITY_TYPES: dict[str, EntityType] = {
    "person": EntityType(
        name="person",
        description="A human individual",
        properties={"name": "str", "email": "str?", "role": "str?"},
        patterns=[
            r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.\s+([A-Z][a-z]+)",
            r"\b([A-Z][a-z]+)\s+(?:said|asked|told|wrote|owns|created|built)",
        ],
    ),
    "project": EntityType(
        name="project",
        description="A software project or product",
        properties={"name": "str", "version": "str?", "repo": "str?", "status": "str?"},
        patterns=[
            r"\b([A-Z][a-zA-Z]+)\s+(?:AI|project|system|platform|app|server|service)",
            r"\b([A-Z][a-zA-Z]+)\s+v?(\d+\.\d+)",
        ],
    ),
    "tool": EntityType(
        name="tool",
        description="A software tool, library, or framework",
        properties={"name": "str", "category": "str?", "version": "str?"},
        patterns=[
            r"\b(python3?|node|docker|git|pytest|ollama|redis|sqlite)\b",
            r"\b([a-z][a-z-]+)\s+(?:library|framework|tool|package)",
        ],
    ),
    "code": EntityType(
        name="code",
        description="A code file, function, or class",
        properties={"path": "str", "language": "str?", "type": "str?"},
        patterns=[
            r"\b([\w/]+\.\w+)\b",  # file paths
            r"\bdef\s+(\w+)",  # function definitions
            r"\bclass\s+(\w+)",  # class definitions
        ],
    ),
    "error": EntityType(
        name="error",
        description="An error, exception, or failure",
        properties={"type": "str", "message": "str", "code": "str?"},
        patterns=[
            r"\b([A-Z]\w+Error):\s*(.+)",
            r"\bError:\s*(.+)",
            r"\bFAILED\b",
        ],
    ),
    "concept": EntityType(
        name="concept",
        description="An abstract concept or technology",
        properties={"name": "str", "category": "str?"},
        patterns=[
            r"\b(MCP|A2A|ACP|AGI|LLM|RAG|FTS5|RRF|GBNF)\b",
            r"\b([a-z][a-z]+(?:\s+[a-z][a-z]+)?)\s+(?:protocol|standard|architecture)",
        ],
    ),
    "command": EntityType(
        name="command",
        description="A shell command or CLI invocation",
        properties={"command": "str", "exit_code": "int?"},
        patterns=[
            r"\$\s+(.+)",
            r"```(?:bash|sh)\s*\n(.+?)```",
        ],
    ),
}

DEFAULT_RELATION_TYPES: dict[str, RelationType] = {
    "owns": RelationType(
        name="owns",
        description="Ownership relationship",
        source_types=["person"],
        target_types=["project", "tool"],
        patterns=[r"(\w+)\s+owns?\s+(\w+)"],
    ),
    "created": RelationType(
        name="created",
        description="Creation relationship",
        source_types=["person"],
        target_types=["project", "tool", "code"],
        patterns=[r"(\w+)\s+(?:created|built|wrote|developed)\s+(\w+)"],
    ),
    "depends_on": RelationType(
        name="depends_on",
        description="Dependency relationship",
        source_types=["project", "code", "tool"],
        target_types=["tool", "project"],
        patterns=[r"(\w+)\s+(?:depends\s+on|requires|uses)\s+(\w+)"],
    ),
    "part_of": RelationType(
        name="part_of",
        description="Containment relationship",
        source_types=["code", "tool"],
        target_types=["project", "system"],
        patterns=[r"(\w+)\s+(?:is\s+part\s+of|belongs\s+to|in)\s+(\w+)"],
    ),
    "instance_of": RelationType(
        name="instance_of",
        description="Type relationship",
        source_types=["project", "tool", "code"],
        target_types=["concept"],
        patterns=[r"(\w+)\s+is\s+(?:a|an)\s+(\w+)"],
    ),
    "uses": RelationType(
        name="uses",
        description="Usage relationship",
        source_types=["person", "project", "tool"],
        target_types=["tool", "concept"],
        patterns=[r"(\w+)\s+uses?\s+(\w+)"],
    ),
    "related_to": RelationType(
        name="related_to",
        description="Generic relationship",
        source_types=[],
        target_types=[],
        patterns=[r"(\w+)\s+(?:related\s+to|connected\s+to|linked\s+to)\s+(\w+)"],
    ),
}


class Ontology:
    """Ontology for structured knowledge representation."""

    def __init__(
        self,
        entity_types: dict[str, EntityType] | None = None,
        relation_types: dict[str, RelationType] | None = None,
    ) -> None:
        self.entity_types = entity_types or dict(DEFAULT_ENTITY_TYPES)
        self.relation_types = relation_types or dict(DEFAULT_RELATION_TYPES)

    def classify(self, text: str) -> ClassificationResult:
        """Classify text against the ontology.

        Extracts entities and relations using pattern matching.
        """
        entities: list[dict] = []
        relations: list[dict] = []
        seen_entities: set[str] = set()

        # Detect entities
        for etype_name, etype in self.entity_types.items():
            for pattern in etype.patterns:
                for m in re.finditer(pattern, text, re.MULTILINE):
                    name = m.group(1) if m.groups() else m.group(0)
                    name = name.strip()
                    if not name or len(name) < 2:
                        continue
                    key = f"{etype_name}:{name.lower()}"
                    if key not in seen_entities:
                        seen_entities.add(key)
                        entities.append(
                            {
                                "name": name,
                                "type": etype_name,
                                "confidence": 0.7,
                            }
                        )

        # Detect relations
        for rtype_name, rtype in self.relation_types.items():
            for pattern in rtype.patterns:
                for m in re.finditer(pattern, text, re.MULTILINE):
                    if len(m.groups()) >= 2:
                        source = m.group(1).strip()
                        target = m.group(2).strip()
                        if source and target:
                            relations.append(
                                {
                                    "source": source,
                                    "target": target,
                                    "type": rtype_name,
                                }
                            )

        return ClassificationResult(entities=entities, relations=relations)

    def validate_entity(self, entity: dict) -> bool:
        """Validate an entity against the ontology."""
        etype = self.entity_types.get(entity.get("type", ""))
        if etype is None:
            return False
        # Check required properties
        for prop_name, prop_type in etype.properties.items():
            if not prop_type.endswith("?") and prop_name not in entity:
                return False
        return True

    def validate_relation(self, relation: dict) -> bool:
        """Validate a relation against the ontology."""
        rtype = self.relation_types.get(relation.get("type", ""))
        if rtype is None:
            return False
        # Check source/target types if specified
        if rtype.source_types:
            source_type = relation.get("source_type", "")
            if source_type and source_type not in rtype.source_types:
                return False
        if rtype.target_types:
            target_type = relation.get("target_type", "")
            if target_type and target_type not in rtype.target_types:
                return False
        return True

    def get_entity_type(self, name: str) -> EntityType | None:
        return self.entity_types.get(name)

    def get_relation_type(self, name: str) -> RelationType | None:
        return self.relation_types.get(name)

    def to_dict(self) -> dict:
        """Serialize ontology for storage."""
        return {
            "entity_types": {
                name: {
                    "name": et.name,
                    "description": et.description,
                    "properties": et.properties,
                    "patterns": et.patterns,
                }
                for name, et in self.entity_types.items()
            },
            "relation_types": {
                name: {
                    "name": rt.name,
                    "description": rt.description,
                    "source_types": rt.source_types,
                    "target_types": rt.target_types,
                    "patterns": rt.patterns,
                }
                for name, rt in self.relation_types.items()
            },
        }


# ── Singleton ───────────────────────────────────────────────────────────

_ontology: Ontology | None = None


def get_ontology() -> Ontology:
    global _ontology
    if _ontology is None:
        _ontology = Ontology()
    return _ontology
