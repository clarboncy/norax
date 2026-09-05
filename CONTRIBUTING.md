# Contributing to Norax

Thank you for contributing. Norax is a production-oriented agent runtime, so changes must preserve safety boundaries, deterministic behavior, and evidence-based completion.

## Development setup

```bash
git clone https://github.com/clarboncy/norax.git
cd norax
cp .env.example .env
uv sync --all-extras
```

Do not put real credentials, private conversation transcripts, runtime memories, databases, logs, screenshots, or host-specific configuration in a contribution.

## Workflow

1. Open an issue for substantial behavior or architecture changes.
2. Create a focused branch.
3. Add a regression test before fixing a bug when practical.
4. Keep changes compact and follow existing abstractions.
5. Run the relevant tests while iterating.
6. Run the complete release checks before opening a pull request.

## Required checks

```bash
uv run ruff check norax tests scripts
uv run ruff format --check norax tests scripts
uv run mypy norax
uv run pytest tests/unit tests/boundary -q
uv run pytest tests/integration -q -m "not host_integration"
uv run python scripts/quality_gate.py
uv build
```

Changes to agent execution should include tests for failure, retry, loop interruption, and verification behavior. Changes to memory or compaction should demonstrate that the active goal, constraints, blockers, and next step survive. Changes to tools must preserve sender-tier and risk-gate enforcement.

## Pull requests

Include:

- the problem and root cause;
- the behavioral change;
- tests and verification performed;
- security, compatibility, and migration implications;
- any remaining limitations.

Do not claim a runtime or deployment outcome based only on a unit test. Include read-back or health-check evidence for live-system changes.

## License

By contributing, you agree that your contributions are licensed under the Apache License 2.0.
