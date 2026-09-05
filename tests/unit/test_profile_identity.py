"""Tests for owner profile identity/source separation (P1-8 reaudit fix).

Verifies that event source labels (like 'cron') cannot overwrite
identity labels (like 'Colby') in user profiles.
"""

from __future__ import annotations

from pathlib import Path

from norax.memory.user_model import UserModel


def test_cron_label_does_not_overwrite_identity(tmp_path: Path) -> None:
    """A cron-sourced message must not change the owner's label from 'Colby' to 'cron'."""
    model = UserModel(root=tmp_path)
    owner_id = "owner-123"

    # First: create profile with identity label
    model.get_or_create(user_id=owner_id, label="Colby", tier="owner")
    profile = model.load(owner_id)
    assert profile.label == "Colby"

    # Simulate cron message with same user_id but label="cron"
    model.get_or_create(user_id=owner_id, label="cron", tier="owner")
    profile = model.load(owner_id)
    assert profile.label == "Colby"  # unchanged!


def test_discord_label_does_not_overwrite_identity(tmp_path: Path) -> None:
    """A discord-sourced message must not change the owner's label."""
    model = UserModel(root=tmp_path)
    uid = "12345"

    model.get_or_create(user_id=uid, label="Alice", tier="user")
    model.get_or_create(user_id=uid, label="discord", tier="user")
    profile = model.load(uid)
    assert profile.label == "Alice"


def test_label_set_on_first_creation(tmp_path: Path) -> None:
    """Label should be set when profile is first created."""
    model = UserModel(root=tmp_path)
    model.get_or_create(user_id="u1", label="Bob", tier="user")
    assert model.load("u1").label == "Bob"


def test_label_defaults_to_user_id_when_empty(tmp_path: Path) -> None:
    """When no label provided, should default to user_id."""
    model = UserModel(root=tmp_path)
    model.get_or_create(user_id="u1", label="", tier="user")
    assert model.load("u1").label == "u1"


def test_label_updated_from_user_id_to_real_name(tmp_path: Path) -> None:
    """If existing label is the raw user_id, a real name should replace it."""
    model = UserModel(root=tmp_path)
    model.get_or_create(user_id="u1", label="", tier="user")
    assert model.load("u1").label == "u1"
    model.get_or_create(user_id="u1", label="Charlie", tier="user")
    assert model.load("u1").label == "Charlie"


def test_source_label_cannot_overwrite_real_name_back_to_user_id(tmp_path: Path) -> None:
    """After a real name is set, even user_id-as-label should not overwrite."""
    model = UserModel(root=tmp_path)
    model.get_or_create(user_id="u1", label="Dave", tier="user")
    # Even if someone passes label="u1" (the raw id), it should not overwrite
    model.get_or_create(user_id="u1", label="u1", tier="user")
    assert model.load("u1").label == "Dave"
