import os

from fastapi.testclient import TestClient

from norax.remote.relay import app


def test_remote_relay_health():
    c = TestClient(app())
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_public_enrollment_is_fail_closed(monkeypatch):
    monkeypatch.delenv("NORAX_REMOTE_ALLOW_PUBLIC_ENROLL", raising=False)
    client = TestClient(app())

    response = client.post("/remote/public/enroll", json={"name": "untrusted"})

    assert response.status_code == 403


def test_control_enrollment_requires_exact_bearer_token(monkeypatch, tmp_path):
    token_path = tmp_path / "control.token"
    token_path.write_text("correct-token", encoding="utf-8")
    token_path.chmod(0o600)
    monkeypatch.delenv("NORAX_REMOTE_CONTROL_TOKEN", raising=False)
    monkeypatch.setenv("NORAX_REMOTE_CONTROL_TOKEN_FILE", os.fspath(token_path))
    client = TestClient(app())

    missing = client.post("/remote/enroll", json={"name": "worker"})
    wrong = client.post(
        "/remote/enroll",
        headers={"Authorization": "Bearer correct-token-extra"},
        json={"name": "worker"},
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401
