import hashlib

from flask import Flask

from incident_workflow import repository, service
from incident_workflow.permissions import is_allowed_transition, workflow_role
from incident_workflow.routes import create_blueprint
from incident_workflow.validators import alert_image_url, utc_date_boundary


def test_workflow_roles_are_bound_to_existing_account_identity():
    assert workflow_role("user", "supervisor") == "supervisor"
    assert workflow_role("user", "operator", 2) == "supervisor"
    assert workflow_role("user", "operator", 3) is None
    assert workflow_role("admin", "admin2") == "cso"
    assert workflow_role("admin", "admin") == "superadmin"
    assert workflow_role("admin", "supervisor") is None
    assert workflow_role("user", "admin2") is None
    assert is_allowed_transition("pending_cso_review", "cso_rejected")
    assert is_allowed_transition("cso_rejected", "pending_cso_review")
    assert not is_allowed_transition("cso_rejected", "completed")
    assert not is_allowed_transition("pending_cso_review", "completed")


def test_alert_image_url_only_allows_alert_endpoint_filenames():
    assert alert_image_url("https://vms.example/alerts/image_1.jpg") == (
        "/alerts/image_1.jpg"
    )
    assert alert_image_url("/alerts/..%2Fsecret.jpg") is None
    assert alert_image_url("/other/image_1.jpg") is None


def test_date_boundaries_are_utc_and_date_end_is_inclusive():
    assert utc_date_boundary("2026-11-04") == 1793750400
    assert utc_date_boundary("2026-11-04", end=True) == 1793836800


def test_incident_bearer_identity_comes_from_server_session(monkeypatch):
    created_sessions = {}

    def authenticate(username, password):
        if (username, password) == ("admin2", "secret"):
            return (42, "admin", "admin2", "password-hash")
        return None

    def create_session(token_hash, account_id, role, expires_at):
        created_sessions[token_hash] = (account_id, role, expires_at)

    def get_session(token_hash):
        session = created_sessions.get(token_hash)
        if session is None:
            return None
        account_id, role, _ = session
        username = {42: "admin2"}.get(account_id)
        return account_id, role, username

    monkeypatch.setattr(repository, "create_session", create_session)
    monkeypatch.setattr(repository, "get_session", get_session)
    monkeypatch.setattr(
        service,
        "list_incidents",
        lambda *args: {"data": [], "pagination": {"page": 1, "limit": 20, "total": 0, "totalPages": 0}},
    )
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(authenticate))
    client = app.test_client()

    login = client.post(
        "/api/incidents/auth/login",
        json={
            "username": "admin2",
            "password": "secret",
            "role": "superadmin",
        },
    )
    assert login.status_code == 200
    response = login.get_json()["data"]
    assert response["role"] == "cso"
    token = response["token"]
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    assert token_hash in created_sessions

    unauthorized = client.get("/api/incidents")
    assert unauthorized.status_code == 401
    listed = client.get(
        "/api/incidents",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert listed.status_code == 200
    forbidden = client.post(
        "/api/incidents/101/supervisor-remark",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "visitorName": "Visitor",
            "whomToMeet": "HR",
            "reason": "Meeting",
            "remark": "Verified",
            "role": "supervisor",
        },
    )
    assert forbidden.status_code == 403


def test_user_id_2_can_log_in_to_incident_workflow(monkeypatch):
    created_sessions = {}

    def authenticate(username, password):
        if password != "secret":
            return None
        account_id = {"operator": 2, "other": 3}.get(username)
        if account_id is None:
            return None
        return account_id, "user", username, "password-hash"

    def create_session(token_hash, account_id, role, expires_at):
        created_sessions[token_hash] = (account_id, role, expires_at)

    def get_session(token_hash):
        session = created_sessions.get(token_hash)
        if session is None:
            return None
        account_id, role, _ = session
        username = {2: "operator", 3: "other"}.get(account_id)
        return account_id, role, username

    monkeypatch.setattr(repository, "create_session", create_session)
    monkeypatch.setattr(repository, "get_session", get_session)
    monkeypatch.setattr(
        service,
        "list_incidents",
        lambda *args: {
            "data": [],
            "pagination": {"page": 1, "limit": 20, "total": 0, "totalPages": 0},
        },
    )
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(authenticate))
    client = app.test_client()

    login = client.post(
        "/api/incidents/auth/login",
        json={"username": "operator", "password": "secret"},
    )
    assert login.status_code == 200
    response = login.get_json()["data"]
    assert response["role"] == "supervisor"
    listed = client.get(
        "/api/incidents",
        headers={"Authorization": f"Bearer {response['token']}"},
    )
    assert listed.status_code == 200

    denied = client.post(
        "/api/incidents/auth/login",
        json={"username": "other", "password": "secret"},
    )
    assert denied.status_code == 403
