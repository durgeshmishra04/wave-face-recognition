"""Flask routes for authenticated incident workflow operations."""

from functools import wraps
import hashlib
import secrets
import time

from flask import Blueprint, g, jsonify, request

from incident_workflow import repository, service
from incident_workflow.permissions import FINAL_ACTIONS, WORKFLOW_STATUSES, workflow_role
from incident_workflow.repository import IncidentNotFound, WorkflowConflict
from incident_workflow.validators import (
    ValidationError,
    incident_id_value,
    optional_text,
    page_value,
    required_text,
    utc_date_boundary,
)

SESSION_TTL_SECONDS = 12 * 60 * 60


def create_blueprint(authenticate_credentials):
    blueprint = Blueprint("incident_workflow", __name__)

    def json_error(message, status):
        return jsonify(service.error(message)), status

    def auth_required(handler):
        @wraps(handler)
        def wrapped(*args, **kwargs):
            authorization = request.headers.get("Authorization", "")
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() != "bearer" or not token.strip():
                return json_error("Bearer token required", 401)
            token = token.strip()
            token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
            session = repository.get_session(token_hash)
            if session is None:
                return json_error("Invalid or expired incident session", 401)
            account_role = "user" if session[1] == "supervisor" else "admin"
            if workflow_role(account_role, session[2], session[0]) != session[1]:
                return json_error("Incident session role is no longer valid", 401)
            g.incident_actor = {
                "account_id": session[0],
                "role": session[1],
                "username": session[2],
                "token_hash": token_hash,
            }
            return handler(*args, **kwargs)

        return wrapped

    def require_role(*roles):
        def decorator(handler):
            @wraps(handler)
            @auth_required
            def wrapped(*args, **kwargs):
                if g.incident_actor["role"] not in roles:
                    return json_error("You are not authorized for this action", 403)
                return handler(*args, **kwargs)

            return wrapped

        return decorator

    @blueprint.post("/api/incidents/auth/login")
    def incident_login():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return json_error("A JSON request body is required", 400)
        try:
            username = required_text(payload, "username", 255)
            password = required_text(payload, "password", 1024)
        except ValidationError as error:
            return json_error(str(error), 400)
        account = authenticate_credentials(username, password)
        if account is None:
            return json_error("Invalid credentials", 401)
        account_role = account[1]
        account_username = account[2]
        role = workflow_role(account_role, account_username, int(account[0]))
        if role is None:
            return json_error("This account has no incident workflow access", 403)

        token = secrets.token_urlsafe(32)
        expires_at = time.time() + SESSION_TTL_SECONDS
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        repository.create_session(token_hash, int(account[0]), role, expires_at)
        return jsonify(
            service.success(
                {
                    "token": token,
                    "token_type": "Bearer",
                    "expires_at": service.utc_iso(expires_at),
                    "role": role,
                    "username": account_username,
                },
                "Incident session created",
            )
        ), 200

    @blueprint.post("/api/incidents/auth/logout")
    @auth_required
    def incident_logout():
        repository.revoke_session(g.incident_actor["token_hash"])
        return jsonify(service.success(None, "Incident session revoked")), 200

    @blueprint.get("/api/incidents")
    @require_role("supervisor", "cso", "superadmin")
    def incident_list():
        try:
            category = request.args.get("category", "unknown_person").strip()
            if category != "unknown_person":
                raise ValidationError("category must be unknown_person")
            page = page_value(request.args.get("page"), "page", 1)
            limit = page_value(request.args.get("limit"), "limit", 20, maximum=100)
            status = request.args.get("status", "").strip() or None
            if status and status not in WORKFLOW_STATUSES:
                raise ValidationError("Invalid status")
            camera_id = request.args.get("camera_id", "").strip() or None
            start_epoch = utc_date_boundary(request.args.get("start_date"))
            end_epoch = utc_date_boundary(request.args.get("end_date"), end=True)
            if start_epoch is not None and end_epoch is not None and start_epoch >= end_epoch:
                raise ValidationError("start_date must be earlier than end_date")
            result = service.list_incidents(
                page, limit, status, camera_id, start_epoch, end_epoch
            )
            return jsonify(
                {"success": True, **result}
            ), 200
        except ValidationError as error:
            return json_error(str(error), 400)

    @blueprint.get("/api/incidents/<incident_id>")
    @require_role("supervisor", "cso", "superadmin")
    def incident_detail(incident_id):
        try:
            validated_id = incident_id_value(incident_id)
            data = service.incident_details(validated_id)
            return jsonify(service.success(data)), 200
        except ValidationError as error:
            return json_error(str(error), 400)
        except IncidentNotFound:
            return json_error("Unknown-person incident not found", 404)

    @blueprint.post("/api/incidents/<incident_id>/supervisor-remark")
    @require_role("supervisor")
    def supervisor_remark(incident_id):
        try:
            validated_id = incident_id_value(incident_id)
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                raise ValidationError("A JSON request body is required")
            fields = {
                "visitor_name": required_text(payload, "visitorName", 255),
                "phone": optional_text(payload, "phone", 50),
                "whom_to_meet": required_text(payload, "whomToMeet", 255),
                "reason": required_text(payload, "reason", 2000),
                "remark": required_text(payload, "remark", 2000),
            }
            repository.submit_supervisor_remark(validated_id, g.incident_actor, fields)
            return jsonify(
                service.success(
                    service.incident_details(validated_id),
                    "Supervisor remark submitted",
                )
            ), 200
        except ValidationError as error:
            return json_error(str(error), 400)
        except IncidentNotFound:
            return json_error("Unknown-person incident not found", 404)
        except WorkflowConflict as error:
            return json_error(str(error), 409)

    @blueprint.post("/api/incidents/<incident_id>/cso-review")
    @require_role("cso")
    def cso_review(incident_id):
        try:
            validated_id = incident_id_value(incident_id)
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                raise ValidationError("A JSON request body is required")
            decision = required_text(payload, "decision", 20).lower()
            if decision not in {"approved", "rejected"}:
                raise ValidationError("decision must be approved or rejected")
            remark = required_text(payload, "remark", 2000)
            repository.submit_cso_review(
                validated_id, g.incident_actor, decision, remark
            )
            return jsonify(
                service.success(
                    service.incident_details(validated_id),
                    "CSO review submitted",
                )
            ), 200
        except ValidationError as error:
            return json_error(str(error), 400)
        except IncidentNotFound:
            return json_error("Unknown-person incident not found", 404)
        except WorkflowConflict as error:
            return json_error(str(error), 409)

    @blueprint.post("/api/incidents/<incident_id>/final-action")
    @require_role("superadmin")
    def final_action(incident_id):
        try:
            validated_id = incident_id_value(incident_id)
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                raise ValidationError("A JSON request body is required")
            action = required_text(payload, "action", 20).lower()
            if action not in FINAL_ACTIONS:
                raise ValidationError(
                    "action must be allow, monitor, escalate, or blacklist"
                )
            remark = required_text(payload, "remark", 2000)
            repository.submit_final_action(
                validated_id, g.incident_actor, action, remark
            )
            return jsonify(
                service.success(
                    service.incident_details(validated_id),
                    "Final incident action submitted",
                )
            ), 200
        except ValidationError as error:
            return json_error(str(error), 400)
        except IncidentNotFound:
            return json_error("Unknown-person incident not found", 404)
        except WorkflowConflict as error:
            return json_error(str(error), 409)

    return blueprint
