"""PostgreSQL persistence for incident workflow records and sessions."""

import json
import time

from database import get_db_connection
from incident_workflow.permissions import is_allowed_transition


def initialize_schema():
    connection = get_db_connection()
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS incident_workflow (
                id SERIAL PRIMARY KEY,
                incident_id INTEGER NOT NULL UNIQUE
                    REFERENCES detection_records(id) ON DELETE CASCADE,
                visitor_name TEXT NOT NULL,
                phone TEXT,
                whom_to_meet TEXT NOT NULL,
                reason TEXT NOT NULL,
                supervisor_remark TEXT NOT NULL,
                supervisor_id INTEGER NOT NULL REFERENCES auth_accounts(id),
                supervisor_submitted_at DOUBLE PRECISION NOT NULL,
                cso_remark TEXT,
                cso_id INTEGER REFERENCES auth_accounts(id),
                cso_status TEXT CHECK (cso_status IN ('approved', 'rejected')),
                cso_reviewed_at DOUBLE PRECISION,
                final_action TEXT
                    CHECK (final_action IN ('allow', 'monitor', 'escalate', 'blacklist')),
                final_remark TEXT,
                final_action_by INTEGER REFERENCES auth_accounts(id),
                final_action_at DOUBLE PRECISION,
                status TEXT NOT NULL CHECK (status IN (
                    'pending_supervisor_remark',
                    'pending_cso_review',
                    'cso_approved',
                    'cso_rejected',
                    'completed'
                )),
                created_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_incident_workflow_status_updated
            ON incident_workflow (status, updated_at DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_incident_workflow_created
            ON incident_workflow (created_at DESC)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS incident_workflow_history (
                id SERIAL PRIMARY KEY,
                incident_id INTEGER NOT NULL
                    REFERENCES detection_records(id) ON DELETE CASCADE,
                actor_id INTEGER NOT NULL REFERENCES auth_accounts(id),
                actor_role TEXT NOT NULL,
                action TEXT NOT NULL,
                remark TEXT,
                details JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at DOUBLE PRECISION NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_incident_workflow_history_incident
            ON incident_workflow_history (incident_id, created_at, id)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS incident_sessions (
                id SERIAL PRIMARY KEY,
                token_hash TEXT NOT NULL UNIQUE,
                account_id INTEGER NOT NULL REFERENCES auth_accounts(id)
                    ON DELETE CASCADE,
                workflow_role TEXT NOT NULL CHECK (workflow_role IN (
                    'supervisor', 'cso', 'superadmin'
                )),
                created_at DOUBLE PRECISION NOT NULL,
                expires_at DOUBLE PRECISION NOT NULL,
                revoked_at DOUBLE PRECISION
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_incident_sessions_expiry
            ON incident_sessions (expires_at)
            """
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _record_history(connection, incident_id, actor, action, remark, details, now):
    connection.execute(
        """
        INSERT INTO incident_workflow_history (
            incident_id, actor_id, actor_role, action, remark, details, created_at
        ) VALUES (?, ?, ?, ?, ?, ?::jsonb, ?)
        """,
        (
            incident_id,
            actor["account_id"],
            actor["role"],
            action,
            remark,
            json.dumps(details),
            now,
        ),
    )


def create_session(token_hash, account_id, role, expires_at):
    now = time.time()
    connection = get_db_connection()
    try:
        connection.execute(
            """
            DELETE FROM incident_sessions
            WHERE expires_at <= ? OR revoked_at IS NOT NULL
            """,
            (now,),
        )
        connection.execute(
            """
            INSERT INTO incident_sessions (
                token_hash, account_id, workflow_role, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (token_hash, account_id, role, now, expires_at),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_session(token_hash):
    connection = get_db_connection()
    try:
        return connection.execute(
            """
            SELECT s.account_id, s.workflow_role, a.username
            FROM incident_sessions AS s
            INNER JOIN auth_accounts AS a ON a.id = s.account_id
            WHERE s.token_hash = ?
              AND s.revoked_at IS NULL
              AND s.expires_at > ?
            """,
            (token_hash, time.time()),
        ).fetchone()
    finally:
        connection.close()


def revoke_session(token_hash):
    connection = get_db_connection()
    try:
        cursor = connection.execute(
            """
            UPDATE incident_sessions
            SET revoked_at = ?
            WHERE token_hash = ? AND revoked_at IS NULL
            """,
            (time.time(), token_hash),
        )
        connection.commit()
        return cursor.rowcount > 0
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _lock_incident(connection, incident_id):
    row = connection.execute(
        """
        SELECT id, detected_at, detection_type, camera_id, camera_name,
               gate_name, image_url, person_name
        FROM detection_records
        WHERE id = ? AND detection_type = 'unknown_person'
        FOR UPDATE
        """,
        (incident_id,),
    ).fetchone()
    if row is None:
        raise IncidentNotFound()
    return row


class IncidentNotFound(Exception):
    pass


class WorkflowConflict(Exception):
    pass


def list_incidents(page, limit, status, camera_id, start_epoch, end_epoch):
    connection = get_db_connection()
    try:
        conditions = ["d.detection_type = 'unknown_person'"]
        params = []
        if status:
            conditions.append(
                "COALESCE(w.status, 'pending_supervisor_remark') = ?"
            )
            params.append(status)
        if camera_id:
            conditions.append("d.camera_id = ?")
            params.append(camera_id)
        if start_epoch is not None:
            conditions.append("d.detected_at >= ?")
            params.append(start_epoch)
        if end_epoch is not None:
            conditions.append("d.detected_at < ?")
            params.append(end_epoch)
        where_clause = " AND ".join(conditions)
        total = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM detection_records AS d
            LEFT JOIN incident_workflow AS w ON w.incident_id = d.id
            WHERE {where_clause}
            """,
            tuple(params),
        ).fetchone()[0]
        offset = (page - 1) * limit
        rows = connection.execute(
            f"""
            SELECT d.id, d.detected_at, d.person_name, d.camera_id,
                   d.camera_name, d.gate_name, d.image_url,
                   COALESCE(w.status, 'pending_supervisor_remark')
            FROM detection_records AS d
            LEFT JOIN incident_workflow AS w ON w.incident_id = d.id
            WHERE {where_clause}
            ORDER BY d.detected_at DESC, d.id DESC
            LIMIT ? OFFSET ?
            """,
            tuple(params + [limit, offset]),
        ).fetchall()
        return total, rows
    finally:
        connection.close()


def get_incident(incident_id):
    connection = get_db_connection()
    try:
        row = connection.execute(
            """
            SELECT d.id, d.detected_at, d.person_name, d.camera_id,
                   d.camera_name, d.gate_name, d.image_url,
                   w.visitor_name, w.phone, w.whom_to_meet, w.reason,
                   w.supervisor_remark, w.supervisor_submitted_at,
                   supervisor.username,
                   w.cso_remark, w.cso_reviewed_at, cso.username,
                   w.cso_status, w.final_action, w.final_remark,
                   w.final_action_at, final_actor.username,
                   COALESCE(w.status, 'pending_supervisor_remark'),
                   w.created_at, w.updated_at
            FROM detection_records AS d
            LEFT JOIN incident_workflow AS w ON w.incident_id = d.id
            LEFT JOIN auth_accounts AS supervisor ON supervisor.id = w.supervisor_id
            LEFT JOIN auth_accounts AS cso ON cso.id = w.cso_id
            LEFT JOIN auth_accounts AS final_actor
                ON final_actor.id = w.final_action_by
            WHERE d.id = ? AND d.detection_type = 'unknown_person'
            """,
            (incident_id,),
        ).fetchone()
        if row is None:
            raise IncidentNotFound()
        history = connection.execute(
            """
            SELECT h.action, h.actor_role, a.username, h.remark,
                   h.details, h.created_at
            FROM incident_workflow_history AS h
            INNER JOIN auth_accounts AS a ON a.id = h.actor_id
            WHERE h.incident_id = ?
            ORDER BY h.created_at, h.id
            """,
            (incident_id,),
        ).fetchall()
        return row, history
    finally:
        connection.close()


def submit_supervisor_remark(incident_id, actor, fields):
    connection = get_db_connection()
    now = time.time()
    try:
        _lock_incident(connection, incident_id)
        existing = connection.execute(
            """
            SELECT status, cso_status
            FROM incident_workflow
            WHERE incident_id = ?
            FOR UPDATE
            """,
            (incident_id,),
        ).fetchone()
        if existing is None:
            if not is_allowed_transition(
                "pending_supervisor_remark", "pending_cso_review"
            ):
                raise WorkflowConflict("Invalid workflow transition")
            connection.execute(
                """
                INSERT INTO incident_workflow (
                    incident_id, visitor_name, phone, whom_to_meet, reason,
                    supervisor_remark, supervisor_id, supervisor_submitted_at,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending_cso_review', ?, ?)
                """,
                (
                    incident_id,
                    fields["visitor_name"],
                    fields["phone"],
                    fields["whom_to_meet"],
                    fields["reason"],
                    fields["remark"],
                    actor["account_id"],
                    now,
                    now,
                    now,
                ),
            )
            action = "supervisor_submitted"
        elif existing[0] == "cso_rejected" and existing[1] == "rejected":
            if not is_allowed_transition(existing[0], "pending_cso_review"):
                raise WorkflowConflict("Invalid workflow transition")
            connection.execute(
                """
                UPDATE incident_workflow
                SET visitor_name = ?, phone = ?, whom_to_meet = ?, reason = ?,
                    supervisor_remark = ?, supervisor_id = ?,
                    supervisor_submitted_at = ?, cso_remark = NULL,
                    cso_id = NULL, cso_status = NULL, cso_reviewed_at = NULL,
                    final_action = NULL, final_remark = NULL,
                    final_action_by = NULL, final_action_at = NULL,
                    status = 'pending_cso_review', updated_at = ?
                WHERE incident_id = ?
                """,
                (
                    fields["visitor_name"],
                    fields["phone"],
                    fields["whom_to_meet"],
                    fields["reason"],
                    fields["remark"],
                    actor["account_id"],
                    now,
                    now,
                    incident_id,
                ),
            )
            action = "supervisor_resubmitted"
        else:
            raise WorkflowConflict("Supervisor remarks have already been submitted")
        _record_history(
            connection,
            incident_id,
            actor,
            action,
            fields["remark"],
            {key: value for key, value in fields.items() if key != "remark"},
            now,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def submit_cso_review(incident_id, actor, decision, remark):
    connection = get_db_connection()
    now = time.time()
    try:
        _lock_incident(connection, incident_id)
        workflow = connection.execute(
            """
            SELECT status FROM incident_workflow
            WHERE incident_id = ? FOR UPDATE
            """,
            (incident_id,),
        ).fetchone()
        if workflow is None:
            raise WorkflowConflict("Supervisor submission is required before review")
        status = "cso_approved" if decision == "approved" else "cso_rejected"
        if not is_allowed_transition(workflow[0], status):
            raise WorkflowConflict("Incident is not awaiting CSO review")
        connection.execute(
            """
            UPDATE incident_workflow
            SET cso_remark = ?, cso_id = ?, cso_status = ?,
                cso_reviewed_at = ?, status = ?, updated_at = ?
            WHERE incident_id = ?
            """,
            (
                remark,
                actor["account_id"],
                decision,
                now,
                status,
                now,
                incident_id,
            ),
        )
        _record_history(
            connection,
            incident_id,
            actor,
            f"cso_{decision}",
            remark,
            {"decision": decision},
            now,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def submit_final_action(incident_id, actor, action, remark):
    connection = get_db_connection()
    now = time.time()
    try:
        _lock_incident(connection, incident_id)
        workflow = connection.execute(
            """
            SELECT status FROM incident_workflow
            WHERE incident_id = ? FOR UPDATE
            """,
            (incident_id,),
        ).fetchone()
        if workflow is None or not is_allowed_transition(workflow[0], "completed"):
            raise WorkflowConflict("Only CSO-approved incidents can be finalized")
        connection.execute(
            """
            UPDATE incident_workflow
            SET final_action = ?, final_remark = ?, final_action_by = ?,
                final_action_at = ?, status = 'completed', updated_at = ?
            WHERE incident_id = ?
            """,
            (action, remark, actor["account_id"], now, now, incident_id),
        )
        _record_history(
            connection,
            incident_id,
            actor,
            "final_action",
            remark,
            {"action": action},
            now,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
