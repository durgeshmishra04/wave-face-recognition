"""Role mapping and workflow transition permissions."""

ACCOUNT_ROLES = {
    ("user", "supervisor"): "supervisor",
    ("admin", "admin2"): "cso",
    ("admin", "admin"): "superadmin",
}

ACCOUNT_ID_ROLES = {
    ("user", 2): "supervisor",
}

WORKFLOW_STATUSES = {
    "pending_supervisor_remark",
    "pending_cso_review",
    "cso_approved",
    "cso_rejected",
    "completed",
}

FINAL_ACTIONS = {"allow", "monitor", "escalate", "blacklist"}

WORKFLOW_TRANSITIONS = {
    "pending_supervisor_remark": {"pending_cso_review"},
    "pending_cso_review": {"cso_approved", "cso_rejected"},
    "cso_rejected": {"pending_cso_review"},
    "cso_approved": {"completed"},
}


def workflow_role(account_role, username, account_id=None):
    """Return the workflow role only for the explicitly mapped account."""
    role = ACCOUNT_ID_ROLES.get((account_role, account_id))
    if role is not None:
        return role
    return ACCOUNT_ROLES.get((account_role, username))


def is_allowed_transition(current_status, next_status):
    return next_status in WORKFLOW_TRANSITIONS.get(current_status, set())
