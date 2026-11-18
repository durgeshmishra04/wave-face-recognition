"""Role mapping and workflow transition permissions."""

ACCOUNT_ROLES = {
    ("user", "supervisor"): "supervisor",
    ("admin", "admin2"): "cso",
    ("admin", "admin"): "superadmin",
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


def workflow_role(account_role, username):
    """Return the workflow role only for the explicitly mapped account."""
    return ACCOUNT_ROLES.get((account_role, username))


def is_allowed_transition(current_status, next_status):
    return next_status in WORKFLOW_TRANSITIONS.get(current_status, set())
