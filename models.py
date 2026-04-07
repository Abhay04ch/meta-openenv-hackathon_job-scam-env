"""
Data models for the Job Scam Detection Environment.

Design rationale
----------------
OpenEnv's ``create_app`` accepts exactly **one** Action class and **one**
Observation class.  To support three task variants (easy / medium / hard)
without forking the server, we use a **unified superset** approach:

* ``JobScamAction``       — one class; its ``action_type`` enum is the union
                            of every action from all three tasks.  Only the
                            action types that are valid for the *current
                            episode's task* will be accepted by the server.
* ``JobScamObservation``  — one class with every possible field across all
                            tasks declared as ``Optional``.  Fields that are
                            irrelevant to the current task are simply ``None``.

Task-specific enumerations are grouped with comments so it is easy to extend
them when the easy / hard task designs are finalised.

Adding a new task
-----------------
1. Add new ``ActionType`` members in the "HARD TASK" / "EASY TASK" blocks.
2. Add new ``ClassificationLabel`` members if the label space differs.
3. Add new Optional fields to ``JobScamObservation`` in the appropriate block.
4. Wire the new values into ``constants.py`` and the environment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations — unified superset across all tasks
# ---------------------------------------------------------------------------
class ActionType(str, Enum):
    """
    All valid action types across every task variant.
    """
    # ── MEDIUM TASK ─────────────────────────
    CLASSIFY = "classify"
    REQUEST_RECRUITER_PROFILE = "request_recruiter_profile"
    REQUEST_COMPANY_PROFILE   = "request_company_profile"
    REQUEST_THREAD_HISTORY    = "request_thread_history"
    REQUEST_JOB_POST_COMMENTS = "request_job_post_comments"

    # ── EASY TASK ────────────
    # TODO: uncomment / add when easy task is designed
    # REQUEST_EASY_FIELD_A = "request_easy_field_a"

    # ── HARD TASK ────────────
    # TODO: uncomment / add when hard task is designed
    # REQUEST_HARD_FIELD_A = "request_hard_field_a"
    # REQUEST_HARD_FIELD_B = "request_hard_field_b"


class ClassificationLabel(str, Enum):
    """
    All valid classification outcomes across every task variant.

    Right now the label space is identical for easy / medium / hard.
    If a task variant needs additional labels, add them here with a comment
    indicating which task introduced them.
    """
    # ── MEDIUM TASK labels (placeholders) ─────────────────────────────────────
    LEGIT             = "legit"
    SUSPICIOUS        = "suspicious"
    SCAM              = "scam"
    INSUFFICIENT_INFO = "insufficient_info"

    # ── EASY TASK labels (placeholders) ─────────────────────────────────────
    # TODO: add easy-specific labels here if needed

    # ── HARD TASK labels (placeholders) ─────────────────────────────────────
    # TODO: add hard-specific labels here if needed


# ---------------------------------------------------------------------------
# Unified Action
# ---------------------------------------------------------------------------
class JobScamAction(Action):
    """
    A single step action submitted by the client — valid for all task variants.

    For information-gathering steps, set ``action_type`` to one of the
    ``REQUEST_*`` variants appropriate for the active task, and leave
    ``label`` as ``None``.

    For the terminal classification step, set ``action_type`` to
    ``CLASSIFY`` and provide a ``ClassificationLabel`` in ``label``.

    The server will reject ``REQUEST_*`` actions that do not belong to the
    current episode's task (e.g. sending a medium-task action during an
    easy-task episode).

    Example — medium info request::

        JobScamAction(action_type="request_company_profile")

    Example — classification (all tasks)::

        JobScamAction(action_type="classify", label="scam")
    """

    action_type: ActionType = Field(
        ...,
        description=(
            "Type of action: a context-field request (task-specific) or "
            "the terminal 'classify' action (shared across all tasks)."
        ),
    )
    label: Optional[ClassificationLabel] = Field(
        default=None,
        description=(
            "Required only when action_type is 'classify'. "
            "Must be one of: legit, suspicious, scam, insufficient_info."
        ),
    )

    @model_validator(mode="after")
    def _label_required_for_classify(self) -> "JobScamAction":
        if self.action_type == ActionType.CLASSIFY and self.label is None:
            raise ValueError(
                "label must be provided when action_type is 'classify'."
            )
        return self


# ---------------------------------------------------------------------------
# Unified Observation (superset of all task-specific fields)
# ---------------------------------------------------------------------------
class JobScamObservation(Observation):
    """
    Observation returned by the environment at every step.

    Fields are grouped by which task / step type populates them.
    Fields not relevant to the current task / step are ``None``.

    ── Shared fields (all tasks, all steps) ─────────────────────────────────
    step_budget        : remaining step budget for the episode
    info               : reward_breakdown + cumulative reward details

    ── Reset fields (all tasks, step 0) ─────────────────────────────────────
    task_name          : which task variant is active ("easy"|"medium"|"hard")
    query_type         : channel type of the job opportunity
    initial_query      : raw text the candidate received
    available_context  : names of hidden context fields the client may request

    ── MEDIUM TASK — information request fields ──────────────────────────────
    requested_field    : name of the context field just returned
    field_content      : raw text content of that field

    ── EASY TASK — information request fields (placeholders) ─────────────────
    # TODO: add easy-specific observation fields here when task is designed

    ── HARD TASK — information request fields (placeholders) ─────────────────
    # TODO: add hard-specific observation fields here when task is designed

    ── Classification / terminal fields (all tasks) ──────────────────────────
    predicted_label    : label the client submitted
    actual_label       : ground-truth label (revealed only at terminal step)
    episode_done       : True when the episode has ended (classification or timeout)
    reason             : "classification" | "timeout"
    """

    # ── Shared / budget ──────────────────────────────────────────────────────
    step_budget: Optional[Dict[str, int]] = Field(
        default=None,
        description="Keys: total, used, remaining.",
    )

    # ── Reset — shared across all tasks ──────────────────────────────────────
    task_name: Optional[str] = Field(
        default=None,
        description=(
            "Task variant active for this episode: 'easy' | 'medium' | 'hard'. "
            "Set at reset and echoed in every subsequent observation."
        ),
    )
    query_type: Optional[str] = Field(
        default=None,
        description="Channel type: job_post | email | whatsapp_msg | telegram_msg.",
    )
    initial_query: Optional[str] = Field(
        default=None,
        description="Raw text of the job opportunity as received by the candidate.",
    )

    # ── MEDIUM TASK ───────────────────────────────
    available_context: Optional[List[str]] = Field(
        default=None,
        description="[Medium] Names of hidden context fields the client may request.",
    )
    requested_field: Optional[str] = Field(
        default=None,
        description="[Medium] Name of the context field that was just returned.",
    )
    field_content: Optional[str] = Field(
        default=None,
        description="[Medium] Raw text content of the requested context field.",
    )

    # ── EASY TASK ─────────────────
    # TODO: add easy-task-specific observation fields here.
    # Example:
    #   easy_summary: Optional[str] = Field(default=None, description="[Easy] ...")

    # ── HARD TASK ─────────────────
    # TODO: add hard-task-specific observation fields here.
    # Example:
    #   hard_social_profile: Optional[str] = Field(default=None, description="[Hard] ...")
    #   hard_payment_history: Optional[str] = Field(default=None, description="[Hard] ...")

    # ── Terminal / classification fields (all tasks) ──────────────────────────
    predicted_label: Optional[str] = Field(
        default=None,
        description="Label submitted by the client.",
    )
    actual_label: Optional[str] = Field(
        default=None,
        description="Ground-truth label revealed only at terminal step.",
    )
    episode_done: Optional[bool] = Field(
        default=None,
        description="True when the episode has ended (classification or timeout).",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason for episode termination: 'classification' | 'timeout'.",
    )
    info: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Contains reward_breakdown (step-level) and cumulative reward "
            "totals so the client can inspect grading details."
        ),
    )