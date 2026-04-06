"""
Data models for the Job Scam Detection Environment.

Defines the action and observation types used by both the environment
server and the client. All types are OpenEnv spec compliant and extend
the base Action / Observation classes from openenv.core.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """All valid action types the client may submit."""

    REQUEST_RECRUITER_PROFILE = "request_recruiter_profile"
    REQUEST_COMPANY_PROFILE   = "request_company_profile"
    REQUEST_THREAD_HISTORY    = "request_thread_history"
    REQUEST_JOB_POST_COMMENTS = "request_job_post_comments"
    CLASSIFY                  = "classify"


class ClassificationLabel(str, Enum):
    """Four mutually exclusive classification outcomes."""

    LEGIT             = "legit"
    SUSPICIOUS        = "suspicious"
    SCAM              = "scam"
    INSUFFICIENT_INFO = "insufficient_info"


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class JobScamAction(Action):
    """
    A single step action submitted by the client.

    For information-gathering steps, set ``action_type`` to one of the
    four ``REQUEST_*`` variants and leave ``label`` as ``None``.

    For the terminal classification step, set ``action_type`` to
    ``CLASSIFY`` and provide a ``ClassificationLabel`` in ``label``.

    Example (info request)::

        JobScamAction(action_type="request_company_profile")

    Example (classification)::

        JobScamAction(action_type="classify", label="scam")
    """

    action_type: ActionType = Field(
        ...,
        description="Type of action: a context request or final classification.",
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
        if self.action_type != ActionType.CLASSIFY and self.label is not None:
            raise ValueError(
                "label must be None for non-classify actions."
            )
        return self


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class JobScamObservation(Observation):
    """
    Observation returned by the environment at every step.

    The fields present depend on the step type:

    Reset (step 0)
    --------------
    - query_type
    - initial_query
    - available_context
    - step_budget

    Information request (steps 1–4)
    --------------------------------
    - requested_field
    - field_content
    - step_budget

    Classification / terminal (classify action or timeout)
    -------------------------------------------------------
    - predicted_label     (classification only)
    - actual_label        (classification only)
    - episode_done        (timeout only)
    - reason              (timeout only, value: "timeout")
    - step_budget

    The ``metadata`` dict (inherited from Observation) always carries
    ``info.reward_breakdown`` and ``info.cumulative`` so the client can
    inspect the grading details without them being part of the primary
    observation contract.
    """

    # ── Reset fields ────────────────────────────────────────────────────────
    query_type: Optional[str] = Field(
        default=None,
        description="Channel type: job_post | email | whatsapp_msg | telegram_msg.",
    )
    initial_query: Optional[str] = Field(
        default=None,
        description="Raw text of the job opportunity as received by the candidate.",
    )
    available_context: Optional[List[str]] = Field(
        default=None,
        description="Names of hidden context fields the client may request.",
    )

    # ── Shared budget ────────────────────────────────────────────────────────
    step_budget: Optional[Dict[str, int]] = Field(
        default=None,
        description="Keys: total, used, remaining.",
    )

    # ── Info request fields ──────────────────────────────────────────────────
    requested_field: Optional[str] = Field(
        default=None,
        description="Name of the context field that was just returned.",
    )
    field_content: Optional[str] = Field(
        default=None,
        description="Raw text content of the requested context field.",
    )

    # ── Terminal classification fields ───────────────────────────────────────
    predicted_label: Optional[str] = Field(
        default=None,
        description="Label submitted by the client.",
    )
    actual_label: Optional[str] = Field(
        default=None,
        description="Ground-truth label revealed only at terminal step.",
    )

    # ── Timeout fields ───────────────────────────────────────────────────────
    episode_done: Optional[bool] = Field(
        default=None,
        description="True only when the episode ends by timeout.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason for episode termination (e.g. 'timeout').",
    )

    # ── Info ─────────────────────────────────────────────────────────────
    info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Contains reward_breakdown and cumulative reward information.",
    )