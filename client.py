"""
Job Scam Detection Environment — typed client.

Wraps the OpenEnv ``EnvClient`` base class with concrete serialisation
and deserialisation logic for ``JobScamAction`` and ``JobScamObservation``.

Usage
-----
::

    from client import JobScamEnv
    from models import ActionType, ClassificationLabel, JobScamAction

    with JobScamEnv(base_url="http://localhost:8000") as env:
        # Start a new episode
        result = env.reset()
        obs = result.observation
        print(obs.query_type, obs.initial_query)

        # Request a context field
        result = env.step(JobScamAction(action_type=ActionType.REQUEST_COMPANY_PROFILE))
        print(result.observation.field_content)
        print("Step reward:", result.reward)
        print("Info:", result.observation.metadata.get("info"))

        # Classify
        result = env.step(
            JobScamAction(
                action_type=ActionType.CLASSIFY,
                label=ClassificationLabel.SCAM,
            )
        )
        print("Done:", result.done)
        print("Episode reward:", result.reward)

Docker usage
------------
::

    client = JobScamEnv.from_docker_image("job_scam_env-env:latest")
    try:
        result = client.reset()
        result = client.step(JobScamAction(action_type="request_company_profile"))
    finally:
        client.close()
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import ActionType, JobScamAction, JobScamObservation
except ImportError:
    from models import ActionType, JobScamAction, JobScamObservation


class JobScamEnv(EnvClient[JobScamAction, JobScamObservation, State]):
    """
    Typed client for the Job Scam Detection Environment.

    Maintains a persistent WebSocket connection to the environment server
    so that each ``step()`` call incurs minimal latency.  Every client
    instance gets its own isolated environment session on the server.
    """

    # ------------------------------------------------------------------ wire

    def _step_payload(self, action: JobScamAction) -> Dict[str, Any]:
        """
        Serialise a ``JobScamAction`` to a JSON-safe dict for transmission.

        The server deserialises this back into a ``JobScamAction`` instance.
        """
        payload: Dict[str, Any] = {"action_type": action.action_type.value}
        if action.label is not None:
            payload["label"] = action.label.value
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[JobScamObservation]:
        """
        Deserialise the server response into a typed ``StepResult``.

        The server sends::

            {
                "observation": { ...JobScamObservation fields... },
                "reward": float,
                "done": bool
            }

        ``metadata`` (containing ``info.reward_breakdown`` and
        ``info.cumulative``) lives inside the ``observation`` sub-object.
        """
        obs_raw = payload.get("observation", {})

        observation = JobScamObservation(
            # ── Reset fields ─────────────────────────────────────────────────
            query_type=obs_raw.get("query_type"),
            initial_query=obs_raw.get("initial_query"),
            available_context=obs_raw.get("available_context"),
            # ── Shared budget ─────────────────────────────────────────────────
            step_budget=obs_raw.get("step_budget"),
            # ── Info request fields ───────────────────────────────────────────
            requested_field=obs_raw.get("requested_field"),
            field_content=obs_raw.get("field_content"),
            # ── Terminal classification fields ────────────────────────────────
            predicted_label=obs_raw.get("predicted_label"),
            actual_label=obs_raw.get("actual_label"),
            # ── Timeout fields ────────────────────────────────────────────────
            episode_done=obs_raw.get("episode_done"),
            reason=obs_raw.get("reason"),
            # ── OpenEnv base fields ───────────────────────────────────────────
            done=payload.get("done", False),
            reward=payload.get("reward"),
            # metadata=obs_raw.get("metadata", {}),
            info=obs_raw.get("info", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Deserialise the server's ``/state`` response into a ``State`` object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )