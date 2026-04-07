"""
Job Scam Detection Environment — typed client.

Wraps the OpenEnv ``EnvClient`` base class with concrete serialisation
and deserialisation logic for ``JobScamAction`` and ``JobScamObservation``.

Multi-task support
------------------
Pass ``task_name`` to ``reset()`` to choose which difficulty variant to run:

    result = await env.reset(task_name="medium")   # default
    result = await env.reset(task_name="easy")
    result = await env.reset(task_name="hard")

The task name is forwarded to the server as part of the reset payload and
echoed back in every observation's ``task_name`` field.

Usage
-----
::

    from client import JobScamEnv
    from models import ActionType, ClassificationLabel, JobScamAction

    async with JobScamEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task_name="medium")
        obs = result.observation
        print(obs.task_name, obs.query_type, obs.initial_query)

        result = await env.step(
            JobScamAction(action_type=ActionType.REQUEST_COMPANY_PROFILE)
        )
        print(result.observation.field_content)
        print("Step reward:", result.reward)

        result = await env.step(
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

    env = await JobScamEnv.from_docker_image("job_scam_env-env:latest")
    async with env:
        result = await env.reset(task_name="medium")
        result = await env.step(
            JobScamAction(action_type="request_company_profile")
        )
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import ActionType, JobScamAction, JobScamObservation
    from .constants import VALID_TASK_NAMES
except ImportError:
    from models import ActionType, JobScamAction, JobScamObservation
    from constants import VALID_TASK_NAMES


class JobScamEnv(EnvClient[JobScamAction, JobScamObservation, State]):
    """
    Typed async client for the Job Scam Detection Environment.

    Maintains a persistent WebSocket connection to the environment server
    so that each ``step()`` call incurs minimal latency.  Every client
    instance gets its own isolated environment session on the server.

    The client overrides ``reset()`` to forward ``task_name`` in the
    HTTP/WebSocket payload so the server knows which task variant to run.
    """

    # ------------------------------------------------------------------ reset

    async def reset(self, task_name: str = "medium") -> StepResult[JobScamObservation]:
        """
        Start a new episode on the server.

        Parameters
        ----------
        task_name:
            Which task variant to activate.  Must be one of:
            ``easy`` | ``medium`` | ``hard``.  Defaults to ``"medium"``.

        Returns
        -------
        StepResult[JobScamObservation]
            The initial observation containing query_type, initial_query,
            available_context, step_budget, and task_name.
        """
        if task_name not in VALID_TASK_NAMES:
            raise ValueError(
                f"Unknown task_name '{task_name}'. "
                f"Must be one of: {VALID_TASK_NAMES}."
            )

        # Send task_name to the server as part of the reset payload.
        # EnvClient.reset() is expected to accept **kwargs that are forwarded
        # to the server's /reset endpoint body.
        return await super().reset(task_name=task_name)

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
        """
        obs_raw = payload.get("observation", {})

        observation = JobScamObservation(
            # ── Shared ───────────────────────────────────────────────────────
            task_name=obs_raw.get("task_name"),
            step_budget=obs_raw.get("step_budget"),

            # ── Reset fields ──────────────────────────────────────────────────
            query_type=obs_raw.get("query_type"),
            initial_query=obs_raw.get("initial_query"),

            # ── MEDIUM TASK — information request fields ───────────────────────
            requested_field=obs_raw.get("requested_field"),
            field_content=obs_raw.get("field_content"),
            available_context=obs_raw.get("available_context"),

            # ── EASY TASK — information request fields (placeholders) ──────────
            # TODO: add easy-specific field deserialization here when designed
            # Example:
            #   easy_summary=obs_raw.get("easy_summary"),

            # ── HARD TASK — information request fields (placeholders) ──────────
            # TODO: add hard-specific field deserialization here when designed
            # Example:
            #   hard_social_profile=obs_raw.get("hard_social_profile"),
            #   hard_payment_history=obs_raw.get("hard_payment_history"),

            # ── Terminal / classification fields ──────────────────────────────
            predicted_label=obs_raw.get("predicted_label"),
            actual_label=obs_raw.get("actual_label"),
            episode_done=obs_raw.get("episode_done"),
            reason=obs_raw.get("reason"),
            info=obs_raw.get("info", {}),

            # ── OpenEnv base fields ───────────────────────────────────────────
            done=payload.get("done", False),
            reward=payload.get("reward"),
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