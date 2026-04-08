"""
Job Scam Detection Environment — core implementation.

Implements the OpenEnv ``Environment`` interface for a step-based
job-scam investigation task.  All grading and reward computation is
fully programmatic (rule-based) — no LLM calls occur inside this file.

Multi-task architecture
-----------------------
The environment supports three difficulty variants:

  easy   — lightweight task (implementation pending)
  medium — original task; uses 4 requestable context fields, 5 steps
  hard   — extended task (implementation pending)

The active task variant is chosen at ``reset()`` time via the
``task_name`` keyword argument.  All task-specific logic is isolated in
methods and constants prefixed with the task name (e.g. ``_medium_*``).

To add a new task
-----------------
1. Add its constants to ``constants.py`` (EASY_*/HARD_* blocks).
2. Add ``_<task>_compute_field_scores``, ``_<task>_handle_field_request``,
   ``_<task>_handle_classify``, ``_<task>_handle_timeout`` methods.
3. Wire them into ``reset()`` / ``step()`` via the if/elif/else dispatch.

Signal score formula (medium task — §3 of architecture doc)
------------------------------------------------------------
  signal_score(field) = (|red_categories| + |green_categories|) /
                        total_unique_categories_in_sample

Reward structure (medium task — §6-9 of architecture doc)
----------------------------------------------------------
  Information request
    signal_reward      = 0.10 × signal_score(field)   [valid & new]
    redundancy_penalty = −0.20                         [already seen]
    irrelevant_penalty = −0.10                         [signal == 0]

  Classification (terminal)
    classification_reward    = REWARD_MATRIX[predicted][ground_truth]
    alpha                    = +0.1 if correct else −0.1
    total_steps_taken_reward = alpha × remaining_steps_at_classification

  Timeout (no classify before budget exhaustion)
    timeout_penalty = −1.5
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4
import json
from pathlib import Path

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ActionType, ClassificationLabel, JobScamAction, JobScamObservation
    from ..constants import (
        VALID_TASK_NAMES,
        # Easy
        EASY_DATASET_FILENAME, EASY_ACTION_TO_FIELD,
        # Medium
        MEDIUM_REWARD_MATRIX, MEDIUM_ACTION_TO_FIELD, MEDIUM_ALL_CONTEXT_FIELDS,
        MEDIUM_MAX_STEPS, MEDIUM_TIMEOUT_PENALTY, MEDIUM_DATASET_FILENAME,
        # Hard
        HARD_DATASET_FILENAME, HARD_ACTION_TO_FIELD,
    )
except ImportError:
    from models import ActionType, ClassificationLabel, JobScamAction, JobScamObservation
    from constants import (
        VALID_TASK_NAMES,
        EASY_DATASET_FILENAME,
        MEDIUM_REWARD_MATRIX, MEDIUM_ACTION_TO_FIELD, MEDIUM_ALL_CONTEXT_FIELDS,
        MEDIUM_MAX_STEPS, MEDIUM_TIMEOUT_PENALTY, MEDIUM_DATASET_FILENAME,
        HARD_DATASET_FILENAME,
    )


# ---------------------------------------------------------------------------
# Dataset loader (shared utility)
# ---------------------------------------------------------------------------
def _resolve_dataset_path(dataset_filename: str) -> Path:
    """
    Resolve dataset path robustly both in local dev and HF Space deployment.
    The JSONL must be shipped with the package for this to work in site-packages.
    """
    current_dir = Path(__file__).resolve().parent

    candidate_paths = [
        current_dir / dataset_filename,
        current_dir.parent / dataset_filename,
        current_dir.parent.parent / dataset_filename,
        Path.cwd() / dataset_filename,
        Path.cwd() / "server" / dataset_filename,
    ]

    for path in candidate_paths:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Dataset file '{dataset_filename}' not found.\n"
        f"Tried:\n" + "\n".join(str(p) for p in candidate_paths)
    )

def _load_dataset(dataset_filename: str) -> List[Dict[str, Any]]:
    """Load a JSONL dataset file. Raises clearly if file is missing or malformed."""
    jsonl_path = _resolve_dataset_path(dataset_filename)
    dataset: List[Dict[str, Any]] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                dataset.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {jsonl_path} at line {line_no}: {e}"
                ) from e

    if not dataset:
        raise ValueError(f"Dataset file {jsonl_path} is empty.")

    return dataset

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class JobScamEnvironment(Environment):
    """
    Step-based job scam investigation environment supporting three task variants.

    Episode lifecycle
    -----------------
    1. ``reset(task_name=...)``  → client receives initial query + available
                                   context list for the chosen task.
    2. Steps 1–N                 → client requests hidden context fields.
    3. Terminal step             → client calls ``classify(label)``; episode
                                   ends with classification + timing reward.
    4. Timeout                   → if the client exhausts all steps without
                                   classifying, a fixed penalty is applied.

    Each episode is independent; ``reset()`` selects a random sample from
    the dataset that corresponds to the chosen task.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode: Dict[str, Any] = {}

        # Active task for the current episode (set in reset())
        self._task_name: str = "medium"

        # ── Datasets — loaded once per process ──────────────────────────────
        # Medium dataset is always loaded (it ships with the package).
        self._MEDIUM_DATASET: List[Dict[str, Any]] = _load_dataset(MEDIUM_DATASET_FILENAME)

        # Easy / hard datasets are loaded lazily on first use so the server
        # starts even when those JSONL files are not yet present.
        self._EASY_DATASET: Optional[List[Dict[str, Any]]] = None
        self._HARD_DATASET: Optional[List[Dict[str, Any]]] = None

    # ---------------------------------------------------------------- reset
    def reset(self, task_name: str = "medium") -> JobScamObservation:
        """
        Start a new episode with a randomly selected dataset sample.

        Parameters
        ----------
        task_name:
            Which task variant to run.  Must be one of: easy, medium, hard.
            Defaults to "medium" for backward compatibility.
        """
        if task_name not in VALID_TASK_NAMES:
            raise ValueError(
                f"Unknown task_name '{task_name}'. "
                f"Must be one of: {VALID_TASK_NAMES}."
            )

        self._task_name = task_name
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # ── Dispatch to task-specific reset logic ────────────────────────────
        if task_name == "easy":
            return self._easy_reset()
        elif task_name == "medium":
            return self._medium_reset()
        elif task_name == "hard":
            return self._hard_reset()

    # ---------------------------------------------------------------- step
    def step(self, action: JobScamAction) -> JobScamObservation:  # type: ignore[override]
        """
        Execute one action and return the resulting observation.

        The action is validated against the active task before dispatch.
        """
        if self._episode.get("done"):
            raise RuntimeError(
                "Episode is already done.  Call reset() to start a new one."
            )

        self._episode["used_steps"] += 1
        self._state.step_count += 1

        # ── Validate that the action type belongs to the active task ─────────
        self._validate_action_for_task(action)

        # ── Dispatch to task-specific step logic ─────────────────────────────
        if self._task_name == "easy":
            ## Implement your own logic ##
            pass

        elif self._task_name == "medium":
            if action.action_type == ActionType.CLASSIFY:
                return self._medium_handle_classify(action)
            return self._medium_handle_field_request(action)

        elif self._task_name == "hard":
            ## Implement your own logic ##
            pass

    @property
    def state(self) -> State:
        return self._state

    # ---------------------------------------------------------------- shared helpers
    def _budget_dict(self) -> Dict[str, int]:
        used      = self._episode["used_steps"]
        max_steps = self._episode["max_steps"]
        return {
            "total":     max_steps,
            "used":      used,
            "remaining": max_steps - used,
        }

    def _validate_action_for_task(self, action: JobScamAction) -> None:
        """
        Raise ValueError if the action_type does not belong to the active task
        (and is not the universal CLASSIFY action).
        """
        if action.action_type == ActionType.CLASSIFY:
            return  # always valid

        if self._task_name == "easy":
            valid = set(EASY_ACTION_TO_FIELD.keys())
        elif self._task_name == "medium":
            valid = set(MEDIUM_ACTION_TO_FIELD.keys())
        elif self._task_name == "hard":
            valid = set(HARD_ACTION_TO_FIELD.keys())
        else:
            valid = set()

        if action.action_type.value not in valid:
            raise ValueError(
                f"Action '{action.action_type.value}' is not valid for "
                f"task '{self._task_name}'. "
                f"Valid actions: {sorted(valid) or 'none defined yet'}."
            )

    # =========================================================================
    # EASY TASK implementation
    # =========================================================================
    def _easy_reset(self) -> JobScamObservation:
        """
        Reset the environment for an easy-task episode.

        TODO: Implement when the easy task design is finalised.
              Replace the NotImplementedError with the actual logic,
              following the same pattern as _medium_reset().
        """
        # Lazy-load the easy dataset on first use
        if self._EASY_DATASET is None:
            self._EASY_DATASET = _load_dataset(EASY_DATASET_FILENAME)

        # TODO: implement easy reset logic here
        raise NotImplementedError(
            "Easy task is not yet implemented. "
            "Add the reset logic in _easy_reset() and update constants.py."
        )

    def _easy_handle_field_request(self, action: JobScamAction) -> JobScamObservation:
        """
        Handle an information-gathering action for the easy task.

        TODO: Implement when the easy task design is finalised.
        """
        raise NotImplementedError("Easy task field request not yet implemented.")

    def _easy_handle_classify(self, action: JobScamAction) -> JobScamObservation:
        """
        Handle the terminal classify action for the easy task.

        TODO: Implement when the easy task design is finalised.
        """
        raise NotImplementedError("Easy task classify not yet implemented.")

    def _easy_handle_timeout(self) -> JobScamObservation:
        """
        Handle step-budget exhaustion for the easy task.

        TODO: Implement when the easy task design is finalised.
        """
        raise NotImplementedError("Easy task timeout not yet implemented.")

    def _easy_compute_field_scores(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute signal scores for each field in an easy-task sample.

        TODO: Implement when the easy task design is finalised.
        """
        raise NotImplementedError("Easy task field score computation not yet implemented.")

    # =========================================================================
    # MEDIUM TASK implementation  (original logic, now under _medium_* names)
    # =========================================================================
    def _medium_reset(self) -> JobScamObservation:
        """Start a new medium-task episode with a randomly selected sample."""
        sample       = random.choice(self._MEDIUM_DATASET)
        field_scores = self._medium_compute_field_scores(sample)

        self._episode = {
            "sample":                        sample,
            "field_scores":                  field_scores,
            "requested_fields":              set(),           # type: Set[str]
            "used_steps":                    0,
            "max_steps":                     MEDIUM_MAX_STEPS,
            "requested_fields_reward_total": 0.0,
            "classification_reward_total":   0.0,
            "total_reward":                  0.0,
            "done":                          False,
        }

        return JobScamObservation(
            task_name=self._task_name,
            query_type=sample["query_type"],
            initial_query=sample["initial_query"],
            available_context=MEDIUM_ALL_CONTEXT_FIELDS,
            step_budget={
                "total":     MEDIUM_MAX_STEPS,
                "used":      0,
                "remaining": MEDIUM_MAX_STEPS,
            },
            episode_done=False,
            done=False,
            reward=0.0,
            info={},
        )

    def _medium_compute_field_scores(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute signal scores for every field in a medium-task sample.

        signal_score(field) = (|red_cats| + |green_cats|) /
                              total_unique_categories_in_sample
        """
        all_cats: Set[str] = set()
        for field_data in sample["fields"].values():
            all_cats.update(field_data.get("red_flag_categories", []))
            all_cats.update(field_data.get("green_flag_categories", []))

        total_unique = len(all_cats)
        if total_unique == 0:
            return {f: 0.0 for f in sample["fields"]}

        scores: Dict[str, float] = {}
        for field_name, field_data in sample["fields"].items():
            n_red   = len(field_data.get("red_flag_categories", []))
            n_green = len(field_data.get("green_flag_categories", []))
            scores[field_name] = round((n_red + n_green) / total_unique, 4)

        return scores

    def _medium_handle_field_request(self, action: JobScamAction) -> JobScamObservation:
        """Handle a context-field request for the medium task."""
        field_name    = MEDIUM_ACTION_TO_FIELD[action.action_type.value]
        sample        = self._episode["sample"]
        field_data    = sample["fields"][field_name]
        field_content = field_data["content"]
        signal        = self._episode["field_scores"].get(field_name, 0.0)
        already_seen  = field_name in self._episode["requested_fields"]

        # ── Compute step reward ──────────────────────────────────────────────
        if already_seen:
            step_reward      = -0.20
            reward_breakdown = {
                "signal_reward":             0.0,
                "redundancy_penalty":       -0.20,
                "irrelevant_field_penalty":  0.0,
                "normalized_reward":         None,
            }
        elif signal == 0.0:
            step_reward      = -0.10
            reward_breakdown = {
                "signal_reward":             0.0,
                "redundancy_penalty":        0.0,
                "irrelevant_field_penalty": -0.05,
                "normalized_reward":         None,
            }
        else:
            step_reward      = round(0.10 * signal, 4)
            reward_breakdown = {
                "signal_reward":            step_reward,
                "redundancy_penalty":       0.0,
                "irrelevant_field_penalty": 0.0,
                "normalized_reward":        None,
            }

        # Mark as seen (even redundant requests are recorded to track the penalty)
        self._episode["requested_fields"].add(field_name)

        # ── Update cumulative totals ─────────────────────────────────────────
        self._episode["requested_fields_reward_total"] = round(
            self._episode["requested_fields_reward_total"] + step_reward, 4
        )
        self._episode["total_reward"] = round(
            self._episode["total_reward"] + step_reward, 4
        )

        budget = self._budget_dict()

        # ── Trigger timeout if this was the last step ────────────────────────
        if budget["remaining"] == 0:
            return self._medium_handle_timeout(extra_info_reward=step_reward)

        return JobScamObservation(
            task_name=self._task_name,
            requested_field=field_name,
            field_content=field_content,
            step_budget=budget,
            episode_done=False,
            done=False,
            reward=step_reward,
            info={
                "reward_breakdown": reward_breakdown,
                "cumulative": {
                    "requested_fields_reward_total": self._episode["requested_fields_reward_total"],
                    "classification_reward_total":   self._episode["classification_reward_total"],
                    "total_reward":                  self._episode["total_reward"],
                },
            },
        )

    def _medium_handle_classify(self, action: JobScamAction) -> JobScamObservation:
        """Handle the terminal classify action for the medium task."""
        predicted    = action.label.value          # type: ignore[union-attr]
        ground_truth = self._episode["sample"]["ground_truth"]
        correct      = predicted == ground_truth
        remaining    = (
            self._episode["max_steps"] - self._episode["used_steps"]
        )

        classification_reward    = MEDIUM_REWARD_MATRIX[predicted][ground_truth]
        alpha                    = 0.1 if correct else -0.1
        total_steps_taken_reward = round(alpha * remaining, 4)
        terminal_reward          = round(
            classification_reward + total_steps_taken_reward, 4
        )

        maxv = 1+((MEDIUM_MAX_STEPS-1)*0.1) # 0.1 refers to alpha
        minv = -maxv
        # MinMax Normalization
        normalized_reward = round((terminal_reward - minv) / (maxv - minv), 4) if maxv != minv else 0.0

        self._episode["classification_reward_total"] = terminal_reward
        self._episode["total_reward"] = round(
            self._episode["total_reward"] + terminal_reward, 4
        )
        self._episode["done"] = True

        return JobScamObservation(
            task_name=self._task_name,
            predicted_label=predicted,
            actual_label=ground_truth,
            step_budget=self._budget_dict(),
            done=True,
            episode_done=True,
            reason="classification",
            reward=normalized_reward,
            info={
                "reward_breakdown": {
                    "classification_reward":    classification_reward,
                    "total_steps_taken_reward": total_steps_taken_reward,
                    "timeout_penalty":          0.0,
                    "normalized_reward":        normalized_reward,
                },
                "cumulative": {
                    "requested_fields_reward_total": self._episode["requested_fields_reward_total"],
                    "classification_reward_total":   self._episode["classification_reward_total"],
                    "total_reward":                  self._episode["total_reward"],
                },
            },
        )

    def _medium_handle_timeout(
        self,
        extra_info_reward: float = 0.0,  # noqa: ARG002 — already applied before call
    ) -> JobScamObservation:
        """
        Called when the client exhausts all steps without classifying (medium task).

        ``extra_info_reward`` has already been added to cumulative totals by
        the caller; only the timeout penalty is added here.
        """
        maxv = 1+((MEDIUM_MAX_STEPS-1)*0.1) # 0.1 refers to alpha
        minv = -maxv
        # MinMax Normalization
        normalized_reward = round((MEDIUM_TIMEOUT_PENALTY - minv) / (maxv - minv), 4) if maxv != minv else 0.0

        self._episode["total_reward"] = round(
            self._episode["total_reward"] + MEDIUM_TIMEOUT_PENALTY, 4
        )
        self._episode["done"] = True

        return JobScamObservation(
            task_name=self._task_name,
            episode_done=True,
            reason="timeout",
            step_budget=self._budget_dict(),
            done=True,
            reward=MEDIUM_TIMEOUT_PENALTY,
            info={
                "reward_breakdown": {
                    "classification_reward":    0.0,
                    "total_steps_taken_reward": 0.0,
                    "timeout_penalty":          MEDIUM_TIMEOUT_PENALTY,
                    "normalized_reward":        normalized_reward,
                },
                "cumulative": {
                    "requested_fields_reward_total": self._episode["requested_fields_reward_total"],
                    "classification_reward_total":   self._episode["classification_reward_total"],
                    "total_reward":                  self._episode["total_reward"],
                },
            },
        )

    # =========================================================================
    # HARD TASK implementation
    # =========================================================================
    def _hard_reset(self) -> JobScamObservation:
        """
        Reset the environment for a hard-task episode.

        TODO: Implement when the hard task design is finalised.
              Replace the NotImplementedError with the actual logic,
              following the same pattern as _medium_reset().
        """
        # Lazy-load the hard dataset on first use
        if self._HARD_DATASET is None:
            self._HARD_DATASET = _load_dataset(HARD_DATASET_FILENAME)

        # TODO: implement hard reset logic here
        raise NotImplementedError(
            "Hard task is not yet implemented. "
            "Add the reset logic in _hard_reset() and update constants.py."
        )

    def _hard_handle_field_request(self, action: JobScamAction) -> JobScamObservation:
        """
        Handle an information-gathering action for the hard task.

        TODO: Implement when the hard task design is finalised.
        """
        raise NotImplementedError("Hard task field request not yet implemented.")

    def _hard_handle_classify(self, action: JobScamAction) -> JobScamObservation:
        """
        Handle the terminal classify action for the hard task.

        TODO: Implement when the hard task design is finalised.
        """
        raise NotImplementedError("Hard task classify not yet implemented.")

    def _hard_handle_timeout(self) -> JobScamObservation:
        """
        Handle step-budget exhaustion for the hard task.

        TODO: Implement when the hard task design is finalised.
        """
        raise NotImplementedError("Hard task timeout not yet implemented.")

    def _hard_compute_field_scores(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute signal scores for each field in a hard-task sample.

        TODO: Implement when the hard task design is finalised.
        """
        raise NotImplementedError("Hard task field score computation not yet implemented.")