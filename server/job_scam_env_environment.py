"""
Job Scam Detection Environment — core implementation.

Implements the OpenEnv ``Environment`` interface for a step-based
job-scam investigation task.  All grading and reward computation is
fully programmatic (rule-based) — no LLM calls occur inside this file.

Architecture reference
----------------------
Each episode loads one dataset sample and tracks:
  - which context fields have been requested
  - how many steps have been used
  - per-field signal scores (computed once at reset, hidden from client)
  - cumulative reward components

The client receives only:
  - the initial query framing
  - field content when requested
  - step rewards
  - the ground-truth label at the terminal classification step

Signal score formula (per architecture doc §3)
----------------------------------------------
  signal_score(field) = (|red_categories| + |green_categories|) /
                        total_unique_categories_in_sample

  where ``total_unique_categories_in_sample`` is the count of distinct
  category strings across ALL fields in the sample row.

Reward structure (per architecture doc §6–9)
--------------------------------------------
  Information request
    signal_reward      = 0.10 × signal_score(field)   [if valid & new]
    redundancy_penalty = −0.20                         [if already seen]
    irrelevant_penalty = −0.10                         [if signal == 0]

  Classification (terminal)
    classification_reward   = REWARD_MATRIX[predicted][ground_truth]
    alpha                   = +0.1 if correct else −0.1
    total_steps_taken_reward = alpha × remaining_steps_at_classification

  Timeout (no classify before budget exhaustion)
    classification_reward    = 0
    total_steps_taken_reward = 0
    timeout_penalty          = −1.5
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4
import json
import os
from pathlib import Path

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ActionType, ClassificationLabel, JobScamAction, JobScamObservation
except ImportError:
    from models import ActionType, ClassificationLabel, JobScamAction, JobScamObservation


# ---------------------------------------------------------------------------
# Classification reward matrix   REWARD_MATRIX[predicted][ground_truth]
# ---------------------------------------------------------------------------

_REWARD_MATRIX: Dict[str, Dict[str, float]] = {
    "legit": {
        "legit":             1.00,
        "suspicious":       -0.30,
        "scam":             -1.00,
        "insufficient_info":-0.20,
    },
    "suspicious": {
        "legit":            -0.10,
        "suspicious":        1.00,
        "scam":             -0.30,
        "insufficient_info":-0.10,
    },
    "scam": {
        "legit":            -0.50,
        "suspicious":       -0.10,
        "scam":              1.00,
        "insufficient_info":-0.30,
    },
    "insufficient_info": {
        "legit":            -0.20,
        "suspicious":       -0.20,
        "scam":             -0.50,
        "insufficient_info": 1.00,
    },
}

# Maps action_type strings to their corresponding field names in the dataset.
_ACTION_TO_FIELD: Dict[str, str] = {
    ActionType.REQUEST_RECRUITER_PROFILE: "recruiter_profile",
    ActionType.REQUEST_COMPANY_PROFILE:   "company_profile",
    ActionType.REQUEST_THREAD_HISTORY:    "thread_history",
    ActionType.REQUEST_JOB_POST_COMMENTS: "job_post_comments",
}

_ALL_CONTEXT_FIELDS: List[str] = list(_ACTION_TO_FIELD.values())
_MAX_STEPS: int = 5
_TIMEOUT_PENALTY: float = -1.5


# ---------------------------------------------------------------------------
# Dataset loader
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
    """
    Load the dataset only once per process.
    Cached so repeated env objects do not reread the file.
    """
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
    Step-based job scam investigation environment.

    Episode lifecycle
    -----------------
    1. ``reset()``   → client receives initial query + available context list.
    2. Steps 1–4     → client requests hidden context fields one at a time.
    3. Terminal step → client calls ``classify(label)``; episode ends with
                       classification reward + timing reward.
    4. Timeout       → if the client exhausts all steps without classifying,
                       episode ends with a fixed −1.5 penalty.

    Each episode is independent; ``reset()`` selects a random sample from
    the embedded dataset.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode: Dict[str, Any] = {}

        # Loaded once per process, reused after that
        # self._EASY_DATASET: List[Dict[str, Any]] = _load_dataset("data_task_easy.jsonl")
        self._MEDIUM_DATASET: List[Dict[str, Any]] = _load_dataset("data_task_medium.jsonl")
        # self._HARD_DATASET: List[Dict[str, Any]] = _load_dataset("data_task_hard.jsonl")

    def reset(self) -> JobScamObservation:
        """Start a new episode with a randomly selected dataset sample."""
        self._state = State(episode_id=str(uuid4()), step_count=0)

        sample = random.choice(self._MEDIUM_DATASET)
        field_scores = self._compute_field_scores(sample)

        self._episode = {
            "sample":                     sample,
            "field_scores":               field_scores,
            "requested_fields":           set(),          # type: Set[str]
            "used_steps":                 0,
            "max_steps":                  _MAX_STEPS,
            "info_reward_total":          0.0,
            "classification_reward_total":0.0,
            "total_reward":               0.0,
            "done":                       False,
        }

        return JobScamObservation(
            query_type=sample["query_type"],
            initial_query=sample["initial_query"],
            available_context=_ALL_CONTEXT_FIELDS,
            step_budget={
                "total":     _MAX_STEPS,
                "used":      0,
                "remaining": _MAX_STEPS,
            },
            episode_done=False,
            done=False,
            reward=0.0,
            info={},
        )

    def step(self, action: JobScamAction) -> JobScamObservation:  # type: ignore[override]
        """
        Execute one action.

        Parameters
        ----------
        action:
            Either a context-field request or a terminal classification.

        Returns
        -------
        JobScamObservation
            Contains the reward for *this* action, ``done`` status, and
            a metadata ``info`` dict with ``reward_breakdown`` and
            ``cumulative`` totals.
        """
        if self._episode.get("done"):
            raise RuntimeError("Episode is already done.  Call reset() to start a new one.")

        self._episode["used_steps"] += 1
        self._state.step_count += 1

        if action.action_type == ActionType.CLASSIFY:
            return self._handle_classify(action)

        return self._handle_field_request(action)

    @property
    def state(self) -> State:
        return self._state

    # ---------------------------------------------------------------- helpers
    def _compute_field_scores(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute signal scores for every field in the sample.

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

    def _budget_dict(self) -> Dict[str, int]:
        used      = self._episode["used_steps"]
        max_steps = self._episode["max_steps"]
        return {
            "total":     max_steps,
            "used":      used,
            "remaining": max_steps - used,
        }

    # ------------------------------------------------------ field request
    def _handle_field_request(self, action: JobScamAction) -> JobScamObservation:
        field_name   = _ACTION_TO_FIELD[action.action_type]
        sample       = self._episode["sample"]
        field_data   = sample["fields"][field_name]
        field_content = field_data["content"]
        signal        = self._episode["field_scores"].get(field_name, 0.0)
        already_seen  = field_name in self._episode["requested_fields"]

        # ── Compute step reward ──────────────────────────────────────────────
        if already_seen:
            step_reward          = -0.20
            reward_breakdown     = {
                "signal_reward":          0.0,
                "redundancy_penalty":    -0.20,
                "irrelevant_field_penalty": 0.0,
            }
        elif signal == 0.0:
            step_reward          = -0.10
            reward_breakdown     = {
                "signal_reward":              0.0,
                "redundancy_penalty":         0.0,
                "irrelevant_field_penalty":  -0.10,
            }
        else:
            step_reward          = round(0.10 * signal, 4)
            reward_breakdown     = {
                "signal_reward":             step_reward,
                "redundancy_penalty":        0.0,
                "irrelevant_field_penalty":  0.0,
            }

        # Mark as seen (even if redundant — it's still "seen")
        self._episode["requested_fields"].add(field_name)

        # ── Update cumulative totals ─────────────────────────────────────────
        self._episode["info_reward_total"] = round(
            self._episode["info_reward_total"] + step_reward, 4
        )
        self._episode["total_reward"] = round(
            self._episode["total_reward"] + step_reward, 4
        )

        budget = self._budget_dict()

        # ── Check for timeout after this action ──────────────────────────────
        if budget["remaining"] == 0:
            return self._handle_timeout(extra_info_reward=step_reward)

        return JobScamObservation(
            requested_field=field_name,
            field_content=field_content,
            step_budget=budget,
            episode_done=False,
            done=False,
            reward=step_reward,
            info={
                "reward_breakdown": reward_breakdown,
                "cumulative": {
                    "requested_fields_reward_total": self._episode["info_reward_total"],
                    "classification_reward_total":   self._episode["classification_reward_total"],
                    "total_reward":                  self._episode["total_reward"],
                },
            },
        )

    # ------------------------------------------------------ classification
    def _handle_classify(self, action: JobScamAction) -> JobScamObservation:
        predicted    = action.label.value          # type: ignore[union-attr]
        ground_truth = self._episode["sample"]["ground_truth"]
        correct      = predicted == ground_truth
        remaining    = self._episode["max_steps"] - self._episode["used_steps"]

        classification_reward    = _REWARD_MATRIX[predicted][ground_truth]
        alpha                    = 0.1 if correct else -0.1
        total_steps_taken_reward = round(alpha * remaining, 4)
        terminal_reward          = round(classification_reward + total_steps_taken_reward, 4)

        self._episode["classification_reward_total"] = terminal_reward
        self._episode["total_reward"] = round(
            self._episode["total_reward"] + terminal_reward, 4
        )
        self._episode["done"] = True

        return JobScamObservation(
            predicted_label=predicted,
            actual_label=ground_truth,
            step_budget=self._budget_dict(),
            done=True,
            episode_done=True,
            reason="classification",
            reward=terminal_reward,
            info={
                "reward_breakdown": {
                    "classification_reward":    classification_reward,
                    "total_steps_taken_reward": total_steps_taken_reward,
                    "timeout_penalty":          0.0,
                },
                "cumulative": {
                    "info_reward_total":          self._episode["info_reward_total"],
                    "classification_reward_total":self._episode["classification_reward_total"],
                    "total_reward":               self._episode["total_reward"],
                },
            },
        )

    # ---------------------------------------------------------- timeout
    def _handle_timeout(self, extra_info_reward: float = 0.0) -> JobScamObservation:
        """
        Called when the client exhausts all steps without classifying.

        Note: ``extra_info_reward`` has already been added to
        ``info_reward_total`` and ``total_reward`` before this method
        is called, so only the timeout penalty itself is added here.
        """
        self._episode["total_reward"] = round(
            self._episode["total_reward"] + _TIMEOUT_PENALTY, 4
        )
        self._episode["done"] = True

        return JobScamObservation(
            episode_done=True,
            reason="timeout",
            step_budget=self._budget_dict(),
            done=True,
            reward=_TIMEOUT_PENALTY,
            info={
                "reward_breakdown": {
                    "classification_reward":    0.0,
                    "total_steps_taken_reward": 0.0,
                    "timeout_penalty":          _TIMEOUT_PENALTY,
                },
                "cumulative": {
                    "requested_fields_reward_total": self._episode["info_reward_total"],
                    "classification_reward_total":   self._episode["classification_reward_total"],
                    "total_reward":                  self._episode["total_reward"],
                },
            },
        )
