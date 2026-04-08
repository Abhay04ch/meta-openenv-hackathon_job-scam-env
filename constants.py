"""
Task-specific constants for the Job Scam Detection Environment.

Each task (easy / medium / hard) has its own prefixed constants block.
Import only the constants you need in the environment implementation.

Adding a new task
-----------------
1. Add a new block below following the same naming pattern.
2. Add the task name to VALID_TASK_NAMES.
3. Wire the new constants into JobScamEnvironment via if/elif/else.
"""

from __future__ import annotations
from typing import Dict, List

# ---------------------------------------------------------------------------
# Registry of valid task names
# ---------------------------------------------------------------------------
VALID_TASK_NAMES: List[str] = ["easy", "medium", "hard"]

# ===========================================================================
# EASY TASK constants  (placeholders — fill in when easy task is designed)
# ===========================================================================
# EASY_MAX_STEPS: int = 3
# EASY_TIMEOUT_PENALTY: float = -1.5
EASY_DATASET_FILENAME: str = "data_task_easy.jsonl"

# EASY_REWARD_MATRIX: Dict[str, Dict[str, float]] = {}
EASY_ACTION_TO_FIELD: Dict[str, str] = {}
# EASY_ALL_CONTEXT_FIELDS: List[str] = []


# ===========================================================================
# MEDIUM TASK constants  (the original / current task)
# ===========================================================================
MEDIUM_MAX_STEPS: int = 5
MEDIUM_TIMEOUT_PENALTY: float = (-1)*(1+((MEDIUM_MAX_STEPS-1)*0.1))
MEDIUM_DATASET_FILENAME: str = "data_task_medium.jsonl"

# Reward matrix: MEDIUM_REWARD_MATRIX[predicted][ground_truth]
MEDIUM_REWARD_MATRIX: Dict[str, Dict[str, float]] = {
    "legit": {
        "legit":              1.00,
        "suspicious":        -0.30,
        "scam":              -1.00,
        "insufficient_info": -0.20,
    },
    "suspicious": {
        "legit":             -0.10,
        "suspicious":         1.00,
        "scam":              -0.30,
        "insufficient_info": -0.10,
    },
    "scam": {
        "legit":             -0.50,
        "suspicious":        -0.10,
        "scam":               1.00,
        "insufficient_info": -0.30,
    },
    "insufficient_info": {
        "legit":             -0.20,
        "suspicious":        -0.20,
        "scam":              -0.50,
        "insufficient_info":  1.00,
    },
}

# Maps ActionType enum values → dataset field names (medium task)
MEDIUM_ACTION_TO_FIELD: Dict[str, str] = {
    "request_recruiter_profile": "recruiter_profile",
    "request_company_profile":   "company_profile",
    "request_thread_history":    "thread_history",
    "request_job_post_comments": "job_post_comments",
}

# Ordered list of all context field names the client may request (medium task)
MEDIUM_ALL_CONTEXT_FIELDS: List[str] = list(MEDIUM_ACTION_TO_FIELD.values())


# ===========================================================================
# HARD TASK constants  (placeholders — fill in when hard task is designed)
# ===========================================================================
# HARD_MAX_STEPS: int = 7
# HARD_TIMEOUT_PENALTY: float = -1.5
HARD_DATASET_FILENAME: str = "data_task_hard.jsonl"

# HARD_REWARD_MATRIX: Dict[str, Dict[str, float]] = {}
HARD_ACTION_TO_FIELD: Dict[str, str] = {}
# HARD_ALL_CONTEXT_FIELDS: List[str] = []