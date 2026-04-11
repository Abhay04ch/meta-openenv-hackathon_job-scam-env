"""
Inference Script — Job Scam Detection Environment
==================================================

An LLM-driven agent that plays episodes of the JobScamEnvironment for ALL
task variants (easy, medium, hard) in a single invocation, emitting
structured [START] / [STEP] / [END] logs that the validator parses.

Required environment variables
-------------------------------
API_BASE_URL   API endpoint for the LLM (e.g. https://router.huggingface.co/v1).
MODEL_NAME     Model identifier (e.g. Qwen/Qwen2.5-72B-Instruct).
HF_TOKEN       Hugging Face / API key.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from client import JobScamEnv
from models import ActionType, ClassificationLabel, JobScamAction, JobScamObservation
from constants import VALID_TASK_NAMES, EASY_MAX_STEPS, MEDIUM_MAX_STEPS, HARD_MAX_STEPS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()
API_KEY:       Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL:  str           = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME:    Optional[str] = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME", "job_scam_env:latest")
HF_SPACE_URL:  Optional[str] = os.getenv("HF_SPACE_URL", "https://sumitwarrior7-job-scam-env.hf.space")
BENCHMARK:   str   = "job_scam_detection"
TEMPERATURE: float = 0.2
MAX_TOKENS:  int   = 300

# ---------------------------------------------------------------------------
# All tasks to run — validator requires one [START]/[END] pair per task
# ---------------------------------------------------------------------------
TASKS: List[tuple[str, str, int]] = [
    ("task_easy",   "easy",   EASY_MAX_STEPS),
    ("task_medium", "medium", MEDIUM_MAX_STEPS),
    ("task_hard",   "hard",   HARD_MAX_STEPS),
]

_VALID_ACTION_TYPES = [a.value for a in ActionType]
_VALID_LABELS       = [l.value for l in ClassificationLabel]

# ---------------------------------------------------------------------------
# Structured log helpers  (validator-required stdout format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    done_str  = "true" if done  else "false"
    error_str = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"done={done_str} error={error_str}",
        flush=True,
    )


def log_end(
    task: str,
    success: bool,
    steps: int,
    score: float,
) -> None:
    success_str  = "true" if success else "false"
    print(
        f"[END] task={task} success={success_str} steps={steps} "
        f"score={score}",
        flush=True,
    )

def normalize_one(x):
    return int(x) if float(x) == 1.0 else x

def normalize_zero(x):
    return int(x) if float(x) == 0.0 else x

# ---------------------------------------------------------------------------
# System prompts (per task)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPTS: Dict[str, str] = {

    "easy": textwrap.dedent("""
        You are an expert job scam detector.

        For this easy task, read the message and classify it as either:
          legit | scam

        Output exactly one JSON object and nothing else.

        Format:
        {"action_type": "classify", "label": "<legit|scam>"}
    """).strip(),

    "medium": textwrap.dedent("""
        You are an expert job-scam investigator.
        Your task is to analyse a job opportunity and classify it as one of:
          legit | suspicious | scam | insufficient_info

        You have up to 5 steps per episode. Each step you must output exactly one
        JSON object — nothing else, no markdown, no explanation.

        AVAILABLE ACTIONS
        -----------------
        Information requests (use these to gather evidence before classifying):
          {"action_type": "request_recruiter_profile"}
          {"action_type": "request_company_profile"}
          {"action_type": "request_thread_history"}
          {"action_type": "request_job_post_comments"}

        Terminal action (must be your last action):
          {"action_type": "classify", "label": "<legit|suspicious|scam|insufficient_info>"}

        STRATEGY GUIDELINES
        -------------------
        - Look for red flags: payment requests, off-platform contact (WhatsApp/Telegram),
          government ID requests, urgency pressure, evasion tactics, gmail sender for MNC.
        - Look for green flags: official domain, no-fee statements, portal-only hiring,
          anti-scam guidance.
        - Classify as soon as you have enough evidence — earlier correct classification
          earns a timing bonus.
        - Do NOT request the same field twice — it incurs a -0.20 penalty.
        - Do NOT use classify before you have enough evidence.
        - Output ONLY a valid JSON object. No surrounding text.
    """).strip(),

    "hard": textwrap.dedent("""
       You are an expert job-scam investigator solving the HARD task.

        Your objective is to maximize reward by:
        1) following the most evidence-efficient investigation path
        2) avoiding redundant tools
        3) avoiding forbidden shortcuts
        4) classifying with the most accurate label possible

        VALID HARD ACTIONS
        ------------------
        {"action_type": "request_sender_profile"}
        {"action_type": "request_organization_profile"}
        {"action_type": "request_shared_channel_history"}
        {"action_type": "request_private_conversation_history"}
        {"action_type": "request_candidate_interaction_history"}
        {"action_type": "request_external_market_signals"}
        {"action_type": "request_attached_artifacts"}
        {"action_type": "request_temporal_context"}

        TERMINAL ACTION
        ---------------
        {"action_type": "classify", "label": "<legit|suspicious|scam|insufficient_info>"}

        TOOL STRATEGY
        -------------
        - Pick the NEXT MOST INFORMATIVE tool only
        - Prefer actions that directly validate the strongest suspicious signal
        - Use the minimum number of tools needed
        - Avoid repeating previously used tools
        - Use external signals before classifying if money, urgency, phishing, Telegram, UPI, shortened links, or off-platform movement is present
        - Use private conversation history when payment, urgency, slot booking, refundable token, or pressure tactics appear
        - Use attached artifacts only when documents, offer letters, or screenshots are mentioned
        - Do NOT over-investigate once enough evidence for classification exists

        CLASSIFICATION STRATEGY
        -----------------------
        - If multiple strong scam indicators align, classify as "scam"
        - If evidence strongly suggests fraud but one critical verification is missing, classify as "suspicious"
        - If official channels strongly confirm authenticity and no payment / urgency indicators exist, classify as "legit"
        - Use "insufficient_info" only when key verification evidence is missing

        OUTPUT RULE
        -----------
        Output ONLY one valid JSON object.
    """).strip(),
}



# ---------------------------------------------------------------------------
# Fallback action priority (per task)
# ---------------------------------------------------------------------------
_FALLBACK_PRIORITY: Dict[str, List[ActionType]] = {
    "easy": [],
    "medium": [
        ActionType.REQUEST_COMPANY_PROFILE,
        ActionType.REQUEST_RECRUITER_PROFILE,
        ActionType.REQUEST_THREAD_HISTORY,
        ActionType.REQUEST_JOB_POST_COMMENTS,
    ],
    "hard": [
        ActionType.REQUEST_SENDER_PROFILE,
        ActionType.REQUEST_ORGANIZATION_PROFILE,
        ActionType.REQUEST_SHARED_CHANNEL_HISTORY,
        ActionType.REQUEST_PRIVATE_CONVERSATION_HISTORY,
        ActionType.REQUEST_CANDIDATE_INTERACTION_HISTORY,
        ActionType.REQUEST_ATTACHED_ARTIFACTS,
        ActionType.REQUEST_TEMPORAL_CONTEXT,
        ActionType.REQUEST_EXTERNAL_MARKET_SIGNALS,
    ],
}

# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------

def _build_user_message(
    step: int,
    obs: JobScamObservation,
    history: List[str],
    task_name: str,
    max_steps: int,
) -> str:
    step_budget = obs.step_budget
    if isinstance(step_budget, dict):
        remaining = step_budget.get("remaining", max_steps - step)
    else:
        remaining = max_steps - step

    parts: List[str] = [
        f"TASK: {obs.task_name or task_name}",
        f"STEP: {step}",
        f"STEPS REMAINING: {remaining}",
        "",
    ]

    if obs.query_type:
        parts.append(f"QUERY TYPE: {obs.query_type}")
    if obs.initial_query:
        parts.append(f"INITIAL QUERY:\n{obs.initial_query}")

    if task_name in  ("medium"):
        if obs.requested_field and obs.field_content:
            parts.append(f"\nFIELD JUST RECEIVED — {obs.requested_field.upper()}:")
            parts.append(obs.field_content)
    elif task_name == "hard":
        parts.append("\nAVAILABLE HARD ACTIONS:")
        for field in obs.available_context or []:
            if field == "sender_profile":
                parts.append('- {"action_type": "request_sender_profile"}')
            elif field == "organization_profile":
                parts.append('- {"action_type": "request_organization_profile"}')
            elif field == "shared_channel_history":
                parts.append('- {"action_type": "request_shared_channel_history"}')
            elif field == "private_conversation_history":
                parts.append('- {"action_type": "request_private_conversation_history"}')
            elif field == "candidate_interaction_history":
                parts.append('- {"action_type": "request_candidate_interaction_history"}')
            elif field == "external_market_signals":
                parts.append('- {"action_type": "request_external_market_signals"}')
            elif field == "attached_artifacts":
                parts.append('- {"action_type": "request_attached_artifacts"}')
            elif field == "temporal_context":
                parts.append('- {"action_type": "request_temporal_context"}')

    if history:
        parts.append("\nACTIONS TAKEN SO FAR:")
        parts.extend(f"  {h}" for h in history)

    parts.append(
        "\nOutput exactly one JSON action object. "
        "Classify only when you have sufficient evidence."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_action(response_text: str) -> Optional[JobScamAction]:
    match = _JSON_RE.search(response_text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    action_type_str: str = data.get("action_type", "")
    if action_type_str not in _VALID_ACTION_TYPES:
        return None

    label_str: Optional[str] = data.get("label")

    if action_type_str == ActionType.CLASSIFY.value:
        if label_str not in _VALID_LABELS:
            return None
        return JobScamAction(
            action_type=ActionType(action_type_str),
            label=ClassificationLabel(label_str),
        )

    if label_str is not None:
        return None  # label must not be present for non-classify actions

    return JobScamAction(action_type=ActionType(action_type_str))


def _fallback_action(task_name: str, requested: List[str]) -> JobScamAction:
    for action_type in _FALLBACK_PRIORITY.get(task_name, []):
        field = action_type.value.replace("request_", "")
        if field not in requested:
            return JobScamAction(action_type=action_type)
    return JobScamAction(
        action_type=ActionType.CLASSIFY,
        label=ClassificationLabel.INSUFFICIENT_INFO,
    )


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

async def run_episode(
    env: JobScamEnv,
    llm_client: OpenAI,
    task_id: str,
    task_name: str,
    max_steps: int,
) -> tuple[bool, int, float, List[float]]:
    """
    Run one episode for the given task variant.

    Returns (success, steps_taken, total_reward, per_step_rewards).
    Emits [STEP] lines to stdout.
    """
    result = await env.reset(task_name=task_name)
    obs    = result.observation

    history:          List[str]   = []
    requested_fields: List[str]   = []
    per_step_rewards: List[float] = []
    total_reward:     float       = 0.0
    success:          bool        = False
    steps_taken:      int         = 0

    initial_query_text = obs.initial_query or ""
    print(f"\n{'='*60}")
    print(f"NEW EPISODE  [task_id={task_id}  task_name={task_name}]")
    print(f"Query type : {obs.query_type}")
    print(f"Initial    : {initial_query_text[:120]}...")
    print(f"{'='*60}")

    system_prompt = _SYSTEM_PROMPTS[task_name]

    for step in range(1, max_steps + 1):
        steps_taken = step
        step_error:  Optional[str] = None

        # ── Build LLM messages ──────────────────────────────────────────────
        user_msg = _build_user_message(step, obs, history, task_name, max_steps)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ]

        # ── Call LLM ───────────────────────────────────────────────────────
        response_text = ""
        try:
            completion = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            step_error = str(exc)

        # ── Parse action ────────────────────────────────────────────────────
        action = _parse_action(response_text)
        if action is None:
            print(
                f"  [Step {step}] Unparseable response: "
                f"{response_text!r:.80}. Using fallback."
            )
            action = _fallback_action(task_name, requested_fields)

        action_str = action.action_type.value
        if action.label:
            action_str += f":{action.label.value}"

        # ── Step environment ────────────────────────────────────────────────
        result       = await env.step(action)
        obs          = result.observation
        reward       = result.reward or 0.0
        total_reward += reward
        per_step_rewards.append(reward)
        normalised_reward = 0

        is_done = bool(result.done)

        # ── Emit structured [STEP] log ──────────────────────────────────────
        log_step(step=step, action=action_str, reward=reward, done=is_done, error=step_error)

        # ── Track history for info-request actions ──────────────────────────
        is_classify = action.action_type in (ActionType.CLASSIFY)
        if not is_classify:
            if task_name == "easy":
                # TODO: derive field name for easy-task actions when designed
                # field_name = action.action_type.value.replace("request_", "")
                pass
            elif task_name == "medium":
                field_name = action.action_type.value.replace("request_", "")
                requested_fields.append(field_name)
                history.append(
                    f"Step {step}: {action.action_type.value}  → reward {reward:+.4f}"
                )

                info = obs.info or {}
                print(f"         Step Reward    : {reward:+.4f}")
                if info.get("reward_breakdown"):
                    print(f"         Breakdown      : {info['reward_breakdown']}")
                if info.get("cumulative"):
                    print(f"         Cumulative     : {info['cumulative']}")
            elif task_name == "hard":
                field_name = action.action_type.value.replace("request_", "")
                requested_fields.append(field_name)

                history.append(
                    f"Step {step}: {action.action_type.value} → reward {reward:+.4f}"
                )

                info = obs.info or {}

                print(f"         Step Reward    : {reward:+.4f}")

                if info.get("reward_breakdown"):
                    print(f"         Breakdown      : {info['reward_breakdown']}")

                if info.get("trajectory"):
                    print(f"         Trajectory     : {info['trajectory']}")

                if info.get("cumulative"):
                    print(f"         Cumulative     : {info['cumulative']}")

        # ── Handle done ─────────────────────────────────────────────────────
        if is_done:
            if task_name == "easy":
                normalised_reward = reward
                normalised_reward = normalize_one(normalize_zero(normalised_reward))
                correct = obs.predicted_label == obs.actual_label
                success = bool(correct)
                print(f"\n  CLASSIFICATION RESULT")
                print(f"    Step reward  : {normalised_reward}")
                print(f"    Predicted    : {obs.predicted_label}")
                print(f"    Actual       : {obs.actual_label}")
                print(f"    Correct      : {correct}\n")
                break
            elif task_name == "medium":
                normalised_reward = reward
                normalised_reward = normalize_one(normalize_zero(normalised_reward))
                info = obs.info or {}
                if obs.reason == "timeout":
                    success = False
                    print(f"\n  TIMEOUT — all steps exhausted without classifying.")
                    print(f"    Timeout penalty: {info['reward_breakdown']['timeout_penalty']:+.4f}")
                    print(f"    Step Reward    : {normalised_reward}")
                    print(f"    Correct   : False")
                else:
                    correct = obs.predicted_label == obs.actual_label
                    success = bool(correct)
                    print(f"\n  CLASSIFICATION RESULT")
                    print(f"    Step reward  : {normalised_reward}")
                    print(f"    Predicted : {obs.predicted_label}")
                    print(f"    Actual    : {obs.actual_label}")
                    print(f"    Correct   : {correct}")
                    if info.get("reward_breakdown"):
                        print(f"    Breakdown   : {info['reward_breakdown']}")
                    if info.get("cumulative"):
                        print(f"    Cumulative  : {info['cumulative']}")
                break
            elif task_name == "hard":
                normalised_reward = reward
                info = obs.info or {}

                if obs.reason == "timeout":
                    success = False
                    print(f"\n  HARD TIMEOUT")
                    print(f"    Step reward : {normalised_reward}")
                else:
                    correct = obs.predicted_label == obs.actual_label
                    success = bool(correct)

                    print(f"\n  HARD CLASSIFICATION RESULT")
                    print(f"    Step reward : {normalised_reward}")
                    print(f"    Predicted   : {obs.predicted_label}")
                    print(f"    Actual      : {obs.actual_label}")
                    print(f"    Correct     : {correct}")

                    if info.get("reward_breakdown"):
                        print(f"    Breakdown   : {info['reward_breakdown']}")

                    if info.get("grading"):
                        print(f"    Grading     : {info['grading']}")

                    if info.get("cumulative"):
                        print(f"    Cumulative  : {info['cumulative']}")

                break


    print(f"\nEpisode finished — task={task_id}")
    return success, steps_taken, normalised_reward, per_step_rewards


# ---------------------------------------------------------------------------
# Main — iterate over ALL tasks in a single invocation
# ---------------------------------------------------------------------------

async def main() -> None:
    if not API_KEY:
        raise EnvironmentError("HF_TOKEN or API_KEY environment variable must be set.")
    if not MODEL_NAME:
        raise EnvironmentError("MODEL_NAME environment variable must be set.")

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # env = await JobScamEnv.from_docker_image(LOCAL_IMAGE_NAME)
    env = JobScamEnv(base_url=HF_SPACE_URL)

    async with env:
        for task_id, task_name, max_steps in TASKS:
            # Emit [START] before the episode
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

            success      = False
            steps_taken  = 0
            # total_reward = 0.0
            # rewards_list: List[float] = []
            try:
                success, steps_taken, normalised_reward, rewards_list = await run_episode(
                    env=env,
                    llm_client=llm_client,
                    task_id=task_id,
                    task_name=task_name,
                    max_steps=max_steps,
                )
            except Exception as exc:
                # Log the crash but always emit [END]
                print(f"  [FATAL] task={task_id} crashed: {exc}", flush=True)
                # if not rewards_list:
                #     rewards_list = [0.0]
            finally:
                log_end(
                    task=task_id,
                    success=success,
                    steps=steps_taken,
                    score=normalised_reward,
                    # rewards=rewards_list if rewards_list else [0.0],
                )
                print("\n\n")


if __name__ == "__main__":
    asyncio.run(main())