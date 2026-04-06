"""
Inference Script — Job Scam Detection Environment
==================================================

An LLM-driven agent that plays one episode of the JobScamEnvironment by
investigating a job opportunity and submitting a classification label.

Required environment variables
-------------------------------
API_BASE_URL   API endpoint for the LLM (e.g. https://router.huggingface.co/v1).
MODEL_NAME     Model identifier (e.g. meta-llama/Llama-3.1-8B-Instruct).
HF_TOKEN       Hugging Face / API key.

The script uses the OpenAI-compatible client for all LLM calls, as required
by the submission spec.

Agent strategy
--------------
1. Parse the initial observation (query type + raw query text).
2. Use the LLM to decide which context field to request next, or to
   classify if enough evidence has been gathered.
3. Parse the LLM response as a JSON action object.
4. Step the environment, print the reward breakdown.
5. Repeat until ``done=True``.

Action JSON format expected from the LLM
-----------------------------------------
Info request::

    {"action_type": "request_company_profile"}

Classification::

    {"action_type": "classify", "label": "scam"}
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME: Optional[str] = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME: Optional[str] = os.getenv("IMAGE_NAME", "job_scam_env-env:latest")

MAX_STEPS:   int   = 5
TEMPERATURE: float = 0.2
MAX_TOKENS:  int   = 300

_VALID_ACTION_TYPES = [a.value for a in ActionType]
_VALID_LABELS       = [l.value for l in ClassificationLabel]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = textwrap.dedent("""
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
""").strip()

def _build_user_message(
    step: int,
    obs: JobScamObservation,
    history: List[str],
) -> str:
    """Construct the user-turn message for the LLM at each step."""
    # Safely extract remaining steps — step_budget may be None or not a dict
    step_budget = obs.step_budget
    if isinstance(step_budget, dict):
        remaining = step_budget.get("remaining", MAX_STEPS - step)
    else:
        remaining = MAX_STEPS - step

    parts: List[str] = [
        f"STEP: {step}",
        f"STEPS REMAINING: {remaining}",
        "",
    ]

    # ── Initial query (always present) ──────────────────────────────────────
    if obs.query_type:
        parts.append(f"QUERY TYPE: {obs.query_type}")
    if obs.initial_query:
        parts.append(f"INITIAL QUERY:\n{obs.initial_query}")

    # ── Field content returned by env ────────────────────────────────────────
    if obs.requested_field and obs.field_content:
        parts.append(f"\nFIELD JUST RECEIVED — {obs.requested_field.upper()}:")
        parts.append(obs.field_content)

    # ── History ──────────────────────────────────────────────────────────────
    if history:
        parts.append("\nACTIONS TAKEN SO FAR:")
        parts.extend(f"  {h}" for h in history)

    # ── Available actions reminder ────────────────────────────────────────────
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
    """
    Extract the first JSON object from the LLM response and validate it
    as a ``JobScamAction``.

    Returns ``None`` if parsing fails (caller will apply a fallback).
    """
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

    if action_type_str == ActionType.CLASSIFY:
        if label_str not in _VALID_LABELS:
            return None
        return JobScamAction(
            action_type=ActionType(action_type_str),
            label=ClassificationLabel(label_str),
        )

    if label_str is not None:
        return None  # label must not be present for non-classify actions

    return JobScamAction(action_type=ActionType(action_type_str))

def _fallback_action(step: int, requested: List[str]) -> JobScamAction:
    """
    Heuristic fallback when the LLM produces an unparseable response.

    Requests the next unrequested field in a priority order, or
    defaults to ``insufficient_info`` if all fields have been requested.
    """
    priority = [
        ActionType.REQUEST_COMPANY_PROFILE,
        ActionType.REQUEST_RECRUITER_PROFILE,
        ActionType.REQUEST_THREAD_HISTORY,
        ActionType.REQUEST_JOB_POST_COMMENTS,
    ]
    for action_type in priority:
        field = action_type.value.replace("request_", "")
        if field not in requested:
            return JobScamAction(action_type=action_type)

    # All fields seen or last step — classify with insufficient_info as safe default
    return JobScamAction(
        action_type=ActionType.CLASSIFY,
        label=ClassificationLabel.INSUFFICIENT_INFO,
    )


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------
async def run_episode(env: JobScamEnv, llm_client: OpenAI) -> float:
    """
    Run a single episode to completion and return the total reward.
    env is the live async client, already connected via `async with` in main().
    """
    result      = await env.reset()   # async reset — must be awaited
    obs         = result.observation
    history: List[str]        = []
    requested_fields: List[str] = []
    total_reward: float         = 0.0
    initial_query_text          = obs.initial_query or ""

    print(f"\n{'='*60}")
    print(f"NEW EPISODE")
    print(f"Query type : {obs.query_type}")
    print(f"Initial    : {initial_query_text[:120]}...")
    print(f"{'='*60}")

    for step in range(1, MAX_STEPS + 1):
        # ── Build LLM messages ────────────────────────────────────────────────
        user_msg = _build_user_message(step, obs, history)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

        # ── Call LLM (OpenAI client is synchronous — this is intentional) ─────
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
            print(f"  [Step {step}] LLM error: {exc}. Using fallback.")
            response_text = ""

        # ── Parse action ──────────────────────────────────────────────────────
        action = _parse_action(response_text)
        if action is None:
            print(f"  [Step {step}] Unparseable response: {response_text!r:.80}. Using fallback.")
            action = _fallback_action(step, requested_fields)

        print(f"\n[Step {step}] Action: {action.action_type.value}"
              + (f"  label={action.label.value}" if action.label else ""))

        # ── Step environment ──────────────────────────────────────────────────
        result      = await env.step(action)  # async step — must be awaited
        obs         = result.observation
        reward      = result.reward or 0.0
        total_reward += reward

        # ── Record for history ────────────────────────────────────────────────
        if action.action_type != ActionType.CLASSIFY:
            field_name = action.action_type.value.replace("request_", "")
            requested_fields.append(field_name)
            history.append(
                f"Step {step}: {action.action_type.value}  "
                f"→ reward {reward:+.4f}"
            )
            info = obs.info or {}
            print(f"         Reward      : {reward:+.4f}")
            if info.get("reward_breakdown"):
                print(f"         Breakdown   : {info['reward_breakdown']}")
            if info.get("cumulative"):
                print(f"         Cumulative  : {info['cumulative']}")

        # ── Handle done ───────────────────────────────────────────────────────
        if result.done:
            info = obs.info or {}
            if obs.reason == "timeout":
                print(f"\n  TIMEOUT — all steps exhausted without classifying.")
                print(f"  Timeout penalty: {reward:+.4f}")
            else:
                print(f"\n  CLASSIFICATION RESULT")
                print(f"    Predicted : {obs.predicted_label}")
                print(f"    Actual    : {obs.actual_label}")
                correct = obs.predicted_label == obs.actual_label
                print(f"    Correct   : {correct}")
                print(f"    Terminal reward  : {reward:+.4f}")
                if info.get("reward_breakdown"):
                    print(f"    Breakdown        : {info['reward_breakdown']}")
                if info.get("cumulative"):
                    print(f"    Episode total    : {info['cumulative'].get('total_reward', '?'):+}")
            break

    print(f"\nEpisode finished.  Total reward: {total_reward:+.4f}")
    return total_reward


async def main() -> None:
    """
    Entry point: spin up the Docker container, run one episode, then tear down.

    Why everything here is async (per official OpenEnv docs):
    ---------------------------------------------------------
    - from_docker_image() is an async classmethod. It starts the Docker
      container and waits for it to be healthy before returning the client.
      Calling it without `await` returns a coroutine object, NOT the client —
      which caused: AttributeError: 'coroutine' object has no attribute 'reset'

    - The env client must be used as an `async with` context manager. This
      establishes the WebSocket connection and guarantees the container is
      stopped and removed when the block exits (even on exception).

    - env.reset() and env.step() are async — they communicate over the
      WebSocket and must be awaited.

    Official docs pattern (echo_env README):
        client = await EchoEnv.from_docker_image("echo-env:latest")
        async with client:
            result = await client.reset()
            result = await client.step(action)
    """
    if not API_KEY:
        raise EnvironmentError("HF_TOKEN or API_KEY environment variable must be set.")
    if not MODEL_NAME:
        raise EnvironmentError("MODEL_NAME environment variable must be set.")

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Step 1: await from_docker_image() — starts the container, returns the client
    env = await JobScamEnv.from_docker_image(IMAGE_NAME)

    # Step 2: async with — opens the WebSocket connection and ensures cleanup
    async with env:
        await run_episode(env, llm_client)


if __name__ == "__main__":
    asyncio.run(main())