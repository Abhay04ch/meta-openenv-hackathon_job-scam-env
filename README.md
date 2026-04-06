---
title: Job Scam Detection Environment
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---

# Job Scam Detection Environment

An OpenEnv-compatible RL environment for training and evaluating agents on job-scam detection.

Each episode presents a real-world job opportunity (WhatsApp message, email, Telegram post, or job board post). The agent investigates by requesting hidden context fields and then submits a classification label. All grading and reward computation is fully programmatic — no LLM is involved in the environment itself.

---

## Quick start

```bash
# Install dependencies
uv sync

# Start the environment server
uv run server

# In a second terminal, run the inference agent
export HF_TOKEN=<your_token>
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

Or with Docker:

```bash
docker build -t job_scam_env-env:latest .
python inference.py   # connects via from_docker_image()
```

---

## Episode structure

```
reset()
  └─ Observation: query_type, initial_query, available_context, step_budget

step(request_<field>)           × 0–4 times
  └─ Observation: field_content, step_budget
     Reward: signal_reward | redundancy_penalty | irrelevant_field_penalty

step(classify(label))           exactly once, terminal
  └─ Observation: predicted_label, actual_label, step_budget
     Reward: classification_reward + total_steps_taken_reward
     done: true

── OR ──

step budget exhausted without classify → timeout
  └─ Reward: −1.5,  done: true
```

Maximum steps per episode: **5** (4 info requests + 1 classify, or fewer).

---

## Action space

| Action | Description |
|---|---|
| `request_recruiter_profile` | Fetch recruiter name, contact, and bio |
| `request_company_profile` | Fetch company domain, hiring policy, anti-scam statements |
| `request_thread_history` | Fetch prior message thread between sender and candidate |
| `request_job_post_comments` | Fetch public comments or forwarded warnings |
| `classify(label)` | Submit final verdict — **terminates the episode** |

Valid labels: `legit` · `suspicious` · `scam` · `insufficient_info`

---

## Reward structure

### Per-step information reward

```
signal_score(field) = (|red_categories| + |green_categories|) /
                      total_unique_categories_in_sample
```

| Condition | Reward |
|---|---|
| Valid, non-redundant field (signal > 0) | `+0.10 × signal_score` |
| Field already requested this episode | `−0.20` |
| Field has zero signal (empty / N/A) | `−0.10` |

---

### Terminal classification reward

```
classification_reward   = REWARD_MATRIX[predicted][ground_truth]
alpha                   = +0.1  if correct  else  −0.1
total_steps_taken_reward = alpha × remaining_steps_at_classification
terminal_reward          = classification_reward + total_steps_taken_reward
```

Classification reward matrix (`REWARD_MATRIX[predicted][gt]`):

|  | gt: legit | gt: suspicious | gt: scam | gt: insuf |
|---|---|---|---|---|
| **pred: legit** | +1.00 | −0.30 | −1.00 | −0.20 |
| **pred: suspicious** | −0.10 | +1.00 | −0.30 | −0.10 |
| **pred: scam** | −0.50 | −0.10 | +1.00 | −0.30 |
| **pred: insuf** | −0.20 | −0.20 | −0.50 | +1.00 |

The asymmetric penalties reflect real-world stakes: calling a scam `legit` is maximally penalised at −1.00.

---

### Timeout

If the agent exhausts all 5 steps without classifying:

```
timeout_penalty = −1.5
```

---

## Observation contract

The client never receives: ground truth label, field signal scores, red/green flag categories, or internal reward equations.

```python
# Reset observation
obs.query_type          # str
obs.initial_query       # str
obs.available_context   # List[str]
obs.step_budget         # {"total": 5, "used": 0, "remaining": 5}

# Info request observation
obs.requested_field     # str
obs.field_content       # str
obs.step_budget         # updated budget

# Terminal (classification)
obs.predicted_label     # str
obs.actual_label        # str
obs.step_budget         # final budget

# Terminal (timeout)
obs.episode_done        # True
obs.reason              # "timeout"

# All steps — reward breakdown in metadata
obs.metadata["info"]["reward_breakdown"]   # dict
obs.metadata["info"]["cumulative"]         # dict
```

---

## Dataset

Four built-in samples cover all four query types and all four GT labels:

| sample_id | query_type | ground_truth |
|---|---|---|
| job_001 | job_post | legit |
| job_002 | email | suspicious |
| job_003 | whatsapp_msg | scam |
| job_004 | telegram_msg | insufficient_info |

The dataset is embedded directly in `server/job_scam_env_environment.py`.

Each sample contains:
- `red_flag_categories`
- `green_flag_categories`

These are used internally to compute signal scores and are never exposed to the agent.

To extend the dataset, add entries to the `_DATASET` list in:
```
server/job_scam_env_environment.py
```

---

## Project structure

```
job_scam_env/
├── server/
│   ├── app.py
│   └── job_scam_env_environment.py
├── client.py
├── models.py
├── inference.py
├── Dockerfile
├── openenv.yaml
└── pyproject.toml
```

---

## Environment variables (inference)

| Variable | Description |
|---|---|
| `API_BASE_URL` | OpenAI-compatible API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` / `API_KEY` | Authentication key |