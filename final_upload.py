"""
Final upload script — writes all missing files and uploads to HuggingFace.
Run: python final_upload.py
"""
from huggingface_hub import HfApi
import os

REPO_ID = "pooja10206/poshanenv"
api = HfApi()

# ----------------------------------------------------------------
# File contents
# ----------------------------------------------------------------

README = '''---
title: PoshanEnv
emoji: 🌾
colorFrom: green
colorTo: orange
sdk: docker
pinned: true
tags:
  - openenv
  - healthcare
  - maternal-health
  - nutrition
  - rural-india
  - reinforcement-learning
---

# 🌾 PoshanEnv — Rural India Maternal & Child Health RL Environment

An OpenEnv-compliant RL environment where an AI agent acts as a frontline
health worker making triage, nutrition assessment, and escalation decisions
for pregnant women and children under 5 in rural India.

## Motivation

India accounts for 17% of global maternal deaths and 35% of the world\'s
stunted children. PoshanEnv trains AI agents to support ASHA workers and
Anganwadi workers — India\'s 1.3 million frontline community health workers.

## Tasks

| Task | Name | Difficulty | Baseline |
|------|------|------------|---------|
| 1 | Urgency Triage | Easy | 0.62 |
| 2 | Dual Assessment | Medium | 0.55 |
| 3 | Multi-Visit Progression | Hard | 0.45 |

### Task 1 — Urgency Triage (Easy)
Rank 5 patients from most to least urgent using WHO/NHM clinical indicators.
Graded with Kendall Tau rank correlation.

### Task 2 — Dual Assessment (Medium)
Simultaneously classify child malnutrition grade (SAM/MAM/Normal) and
maternal risk (high/medium/low), then prescribe correct interventions.

### Task 3 — Multi-Visit Progression (Hard)
Track a household across 4 visits as condition deteriorates.
Escalate at the right visit — not too early, not too late.

## Action Space

```json
{"action_type": "rank_urgency", "urgency_ranking": ["id1","id2","id3","id4","id5"]}
{"action_type": "assess_and_plan", "assessments": {"mid": "high", "cid": "SAM"}, "interventions": {"mid": "refer_to_hospital_immediately", "cid": "refer_to_NRC"}}
{"action_type": "intervene", "escalate": true, "escalate_reason": "BP rising"}
```

## API

Base URL: `https://pooja10206-poshanenv.hf.space`

- `POST /reset` — start episode
- `POST /step` — take action
- `GET /state` — current state
- `GET /tasks` — list tasks
- `GET /health` — liveness check

## Quick Start

```python
import requests
BASE = "https://pooja10206-poshanenv.hf.space"
obs = requests.post(f"{BASE}/reset", json={"task_id": 1}).json()
result = requests.post(f"{BASE}/step", json={
    "action_type": "rank_urgency",
    "urgency_ranking": ["id1","id2","id3","id4","id5"]
}).json()
print(result["reward"])
```

## Run Locally

```bash
docker build -t poshanenv .
docker run -p 7860:7860 poshanenv
```

## Clinical References

- WHO MUAC thresholds (SAM < 11.5cm, MAM 11.5-12.4cm)
- IMNCI danger signs for children under 5
- India NHM maternal risk criteria (BP >= 140/90, Hb < 7.0)

## Project Structure

```
poshanenv/
├── inference.py
├── requirements.txt
├── openenv.yaml
└── server/
    ├── Dockerfile
    ├── main.py
    ├── env/core.py, models.py, state.py
    ├── tasks/task1, task2, task3
    ├── graders/grader1, grader2, grader3
    └── data/clinical_protocols.py, patient_generator.py
```

Built for Meta x HuggingFace x PyTorch OpenEnv AI Hackathon 2026.
'''

INFERENCE = '''"""
PoshanEnv - Baseline Inference Script
"""
import os
import json
import requests
from typing import List, Optional
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
POSHAN_URL   = os.getenv("POSHAN_URL", "https://pooja10206-poshanenv.hf.space")
TASK_ID      = int(os.getenv("TASK_ID", "0"))
MAX_STEPS    = 8
TEMPERATURE  = 0.2
MAX_TOKENS   = 512

TASK_NAMES = {1: "urgency-triage", 2: "dual-assessment", 3: "multi-visit-progression"}

def log_start(task, model):
    print(f"[START] task={task} env=PoshanEnv model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

SYSTEM_PROMPT = """You are an expert rural health worker in India.
WHO MUAC: SAM<11.5cm, MAM 11.5-12.4cm, Normal>=12.5cm
Maternal risk: BP>=140/90=high, Hb<7=high
Interventions: SAM->refer_to_NRC, high->refer_to_hospital_immediately

Task 1: {"action_type":"rank_urgency","urgency_ranking":["id1","id2","id3","id4","id5"]}
Task 2: {"action_type":"assess_and_plan","assessments":{"mid":"high","cid":"SAM"},"interventions":{"mid":"refer_to_hospital_immediately","cid":"refer_to_NRC"}}
Task 3: {"action_type":"intervene","escalate":true,"escalate_reason":"reason"}

Respond with ONLY the JSON object."""

def env_reset(task_id=None):
    payload = {"task_id": task_id} if task_id else {}
    r = requests.post(f"{POSHAN_URL}/reset", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action):
    r = requests.post(f"{POSHAN_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()

def call_llm(message, task_id):
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = completion.choices[0].message.content or "{}"
        raw = raw.strip().strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        fallbacks = {
            1: {"action_type": "rank_urgency", "urgency_ranking": []},
            2: {"action_type": "assess_and_plan", "assessments": {}, "interventions": {}},
            3: {"action_type": "intervene", "escalate": False},
        }
        return fallbacks.get(task_id, {"action_type": "noop"})

def run_episode(task_id):
    task_name = TASK_NAMES.get(task_id, f"task-{task_id}")
    log_start(task=task_name, model=MODEL_NAME)
    rewards = []
    steps_taken = 0
    success = False
    error = None
    try:
        reset_result = env_reset(task_id)
        obs = reset_result.get("observation", {})
        message = obs.get("message", str(obs))
        for step in range(1, MAX_STEPS + 1):
            action_dict = call_llm(message, task_id)
            action_str = json.dumps(action_dict)
            result = env_step(action_dict)
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            success = result.get("success", False)
            info = result.get("info", {})
            obs = result.get("observation", {})
            message = obs.get("message", "")
            error = info.get("feedback") if not done else None
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str[:120], reward=reward, done=done, error=error)
            if done:
                break
    except Exception as e:
        error = str(e)
        print(f"[DEBUG] Episode error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)
    return {"task_id": task_id, "success": success, "steps": steps_taken, "rewards": rewards}

def main():
    tasks_to_run = [TASK_ID] if TASK_ID in (1, 2, 3) else [1, 2, 3]
    all_results = []
    for tid in tasks_to_run:
        print(f"\\n{'='*50}", flush=True)
        print(f"Running Task {tid}: {TASK_NAMES[tid]}", flush=True)
        print(\'=\'*50, flush=True)
        result = run_episode(tid)
        all_results.append(result)
    print(f"\\n{'='*50}", flush=True)
    print("SUMMARY", flush=True)
    print(\'=\'*50, flush=True)
    for r in all_results:
        avg = sum(r["rewards"]) / len(r["rewards"]) if r["rewards"] else 0.0
        print(f"Task {r[\'task_id\']} ({TASK_NAMES[r[\'task_id\']]}): success={r[\'success\']} steps={r[\'steps\']} avg_reward={avg:.2f}", flush=True)

if __name__ == "__main__":
    main()
'''

# ----------------------------------------------------------------
# Write files locally
# ----------------------------------------------------------------

base = r"C:\Users\DELL\poshanenv"

files = {
    "README.md": README,
    "inference.py": INFERENCE,
}

for filename, content in files.items():
    path = os.path.join(base, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Written: {path}")

# ----------------------------------------------------------------
# Upload to HuggingFace
# ----------------------------------------------------------------

print("\nUploading to HuggingFace...")
for filename in files.keys():
    path = os.path.join(base, filename)
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=filename,
        repo_id=REPO_ID,
        repo_type="space",
    )
    print(f"Uploaded: {filename}")

print(f"\nAll done! https://huggingface.co/spaces/{REPO_ID}")
