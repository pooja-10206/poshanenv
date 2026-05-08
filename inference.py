"""
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
        print(f"\n{'='*50}", flush=True)
        print(f"Running Task {tid}: {TASK_NAMES[tid]}", flush=True)
        print('='*50, flush=True)
        result = run_episode(tid)
        all_results.append(result)
    print(f"\n{'='*50}", flush=True)
    print("SUMMARY", flush=True)
    print('='*50, flush=True)
    for r in all_results:
        avg = sum(r["rewards"]) / len(r["rewards"]) if r["rewards"] else 0.0
        print(f"Task {r['task_id']} ({TASK_NAMES[r['task_id']]}): success={r['success']} steps={r['steps']} avg_reward={avg:.2f}", flush=True)

if __name__ == "__main__":
    main()
