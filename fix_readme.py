# 🌾 PoshanEnv — Rural India Maternal & Child Health RL Environment

An OpenEnv-compliant reinforcement learning environment where an AI agent acts as a frontline health worker (ASHA/Anganwadi) making triage, nutrition assessment, and escalation decisions for pregnant women and children under 5 in rural India.

Built for the **Meta × HuggingFace × PyTorch OpenEnv AI Hackathon 2026.**

🤗 **Live on HuggingFace Spaces:** [pooja10206/poshanenv](https://huggingface.co/spaces/pooja10206/poshanenv)

---

## 🎯 Motivation

India accounts for **17% of global maternal deaths** and **35% of the world's stunted children**. PoshanEnv trains AI agents to support ASHA workers and Anganwadi workers — India's 1.3 million frontline community health workers — by simulating real clinical decision-making scenarios grounded in WHO, IMNCI, and NHM protocols.

---

## 🧩 Tasks

| Task | Name | Difficulty | Max Steps | Baseline Score |
|------|------|------------|-----------|----------------|
| 1 | Urgency Triage | Easy | 1 | 0.62 |
| 2 | Dual Assessment | Medium | 2 | 0.55 |
| 3 | Multi-Visit Progression | Hard | 4 | 0.45 |

### Task 1 — Urgency Triage (Easy)
Rank 5 patients (pregnant mothers + children under 5) from most to least urgent using WHO/NHM clinical indicators. Scored using Kendall Tau rank correlation.

### Task 2 — Dual Assessment (Medium)
Simultaneously classify child malnutrition grade (SAM/MAM/Normal) and maternal risk level (high/medium/low), then prescribe the correct intervention for each.

### Task 3 — Multi-Visit Progression (Hard)
Track a household across 4 visits as their condition deteriorates. Detect the worsening trend and escalate at the right visit — not too early (false alarm), not too late (missed event). Optimal escalation is at visit 2 or 3.

---

## 🔁 API Endpoints

Base URL: `https://pooja10206-poshanenv.hf.space`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset` | Start a new episode (optionally specify `task_id`) |
| POST | `/step` | Take one action, get reward + next observation |
| GET | `/state` | Current episode state |
| GET | `/tasks` | List all tasks with metadata |
| GET | `/health` | Liveness check |

---

## ⚡ Quick Start

```python
import requests

BASE = "https://pooja10206-poshanenv.hf.space"

# Start episode
obs = requests.post(f"{BASE}/reset", json={"task_id": 1}).json()
print(obs["observation"]["message"])

# Take action (Task 1 — rank urgency)
result = requests.post(f"{BASE}/step", json={
    "action_type": "rank_urgency",
    "urgency_ranking": ["id1", "id2", "id3", "id4", "id5"]
}).json()

print(f"Reward: {result['reward']}")
print(f"Done: {result['done']}")
```

---

## 🎬 Action Space

**Task 1 — Urgency Triage**
```json
{
  "action_type": "rank_urgency",
  "urgency_ranking": ["patient_id1", "patient_id2", "patient_id3", "patient_id4", "patient_id5"]
}
```

**Task 2 — Dual Assessment**
```json
{
  "action_type": "assess_and_plan",
  "assessments": {"mother_id": "high", "child_id": "SAM"},
  "interventions": {"mother_id": "refer_to_hospital_immediately", "child_id": "refer_to_NRC"}
}
```

**Task 3 — Multi-Visit Progression**
```json
{
  "action_type": "intervene",
  "escalate": true,
  "escalate_reason": "BP rising across visits, Hb critically low"
}
```

---

## 🏥 Clinical Protocols Used

| Protocol | Threshold |
|----------|-----------|
| WHO MUAC | SAM < 11.5cm, MAM 11.5–12.4cm, Normal ≥ 12.5cm |
| Maternal BP | ≥ 140/90 = high risk |
| Hemoglobin | < 7.0 g/dL = high risk |
| Intervention: SAM | Refer to NRC (Nutrition Rehabilitation Centre) |
| Intervention: High Maternal Risk | Refer to hospital immediately |

---

## 🏃 Run Locally with Docker

```bash
git clone https://github.com/pooja-10206/PoshanEnv.git
cd PoshanEnv
docker build -t poshanenv .
docker run -p 7860:7860 poshanenv
```

Then open `http://localhost:7860`

---

## 🧠 Run Baseline Inference

```bash
pip install -r requirements.txt

# Run all 3 tasks
python inference.py

# Run specific task
TASK_ID=1 python inference.py
```

Uses `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router by default. Override with:
```bash
MODEL_NAME=your-model HF_TOKEN=your-token python inference.py
```

---

## 📁 Project Structure

```
PoshanEnv/
├── inference.py              # Baseline LLM agent script
├── requirements.txt
├── openenv.yaml              # OpenEnv spec metadata
├── Dockerfile
└── server/
    ├── main.py               # FastAPI server (all endpoints)
    ├── env/
    │   ├── core.py           # PoshanEnv class — reset(), step(), state()
    │   ├── models.py         # Pydantic models (Observation, Action, Reward)
    │   └── state.py          # Episode state tracking
    ├── tasks/
    │   ├── task1_triage.py
    │   ├── task2_dual_assessment.py
    │   └── task3_progression.py
    ├── graders/
    │   ├── grader1_triage.py
    │   ├── grader2_dual.py
    │   └── grader3_progression.py
    └── data/
        ├── patient_generator.py
        └── clinical_protocols.py
```

---

## 👩‍💻 Author

**Pooja Lingwat**
B.Tech CSE  — MIT World Peace University, Pune
Research under Dr. Shamla Mantri
[LinkedIn](https://linkedin.com/in/pooja-lingwat-4a04b2255)

---

> *"Every data point here is a real clinical threshold. Every wrong action by the agent mirrors a missed diagnosis in the field."*
