---
title: PoshanEnv
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: true
tags:
  - openenv
  - healthcare
  - maternal-health
  - rural-india
---

# PoshanEnv - Rural India Maternal and Child Health RL Environment

An OpenEnv-compliant RL environment where an AI agent acts as a frontline
health worker making triage, nutrition assessment, and escalation decisions
for pregnant women and children under 5 in rural India.

## Tasks

| Task | Name | Difficulty | Baseline Score |
|------|------|------------|----------------|
| 1 | Urgency Triage | Easy | 0.62 |
| 2 | Dual Assessment | Medium | 0.55 |
| 3 | Multi-Visit Progression | Hard | 0.45 |

## API

Base URL: https://pooja10206-poshanenv.hf.space

- POST /reset - start episode
- POST /step - take action
- GET /state - current state
- GET /tasks - list tasks
- GET /health - liveness check

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

### Option A: PowerShell (recommended on Windows)

From the repo root:

```powershell
.\run_server.ps1
```

Then open:
- `http://127.0.0.1:7860/` (should return `{"message":"OK"}`)
- `http://127.0.0.1:7860/docs`
- `http://127.0.0.1:7860/health`

### Option B: Docker

```bash
docker build -t poshanenv .
docker run -p 7860:7860 poshanenv
```

## Clinical References

- WHO MUAC thresholds: SAM less than 11.5cm, MAM 11.5-12.4cm
- IMNCI danger signs for children under 5
- India NHM maternal risk criteria

Built for Meta x HuggingFace x PyTorch OpenEnv AI Hackathon 2026.
