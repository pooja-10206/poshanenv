"""
Fixes README.md and uploads to HuggingFace.
Run: python fix_readme.py
"""
from huggingface_hub import HfApi
import os

REPO_ID = "pooja10206/poshanenv"
api = HfApi()

README = """---
title: PoshanEnv
emoji: \U0001f33e
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

```bash
docker build -t poshanenv .
docker run -p 7860:7860 poshanenv
```

## Clinical References

- WHO MUAC thresholds: SAM less than 11.5cm, MAM 11.5-12.4cm
- IMNCI danger signs for children under 5
- India NHM maternal risk criteria

Built for Meta x HuggingFace x PyTorch OpenEnv AI Hackathon 2026.
"""

# Write locally
path = r"C:\Users\DELL\poshanenv\README.md"
with open(path, "w", encoding="utf-8") as f:
    f.write(README)
print("README written locally.")

# Upload
api.upload_file(
    path_or_fileobj=path,
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="space",
)
print("README uploaded!")
print(f"Check: https://huggingface.co/spaces/{REPO_ID}")
