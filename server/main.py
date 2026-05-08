"""
PoshanEnv — FastAPI Server
Exposes the OpenEnv spec endpoints:
  POST /reset  — start new episode
  POST /step   — take one action
  GET  /state  — current episode state
  GET  /health — liveness check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import traceback

from env.core import PoshanEnv
from env.models import PoshanAction, PoshanObservation, StepResult

app = FastAPI(
    title="PoshanEnv",
    description=(
        "Rural India Maternal & Child Health RL Environment. "
        "An AI agent acts as a frontline health worker making "
        "triage, assessment, and escalation decisions."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Convenience root endpoint (prevents {"detail":"Not Found"} on /)
@app.get("/")
def read_root():
    return {"message": "OK"}

# One environment instance per server (single-session mode)
env = PoshanEnv()


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[int] = None   # 1, 2, or 3. None = random


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    task_id: int
    episode_id: str
    message: str


class StepRequest(BaseModel):
    action_type: str
    urgency_ranking: Optional[list] = None
    assessments: Optional[Dict[str, str]] = None
    interventions: Optional[Dict[str, str]] = None
    escalate: Optional[bool] = None
    escalate_reason: Optional[str] = None
    follow_up_days: Optional[int] = None


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    success: bool
    info: Dict[str, Any]


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok", "env": "PoshanEnv", "version": "1.0.0"}


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    """
    Start a new episode.
    Optionally specify task_id (1, 2, or 3).
    Returns initial observation.
    """
    try:
        obs = env.reset(task_id=request.task_id)
        return ResetResponse(
            observation=obs.model_dump(),
            task_id=obs.task_id,
            episode_id=obs.episode_id,
            message=obs.message,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(request: StepRequest):
    """
    Take one action in the current episode.
    Returns observation, reward, done flag, and grader feedback.
    """
    try:
        action = PoshanAction(
            action_type=request.action_type,
            urgency_ranking=request.urgency_ranking,
            assessments=request.assessments,
            interventions=request.interventions,
            escalate=request.escalate,
            escalate_reason=request.escalate_reason,
            follow_up_days=request.follow_up_days,
        )
        result: StepResult = env.step(action)

        return StepResponse(
            observation=result.observation.model_dump(),
            reward=result.reward,
            done=result.done,
            success=env.state().get("success", False),
            info=result.info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get("/state")
def state():
    """Return current episode state."""
    return env.state()


# ------------------------------------------------------------------
# OpenEnv spec: /tasks endpoint — list all tasks with metadata
# ------------------------------------------------------------------

@app.get("/tasks")
def list_tasks():
    """List all tasks with descriptions and difficulty."""
    return {
        "tasks": [
            {
                "task_id": 1,
                "name": "Urgency Triage",
                "difficulty": "easy",
                "description": (
                    "Rank 5 patients (pregnant mothers + children under 5) "
                    "from most to least urgent using clinical indicators."
                ),
                "max_steps": 1,
                "score_range": [0.0, 1.0],
                "action_type": "rank_urgency",
            },
            {
                "task_id": 2,
                "name": "Dual Assessment",
                "difficulty": "medium",
                "description": (
                    "Simultaneously classify maternal risk level and child "
                    "malnutrition grade, then prescribe correct interventions."
                ),
                "max_steps": 2,
                "score_range": [0.0, 1.0],
                "action_type": "assess_and_plan",
            },
            {
                "task_id": 3,
                "name": "Multi-Visit Progression",
                "difficulty": "hard",
                "description": (
                    "Track a household across 4 visits as condition deteriorates. "
                    "Detect the trend and escalate at the right visit — "
                    "not too early (false alarm), not too late (missed event)."
                ),
                "max_steps": 4,
                "score_range": [0.0, 1.0],
                "action_type": "intervene",
            },
        ]
    }


# ------------------------------------------------------------------
# Run directly
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
