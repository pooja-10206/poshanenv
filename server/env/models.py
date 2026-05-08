"""
PoshanEnv — Typed Pydantic models for Observation, Action, Reward.
Implements the OpenEnv spec: every field is typed and documented.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class PatientProfile(BaseModel):
    """A single patient (mother or child) in the household."""
    patient_id: str
    patient_type: str                  # "mother" | "child"
    name: str
    age_years: float
    weight_kg: float
    height_cm: float

    # Maternal fields (present when patient_type == "mother")
    gestational_weeks: Optional[int] = None
    anc_visits_completed: Optional[int] = None
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None
    hemoglobin: Optional[float] = None   # g/dL
    symptoms: Optional[List[str]] = None  # e.g. ["headache", "swelling"]

    # Child fields (present when patient_type == "child")
    muac_cm: Optional[float] = None      # mid-upper arm circumference
    age_months: Optional[int] = None
    oedema: Optional[bool] = None
    dietary_recall: Optional[str] = None  # 24-hr diet description


class HouseholdContext(BaseModel):
    """Rural India household context."""
    household_id: str
    village: str
    district: str
    state: str
    distance_to_phc_km: float            # distance to Primary Health Centre
    has_toilet: bool
    monthly_income_inr: int
    visit_number: int                    # 1–4 for Task 3 progression


class PoshanObservation(BaseModel):
    """
    Full observation returned by reset() and step().
    Agent reads this and decides what action to take.
    """
    task_id: int                         # 1, 2, or 3
    episode_id: str
    step: int
    patients: List[PatientProfile]
    household: HouseholdContext
    previous_actions: List[str] = Field(default_factory=list)
    visit_notes: Optional[str] = None   # notes from previous visit (Task 3)
    message: str = ""                   # human-readable context for agent


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class PoshanAction(BaseModel):
    """
    Action taken by the agent.

    For Task 1 (triage): provide urgency_ranking — ordered list of patient_ids
                         from most to least urgent.
    For Task 2 (dual assessment): provide assessments dict mapping patient_id
                         to their assessment result.
    For Task 3 (progression): provide intervention + escalate flag.

    action_type options:
        "rank_urgency"       — Task 1
        "assess_and_plan"    — Task 2
        "intervene"          — Task 3
        "request_more_info"  — any task (penalised if overused)
        "noop"               — do nothing (penalised)
    """
    action_type: str
    urgency_ranking: Optional[List[str]] = None   # Task 1: ordered patient_ids
    assessments: Optional[Dict[str, str]] = None  # Task 2: {patient_id: grade}
    interventions: Optional[Dict[str, str]] = None # Task 2/3: {patient_id: plan}
    escalate: Optional[bool] = None               # Task 3: trigger referral
    escalate_reason: Optional[str] = None
    follow_up_days: Optional[int] = None          # schedule next visit


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class PoshanReward(BaseModel):
    """
    Reward signal returned after each step.
    Total reward is always in [0.0, 1.0].
    """
    total: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    feedback: str = ""     # human-readable explanation of score
    done: bool = False
    success: bool = False


# ---------------------------------------------------------------------------
# Step result (wraps everything the env returns)
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: PoshanObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
