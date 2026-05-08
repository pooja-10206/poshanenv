"""
PoshanEnv — Clinical Protocols
Ground truth rules based on:
  - WHO MUAC thresholds for malnutrition
  - IMNCI danger signs for children
  - NHM maternal risk criteria (India)
  - WHO ANC guidelines

These are used by graders to score agent decisions.
"""

from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Child Malnutrition — WHO MUAC thresholds (age 6–59 months)
# ---------------------------------------------------------------------------

def classify_muac(muac_cm: float, oedema: bool = False) -> str:
    """
    Classify child malnutrition grade by MUAC and oedema.
    Returns: "SAM" | "MAM" | "Normal"
    """
    if oedema:
        return "SAM"  # bilateral pitting oedema = SAM regardless of MUAC
    if muac_cm < 11.5:
        return "SAM"   # Severe Acute Malnutrition
    elif muac_cm < 12.5:
        return "MAM"   # Moderate Acute Malnutrition
    else:
        return "Normal"


def get_child_intervention(grade: str) -> str:
    """
    Recommended intervention per malnutrition grade.
    Returns: protocol string
    """
    protocols = {
        "SAM": "refer_to_NRC",        # Nutrition Rehabilitation Centre
        "MAM": "supplement_RUTF",     # Ready-to-Use Therapeutic Food
        "Normal": "routine_followup",
    }
    return protocols.get(grade, "routine_followup")


CHILD_DANGER_SIGNS = [
    "unable_to_feed",
    "convulsions",
    "lethargic",
    "vomiting_everything",
    "chest_indrawing",
    "stridor",
]

def has_child_danger_sign(symptoms: List[str]) -> bool:
    return any(s in CHILD_DANGER_SIGNS for s in symptoms)


# ---------------------------------------------------------------------------
# Maternal Risk — NHM India criteria
# ---------------------------------------------------------------------------

def classify_maternal_risk(
    gestational_weeks: int,
    systolic_bp: int,
    diastolic_bp: int,
    hemoglobin: float,
    anc_visits: int,
    symptoms: List[str],
) -> str:
    """
    Classify maternal risk level.
    Returns: "high" | "medium" | "low"
    """
    # High risk triggers
    if systolic_bp >= 140 or diastolic_bp >= 90:
        return "high"   # Hypertension / pre-eclampsia
    if hemoglobin < 7.0:
        return "high"   # Severe anaemia
    if gestational_weeks >= 36 and anc_visits < 3:
        return "high"   # Near term, under-monitored
    if any(s in symptoms for s in [
        "severe_headache", "blurred_vision",
        "swelling_face", "reduced_fetal_movement",
        "vaginal_bleeding", "convulsions"
    ]):
        return "high"

    # Medium risk triggers
    if 7.0 <= hemoglobin < 10.0:
        return "medium"   # Moderate anaemia
    if 130 <= systolic_bp < 140 or 85 <= diastolic_bp < 90:
        return "medium"   # Borderline BP
    if anc_visits < 2 and gestational_weeks > 20:
        return "medium"
    if any(s in symptoms for s in [
        "mild_headache", "ankle_swelling", "fatigue", "dizziness"
    ]):
        return "medium"

    return "low"


def get_maternal_intervention(risk: str) -> str:
    interventions = {
        "high": "refer_to_hospital_immediately",
        "medium": "refer_to_PHC",
        "low": "routine_ANC_followup",
    }
    return interventions.get(risk, "routine_ANC_followup")


# ---------------------------------------------------------------------------
# Urgency scoring — used by Task 1 grader
# ---------------------------------------------------------------------------

def compute_urgency_score(patient: dict) -> float:
    """
    Compute a numeric urgency score for a patient profile dict.
    Higher = more urgent. Used to build ground-truth ranking.
    """
    score = 0.0

    if patient.get("patient_type") == "mother":
        bp_s = patient.get("systolic_bp", 110)
        bp_d = patient.get("diastolic_bp", 70)
        hb   = patient.get("hemoglobin", 11.0)
        syms = patient.get("symptoms", [])
        weeks = patient.get("gestational_weeks", 20)

        if bp_s >= 140 or bp_d >= 90:  score += 40
        elif bp_s >= 130:              score += 20
        if hb < 7.0:                   score += 35
        elif hb < 10.0:                score += 15
        if weeks >= 36:                score += 10
        if "vaginal_bleeding" in syms: score += 45
        if "convulsions" in syms:      score += 50
        if "severe_headache" in syms:  score += 25

    elif patient.get("patient_type") == "child":
        muac   = patient.get("muac_cm", 13.0)
        oedema = patient.get("oedema", False)
        syms   = patient.get("symptoms", [])
        age_m  = patient.get("age_months", 24)

        if oedema:           score += 50
        if muac < 11.5:      score += 40
        elif muac < 12.5:    score += 20
        if age_m < 6:        score += 15   # very young = higher risk
        for s in syms:
            if s in CHILD_DANGER_SIGNS:
                score += 30

    return score


# ---------------------------------------------------------------------------
# Task 3 — Escalation timing thresholds
# ---------------------------------------------------------------------------

# Ground truth: at which visit should escalation happen given deterioration
# Keys are (initial_grade, deterioration_rate): value is latest acceptable visit
ESCALATION_DEADLINES: Dict[Tuple[str, str], int] = {
    ("MAM", "fast"):   2,   # should escalate by visit 2
    ("MAM", "slow"):   3,   # by visit 3
    ("medium", "fast"): 2,
    ("medium", "slow"): 3,
    ("high", "any"):   1,   # immediate
}

def score_escalation_timing(
    escalated_at_visit: int,
    initial_grade: str,
    deterioration_rate: str,
) -> float:
    """
    Score how well-timed the escalation was.
    Returns 0.0–1.0.
    """
    key = (initial_grade, deterioration_rate)
    alt_key = (initial_grade, "any")
    deadline = ESCALATION_DEADLINES.get(key) or ESCALATION_DEADLINES.get(alt_key, 3)

    if escalated_at_visit <= deadline:
        return 1.0
    elif escalated_at_visit == deadline + 1:
        return 0.5   # one visit late — partial credit
    else:
        return 0.0   # too late
