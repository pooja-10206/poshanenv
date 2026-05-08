"""
PoshanEnv — Grader 3: Multi-Visit Progression
Scores the agent's escalation decision across 4 visits.

Reward logic per step:
  - Correct monitoring (no escalation when not needed): +0.15
  - Correct escalation at right visit (visit 2 or 3): +1.0
  - Late escalation (visit 4): +0.4
  - False alarm (escalating at visit 1): -0.2 → 0.0
  - Missing escalation entirely (done without escalating): 0.0
  - Correct intervention noted: small bonus +0.1
Score range: 0.0 – 1.0 (clamped)
"""

from typing import Tuple, Dict, Any

from env.models import PoshanAction


def grade_progression_step(
    visit: int,
    action: PoshanAction,
    task_data: Dict[str, Any],
) -> Tuple[float, str, Dict[str, float]]:
    """
    Grade one step (visit) in Task 3.

    visit     — current visit number (1–4)
    action    — agent's action
    task_data — episode task data from state
    """
    escalated_now      = action.escalate is True
    escalated_at       = task_data.get("escalated_at_visit")
    escalation_deadline = task_data.get("escalation_deadline", 3)

    breakdown: Dict[str, float] = {}
    feedback_parts = []

    # ------------------------------------------------------------------
    # Visit 1 — condition is borderline, should NOT escalate yet
    # ------------------------------------------------------------------
    if visit == 1:
        if escalated_now:
            # False alarm — condition not yet critical at visit 1
            score = 0.0
            feedback_parts.append(
                "False alarm at visit 1. Condition was borderline — "
                "monitoring was the correct action."
            )
        else:
            score = 0.15
            feedback_parts.append("Good. Correct monitoring at visit 1.")
        breakdown["visit_1"] = score
        return round(score, 4), " ".join(feedback_parts), breakdown

    # ------------------------------------------------------------------
    # Visit 2 — early signs of deterioration, escalation valid
    # ------------------------------------------------------------------
    if visit == 2:
        if escalated_now:
            # Early but valid escalation
            score = 1.0
            feedback_parts.append(
                "Excellent! Early escalation at visit 2. "
                "Detected deterioration trend correctly."
            )
        else:
            score = 0.15
            feedback_parts.append(
                "Continued monitoring at visit 2. "
                "Watch for worsening Hb and MUAC trend."
            )
        breakdown["visit_2"] = score
        return round(score, 4), " ".join(feedback_parts), breakdown

    # ------------------------------------------------------------------
    # Visit 3 — clear deterioration, should escalate
    # ------------------------------------------------------------------
    if visit == 3:
        if escalated_now:
            score = 0.85
            feedback_parts.append(
                "Good escalation at visit 3. "
                "Mother BP rising, child crossed SAM threshold."
            )
        else:
            # Missed the clearest signal — penalise
            score = 0.0
            feedback_parts.append(
                "Missed escalation at visit 3. "
                "BP ≥138 and child MUAC in SAM range required immediate action."
            )
        breakdown["visit_3"] = score
        return round(score, 4), " ".join(feedback_parts), breakdown

    # ------------------------------------------------------------------
    # Visit 4 — critical, last chance
    # ------------------------------------------------------------------
    if visit == 4:
        already_escalated = escalated_at is not None and escalated_at < 4
        if already_escalated:
            # Already escalated earlier — this step shouldn't be reached
            score = 0.0
            feedback_parts.append("Episode should have ended after escalation.")
        elif escalated_now:
            # Very late — partial credit
            score = 0.3
            feedback_parts.append(
                "Late escalation at visit 4. "
                "Oedema + BP crisis required action at visit 3 at latest."
            )
        else:
            # Never escalated — failure
            score = 0.0
            feedback_parts.append(
                "Failed to escalate across 4 visits. "
                "Mother showed pre-eclampsia signs; child had bilateral oedema (SAM)."
            )
        breakdown["visit_4"] = score
        return round(score, 4), " ".join(feedback_parts), breakdown

    # Fallback
    return 0.0, "Unknown visit.", {}


def compute_final_episode_score(task_data: Dict[str, Any]) -> float:
    """
    Compute the overall episode score for Task 3 after all visits.
    Used by the environment to report final success.
    """
    escalated_at = task_data.get("escalated_at_visit")
    if escalated_at is None:
        return 0.0
    elif escalated_at == 1:
        return 0.0   # false alarm
    elif escalated_at == 2:
        return 1.0   # optimal
    elif escalated_at == 3:
        return 0.85  # good
    elif escalated_at == 4:
        return 0.3   # late
    return 0.0
