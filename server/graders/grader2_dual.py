"""
PoshanEnv — Grader 2: Dual Assessment
Scores the agent's simultaneous maternal risk + child malnutrition
classification and intervention recommendations.
Score range: 0.0 – 1.0
"""

from typing import Dict, Tuple


# Valid values for validation
VALID_CHILD_GRADES    = {"SAM", "MAM", "Normal"}
VALID_MATERNAL_RISKS  = {"high", "medium", "low"}
VALID_CHILD_INTERVENTIONS = {
    "refer_to_NRC", "supplement_RUTF", "routine_followup"
}
VALID_MATERNAL_INTERVENTIONS = {
    "refer_to_hospital_immediately", "refer_to_PHC", "routine_ANC_followup"
}


def grade_dual_assessment(
    assessments: Dict[str, str],
    interventions: Dict[str, str],
    ground_truth: Dict[str, str],
    mother_id: str,
    child_id: str,
) -> Tuple[float, str, Dict[str, float]]:
    """
    Grade Task 2 dual assessment.

    assessments  — {patient_id: grade_string}
    interventions — {patient_id: intervention_string}
    ground_truth — dict with child_grade, maternal_risk,
                   child_intervention, maternal_intervention
    """
    breakdown: Dict[str, float] = {}
    feedback_parts = []

    # ------------------------------------------------------------------
    # 1. Child malnutrition grade (25%)
    # ------------------------------------------------------------------
    child_pred = (assessments.get(child_id) or "").strip()
    child_gt   = ground_truth["child_grade"]

    if child_pred == child_gt:
        child_grade_score = 1.0
        feedback_parts.append(f"Child grade correct ({child_gt}).")
    elif child_pred in VALID_CHILD_GRADES:
        # Adjacent grade = partial credit
        adjacency = {"SAM": ["MAM"], "MAM": ["SAM", "Normal"], "Normal": ["MAM"]}
        child_grade_score = 0.4 if child_pred in adjacency.get(child_gt, []) else 0.0
        feedback_parts.append(
            f"Child grade incorrect. Predicted {child_pred}, expected {child_gt}."
        )
    else:
        child_grade_score = 0.0
        feedback_parts.append(f"Invalid child grade '{child_pred}'.")
    breakdown["child_grade"] = child_grade_score

    # ------------------------------------------------------------------
    # 2. Maternal risk level (25%)
    # ------------------------------------------------------------------
    maternal_pred = (assessments.get(mother_id) or "").strip().lower()
    maternal_gt   = ground_truth["maternal_risk"]

    if maternal_pred == maternal_gt:
        maternal_risk_score = 1.0
        feedback_parts.append(f"Maternal risk correct ({maternal_gt}).")
    elif maternal_pred in VALID_MATERNAL_RISKS:
        # Adjacent level = partial credit
        adjacency = {"high": ["medium"], "medium": ["high", "low"], "low": ["medium"]}
        maternal_risk_score = 0.4 if maternal_pred in adjacency.get(maternal_gt, []) else 0.0
        feedback_parts.append(
            f"Maternal risk incorrect. Predicted {maternal_pred}, expected {maternal_gt}."
        )
    else:
        maternal_risk_score = 0.0
        feedback_parts.append(f"Invalid maternal risk '{maternal_pred}'.")
    breakdown["maternal_risk"] = maternal_risk_score

    # ------------------------------------------------------------------
    # 3. Child intervention (25%)
    # ------------------------------------------------------------------
    child_intv_pred = (interventions.get(child_id) or "").strip()
    child_intv_gt   = ground_truth["child_intervention"]

    if child_intv_pred == child_intv_gt:
        child_intv_score = 1.0
        feedback_parts.append(f"Child intervention correct ({child_intv_gt}).")
    elif child_intv_pred in VALID_CHILD_INTERVENTIONS:
        child_intv_score = 0.2
        feedback_parts.append(
            f"Child intervention suboptimal. Expected {child_intv_gt}."
        )
    else:
        child_intv_score = 0.0
        feedback_parts.append(f"Invalid child intervention '{child_intv_pred}'.")
    breakdown["child_intervention"] = child_intv_score

    # ------------------------------------------------------------------
    # 4. Maternal intervention (25%)
    # ------------------------------------------------------------------
    maternal_intv_pred = (interventions.get(mother_id) or "").strip()
    maternal_intv_gt   = ground_truth["maternal_intervention"]

    if maternal_intv_pred == maternal_intv_gt:
        maternal_intv_score = 1.0
        feedback_parts.append(f"Maternal intervention correct ({maternal_intv_gt}).")
    elif maternal_intv_pred in VALID_MATERNAL_INTERVENTIONS:
        maternal_intv_score = 0.2
        feedback_parts.append(
            f"Maternal intervention suboptimal. Expected {maternal_intv_gt}."
        )
    else:
        maternal_intv_score = 0.0
        feedback_parts.append(f"Invalid maternal intervention '{maternal_intv_pred}'.")
    breakdown["maternal_intervention"] = maternal_intv_score

    # ------------------------------------------------------------------
    # Total (equal weights)
    # ------------------------------------------------------------------
    total = round(
        0.25 * child_grade_score +
        0.25 * maternal_risk_score +
        0.25 * child_intv_score +
        0.25 * maternal_intv_score,
        4
    )
    breakdown["total"] = total

    feedback = " | ".join(feedback_parts)
    return total, feedback, breakdown
