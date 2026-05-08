"""
PoshanEnv — Grader 1: Urgency Triage
Scores the agent's patient urgency ranking against WHO/NHM ground truth.
Uses Kendall Tau rank correlation — partial credit for near-correct orderings.
Score range: 0.0 – 1.0
"""

from typing import List, Tuple, Dict


def kendall_tau_score(predicted: List[str], ground_truth: List[str]) -> float:
    """
    Compute normalised Kendall Tau distance between two rankings.
    Returns 1.0 for perfect match, 0.0 for completely reversed.
    Handles missing / extra IDs gracefully.
    """
    # Only score IDs present in both lists
    common = [p for p in ground_truth if p in predicted]
    if len(common) < 2:
        return 0.0

    pred_order   = [predicted.index(p) for p in common]
    truth_order  = list(range(len(common)))  # ground truth is already sorted

    n = len(common)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            pred_rel  = pred_order[i]  < pred_order[j]
            truth_rel = truth_order[i] < truth_order[j]
            if pred_rel == truth_rel:
                concordant += 1
            else:
                discordant += 1

    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return 1.0
    tau = (concordant - discordant) / total_pairs
    # Normalise from [-1, 1] to [0, 1]
    return round((tau + 1) / 2, 4)


def top_k_accuracy(predicted: List[str], ground_truth: List[str], k: int = 2) -> float:
    """
    Bonus: did the agent correctly identify the top-k most urgent patients?
    """
    if not predicted or not ground_truth:
        return 0.0
    top_pred  = set(predicted[:k])
    top_truth = set(ground_truth[:k])
    overlap   = len(top_pred & top_truth)
    return round(overlap / k, 4)


def grade_triage(
    predicted_ranking: List[str],
    ground_truth_ranking: List[str],
) -> Tuple[float, str, Dict[str, float]]:
    """
    Main grading function for Task 1.

    Returns:
        total_score (float 0.0–1.0)
        feedback    (str — human readable)
        breakdown   (dict — score components)
    """
    if not predicted_ranking:
        return 0.0, "No ranking provided.", {"kendall_tau": 0.0, "top2_accuracy": 0.0}

    tau_score   = kendall_tau_score(predicted_ranking, ground_truth_ranking)
    top2_score  = top_k_accuracy(predicted_ranking, ground_truth_ranking, k=2)

    # Weighted total: ranking order matters more than top-2 bonus
    total = round(0.7 * tau_score + 0.3 * top2_score, 4)

    # Build feedback
    if total >= 0.9:
        feedback = "Excellent triage. Near-perfect urgency ordering."
    elif total >= 0.7:
        feedback = "Good triage. Most urgent patients correctly prioritised."
    elif total >= 0.5:
        feedback = "Partial credit. Some urgency misordering detected."
    else:
        feedback = (
            "Poor triage. Review BP thresholds (>=140/90 = high risk), "
            "Hb thresholds (<7 = severe anaemia), and MUAC (<11.5 = SAM)."
        )

    breakdown = {
        "kendall_tau":    tau_score,
        "top2_accuracy":  top2_score,
        "total":          total,
    }

    return total, feedback, breakdown
