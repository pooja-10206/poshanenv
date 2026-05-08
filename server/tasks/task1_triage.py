"""
PoshanEnv — Task 1: Urgency Triage (Easy)
Agent receives 5 household patient profiles and must rank them
from most to least urgent. Graded against WHO/NHM clinical protocols.
"""

import uuid
from typing import Dict, Any

from env.models import PoshanObservation, PoshanAction, HouseholdContext
from env.state import EpisodeState
from data.patient_generator import generate_task1_patients
from data.clinical_protocols import compute_urgency_score


class Task1Triage:
    task_id = 1
    max_steps = 1   # Single decision task

    def reset(self, state: EpisodeState) -> PoshanObservation:
        """Generate a fresh triage episode."""
        patients = generate_task1_patients(n=5)

        # Compute ground truth urgency order
        scored = sorted(
            patients,
            key=lambda p: compute_urgency_score(p.model_dump()),
            reverse=True
        )
        ground_truth_ranking = [p.patient_id for p in scored]

        # Store in state for grader
        state.task_data = {
            "patients": patients,
            "ground_truth_ranking": ground_truth_ranking,
            "patient_ids": [p.patient_id for p in patients],
        }

        # Build a simple household context
        from data.patient_generator import generate_household
        hh = generate_household(visit_number=1)

        # Build message for agent
        patient_summaries = []
        for p in patients:
            if p.patient_type == "mother":
                patient_summaries.append(
                    f"- {p.name} (ID:{p.patient_id}) | Pregnant {p.gestational_weeks}wks | "
                    f"BP:{p.systolic_bp}/{p.diastolic_bp} | Hb:{p.hemoglobin} | "
                    f"Symptoms:{p.symptoms}"
                )
            else:
                patient_summaries.append(
                    f"- {p.name} (ID:{p.patient_id}) | Child {p.age_months}mo | "
                    f"MUAC:{p.muac_cm}cm | Oedema:{p.oedema} | "
                    f"Symptoms:{p.symptoms}"
                )

        msg = (
            "TASK: Rank these 5 patients from most to least urgent.\n"
            "Use action_type='rank_urgency' and provide urgency_ranking "
            "as an ordered list of patient IDs.\n\n"
            "Patients:\n" + "\n".join(patient_summaries)
        )

        return PoshanObservation(
            task_id=self.task_id,
            episode_id=state.episode_id,
            step=state.step,
            patients=patients,
            household=hh,
            message=msg,
        )

    def step(
        self, action: PoshanAction, state: EpisodeState
    ) -> tuple:
        """
        Process agent's urgency ranking.
        Returns (observation, reward, done, info)
        """
        from graders.grader1_triage import grade_triage

        reward, feedback, breakdown = grade_triage(
            predicted_ranking=action.urgency_ranking or [],
            ground_truth_ranking=state.task_data["ground_truth_ranking"],
        )

        state.done = True
        state.success = reward >= 0.7

        # Final observation
        from data.patient_generator import generate_household
        hh = generate_household(visit_number=1)

        obs = PoshanObservation(
            task_id=self.task_id,
            episode_id=state.episode_id,
            step=state.step,
            patients=state.task_data["patients"],
            household=hh,
            previous_actions=[str(action.model_dump())],
            message=f"Episode complete. Score: {reward:.2f}. {feedback}",
        )

        return obs, reward, True, {"breakdown": breakdown, "feedback": feedback}
