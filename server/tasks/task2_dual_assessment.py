"""
PoshanEnv — Task 2: Dual Assessment (Medium)
Agent receives one household with a pregnant mother + child under 5.
Must simultaneously classify malnutrition grade AND maternal risk,
then prescribe correct interventions for both.
"""

from env.models import PoshanObservation, PoshanAction
from env.state import EpisodeState
from data.patient_generator import generate_task2_household
from data.clinical_protocols import (
    classify_muac, classify_maternal_risk,
    get_child_intervention, get_maternal_intervention,
)


class Task2DualAssessment:
    task_id = 2
    max_steps = 2   # assess then intervene

    def reset(self, state: EpisodeState) -> PoshanObservation:
        mother, child, hh = generate_task2_household()

        # Compute ground truth
        child_grade = classify_muac(child.muac_cm, child.oedema)
        maternal_risk = classify_maternal_risk(
            gestational_weeks=mother.gestational_weeks,
            systolic_bp=mother.systolic_bp,
            diastolic_bp=mother.diastolic_bp,
            hemoglobin=mother.hemoglobin,
            anc_visits=mother.anc_visits_completed,
            symptoms=mother.symptoms or [],
        )

        state.task_data = {
            "mother": mother,
            "child": child,
            "household": hh,
            "ground_truth": {
                "child_grade": child_grade,
                "child_intervention": get_child_intervention(child_grade),
                "maternal_risk": maternal_risk,
                "maternal_intervention": get_maternal_intervention(maternal_risk),
            },
            "assessed": False,
        }

        msg = (
            "TASK: Assess both patients and provide a care plan.\n"
            "Step 1 — use action_type='assess_and_plan' with:\n"
            "  assessments: {patient_id: grade} for each patient\n"
            "  interventions: {patient_id: intervention} for each patient\n\n"
            f"Mother {mother.name} (ID:{mother.patient_id}):\n"
            f"  Pregnant {mother.gestational_weeks} weeks | "
            f"ANC visits: {mother.anc_visits_completed}\n"
            f"  BP: {mother.systolic_bp}/{mother.diastolic_bp} | "
            f"Hb: {mother.hemoglobin} g/dL\n"
            f"  Symptoms: {mother.symptoms}\n\n"
            f"Child {child.name} (ID:{child.patient_id}):\n"
            f"  Age: {child.age_months} months | "
            f"Weight: {child.weight_kg} kg\n"
            f"  MUAC: {child.muac_cm} cm | Oedema: {child.oedema}\n"
            f"  Diet: {child.dietary_recall}\n"
            f"  Symptoms: {child.symptoms}\n\n"
            "Valid assessment grades — Child: SAM, MAM, Normal\n"
            "Valid maternal risk levels: high, medium, low\n"
            "Valid interventions — Child: refer_to_NRC, supplement_RUTF, routine_followup\n"
            "Valid interventions — Mother: refer_to_hospital_immediately, "
            "refer_to_PHC, routine_ANC_followup"
        )

        return PoshanObservation(
            task_id=self.task_id,
            episode_id=state.episode_id,
            step=state.step,
            patients=[mother, child],
            household=hh,
            message=msg,
        )

    def step(self, action: PoshanAction, state: EpisodeState) -> tuple:
        from graders.grader2_dual import grade_dual_assessment

        mother = state.task_data["mother"]
        child  = state.task_data["child"]
        gt     = state.task_data["ground_truth"]

        reward, feedback, breakdown = grade_dual_assessment(
            assessments=action.assessments or {},
            interventions=action.interventions or {},
            ground_truth=gt,
            mother_id=mother.patient_id,
            child_id=child.patient_id,
        )

        state.done = True
        state.success = reward >= 0.6

        hh = state.task_data["household"]
        obs = PoshanObservation(
            task_id=self.task_id,
            episode_id=state.episode_id,
            step=state.step,
            patients=[mother, child],
            household=hh,
            previous_actions=[str(action.model_dump())],
            message=f"Episode complete. Score: {reward:.2f}. {feedback}",
        )

        return obs, reward, True, {"breakdown": breakdown, "feedback": feedback}
