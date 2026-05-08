"""
PoshanEnv — Task 3: Multi-Visit Progression (Hard)
Agent manages a household across 4 sequential visits.
Condition deteriorates — agent must detect the trend and
escalate at the right visit. Too early = false alarm penalty.
Too late = missed escalation penalty.
"""

from env.models import PoshanObservation, PoshanAction
from env.state import EpisodeState
from data.patient_generator import generate_task3_progression


class Task3Progression:
    task_id = 3
    max_steps = 4   # one step per visit

    def reset(self, state: EpisodeState) -> PoshanObservation:
        visits = generate_task3_progression()

        state.task_data = {
            "visits": visits,       # list of (mother, child, hh, notes)
            "current_visit": 1,
            "escalated_at_visit": None,
            "escalation_deadline": 3,   # should escalate by visit 3
            "deterioration_rate": "fast",
        }
        state.current_visit = 1
        state.max_steps = 4

        return self._build_observation(state, visit_index=0)

    def step(self, action: PoshanAction, state: EpisodeState) -> tuple:
        from graders.grader3_progression import grade_progression_step

        visit_index = state.current_visit - 1
        escalated_now = action.escalate is True

        if escalated_now and state.task_data["escalated_at_visit"] is None:
            state.task_data["escalated_at_visit"] = state.current_visit
            state.escalated_at_visit = state.current_visit

        reward, feedback, breakdown = grade_progression_step(
            visit=state.current_visit,
            action=action,
            task_data=state.task_data,
        )

        state.current_visit += 1
        done = (
            state.current_visit > 4
            or escalated_now
            or state.step >= state.max_steps - 1
        )
        state.done = done

        if done:
            # Final success: escalated at right time
            escalated_at = state.task_data.get("escalated_at_visit")
            state.success = (escalated_at is not None and escalated_at <= 3)

        # Build next observation (or final)
        if not done and state.current_visit <= 4:
            obs = self._build_observation(state, visit_index=state.current_visit - 1)
        else:
            visits = state.task_data["visits"]
            last_mother, last_child, last_hh, last_notes = visits[-1]
            obs = PoshanObservation(
                task_id=self.task_id,
                episode_id=state.episode_id,
                step=state.step,
                patients=[last_mother, last_child],
                household=last_hh,
                previous_actions=state.action_history,
                visit_notes=last_notes,
                message=f"Episode complete. {feedback}",
            )

        return obs, reward, done, {"breakdown": breakdown, "feedback": feedback}

    def _build_observation(
        self, state: EpisodeState, visit_index: int
    ) -> PoshanObservation:
        visits = state.task_data["visits"]
        mother, child, hh, notes = visits[visit_index]
        visit_num = visit_index + 1

        # Build trend hint from previous visits
        prev_notes = []
        for i in range(visit_index):
            _, _, _, n = visits[i]
            prev_notes.append(f"Visit {i+1}: {n}")

        trend_info = "\n".join(prev_notes) if prev_notes else "No previous visits."

        msg = (
            f"VISIT {visit_num} of 4\n\n"
            f"Previous visit notes:\n{trend_info}\n\n"
            f"Current findings:\n{notes}\n\n"
            f"Mother {mother.name} (ID:{mother.patient_id}):\n"
            f"  BP: {mother.systolic_bp}/{mother.diastolic_bp} | "
            f"Hb: {mother.hemoglobin} g/dL\n"
            f"  Symptoms: {mother.symptoms}\n\n"
            f"Child {child.name} (ID:{child.patient_id}):\n"
            f"  MUAC: {child.muac_cm} cm | Oedema: {child.oedema}\n"
            f"  Symptoms: {child.symptoms}\n\n"
            "Use action_type='intervene'. Set escalate=true if you believe "
            "the case needs immediate referral to hospital/NRC. "
            "Set escalate=false to continue monitoring."
        )

        return PoshanObservation(
            task_id=self.task_id,
            episode_id=state.episode_id,
            step=state.step,
            patients=[mother, child],
            household=hh,
            previous_actions=state.action_history,
            visit_notes=notes,
            message=msg,
        )
