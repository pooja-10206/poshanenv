"""
PoshanEnv — Core Environment
Implements the OpenEnv spec: reset(), step(), state().
Routes to the correct task based on task_id.
"""

import random
from typing import Optional, Dict, Any

from env.models import PoshanObservation, PoshanAction, StepResult
from env.state import EpisodeState
from tasks.task1_triage import Task1Triage
from tasks.task2_dual_assessment import Task2DualAssessment
from tasks.task3_progression import Task3Progression


TASKS = {
    1: Task1Triage(),
    2: Task2DualAssessment(),
    3: Task3Progression(),
}


class PoshanEnv:
    """
    PoshanEnv — Rural India Maternal & Child Health Environment.

    Three tasks of increasing difficulty:
      Task 1 (Easy)   — urgency triage of 5 patients
      Task 2 (Medium) — dual maternal + child assessment
      Task 3 (Hard)   — 4-visit deterioration detection

    OpenEnv API:
      reset(task_id)  → PoshanObservation
      step(action)    → StepResult
      state()         → dict
    """

    def __init__(self):
        self._state: Optional[EpisodeState] = None
        self._current_task = None

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[int] = None) -> PoshanObservation:
        """
        Start a new episode.
        task_id: 1, 2, or 3. If None, picks randomly.
        """
        if task_id is None:
            task_id = random.choice([1, 2, 3])

        if task_id not in TASKS:
            raise ValueError(f"Invalid task_id {task_id}. Must be 1, 2, or 3.")

        self._state = EpisodeState(task_id=task_id)
        self._current_task = TASKS[task_id]

        obs = self._current_task.reset(self._state)
        return obs

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: PoshanAction) -> StepResult:
        """
        Process one agent action.
        Returns StepResult with observation, reward, done, info.
        """
        if self._state is None or self._state.done:
            raise RuntimeError(
                "Environment not ready. Call reset() before step()."
            )

        # Record action
        self._state.record_action(action.action_type)

        # Handle noop / request_more_info with penalty
        if action.action_type == "noop":
            obs = self._build_penalty_obs("noop action — no progress.")
            reward = 0.0
            self._state.record_reward(reward)
            self._state.increment_step()
            return StepResult(
                observation=obs,
                reward=reward,
                done=self._state.done,
                info={"feedback": "noop penalised"},
            )

        # Delegate to task
        obs, reward, done, info = self._current_task.step(action, self._state)

        self._state.record_reward(reward)
        self._state.increment_step()
        if done:
            self._state.done = True

        return StepResult(
            observation=obs,
            reward=reward,
            done=self._state.done,
            info=info,
        )

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        """Return current episode state as a dict."""
        if self._state is None:
            return {"status": "not_started"}
        return self._state.to_dict()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_penalty_obs(self, reason: str) -> PoshanObservation:
        """Build a minimal observation for penalised actions."""
        from data.patient_generator import generate_household
        from env.models import PatientProfile
        hh = generate_household()
        return PoshanObservation(
            task_id=self._state.task_id,
            episode_id=self._state.episode_id,
            step=self._state.step,
            patients=[],
            household=hh,
            previous_actions=self._state.action_history,
            message=f"Penalty: {reason}",
        )
