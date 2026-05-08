"""
PoshanEnv — Episode State Manager
Tracks all state for a single running episode.
"""

import uuid
from typing import Optional, List, Any, Dict
from dataclasses import dataclass, field


@dataclass
class EpisodeState:
    """Holds the full state of one episode."""

    episode_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    task_id: int = 1
    step: int = 0
    max_steps: int = 8
    done: bool = False
    success: bool = False
    total_reward: float = 0.0
    rewards: List[float] = field(default_factory=list)

    # Current task data (set by task on reset)
    task_data: Dict[str, Any] = field(default_factory=dict)

    # Action history
    action_history: List[str] = field(default_factory=list)

    # For Task 3 multi-visit progression
    current_visit: int = 1
    escalated_at_visit: Optional[int] = None

    def record_action(self, action_str: str):
        self.action_history.append(f"step{self.step}: {action_str}")

    def record_reward(self, reward: float):
        self.rewards.append(reward)
        self.total_reward += reward

    def increment_step(self):
        self.step += 1
        if self.step >= self.max_steps:
            self.done = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "step": self.step,
            "max_steps": self.max_steps,
            "done": self.done,
            "success": self.success,
            "total_reward": round(self.total_reward, 4),
            "rewards": self.rewards,
            "action_history": self.action_history,
            "current_visit": self.current_visit,
            "escalated_at_visit": self.escalated_at_visit,
        }
