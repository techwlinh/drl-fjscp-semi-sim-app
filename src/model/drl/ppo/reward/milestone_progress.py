from typing import Any, Tuple
from src.model.drl.ppo.reward.base import BaseRewardStrategy


class MilestoneProgressReward(BaseRewardStrategy):
    """
    Milestone Progress Reward Strategy.
    Provides proportional progress rewards per operation step completed, plus milestone due-date
    bonuses/penalties based on fractional lot progress (step_idx / total_steps).
    """

    def compute_reward(
        self,
        job_idx: int,
        best_finish_time: float,
        delta_makespan: float,
        best_setup_time: float,
        best_idle_time: float,
        is_final_step: bool,
    ) -> Tuple[float, float]:
        step_idx = self.env.job_step_counts[job_idx]  # 1-indexed (already incremented)
        total_steps = self.env.job_total_steps[job_idx]
        due = float(self.env.job_due_dates[job_idx])
        weight = float(self.env.job_weights[job_idx])

        # Fractional milestone due date for step k
        milestone_due = due * (float(step_idx) / float(total_steps))
        milestone_tardiness = max(0.0, float(best_finish_time) - milestone_due)

        if is_final_step:
            actual_tard = weight * max(0.0, float(best_finish_time) - due)
            self.env.total_weighted_tardiness += actual_tard

        # Positive progress reward (0 to 1 per lot over episode)
        progress_reward = 1.0 / float(total_steps)

        # Milestone tardiness penalty
        milestone_penalty = weight * (milestone_tardiness / 100.0)

        raw_reward = (
            progress_reward
            - self.obj_config.weight_tardiness * milestone_penalty
            - self.obj_config.weight_makespan * (delta_makespan / 10.0)
            - self.obj_config.weight_setup * best_setup_time
            - self.obj_config.weight_idle * (best_idle_time / 10.0)
        )
        reward = raw_reward / (self.config.reward_scale / 100.0)
        return float(raw_reward), float(reward)
