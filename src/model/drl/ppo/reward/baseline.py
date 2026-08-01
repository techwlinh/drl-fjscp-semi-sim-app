from typing import Tuple
from src.model.drl.ppo.reward.base import BaseRewardStrategy


class BaselineReward(BaseRewardStrategy):
    """
    Baseline Sparse/Delayed Reward Strategy.
    Computes tardiness penalty ONLY when a job completes its final operation step.
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
        delta_tardiness = 0.0
        if is_final_step:
            due = float(self.env.job_due_dates[job_idx])
            weight = float(self.env.job_weights[job_idx])
            tardiness = max(0.0, float(best_finish_time) - due)
            delta_tardiness = weight * tardiness
            self.env.total_weighted_tardiness += delta_tardiness

        raw_reward = - (
            self.obj_config.weight_makespan * delta_makespan
            + self.obj_config.weight_setup * best_setup_time
            + self.obj_config.weight_tardiness * (delta_tardiness / float(self.env.num_jobs))
            + self.obj_config.weight_idle * (best_idle_time / 10.0)
        )
        reward = raw_reward / self.config.reward_scale
        return float(raw_reward), float(reward)
