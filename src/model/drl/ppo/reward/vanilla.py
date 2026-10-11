from typing import Tuple
from src.model.drl.ppo.reward.base import BaseRewardStrategy


class VanillaReward(BaseRewardStrategy):
    """
    Vanilla/Normal Reward Strategy (Unshaped Classic Makespan-only).
    Directly penalizes incremental makespan growth: r_t = - delta_makespan.
    Accumulates true tardiness at job completion for KPI logging without shaping reward.
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
        if is_final_step:
            due = float(self.env.job_due_dates[job_idx])
            weight = float(self.env.job_weights[job_idx])
            tardiness = max(0.0, float(best_finish_time) - due)
            self.env.total_weighted_tardiness += weight * tardiness

        raw_reward = -delta_makespan
        reward = raw_reward / self.config.reward_scale
        return float(raw_reward), float(reward)
