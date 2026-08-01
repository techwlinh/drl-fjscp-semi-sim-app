from typing import Any, Tuple
import numpy as np
from src.model.drl.ppo.reward.base import BaseRewardStrategy


class PotentialBasedRewardShaping(BaseRewardStrategy):
    """
    Potential-Based Reward Shaping (PBRS) Strategy.
    Uses potential function Phi(s) derived from Critical Path Lower Bound and Expected Tardiness:
        Phi(s) = - [ w_mksp * LB_makespan(s) + w_tard * (Total_Est_Tardiness(s) / num_jobs) ] / 100.0
    Shaped Reward: r'_t = r_t^step + gamma * Phi(s_{t+1}) - Phi(s_t)
    
    Guarantees policy invariance under optimal RL theory while producing dense gradient updates.
    """

    def __init__(self, env: Any):
        super().__init__(env)
        self.gamma = getattr(self.config, "gamma", 0.99)
        self.prev_potential = 0.0
        self._init_rem_proc_tables()

    def _init_rem_proc_tables(self):
        self.rem_proc_lookup = []
        for j_idx, job in enumerate(self.env.jobs_list):
            recipe = self.env.dataset.product_recipes[job.product_type]
            step_times = [s.nominal_processing_time for s in recipe.steps]
            rem = [sum(step_times[k:]) for k in range(len(step_times))]
            rem.append(0.0)
            self.rem_proc_lookup.append(rem)

    def _compute_potential(self) -> float:
        est_completion_times = []
        total_est_tardiness = 0.0
        for j_idx in range(self.env.num_jobs):
            step_idx = self.env.job_step_counts[j_idx]
            curr_time = float(self.env.job_current_times[j_idx])
            rem_proc = self.rem_proc_lookup[j_idx][step_idx]
            ect = curr_time + rem_proc
            est_completion_times.append(ect)

            due = float(self.env.job_due_dates[j_idx])
            weight = float(self.env.job_weights[j_idx])
            total_est_tardiness += weight * max(0.0, ect - due)

        lb_makespan = max(est_completion_times) if est_completion_times else 0.0
        potential = - (
            self.obj_config.weight_makespan * lb_makespan
            + self.obj_config.weight_tardiness * (total_est_tardiness / float(self.env.num_jobs))
        ) / 100.0
        return float(potential)

    def reset(self):
        self.prev_potential = self._compute_potential()

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

        # Base step penalty for setup and idle
        step_base = - (
            self.obj_config.weight_setup * best_setup_time
            + self.obj_config.weight_idle * (best_idle_time / 10.0)
        )

        curr_potential = self._compute_potential()
        pbrs_shaping = self.gamma * curr_potential - self.prev_potential
        self.prev_potential = curr_potential

        raw_reward = step_base + pbrs_shaping
        reward = raw_reward / self.config.reward_scale
        return float(raw_reward), float(reward)
