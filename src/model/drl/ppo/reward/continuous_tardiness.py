from typing import Any, Tuple
import numpy as np
from src.model.drl.ppo.reward.base import BaseRewardStrategy


class ContinuousTardinessReward(BaseRewardStrategy):
    """
    Continuous Estimated Tardiness Reward Strategy.
    Provides immediate per-step feedback whenever an operation delay increases
    the estimated completion time (ECT) and expected tardiness of a job.
    """

    def __init__(self, env: Any):
        super().__init__(env)
        self.curr_est_tardiness = np.zeros(self.env.num_jobs, dtype=np.float32)
        self._init_rem_proc_tables()

    def _init_rem_proc_tables(self):
        """Precompute remaining processing time lookup for fast step evaluation."""
        self.rem_proc_lookup = []
        for j_idx, job in enumerate(self.env.jobs_list):
            recipe = self.env.dataset.product_recipes[job.product_type]
            step_times = [s.nominal_processing_time for s in recipe.steps]
            # rem[k] = sum of nominal processing times from step k onwards
            rem = [sum(step_times[k:]) for k in range(len(step_times))]
            rem.append(0.0)  # After final step
            self.rem_proc_lookup.append(rem)

    def reset(self):
        """Initialize estimated completion times and estimated tardiness for all jobs."""
        for j_idx in range(self.env.num_jobs):
            due = float(self.env.job_due_dates[j_idx])
            weight = float(self.env.job_weights[j_idx])
            initial_ect = self.rem_proc_lookup[j_idx][0]
            self.curr_est_tardiness[j_idx] = weight * max(0.0, initial_ect - due)

    def compute_reward(
        self,
        job_idx: int,
        best_finish_time: float,
        delta_makespan: float,
        best_setup_time: float,
        best_idle_time: float,
        is_final_step: bool,
    ) -> Tuple[float, float]:
        next_step_idx = self.env.job_step_counts[job_idx]  # Note: already incremented in env.step
        due = float(self.env.job_due_dates[job_idx])
        weight = float(self.env.job_weights[job_idx])

        rem_proc = self.rem_proc_lookup[job_idx][next_step_idx]
        new_ect = best_finish_time + rem_proc
        new_est_tard = weight * max(0.0, new_ect - due)

        # Immediate change in estimated tardiness for this job
        delta_est_tard = max(0.0, new_est_tard - self.curr_est_tardiness[job_idx])
        self.curr_est_tardiness[job_idx] = new_est_tard

        if is_final_step:
            actual_tard = weight * max(0.0, float(best_finish_time) - due)
            self.env.total_weighted_tardiness += actual_tard

        raw_reward = - (
            self.obj_config.weight_makespan * delta_makespan
            + self.obj_config.weight_setup * best_setup_time
            + self.obj_config.weight_tardiness * (delta_est_tard / float(self.env.num_jobs))
            + self.obj_config.weight_idle * (best_idle_time / 10.0)
        )
        # Scaling convention: all reward strategies divide by reward_scale (default 1000).
        # This is the canonical convention every strategy must follow; RC6 (Task 10) will retune the value later.
        reward = raw_reward / self.config.reward_scale
        return float(raw_reward), float(reward)
