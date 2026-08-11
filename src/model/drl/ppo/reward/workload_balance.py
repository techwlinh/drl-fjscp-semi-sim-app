from typing import Any, Tuple
import numpy as np
from src.model.drl.ppo.reward.base import BaseRewardStrategy


class WorkloadBalanceReward(BaseRewardStrategy):
    """
    Workload Balance & Bottleneck Penalty Reward Strategy.
    Combines continuous estimated tardiness with a workload imbalance penalty
    when scheduling onto heavily utilized tools.
    """

    def __init__(self, env: Any):
        super().__init__(env)
        self.curr_est_tardiness: np.ndarray = np.zeros(self.env.num_jobs, dtype=np.float32)
        self.prev_workload_var = 0.0
        self._init_rem_proc_tables()

    def _init_rem_proc_tables(self):
        self.rem_proc_lookup: list[list[float]] = []
        for job in self.env.jobs_list:
            recipe = self.env.dataset.product_recipes[job.product_type]
            step_times = [s.nominal_processing_time for s in recipe.steps]
            rem = [float(sum(step_times[k:])) for k in range(len(step_times))]
            rem.append(0.0)
            self.rem_proc_lookup.append(rem)

    def reset(self):
        for j_idx in range(self.env.num_jobs):
            due = float(self.env.job_due_dates[j_idx])
            weight = float(self.env.job_weights[j_idx])
            initial_ect = self.rem_proc_lookup[j_idx][0]
            self.curr_est_tardiness[j_idx] = weight * max(0.0, initial_ect - due)
        self.prev_workload_var = 0.0

    def compute_reward(
        self,
        job_idx: int,
        best_finish_time: float,
        delta_makespan: float,
        best_setup_time: float,
        best_idle_time: float,
        is_final_step: bool,
    ) -> Tuple[float, float]:
        next_step_idx = self.env.job_step_counts[job_idx]
        due = float(self.env.job_due_dates[job_idx])
        weight = float(self.env.job_weights[job_idx])

        rem_proc = self.rem_proc_lookup[job_idx][next_step_idx]
        new_ect = best_finish_time + rem_proc
        new_est_tard = weight * max(0.0, new_ect - due)

        delta_est_tard = max(0.0, new_est_tard - self.curr_est_tardiness[job_idx])
        self.curr_est_tardiness[job_idx] = new_est_tard

        if is_final_step:
            actual_tard = weight * max(0.0, float(best_finish_time) - due)
            self.env.total_weighted_tardiness += actual_tard

        # Calculate tool workload imbalance (variance of tool completion times)
        tool_times = list(self.env.tool_available_times.values())
        # RC5 fix: was absolute std (accumulated every step); now delta to prevent reward drift
        curr_var = float(np.std(tool_times)) / 100.0 if tool_times else 0.0
        delta_var = max(0.0, curr_var - self.prev_workload_var)
        self.prev_workload_var = curr_var

        raw_reward = - (
            self.obj_config.weight_makespan * delta_makespan
            + self.obj_config.weight_setup * best_setup_time
            + self.obj_config.weight_tardiness * (delta_est_tard / float(self.env.num_jobs))
            + self.obj_config.weight_idle * (best_idle_time / 10.0)
            + 0.05 * delta_var
        )
        reward = raw_reward / self.config.reward_scale
        return float(raw_reward), float(reward)
