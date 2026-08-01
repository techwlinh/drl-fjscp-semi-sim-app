from typing import Any, Dict, List, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.schema.data import DatasetOutputModel
from src.fab.decoder.fjsp import FJSPDecoder
from src.fab.decoder.numba import NumbaFJSPDecoder
from src.model.drl.ppo.config import PPOConfig
from src.model.meta.ga.types import Chromosome, ScheduledTask
from src.config.experiment import ObjectiveConfig


class FJSPEnv(gym.Env):
    """
    Gymnasium environment for Flexible Job Shop Scheduling Problem (FJSP).
    Supports step-by-step dispatching actions with Action Masking and Dense Reward Shaping.
    """
    metadata = {"render_modes": []}

    def __init__(self, dataset: DatasetOutputModel, config: PPOConfig = None):
        super().__init__()
        self.dataset = dataset
        self.config = config or PPOConfig()
        self.obj_config = ObjectiveConfig()
        self.decoder = FJSPDecoder(dataset)

        if self.config.use_numba:
            self.numba_decoder = NumbaFJSPDecoder(dataset)

        self.num_jobs = len(dataset.job_list)
        self.jobs_list = dataset.job_list
        self.tool_info = self.decoder.tool_info
        self.tool_ids = list(self.tool_info.keys())
        self.num_tools = len(self.tool_ids)
        self.tool_idx_map = {t_id: idx for idx, t_id in enumerate(self.tool_ids)}

        # Job steps count
        self.job_total_steps = np.array(
            [len(dataset.product_recipes[j.product_type].steps) for j in dataset.job_list],
            dtype=np.int32,
        )
        self.total_ops = int(np.sum(self.job_total_steps))

        # Total raw processing times per job
        self.job_raw_times = np.array(
            [j.total_raw_processing_time for j in dataset.job_list], dtype=np.float32
        )
        self.job_due_dates = np.array([j.due_date for j in dataset.job_list], dtype=np.float32)
        self.job_weights = np.array([j.priority_weight for j in dataset.job_list], dtype=np.float32)

        # Gym Action & Observation spaces
        self.action_space = spaces.Discrete(self.num_jobs)

        # Observation vector dimension:
        # Per job (4 features): [progress_ratio, remaining_proc_time, slack_time, priority_weight]
        # Per tool (2 features): [available_time, recipe_idx]
        # Global (2 features): [current_max_time, completion_ratio]
        self.obs_dim = self.num_jobs * 4 + self.num_tools * 2 + 2
        self.observation_space = spaces.Box(
            low=-10000.0, high=10000.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # Sequence of scheduled actions recorded during episode
        self.os_sequence: List[int] = []
        self.ms_sequence: List[int] = []

    def reset(
        self, seed: int = None, options: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.job_step_counts = np.zeros(self.num_jobs, dtype=np.int32)
        self.job_current_times = np.zeros(self.num_jobs, dtype=np.float32)
        self.job_current_locs = ["Central_Stockroom"] * self.num_jobs

        self.tool_available_times = {t_id: 0.0 for t_id in self.tool_ids}
        self.tool_current_recipes = {
            t_id: info["initial_setup"] for t_id, info in self.tool_info.items()
        }

        self.scheduled_ops_count = 0
        self.current_makespan = 0.0
        self.total_setup_time = 0.0
        self.total_weighted_tardiness = 0.0

        self.os_sequence = []
        self.ms_sequence = [-1] * self.total_ops

        obs = self._get_observation()
        info = {"action_mask": self.get_action_mask()}
        return obs, info

    def get_action_mask(self) -> np.ndarray:
        """Binary action mask: 1 if job has remaining unscheduled steps, 0 otherwise."""
        mask = (self.job_step_counts < self.job_total_steps).astype(np.float32)
        # If all jobs finished (terminal), return all ones to avoid zero probability division
        if np.sum(mask) == 0:
            mask = np.ones(self.num_jobs, dtype=np.float32)
        return mask

    def _get_observation(self) -> np.ndarray:
        job_feats = []
        for j_idx in range(self.num_jobs):
            step_idx = self.job_step_counts[j_idx]
            total_s = self.job_total_steps[j_idx]
            prog_ratio = float(step_idx) / float(total_s)

            # Remaining raw processing time from current step onwards
            recipe = self.dataset.product_recipes[self.jobs_list[j_idx].product_type]
            rem_proc = sum(s.nominal_processing_time for s in recipe.steps[step_idx:])

            # Slack time
            curr_time = float(self.job_current_times[j_idx])
            slack = float(self.job_due_dates[j_idx]) - (curr_time + rem_proc)
            weight = float(self.job_weights[j_idx])

            job_feats.extend([prog_ratio, rem_proc / 100.0, slack / 100.0, weight])

        tool_feats = []
        for t_id in self.tool_ids:
            avail = self.tool_available_times[t_id] / 100.0
            rec = float(hash(self.tool_current_recipes[t_id]) % 100) / 100.0
            tool_feats.extend([avail, rec])

        global_feats = [
            self.current_makespan / 100.0,
            float(self.scheduled_ops_count) / float(self.total_ops),
        ]

        obs = np.array(job_feats + tool_feats + global_feats, dtype=np.float32)
        return obs

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        job_idx = int(action)

        # Fallback if invalid action selected
        if self.job_step_counts[job_idx] >= self.job_total_steps[job_idx]:
            valid_jobs = np.where(self.job_step_counts < self.job_total_steps)[0]
            if len(valid_jobs) > 0:
                job_idx = int(valid_jobs[0])

        step_idx = self.job_step_counts[job_idx]
        global_op_idx = self.decoder.job_op_indices[job_idx][step_idx]
        op_data = self.decoder.op_info[global_op_idx]

        # Machine selection strategy: pick valid tool with earliest finish time
        valid_tools = op_data["valid_tools"]
        best_tool_id = valid_tools[0]
        best_tool_idx_in_valid = 0
        best_finish_time = float("inf")
        best_setup_time = 0.0
        best_idle_time = 0.0

        for idx_in_v, t_id in enumerate(valid_tools):
            tool_meta = self.tool_info[t_id]
            curr_loc = self.job_current_locs[job_idx]
            target_ws = tool_meta["ws_id"]
            travel_time = self.decoder.get_travel_time(curr_loc, target_ws)

            arr_time = self.job_current_times[job_idx] + travel_time
            t_ready = self.tool_available_times[t_id]
            idle_t = max(0.0, arr_time - t_ready)

            prev_rec = self.tool_current_recipes[t_id]
            curr_rec = op_data["product_type"]
            setup_t = self.decoder.get_setup_time(op_data["wsg_id"], prev_rec, curr_rec)

            start_t = max(arr_time, t_ready) + setup_t
            end_t = start_t + op_data["nominal_proc_time"]

            if end_t < best_finish_time:
                best_finish_time = end_t
                best_tool_id = t_id
                best_tool_idx_in_valid = idx_in_v
                best_setup_time = setup_t
                best_idle_time = idle_t

        # Apply state changes for selected tool & job
        self.ms_sequence[global_op_idx] = best_tool_idx_in_valid
        self.os_sequence.append(job_idx)

        tool_meta = self.tool_info[best_tool_id]
        curr_loc = self.job_current_locs[job_idx]
        target_ws = tool_meta["ws_id"]

        prev_makespan = self.current_makespan
        self.current_makespan = max(self.current_makespan, best_finish_time)
        delta_makespan = self.current_makespan - prev_makespan

        self.tool_available_times[best_tool_id] = best_finish_time
        self.tool_current_recipes[best_tool_id] = op_data["product_type"]
        self.job_current_times[job_idx] = best_finish_time
        self.job_current_locs[job_idx] = target_ws
        self.job_step_counts[job_idx] += 1
        self.scheduled_ops_count += 1
        self.total_setup_time += best_setup_time

        # Calculate incremental tardiness penalty if job completes final step
        is_final_step = self.job_step_counts[job_idx] == self.job_total_steps[job_idx]
        delta_tardiness = 0.0
        if is_final_step:
            due = self.job_due_dates[job_idx]
            weight = self.job_weights[job_idx]
            tardiness = max(0.0, float(best_finish_time) - due)
            delta_tardiness = weight * tardiness
            self.total_weighted_tardiness += delta_tardiness

        # Scaled Dense Reward Calculation (Tardiness normalized per job)
        raw_reward = - (
            self.obj_config.weight_makespan * delta_makespan
            + self.obj_config.weight_setup * best_setup_time
            + self.obj_config.weight_tardiness * (delta_tardiness / self.num_jobs)
            + self.obj_config.weight_idle * (best_idle_time / 10.0)
        )
        reward = raw_reward / self.config.reward_scale

        terminated = self.scheduled_ops_count >= self.total_ops
        truncated = False

        obs = self._get_observation()
        info = {
            "action_mask": self.get_action_mask(),
            "makespan": self.current_makespan,
            "total_setup_time": self.total_setup_time,
            "total_tardiness": self.total_weighted_tardiness,
            "raw_reward": float(raw_reward),
        }

        return obs, float(reward), terminated, truncated, info

    def get_chromosome(self) -> Chromosome:
        """Construct full Chromosome (os, ms) from environment episode sequence."""
        # Replace missing ms entries with default 0 if incomplete
        ms = [m if m >= 0 else 0 for m in self.ms_sequence]
        return Chromosome(os=self.os_sequence, ms=ms)
