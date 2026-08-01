from typing import List, Tuple
import numpy as np

from src.schema.data import DatasetOutputModel
from src.fab.decoder.fjsp import FJSPDecoder
from src.fab.decoder.numba import NumbaFJSPDecoder
from src.fab.objective import compute_weighted_fitness
from src.model.drl.ppo.agent import PPOAgent
from src.model.drl.ppo.config import PPOConfig
from src.model.drl.ppo.env import FJSPEnv
from src.model.meta.ga.types import Chromosome, ScheduledTask


class PPOOptimizer:
    """
    Optimizer and Trainer for DRL PPO Scheduler on FJSP datasets.
    """

    def __init__(self, dataset: DatasetOutputModel, config: PPOConfig = None):
        self.dataset = dataset
        self.config = config or PPOConfig()

        self.env = FJSPEnv(dataset, self.config)
        self.obs_dim = self.env.obs_dim
        self.action_dim = self.env.action_space.n

        self.agent = PPOAgent(self.obs_dim, self.action_dim, self.config)
        self.decoder = FJSPDecoder(dataset)

        if self.config.use_numba:
            self.numba_decoder = NumbaFJSPDecoder(dataset)

    def train(self) -> Tuple[Chromosome, List[dict]]:
        """Run PPO training loop across episodes and return (best_chromo, history)."""
        history: List[dict] = []
        best_fitness = float("inf")
        best_chromo = None

        if self.config.resume_training:
            loaded = self.agent.load_model(self.config.model_checkpoint_path)
            if loaded:
                print("Resuming PPO training from existing model checkpoint...")
            else:
                print("No checkpoint found. Starting PPO training from scratch...")
        else:
            print("Starting PPO training from scratch...")

        print(
            f"Dataset: '{self.config.dataset_path}' | Episodes: {self.config.num_episodes}"
        )

        for episode in range(1, self.config.num_episodes + 1):
            obs, info = self.env.reset()
            action_mask = info["action_mask"]

            obs_list = []
            mask_list = []
            action_list = []
            log_prob_list = []
            value_list = []
            reward_list = []
            done_list = []

            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, log_prob, value = self.agent.select_action(
                    obs, action_mask, deterministic=False
                )

                next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                next_mask = next_info["action_mask"]

                obs_list.append(obs)
                mask_list.append(action_mask)
                action_list.append(action)
                log_prob_list.append(log_prob)
                value_list.append(value)
                reward_list.append(reward)
                done_list.append(terminated or truncated)

                obs = next_obs
                action_mask = next_mask

            # GAE computation & PPO policy update
            advantages, returns = self.agent.compute_gae(reward_list, value_list, done_list)
            losses = self.agent.update(
                obs_list, mask_list, action_list, log_prob_list, returns, advantages
            )

            # Evaluate episode schedule
            chromo = self.env.get_chromosome()
            if self.config.use_numba and hasattr(self, "numba_decoder"):
                makespan, tardiness, setup_time = self.numba_decoder.decode_fitness(chromo)
            else:
                _, makespan, tardiness, setup_time, _ = self.decoder.decode(chromo)

            chromo.makespan = makespan
            chromo.total_tardiness = tardiness
            chromo.total_setup_time = setup_time
            chromo.fitness = compute_weighted_fitness(
                makespan,
                tardiness,
                setup_time,
                weight_makespan=self.config.weight_makespan,
                weight_tardiness=self.config.weight_tardiness,
                weight_setup=self.config.weight_setup,
            )

            if chromo.fitness < best_fitness:
                best_fitness = chromo.fitness
                best_chromo = chromo
                self.agent.save_model(self.config.model_checkpoint_path)

            history.append(
                {
                    "episode": episode,
                    "fitness": chromo.fitness,
                    "makespan": makespan,
                    "tardiness": tardiness,
                    "setup_time": setup_time,
                    "total_reward": sum(reward_list),
                    "policy_loss": losses["policy_loss"],
                    "value_loss": losses["value_loss"],
                }
            )

            if episode % 20 == 0 or episode == self.config.num_episodes:
                print(
                    f"Episode {episode}/{self.config.num_episodes} - "
                    f"Best Fitness: {best_fitness:.2f} (Curr Fitness: {chromo.fitness:.2f}, "
                    f"Makespan: {makespan:.1f}m, Tardiness: {tardiness:.1f}m, Setup: {setup_time:.1f}m)"
                )

        return best_chromo, history

    def predict(self, num_samples: int = 15) -> Tuple[Chromosome, List[ScheduledTask], dict]:
        """
        Evaluate trained PPO policy across rollouts and return the best complete schedule & metrics.
        """
        self.agent.load_model(self.config.model_checkpoint_path)

        best_chromo = None
        best_fitness = float("inf")

        # Evaluate deterministic + stochastic policy rollouts
        for sample_i in range(num_samples):
            deterministic = (sample_i == 0)
            obs, info = self.env.reset()
            action_mask = info["action_mask"]

            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _, _ = self.agent.select_action(obs, action_mask, deterministic=deterministic)
                obs, _, terminated, truncated, info = self.env.step(action)
                action_mask = info["action_mask"]

            chromo = self.env.get_chromosome()
            if self.config.use_numba and hasattr(self, "numba_decoder"):
                makespan, tardiness, setup_time = self.numba_decoder.decode_fitness(chromo)
            else:
                _, makespan, tardiness, setup_time, _ = self.decoder.decode(chromo)

            chromo.makespan = makespan
            chromo.total_tardiness = tardiness
            chromo.total_setup_time = setup_time
            chromo.fitness = compute_weighted_fitness(
                makespan,
                tardiness,
                setup_time,
                weight_makespan=self.config.weight_makespan,
                weight_tardiness=self.config.weight_tardiness,
                weight_setup=self.config.weight_setup,
            )

            if chromo.fitness < best_fitness:
                best_fitness = chromo.fitness
                best_chromo = chromo

        tasks, makespan, tardiness, setup_time, _ = self.decoder.decode(best_chromo)

        # Compute tool utilization and tardy job rate
        total_tools = sum(
            len(ws.tools)
            for area in self.dataset.factory_infrastructure.areas
            for wsg in area.workstation_groups
            for ws in wsg.workstations
        )
        tool_busy = {}
        tardy_map = {}
        for task in tasks:
            tool_busy[task.tool_id] = (
                tool_busy.get(task.tool_id, 0.0)
                + (task.proc_end - task.proc_start)
                + (task.setup_end - task.setup_start)
            )
            tardy_map[task.job_id] = task.tardiness

        avg_utilization = (
            sum(tool_busy.values()) / (total_tools * makespan)
            if (total_tools * makespan) > 0
            else 0.0
        )
        tardy_jobs = sum(1 for t in tardy_map.values() if t > 0)
        on_time_rate = round((len(tardy_map) - tardy_jobs) / len(tardy_map) * 100, 1)

        metrics = {
            "name": "PPO Deep Reinforcement Learning (Proposed DRL)",
            "makespan": makespan,
            "total_weighted_tardiness": tardiness,
            "total_setup_time": setup_time,
            "fitness": chromo.fitness,
            "tardy_jobs": tardy_jobs,
            "on_time_rate_percent": on_time_rate,
            "avg_tool_utilization_percent": round(avg_utilization * 100, 1),
        }

        return chromo, tasks, metrics
