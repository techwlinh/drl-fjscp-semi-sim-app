from typing import List, Tuple
import numpy as np
from tqdm import tqdm

from src.schema.data import DatasetOutputModel
from src.fab.decoder.fjsp import FJSPDecoder
from src.fab.decoder.numba import NumbaFJSPDecoder
from src.fab.objective import compute_weighted_fitness
from src.model.drl.ppo.agent import PPOAgent
from src.model.drl.ppo.config import PPOConfig
from src.model.drl.ppo.env import FJSPEnv
from src.model.meta.ga.types import Chromosome, ScheduledTask
from src.config.experiment import ObjectiveConfig


class PPOOptimizer:
    """
    Optimizer and Trainer for DRL PPO Scheduler on FJSP datasets.
    """

    def __init__(self, dataset: DatasetOutputModel, config: PPOConfig = None):
        self.dataset = dataset
        self.config = config or PPOConfig()
        self.obj_config = ObjectiveConfig()

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

        import torch

        pbar = tqdm(
            range(1, self.config.num_episodes + 1),
            desc=f"PPO ({self.agent.device.type.upper()})",
            unit="ep",
        )
        for episode in pbar:
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
                action, log_prob_t, value_t = self.agent.select_action(
                    obs, action_mask, deterministic=False
                )

                next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                next_mask = next_info["action_mask"]

                obs_list.append(torch.tensor(obs, dtype=torch.float32, device=self.agent.device))
                mask_list.append(torch.tensor(action_mask, dtype=torch.float32, device=self.agent.device))
                action_list.append(torch.tensor(action, dtype=torch.int64, device=self.agent.device))
                log_prob_list.append(log_prob_t)
                value_list.append(value_t)
                reward_list.append(reward)
                done_list.append(terminated or truncated)

                obs = next_obs
                action_mask = next_mask

            # Stack tensors on GPU device
            obs_t = torch.stack(obs_list)
            mask_t = torch.stack(mask_list)
            actions_t = torch.stack(action_list)
            log_probs_t = torch.stack(log_prob_list).reshape(-1)
            values_t = torch.stack(value_list).reshape(-1)
            rewards_t = torch.tensor(reward_list, dtype=torch.float32, device=self.agent.device)
            dones_t = torch.tensor(done_list, dtype=torch.bool, device=self.agent.device)

            # GPU-native GAE computation & PPO policy update
            advantages_t, returns_t = self.agent.compute_gae_gpu(rewards_t, values_t, dones_t)
            losses = self.agent.update_gpu(
                obs_t, mask_t, actions_t, log_probs_t, returns_t, advantages_t
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
                weight_makespan=self.obj_config.weight_makespan,
                weight_tardiness=self.obj_config.weight_tardiness,
                weight_setup=self.obj_config.weight_setup,
                num_jobs=self.env.num_jobs,
            )

            if chromo.fitness < best_fitness:
                best_fitness = chromo.fitness
                best_chromo = chromo
                self.agent.save_model(self.config.model_checkpoint_path)

            total_reward = float(rewards_t.sum().item())
            history.append(
                {
                    "episode": episode,
                    "fitness": chromo.fitness,
                    "makespan": makespan,
                    "tardiness": tardiness,
                    "setup_time": setup_time,
                    "total_reward": total_reward,
                    "policy_loss": losses["policy_loss"],
                    "value_loss": losses["value_loss"],
                }
            )

            pbar.set_postfix(
                {
                    "R_tot": f"{total_reward:.1f}",
                    "Fit": f"{chromo.fitness:.1f}",
                    "Best": f"{best_fitness:.1f}",
                    "MS": f"{makespan:.1f}",
                }
            )

            if episode % 20 == 0 or episode == self.config.num_episodes:
                # Deterministic evaluation rollout to measure true policy quality
                eval_obs, eval_info = self.env.reset()
                eval_mask = eval_info["action_mask"]
                eval_done = False
                while not eval_done:
                    act, _, _ = self.agent.select_action(eval_obs, eval_mask, deterministic=True)
                    eval_obs, _, term, trunc, eval_info = self.env.step(act)
                    eval_mask = eval_info["action_mask"]
                    eval_done = term or trunc

                eval_chromo = self.env.get_chromosome()
                if self.config.use_numba and hasattr(self, "numba_decoder"):
                    e_ms, e_tard, e_setup = self.numba_decoder.decode_fitness(eval_chromo)
                else:
                    _, e_ms, e_tard, e_setup, _ = self.decoder.decode(eval_chromo)

                eval_fitness = compute_weighted_fitness(
                    e_ms,
                    e_tard,
                    e_setup,
                    weight_makespan=self.obj_config.weight_makespan,
                    weight_tardiness=self.obj_config.weight_tardiness,
                    weight_setup=self.obj_config.weight_setup,
                    num_jobs=self.env.num_jobs,
                )

                tqdm.write(
                    f"Episode {episode}/{self.config.num_episodes} - "
                    f"Eval Fitness: {eval_fitness:.2f} (Makespan: {e_ms:.1f}m, Tardiness: {e_tard:.1f}m) | "
                    f"Best: {best_fitness:.2f} (Train Stochastic: {chromo.fitness:.2f})"
                )

        self.history = history
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
                weight_makespan=self.obj_config.weight_makespan,
                weight_tardiness=self.obj_config.weight_tardiness,
                weight_setup=self.obj_config.weight_setup,
                num_jobs=self.env.num_jobs,
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
            "name": "PPO",
            "makespan": makespan,
            "total_weighted_tardiness": tardiness,
            "total_setup_time": setup_time,
            "fitness": chromo.fitness,
            "tardy_jobs": tardy_jobs,
            "on_time_rate_percent": on_time_rate,
            "avg_tool_utilization_percent": round(avg_utilization * 100, 1),
        }

        return chromo, tasks, metrics
