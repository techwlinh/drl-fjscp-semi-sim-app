from numbers import Integral
from typing import cast
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


HistoryEntry = dict[str, float | int]
PredictionMetrics = dict[str, float | int | str]


class PPOOptimizer:
    """
    Optimizer and Trainer for DRL PPO Scheduler on FJSP datasets.
    """

    def __init__(self, dataset: DatasetOutputModel, config: PPOConfig | None = None):
        self.dataset: DatasetOutputModel = dataset
        self.config: PPOConfig = config or PPOConfig()
        self.obj_config: ObjectiveConfig = ObjectiveConfig()
        self.history: list[HistoryEntry] = []

        self.env: FJSPEnv = FJSPEnv(dataset, self.config)
        self.obs_dim: int = self.env.obs_dim
        action_space = cast(object, self.env.action_space)
        action_dim = getattr(action_space, "n", None)
        if not isinstance(action_dim, Integral):
            raise TypeError("PPOOptimizer requires an integral discrete action size")
        self.action_dim: int = int(action_dim)

        self.agent: PPOAgent = PPOAgent(self.obs_dim, self.action_dim, self.config)
        self.decoder: FJSPDecoder = FJSPDecoder(dataset)

        if self.config.use_numba:
            self.numba_decoder: NumbaFJSPDecoder = NumbaFJSPDecoder(dataset)

    def train(self) -> tuple[Chromosome, list[HistoryEntry]]:
        """Run PPO training loop across episodes and return (best_chromo, history)."""
        history: list[HistoryEntry] = []
        best_fitness = float("inf")
        best_chromo: Chromosome | None = None
        last_losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}
        pending_history: list[HistoryEntry] = []
        last_det_fitness: float | None = None
        last_det_makespan: float | None = None
        last_det_tardiness: float | None = None
        last_det_setup_time: float | None = None

        batch_obs_list: list[np.ndarray] = []
        batch_mask_list: list[np.ndarray] = []
        batch_action_list: list[int] = []
        batch_log_prob_list: list[float] = []
        batch_advantages: list[np.ndarray] = []
        batch_returns: list[np.ndarray] = []
        batch_step_counts: list[int] = []
        rollout_count = 0

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

        episode_pbar = tqdm(
            range(1, self.config.num_episodes + 1),
            desc=f"PPO ({self.agent.device.type.upper()})",
            unit="ep",
        )
        for episode in episode_pbar:
            obs, info = self.env.reset()
            action_mask = cast(np.ndarray, info["action_mask"])

            obs_list: list[np.ndarray] = []
            mask_list: list[np.ndarray] = []
            action_list: list[int] = []
            log_prob_list: list[float] = []
            value_list: list[float] = []
            reward_list: list[float] = []
            raw_reward_list: list[float] = []
            done_list: list[bool] = []

            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, log_prob, value = self.agent.select_action(
                    obs, action_mask, deterministic=False
                )

                next_obs, reward, terminated, truncated, next_info = self.env.step(
                    action
                )
                next_mask = cast(np.ndarray, next_info["action_mask"])

                obs_list.append(obs)
                mask_list.append(action_mask)
                action_list.append(action)
                log_prob_list.append(log_prob)
                value_list.append(value)
                reward_list.append(reward)
                raw_reward: float = cast(float, next_info["raw_reward"])
                raw_reward_list.append(raw_reward)
                done_list.append(terminated or truncated)

                obs = next_obs
                action_mask = next_mask

            if not done_list or not done_list[-1]:
                raise RuntimeError(
                    f"Episode {episode} did not terminate cleanly for per-episode GAE computation"
                )

            advantages, returns = self.agent.compute_gae(
                reward_list, value_list, done_list, next_value=0.0
            )

            batch_obs_list.extend(obs_list)
            batch_mask_list.extend(mask_list)
            batch_action_list.extend(action_list)
            batch_log_prob_list.extend(log_prob_list)
            batch_advantages.append(advantages)
            batch_returns.append(returns)
            batch_step_counts.append(len(obs_list))
            rollout_count += 1

            total_reward = sum(raw_reward_list)
            should_eval = episode % self.config.eval_every == 0
            if should_eval:
                eval_obs, eval_info = self.env.reset()
                eval_mask = cast(np.ndarray, eval_info["action_mask"])
                eval_terminated = False
                eval_truncated = False

                while not (eval_terminated or eval_truncated):
                    eval_action, _, _ = self.agent.select_action(
                        eval_obs, eval_mask, deterministic=True
                    )
                    eval_obs, _, eval_terminated, eval_truncated, eval_info = self.env.step(
                        eval_action
                    )
                    eval_mask = cast(np.ndarray, eval_info["action_mask"])

                eval_chromo = self.env.get_chromosome()
                eval_decode_chromo = Chromosome(os=eval_chromo.os, ms=[])
                if self.config.use_numba and hasattr(self, "numba_decoder"):
                    makespan, tardiness, setup_time = self.numba_decoder.decode_fitness(
                        eval_decode_chromo
                    )
                else:
                    _, makespan, tardiness, setup_time, _ = self.decoder.decode(
                        eval_decode_chromo
                    )

                eval_chromo.makespan = makespan
                eval_chromo.total_tardiness = tardiness
                eval_chromo.total_setup_time = setup_time
                eval_chromo.fitness = compute_weighted_fitness(
                    makespan,
                    tardiness,
                    setup_time,
                    weight_makespan=self.obj_config.weight_makespan,
                    weight_tardiness=self.obj_config.weight_tardiness,
                    weight_setup=self.obj_config.weight_setup,
                    num_jobs=self.env.num_jobs,
                )

                last_det_fitness = eval_chromo.fitness
                last_det_makespan = makespan
                last_det_tardiness = tardiness
                last_det_setup_time = setup_time

                if eval_chromo.fitness < best_fitness:
                    best_fitness = eval_chromo.fitness
                    best_chromo = eval_chromo
                    self.agent.save_model(self.config.model_checkpoint_path)

            history_fitness = (
                last_det_fitness if last_det_fitness is not None else float("nan")
            )
            history_makespan = (
                last_det_makespan if last_det_makespan is not None else float("nan")
            )
            history_tardiness = (
                last_det_tardiness if last_det_tardiness is not None else float("nan")
            )
            history_setup_time = (
                last_det_setup_time if last_det_setup_time is not None else float("nan")
            )
            best_display = (
                f"{best_fitness:.2f}" if best_chromo is not None else "n/a"
            )
            episode_pbar.set_postfix(  # pyright: ignore[reportUnknownMemberType]
                {
                    "reward": f"{total_reward:.2f}",
                    "det_fit": (
                        f"{last_det_fitness:.2f}"
                        if last_det_fitness is not None
                        else "n/a"
                    ),
                    "best": best_display,
                }
            )
            pending_history.append(
                {
                    "episode": episode,
                    "fitness": history_fitness,
                    "best_fitness": best_fitness,
                    "makespan": history_makespan,
                    "tardiness": history_tardiness,
                    "setup_time": history_setup_time,
                    "total_reward": total_reward,
                    "policy_loss": last_losses["policy_loss"],
                    "value_loss": last_losses["value_loss"],
                }
            )

            should_update = (
                rollout_count >= self.config.rollouts_per_update
                or episode == self.config.num_episodes
            )
            if should_update:
                all_advantages = np.concatenate(batch_advantages).astype(np.float32, copy=False)
                all_returns = np.concatenate(batch_returns).astype(np.float32, copy=False)
                expected_total_steps = sum(batch_step_counts)
                assert len(all_advantages) == expected_total_steps
                assert len(all_returns) == expected_total_steps

                last_losses = self.agent.update(
                    batch_obs_list,
                    batch_mask_list,
                    batch_action_list,
                    batch_log_prob_list,
                    all_returns,
                    all_advantages,
                )

                for entry in pending_history:
                    entry["policy_loss"] = last_losses["policy_loss"]
                    entry["value_loss"] = last_losses["value_loss"]
                history.extend(pending_history)

                batch_obs_list = []
                batch_mask_list = []
                batch_action_list = []
                batch_log_prob_list = []
                batch_advantages = []
                batch_returns = []
                batch_step_counts = []
                pending_history = []
                rollout_count = 0

        if best_chromo is None:
            raise RuntimeError("PPO training produced no chromosome")

        self.history = history
        return best_chromo, history

    def predict(
        self, num_samples: int = 15
    ) -> tuple[Chromosome, list[ScheduledTask], PredictionMetrics]:
        """
        Evaluate trained PPO policy across rollouts and return the best complete schedule & metrics.
        """
        _ = self.agent.load_model(self.config.model_checkpoint_path)

        best_chromo: Chromosome | None = None
        best_fitness = float("inf")

        # Evaluate deterministic + stochastic policy rollouts
        for sample_i in range(num_samples):
            deterministic = sample_i == 0
            obs, info = self.env.reset()
            action_mask = cast(np.ndarray, info["action_mask"])

            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _, _ = self.agent.select_action(
                    obs, action_mask, deterministic=deterministic
                )
                obs, _, terminated, truncated, info = self.env.step(action)
                action_mask = cast(np.ndarray, info["action_mask"])

            chromo = self.env.get_chromosome()
            eval_chromo = Chromosome(os=chromo.os, ms=[])
            if self.config.use_numba and hasattr(self, "numba_decoder"):
                makespan, tardiness, setup_time = self.numba_decoder.decode_fitness(
                    eval_chromo
                )
            else:
                _, makespan, tardiness, setup_time, _ = self.decoder.decode(eval_chromo)

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

        if best_chromo is None:
            raise RuntimeError("PPO prediction produced no chromosome")

        tasks, makespan, tardiness, setup_time, _ = self.decoder.decode(best_chromo)

        # Compute tool utilization and tardy job rate
        total_tools = sum(
            len(ws.tools)
            for area in self.dataset.factory_infrastructure.areas
            for wsg in area.workstation_groups
            for ws in wsg.workstations
        )
        tool_busy: dict[str, float] = {}
        tardy_map: dict[str, float] = {}
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

        metrics: PredictionMetrics = {
            "name": "PPO",
            "makespan": makespan,
            "total_weighted_tardiness": tardiness,
            "total_setup_time": setup_time,
            "fitness": best_chromo.fitness,
            "tardy_jobs": tardy_jobs,
            "on_time_rate_percent": on_time_rate,
            "avg_tool_utilization_percent": round(avg_utilization * 100, 1),
        }

        return best_chromo, tasks, metrics
