from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.model.drl.ppo.config import PPOConfig
from src.model.drl.ppo.network import ActorCritic


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent with GAE & Action Masking.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig = None):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or PPOConfig()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)

    def select_action(
        self, obs: np.ndarray, action_mask: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action_t, log_prob_t, value_t = self.network.act(
                obs_t, mask_t, deterministic=deterministic
            )

        return action_t.item(), log_prob_t.item(), value_t.item()

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation (GAE) and Returns."""
        rewards_arr = np.array(rewards, dtype=np.float32)
        values_arr = np.array(values + [next_value], dtype=np.float32)
        dones_arr = np.array(dones, dtype=np.float32)

        advantages = np.zeros_like(rewards_arr)
        last_gae_lam = 0.0

        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones_arr[t]
            delta = rewards_arr[t] + self.config.gamma * values_arr[t + 1] * non_terminal - values_arr[t]
            advantages[t] = last_gae_lam = (
                delta + self.config.gamma * self.config.gae_lambda * non_terminal * last_gae_lam
            )

        returns = advantages + values_arr[:-1]
        return advantages, returns

    def update(
        self,
        obs_list: List[np.ndarray],
        mask_list: List[np.ndarray],
        action_list: List[int],
        old_log_prob_list: List[float],
        returns: np.ndarray,
        advantages: np.ndarray,
    ) -> Dict[str, float]:
        """Perform PPO gradient updates across minibatch epochs."""
        obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(np.array(mask_list), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(np.array(action_list), dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.tensor(
            np.array(old_log_prob_list), dtype=torch.float32, device=self.device
        )
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # Normalize advantages
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        dataset_size = len(obs_list)
        batch_size = min(self.config.batch_size, dataset_size)

        policy_losses = []
        value_losses = []
        entropy_losses = []

        for _ in range(self.config.ppo_epochs):
            permutation = torch.randperm(dataset_size)
            for start_idx in range(0, dataset_size, batch_size):
                batch_indices = permutation[start_idx : start_idx + batch_size]

                b_obs = obs_t[batch_indices]
                b_mask = mask_t[batch_indices]
                b_actions = actions_t[batch_indices]
                b_old_log_probs = old_log_probs_t[batch_indices]
                b_returns = returns_t[batch_indices]
                b_advantages = advantages_t[batch_indices]

                log_probs, values, entropy = self.network.evaluate_actions(
                    b_obs, b_mask, b_actions
                )

                # Ratio for PPO clip objective
                ratios = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratios * b_advantages
                surr2 = (
                    torch.clamp(ratios, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
                    * b_advantages
                )

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.config.c_value * F.mse_loss(values, b_returns)
                entropy_loss = -self.config.c_entropy * entropy.mean()

                loss = policy_loss + value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy_loss": float(np.mean(entropy_losses)),
        }

    def save_model(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), path)
        print(f"PPO model checkpoint saved to: {filepath}")

    def load_model(self, filepath: str) -> bool:
        path = Path(filepath)
        if not path.exists():
            print(f"Warning: Checkpoint not found at {filepath}")
            return False
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.eval()
        print(f"PPO model weights loaded from: {filepath}")
        return True
