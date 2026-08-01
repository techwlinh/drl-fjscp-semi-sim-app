"""
run_ppo.py - PPO Deep Reinforcement Learning Training Script

Trains the PPO policy network on the FJSP dataset
and saves checkpoint weights to data/ppo_model.pt.

Usage:
    uv run src/scripts/run_ppo.py
"""
import json

from src.schema.data import DatasetOutputModel
from src.model.drl.ppo.config import PPOConfig
from src.model.drl.ppo.optimizer import PPOOptimizer


def main() -> None:
    config = PPOConfig()

    # Load dataset
    with open(config.dataset_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)
    dataset = DatasetOutputModel.model_validate(raw_json)

    print(f"=== PPO Training ({config.num_episodes} episodes) ===")
    optimizer = PPOOptimizer(dataset, config)
    optimizer.train()
    print(f"✅ Training complete. Checkpoint saved to: {config.model_checkpoint_path}")


if __name__ == "__main__":
    main()
