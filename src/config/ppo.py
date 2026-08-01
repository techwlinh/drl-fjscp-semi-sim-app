from pydantic import BaseModel, Field


class PPOConfig(BaseModel):
    algorithm_name: str = Field(default="ppo", description="Algorithm identifier for exports")
    mode: str = Field(default="compare", description="Execution mode: 'train', 'predict', or 'compare'")
    dataset_path: str = Field(default="data/fjsp_dataset_seed42.json", description="Input dataset JSON path")
    output_path: str = Field(default="experiments/ppo/schedule.json", description="Output schedule JSON path")
    experiments_dir: str = Field(default="experiments", description="Experiments directory")
    model_checkpoint_path: str = Field(default="experiments/ppo/ppo_model.pt", description="Path to save/load trained PPO weights")
    resume_training: bool = Field(default=False, description="If True, load existing checkpoint before training")
    force_retrain: bool = Field(default=False, description="In compare mode, force training from scratch")

    # Hyperparameters for PPO
    learning_rate: float = Field(default=3e-4, description="Adam optimizer learning rate")
    gamma: float = Field(default=0.99, description="Discount factor gamma")
    gae_lambda: float = Field(default=0.95, description="GAE lambda parameter")
    clip_eps: float = Field(default=0.2, description="PPO clipping epsilon parameter")
    c_value: float = Field(default=0.5, description="Value loss coefficient")
    c_entropy: float = Field(default=0.01, description="Entropy regularization coefficient")

    # Training control
    num_episodes: int = Field(default=300, description="Total training episodes")
    batch_size: int = Field(default=64, description="Minibatch size for PPO updates")
    ppo_epochs: int = Field(default=5, description="Number of PPO optimization epochs per update")

    # Reward shaping weights
    weight_makespan: float = Field(default=0.3, description="Weight for makespan penalty")
    weight_tardiness: float = Field(default=0.5, description="Weight for tardiness penalty")
    weight_setup: float = Field(default=0.2, description="Weight for setup time penalty")
    weight_idle: float = Field(default=0.1, description="Weight for machine idle time penalty")

    # Numba JIT Acceleration
    use_numba: bool = Field(default=True, description="Enable Numba acceleration for state transitions")
