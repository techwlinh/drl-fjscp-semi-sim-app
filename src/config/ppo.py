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
    # RC2: gamma reduced from 0.999 to 0.99 for long-episode credit assignment (old: 0.999)
    gamma: float = Field(default=0.99, description="Discount factor gamma")
    gae_lambda: float = Field(default=0.95, description="GAE lambda parameter")
    clip_eps: float = Field(default=0.2, description="PPO clipping epsilon parameter")
    c_value: float = Field(default=0.5, description="Value loss coefficient")
    c_entropy: float = Field(default=0.02, description="Entropy regularization coefficient")

    # Training control
    num_episodes: int = Field(default=300, description="Total training episodes")
    eval_every: int = Field(default=1, ge=1, description="Deterministic eval frequency in episodes (1=every episode)")
    batch_size: int = Field(default=256, description="Minibatch size for PPO updates")
    rollouts_per_update: int = Field(default=4, description="Number of episodes to accumulate before each PPO update (RC4; old: 1)")
    # RC4: reduced from 3 to 2 to respect trust region over larger batch
    ppo_epochs: int = Field(default=2, description="Number of PPO optimization epochs per update")

    # Reward shaping options
    reward_strategy: str = Field(
        default="continuous_tardiness",
        description="Reward strategy: 'baseline', 'continuous_tardiness', 'pbrs', 'milestone_progress', 'workload_balance'",
    )
    # RC6: reward_scale reduced from 1000 to 100 (raw_reward mean~-25, scaled mean~-0.25; old: 1000)
    reward_scale: float = Field(default=100.0, description="Scaling factor for step rewards to stabilize gradients")

    # State representation options
    state_strategy: str = Field(
        default="enhanced",
        description="State representation strategy: 'baseline' (4 job, 2 tool, 2 global) or 'enhanced' (7 job, 3 WS, 3 global)",
    )

    # Numba JIT Acceleration
    use_numba: bool = Field(default=True, description="Enable Numba acceleration for state transitions")

    # RC6 LR-decay scaffolding (Task 11 will enable by default; old: no decay)
    # RC6: lr_decay enabled by default (old: False)
    lr_decay: bool = Field(default=True, description="Enable linear LR decay over training")
    lr_final_fraction: float = Field(default=0.1, description="Final LR as fraction of initial (used when lr_decay=True)")
    lr_decay_steps: int = Field(default=0, description="Total update steps for LR decay schedule (0=auto from num_episodes)")
