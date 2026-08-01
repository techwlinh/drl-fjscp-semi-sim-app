from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np


class BaseRewardStrategy(ABC):
    """
    Abstract Base Class for FJSP PPO Reward Strategies.
    Decouples reward shaping logic from the environment step function.
    """

    def __init__(self, env: Any):
        self.env = env
        self.config = env.config
        self.obj_config = env.obj_config

    def reset(self):
        """Called when environment resets."""
        pass

    @abstractmethod
    def compute_reward(
        self,
        job_idx: int,
        best_finish_time: float,
        delta_makespan: float,
        best_setup_time: float,
        best_idle_time: float,
        is_final_step: bool,
    ) -> Tuple[float, float]:
        """
        Compute step reward.
        
        Returns:
            Tuple[raw_reward, scaled_reward]
        """
        pass
