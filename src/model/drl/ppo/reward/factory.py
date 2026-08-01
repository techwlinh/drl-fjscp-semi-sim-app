from typing import Any
from src.model.drl.ppo.reward.base import BaseRewardStrategy
from src.model.drl.ppo.reward.baseline import BaselineReward
from src.model.drl.ppo.reward.continuous_tardiness import ContinuousTardinessReward
from src.model.drl.ppo.reward.pbrs import PotentialBasedRewardShaping
from src.model.drl.ppo.reward.milestone_progress import MilestoneProgressReward
from src.model.drl.ppo.reward.workload_balance import WorkloadBalanceReward


REWARD_STRATEGIES = {
    "baseline": BaselineReward,
    "continuous_tardiness": ContinuousTardinessReward,
    "pbrs": PotentialBasedRewardShaping,
    "milestone_progress": MilestoneProgressReward,
    "workload_balance": WorkloadBalanceReward,
}


def get_reward_strategy(name: str, env: Any) -> BaseRewardStrategy:
    """
    Factory function to instantiate reward strategy by name.
    
    Args:
        name: Name of strategy ('baseline', 'continuous_tardiness', 'pbrs', 'milestone_progress', 'workload_balance')
        env: FJSPEnv instance
        
    Returns:
        BaseRewardStrategy instance
    """
    strategy_cls = REWARD_STRATEGIES.get(name.lower(), ContinuousTardinessReward)
    return strategy_cls(env)
