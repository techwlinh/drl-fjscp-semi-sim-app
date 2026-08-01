from dataclasses import dataclass
import numpy as np
from numba import njit
from src.config.experiment import ObjectiveConfig



@njit(fastmath=True)
def calculate_numba_makespan(job_completion_times: np.ndarray) -> float:
    """Calculate Makespan C_max (maximum completion time across all jobs)."""
    makespan = 0.0
    for j in range(len(job_completion_times)):
        if job_completion_times[j] > makespan:
            makespan = job_completion_times[j]
    return makespan


@njit(fastmath=True)
def calculate_numba_weighted_tardiness(
    job_completion_times: np.ndarray,
    job_due_dates: np.ndarray,
    job_priority_weights: np.ndarray,
) -> float:
    """Calculate Total Weighted Tardiness sum(w_i * max(0, C_i - d_i))."""
    total_tardiness = 0.0
    for j in range(len(job_completion_times)):
        tardiness = job_completion_times[j] - job_due_dates[j]
        if tardiness > 0.0:
            total_tardiness += job_priority_weights[j] * tardiness
    return total_tardiness


@njit(fastmath=True)
def calculate_numba_setup_cost(total_setup_time: float, cost_per_unit_time: float = 1.0) -> float:
    """Calculate setup cost from accumulated setup time and cost multiplier."""
    return total_setup_time * cost_per_unit_time


def calculate_setup_cost(total_setup_time: float, cost_per_unit_time: float = 1.0) -> float:
    """Calculate setup cost from accumulated setup time and cost multiplier."""
    return round(total_setup_time * cost_per_unit_time, 2)


@njit(fastmath=True)
def compute_numba_weighted_fitness(
    makespan: float,
    tardiness: float,
    setup_time: float,
    weight_makespan: float = 1.0,
    weight_tardiness: float = 2.0,
    weight_setup: float = 0.1,
    cost_per_unit_setup: float = 1.0,
    num_jobs: int = 100,
) -> float:
    """Calculate scalar fitness using Numba JIT (incorporates setup cost and avg tardiness)."""
    setup_cost = calculate_numba_setup_cost(setup_time, cost_per_unit_setup)
    tardiness_term = tardiness / num_jobs if num_jobs > 0 else tardiness
    return (
        weight_makespan * makespan
        + weight_tardiness * tardiness_term
        + weight_setup * setup_cost
    )


def compute_weighted_fitness(
    makespan: float,
    tardiness: float,
    setup_time: float,
    weight_makespan: float = 1.0,
    weight_tardiness: float = 2.0,
    weight_setup: float = 0.1,
    cost_per_unit_setup: float = 1.0,
    num_jobs: int = 100,
) -> float:
    """Calculate combined scalar fitness objective from metrics and weights."""
    setup_cost = calculate_setup_cost(setup_time, cost_per_unit_setup)
    tardiness_term = tardiness / num_jobs if (num_jobs and num_jobs > 0) else tardiness
    fitness = (
        weight_makespan * makespan
        + weight_tardiness * tardiness_term
        + weight_setup * setup_cost
    )
    return round(fitness, 2)


class ObjectiveEvaluator:
    """Evaluates multi-objective FJSP performance criteria."""

    def __init__(self, weights: ObjectiveConfig = None):
        self.weights = ObjectiveConfig()

    def evaluate(self, makespan: float, tardiness: float, setup_time: float, num_jobs: int = 100) -> float:
        """Compute scalar fitness score using configured objective weights."""
        return compute_weighted_fitness(
            makespan,
            tardiness,
            setup_time,
            weight_makespan=self.weights.weight_makespan,
            weight_tardiness=self.weights.weight_tardiness,
            weight_setup=self.weights.weight_setup,
            num_jobs=num_jobs,
        )
