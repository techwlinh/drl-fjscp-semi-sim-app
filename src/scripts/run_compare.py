"""
run_compare.py - Benchmark Comparison Script

Compares PPO (DRL) vs GA (Metaheuristic) vs Heuristic dispatching rules.
If PPO weights (ppo_model.pt) don't exist yet, trains PPO first.

Usage:
    uv run src/scripts/run_compare.py
"""
import json
from pathlib import Path

from src.schema.data import DatasetOutputModel
from src.model.heuristics.dispatching import HeuristicScheduler
from src.model.meta.ga.config import GAConfig
from src.model.meta.ga.exporter import export_schedule_results
from src.model.meta.ga.optimizer import GAOptimizer
from src.model.drl.ppo.config import PPOConfig
from src.model.drl.ppo.optimizer import PPOOptimizer


def _compute_schedule_metrics(tasks, makespan, dataset):
    """Compute on-time rate and tool utilization from scheduled tasks."""
    total_tools = sum(
        len(ws.tools)
        for area in dataset.factory_infrastructure.areas
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

    utilization = (
        sum(tool_busy.values()) / (total_tools * makespan)
        if (total_tools * makespan) > 0
        else 0.0
    )
    tardy_jobs = sum(1 for t in tardy_map.values() if t > 0)
    on_time = round((len(tardy_map) - tardy_jobs) / max(len(tardy_map), 1) * 100, 1)

    return {
        "tardy_jobs": tardy_jobs,
        "on_time_rate_percent": on_time,
        "avg_tool_utilization_percent": round(utilization * 100, 1),
    }


def _print_summary_table(all_comparisons):
    """Print formatted comparison table to stdout."""
    print("\n" + "=" * 80)
    print(f"{'Algorithm / Policy':<35} | {'Makespan (m)':<12} | {'Tardiness (m)':<13} | {'Setup (m)':<10} | {'Fitness':<8}")
    print("-" * 80)
    for _key, v in all_comparisons.items():
        print(
            f"{v['name']:<35} | {v['makespan']:<12.1f} | "
            f"{v['total_weighted_tardiness']:<13.1f} | {v['total_setup_time']:<10.1f} | "
            f"{v['fitness']:<8.2f}"
        )
    print("=" * 80 + "\n")


def main() -> None:
    ppo_config = PPOConfig()
    ga_config = GAConfig(dataset_path=ppo_config.dataset_path)

    # Load dataset
    with open(ppo_config.dataset_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)
    dataset = DatasetOutputModel.model_validate(raw_json)

    print("=== Comprehensive Benchmark: PPO vs GA vs Heuristics ===")

    # 1. PPO — train if no weights exist, then predict
    optimizer = PPOOptimizer(dataset, ppo_config)
    checkpoint_path = Path(ppo_config.model_checkpoint_path)

    if not checkpoint_path.exists():
        print("PPO weights not found. Training PPO model first...")
        optimizer.train()
    else:
        print(f"PPO weights loaded from: {ppo_config.model_checkpoint_path}")

    best_ppo_chromo, ppo_tasks, ppo_metrics = optimizer.predict()

    # 2. GA Metaheuristic
    ga_optimizer = GAOptimizer(dataset, ga_config)
    best_ga_chromo, ga_tasks, ga_history = ga_optimizer.run()

    # 3. Heuristic Dispatching Rules
    heuristic_scheduler = HeuristicScheduler(dataset, ga_config)
    heuristic_results = heuristic_scheduler.run_all()

    # Compute full GA metrics (on-time %, utilization %)
    ga_extra = _compute_schedule_metrics(ga_tasks, best_ga_chromo.makespan, dataset)
    ga_metrics = {
        "name": "GA Metaheuristic",
        "makespan": best_ga_chromo.makespan,
        "total_weighted_tardiness": best_ga_chromo.total_tardiness,
        "total_setup_time": best_ga_chromo.total_setup_time,
        "fitness": best_ga_chromo.fitness,
        **ga_extra,
    }

    # Combine all benchmarks
    all_comparisons = {"ppo": ppo_metrics, "ga": ga_metrics}
    all_comparisons.update(heuristic_results)

    _print_summary_table(all_comparisons)

    # Export PPO schedule -> experiments/ppo/schedule.json
    export_schedule_results(
        best_ppo_chromo,
        ppo_tasks,
        dataset,
        ppo_config.output_path,
        heuristic_comparisons=all_comparisons,
    )

    # Export GA schedule -> experiments/ga/schedule.json
    export_schedule_results(
        best_ga_chromo,
        ga_tasks,
        dataset,
        ga_config.output_path,
        history=ga_history,
        heuristic_comparisons=all_comparisons,
    )

    print(f"✅ PPO results exported to: {ppo_config.output_path}")
    print(f"✅ GA  results exported to: {ga_config.output_path}")


if __name__ == "__main__":
    main()