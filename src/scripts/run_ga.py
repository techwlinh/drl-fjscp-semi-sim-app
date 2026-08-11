"""
run_ga.py - GA Metaheuristic Optimizer Script

Runs the Genetic Algorithm optimizer on the FJSP dataset,
benchmarks against dispatching heuristics, and exports results.

Usage:
    uv run src/scripts/run_ga.py
"""
import json

from src.schema.data import DatasetOutputModel
from src.model.heuristics.dispatching import HeuristicScheduler
from src.model.meta.ga.config import GAConfig
from src.model.meta.ga.exporter import export_schedule_results
from src.model.meta.ga.optimizer import GAOptimizer


def main() -> None:
    config = GAConfig()

    # Load dataset
    with open(config.dataset_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)
    dataset = DatasetOutputModel.model_validate(raw_json)

    print(
        f"=== GA Optimization ({config.generations} generations, pop={config.pop_size}) ===\n"
        f"Dataset: {config.dataset_path} | "
        f"{len(dataset.job_list)} jobs, {len(dataset.factory_infrastructure.areas)} areas"
    )

    # Optimize
    optimizer = GAOptimizer(dataset, config)
    best_chromo, tasks, history = optimizer.run()

    # Heuristic benchmarks
    print("\nRunning Heuristic Dispatching Rule Benchmarks (FIFO, EDD, SPT, CR)...")
    heuristic_scheduler = HeuristicScheduler(dataset, config)
    heuristic_results = heuristic_scheduler.run_all()

    # Export results -> experiments/ga/schedule.json
    export_schedule_results(
        best_chromo,
        tasks,
        dataset,
        config.output_path,
        history=history,
        heuristic_comparisons=heuristic_results,
    )

    print(f"\n✅ GA results exported to: {config.output_path}")
    print(f"   Fitness: {best_chromo.fitness:.2f} | Makespan: {best_chromo.makespan:.1f}m | "
          f"Tardiness: {best_chromo.total_tardiness:.1f}m | Setup: {best_chromo.total_setup_time:.1f}m")


if __name__ == "__main__":
    main()
