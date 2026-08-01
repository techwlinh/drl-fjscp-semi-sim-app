import argparse
import json

from src.schema.data import DatasetOutputModel
from src.model.heuristics.dispatching import HeuristicScheduler
from src.model.meta.ga.config import GAConfig
from src.model.meta.ga.exporter import export_schedule_results
from src.model.meta.ga.optimizer import GAOptimizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GA Metaheuristic Scheduler for Semiconductor Fab FJSP"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/fjsp_dataset_seed42.json",
        help="Input dataset JSON file path",
    )
    parser.add_argument(
        "--generations", type=int, default=50, help="Number of GA generations"
    )
    parser.add_argument(
        "--pop-size", type=int, default=60, help="Population size"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/ga_schedule_results.json",
        help="Output schedule JSON file path",
    )

    args = parser.parse_args()

    # Load dataset JSON
    with open(args.dataset, "r", encoding="utf-8") as f:
        raw_json = json.load(f)
    dataset = DatasetOutputModel.model_validate(raw_json)

    config = GAConfig(pop_size=args.pop_size, generations=args.generations)
    print(
        f"Starting GA Optimization on dataset '{args.dataset}' "
        f"({len(dataset.job_list)} jobs, {len(dataset.factory_infrastructure.areas)} areas)..."
    )

    optimizer = GAOptimizer(dataset, config)
    best_chromo, tasks, history = optimizer.run()

    print("Running 4 Heuristic Dispatching Rule Benchmarks (FIFO, EDD, SPT, CR)...")
    heuristic_scheduler = HeuristicScheduler(dataset, config)
    heuristic_results = heuristic_scheduler.run_all()

    export_schedule_results(
        best_chromo,
        tasks,
        dataset,
        args.output,
        history=history,
        heuristic_comparisons=heuristic_results,
    )


if __name__ == "__main__":
    main()
