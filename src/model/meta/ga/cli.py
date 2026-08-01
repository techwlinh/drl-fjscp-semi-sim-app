import json
from datetime import datetime
from pathlib import Path

from src.schema.data import DatasetOutputModel
from src.model.heuristics.dispatching import HeuristicScheduler
from src.model.meta.ga.config import GAConfig
from src.model.meta.ga.exporter import export_schedule_results
from src.model.meta.ga.optimizer import GAOptimizer


def main() -> None:
    config = GAConfig()

    # Load dataset JSON
    with open(config.dataset_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)
    dataset = DatasetOutputModel.model_validate(raw_json)

    print(
        f"Starting GA Optimization on dataset '{config.dataset_path}' "
        f"({len(dataset.job_list)} jobs, {len(dataset.factory_infrastructure.areas)} areas)..."
    )

    optimizer = GAOptimizer(dataset, config)
    best_chromo, tasks, history = optimizer.run()

    print("Running 4 Heuristic Dispatching Rule Benchmarks (FIFO, EDD, SPT, CR)...")
    heuristic_scheduler = HeuristicScheduler(dataset, config)
    heuristic_results = heuristic_scheduler.run_all()

    # Generate timestamped experiment filepath: experiments/{alg}_{timestamp}.json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_filepath = Path(config.experiments_dir) / f"{config.algorithm_name}_{timestamp}.json"

    # Export to timestamped experiment file
    export_schedule_results(
        best_chromo,
        tasks,
        dataset,
        str(exp_filepath),
        history=history,
        heuristic_comparisons=heuristic_results,
    )

    # Export to standard default output path (and update web_viz)
    export_schedule_results(
        best_chromo,
        tasks,
        dataset,
        config.output_path,
        history=history,
        heuristic_comparisons=heuristic_results,
    )


if __name__ == "__main__":
    main()
