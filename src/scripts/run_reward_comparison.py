"""
run_reward_comparison.py - Comprehensive Multi-Reward & GA Benchmark Suite

Executes and compares:
1. All 5 PPO Reward Strategies ('baseline', 'continuous_tardiness', 'pbrs', 'milestone_progress', 'workload_balance')
2. GA Metaheuristic (checks for existing schedule; prompts or reuses)
3. Classic Heuristic Dispatching Rules (FIFO, EDD, SPT, Priority CR)

Exports individual experiment subfolders:
- experiments/ppo_baseline/schedule.json
- experiments/ppo_continuous_tardiness/schedule.json
- experiments/ppo_pbrs/schedule.json
- experiments/ppo_milestone_progress/schedule.json
- experiments/ppo_workload_balance/schedule.json
- experiments/ga/schedule.json

Updates experiments/manifest.json for multi-model Gantt & progression charts in web_viz.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.schema.data import DatasetOutputModel
from src.config.ppo import PPOConfig
from src.model.drl.ppo.optimizer import PPOOptimizer
from src.model.drl.ppo.reward import REWARD_STRATEGIES
from src.model.heuristics.dispatching import HeuristicScheduler
from src.model.meta.ga.config import GAConfig
from src.model.meta.ga.exporter import export_schedule_results, update_experiments_manifest
from src.model.meta.ga.optimizer import GAOptimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO Reward & State Design Comparison Benchmark Suite")
    parser.add_argument(
        "-e", "--episodes", type=int, default=1000, help="Number of training episodes per PPO strategy"
    )
    parser.add_argument(
        "-f", "--force-ga", action="store_true", help="Force re-running GA even if existing schedule exists"
    )
    parser.add_argument(
        "--non-interactive", action="store_true", help="Run non-interactively (default to reusing GA if present)"
    )
    parser.add_argument(
        "-s", "--state-strategies", nargs="+", default=["enhanced", "baseline"],
        choices=["enhanced", "baseline"], help="State design strategies to evaluate ('enhanced', 'baseline')"
    )
    parser.add_argument(
        "-r", "--reward-strategies", nargs="+", default=None,
        help="Specific reward strategies to evaluate (default: all registered reward strategies)"
    )
    return parser.parse_args()


def run_benchmark():
    args = parse_args()
    num_episodes = args.episodes

    dataset_path = "data/fjsp_dataset_seed42.json"
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} not found.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data_json = json.load(f)
    dataset = DatasetOutputModel(**data_json)

    ga_config = GAConfig(dataset_path=dataset_path)
    workspace_root = Path(".")

    print("=" * 75)
    print("=== COMPREHENSIVE PPO REWARD & STATE DESIGN BENCHMARK SUITE ===")
    print("=" * 75)

    all_comparisons = {}

    # 1. GA Metaheuristic Check & Run
    ga_schedule_path = Path("experiments/ga/schedule.json")
    should_run_ga = True

    if ga_schedule_path.exists() and not args.force_ga:
        if args.non_interactive:
            should_run_ga = False
        else:
            try:
                user_choice = input("\n[?] GA schedule already exists in 'experiments/ga/schedule.json'. Re-run GA? [y/N]: ").strip().lower()
                if user_choice not in ["y", "yes"]:
                    should_run_ga = False
            except (KeyboardInterrupt, EOFError):
                should_run_ga = False

    if should_run_ga:
        print("\n---> Running GA Metaheuristic Optimizer...")
        ga_optimizer = GAOptimizer(dataset, ga_config)
        best_ga_chromo, ga_tasks, ga_history = ga_optimizer.run()

        export_schedule_results(
            best_ga_chromo,
            ga_tasks,
            dataset,
            ga_config.output_path,
            history=ga_history,
        )
        print(f"✅ GA results exported to: {ga_config.output_path}")
    else:
        print("\n---> Reusing existing GA Metaheuristic schedule from experiments/ga/schedule.json")

    # Load GA schedule info if present
    if ga_schedule_path.exists():
        with open(ga_schedule_path, "r", encoding="utf-8") as f:
            ga_json = json.load(f)
        kpis = ga_json.get("kpis", {})
        all_comparisons["ga"] = {
            "name": "GA Metaheuristic",
            "makespan": float(kpis.get("makespan", 0.0)),
            "total_weighted_tardiness": float(kpis.get("total_weighted_tardiness", 0.0)),
            "total_setup_time": float(kpis.get("total_setup_time", 0.0)),
            "fitness": float(kpis.get("fitness", 0.0)),
        }

    # 2. Heuristic Dispatching Rules
    print("\n---> Running Classic Heuristic Dispatching Rules...")
    heuristic_scheduler = HeuristicScheduler(dataset, ga_config)
    heuristic_results = heuristic_scheduler.run_all()
    all_comparisons.update(heuristic_results)

    # 3. PPO State Strategies & Reward Strategies Comparison
    state_strategies = args.state_strategies
    reward_strategies = args.reward_strategies or list(REWARD_STRATEGIES.keys())

    total_combos = len(state_strategies) * len(reward_strategies)
    print(
        f"\n---> Evaluating {total_combos} PPO Combinations "
        f"({len(state_strategies)} State Designs x {len(reward_strategies)} Reward Strategies, {num_episodes} Episodes each)..."
    )

    for state_strat in state_strategies:
        for strat_name in reward_strategies:
            if len(state_strategies) == 1 and state_strat == "enhanced":
                folder_name = f"ppo_{strat_name}"
            else:
                folder_name = f"ppo_{state_strat}_{strat_name}"

            output_dir = Path(f"experiments/{folder_name}")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = str(output_dir / "schedule.json")
            model_path = str(output_dir / "ppo_model.pt")

            print(f"\n" + "-" * 60)
            print(f"🤖 Training PPO [State: {state_strat.upper()} | Reward: {strat_name.upper()}] -> {output_dir}")
            print("-" * 60)

            # Set fixed random seeds for fair evaluation across strategies
            torch.manual_seed(42)
            np.random.seed(42)

            ppo_config = PPOConfig(
                num_episodes=num_episodes,
                state_strategy=state_strat,
                reward_strategy=strat_name,
                output_path=output_path,
                model_checkpoint_path=model_path,
                use_numba=True,
            )

            start_time = time.time()
            optimizer = PPOOptimizer(dataset, ppo_config)
            optimizer.train()
            elapsed = time.time() - start_time

            # Predict best schedule using trained model
            best_ppo_chromo, ppo_tasks, ppo_metrics = optimizer.predict()
            state_label = "Enhanced" if state_strat == "enhanced" else "Baseline"
            ppo_metrics["name"] = f"PPO [{state_label} State] ({strat_name.replace('_', ' ').title()})"
            ppo_metrics["training_time_sec"] = round(float(elapsed), 2)

            # Export schedule result for this strategy
            export_schedule_results(
                best_ppo_chromo,
                ppo_tasks,
                dataset,
                output_path,
                history=getattr(optimizer, "history", []),
                heuristic_comparisons=all_comparisons,
            )

            all_comparisons[folder_name] = ppo_metrics
            print(
                f"✅ [{state_strat}/{strat_name}] Complete: Fitness={best_ppo_chromo.fitness:.2f}, "
                f"Makespan={best_ppo_chromo.makespan:.1f}m, Tardiness={best_ppo_chromo.total_tardiness:.1f}m, "
                f"Time={elapsed:.1f}s"
            )

    # 4. Update experiments manifest for web_viz
    update_experiments_manifest(workspace_root)

    # 5. Print Summary Comparison Table
    print("\n" + "=" * 95)
    print("FINAL BENCHMARK COMPARISON SUMMARY")
    print("=" * 95)
    print(f"{'Algorithm / State & Reward Strategy':<45} | {'Makespan (m)':<12} | {'Tardiness (m)':<13} | {'Setup (m)':<10} | {'Fitness':<8}")
    print("-" * 95)

    sorted_items = sorted(all_comparisons.items(), key=lambda x: x[1].get("fitness", float("inf")))
    for key, v in sorted_items:
        print(
            f"{v.get('name', key):<45} | {float(v.get('makespan', 0)):<12.1f} | "
            f"{float(v.get('total_weighted_tardiness', 0)):<13.1f} | {float(v.get('total_setup_time', 0)):<10.1f} | "
            f"{float(v.get('fitness', 0)):<8.2f}"
        )
    print("=" * 95)
    print("\n✅ All experiments updated in 'experiments/' directory.")
    print("📊 Run 'npm run dev' inside 'web_viz' to compare Gantt charts and progress curves!\n")


if __name__ == "__main__":
    run_benchmark()
