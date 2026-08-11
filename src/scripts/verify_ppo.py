"""
verify_ppo.py — PPO training verification harness (QA gate).

Runs PPO training for N episodes on a given dataset across multiple seeds,
then prints grep-able assertion lines about reward slope and fitness.

Usage:
    python -m src.scripts.verify_ppo \\
        --instance data/fjsp_dataset_small12.json \\
        --episodes 20 \\
        --seeds 42 43 \\
        --reward continuous_tardiness \\
        --state enhanced

Output lines (grep-able):
    EP ep=<n> seed=<s> reward=<r:.4f>
    SLOPE seed=<s> first=<f:.4f> last=<l:.4f> POSITIVE=<True/False>
    FIT ppo=<f:.2f> setup_ppo=<sp:.1f> [ga=<g:.2f> setup_ga=<sg:.1f>]

Exit code:
    0  — all seeds have POSITIVE=True
    1  — at least one seed has POSITIVE=False
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import List

import numpy as np
import torch

# Ensure project root is on sys.path when run as __main__
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schema.data import DatasetOutputModel
from src.config.ppo import PPOConfig
from src.model.drl.ppo.optimizer import PPOOptimizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PPO verification harness — reward slope + fitness QA gate"
    )
    p.add_argument(
        "--instance",
        required=True,
        metavar="PATH",
        help="Path to dataset JSON (e.g. data/fjsp_dataset_small12.json)",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=20,
        metavar="N",
        help="Number of training episodes per seed (default: 20)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        metavar="S",
        help="Random seeds to evaluate (default: 42)",
    )
    p.add_argument(
        "--reward",
        default="continuous_tardiness",
        metavar="STRATEGY",
        help="Reward strategy (default: continuous_tardiness)",
    )
    p.add_argument(
        "--state",
        default="enhanced",
        metavar="STRATEGY",
        help="State strategy (default: enhanced)",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataset(path: str) -> DatasetOutputModel:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return DatasetOutputModel(**data)


def _load_ga_kpis(ga_json_path: str = "experiments/ga/schedule.json"):
    """Return (fitness, setup_time) from GA schedule JSON, or None if absent."""
    p = Path(ga_json_path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    kpis = doc.get("kpis", {})
    fitness = kpis.get("fitness")
    setup = kpis.get("total_setup_time")
    if fitness is None or setup is None:
        return None
    return float(fitness), float(setup)


def _slope_windows(rewards: List[float], episodes: int):
    """Return (first_window_mean, last_window_mean, is_positive)."""
    k = max(3, episodes // 4)
    first = mean(rewards[:k])
    last = mean(rewards[-k:])
    return first, last, last > first


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)

    # Validate instance path
    if not Path(args.instance).exists():
        print(f"ERROR: dataset not found: {args.instance}", file=sys.stderr)
        return 2

    dataset = _load_dataset(args.instance)
    ga_kpis = _load_ga_kpis()

    all_positive: List[bool] = []
    best_ppo_fitness: float | None = None
    best_ppo_setup: float | None = None

    for seed in args.seeds:
        # Reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        ppo_config = PPOConfig(
            dataset_path=args.instance,
            num_episodes=args.episodes,
            reward_strategy=args.reward,
            state_strategy=args.state,
            # Avoid writing checkpoints / output during QA runs
            output_path=f"/dev/null",
            model_checkpoint_path=f"experiments/verify_ppo_seed{seed}/ppo_model.pt",
            resume_training=False,
            use_numba=True,
        )

        optimizer = PPOOptimizer(dataset, ppo_config)
        best_chromo, history = optimizer.train()

        # Per-episode reward lines
        rewards: List[float] = []
        for entry in history:
            ep = entry["episode"]
            r = float(entry["total_reward"])
            rewards.append(r)
            print(f"EP ep={ep} seed={seed} reward={r:.4f}", flush=True)

        # Slope assertion
        first, last, positive = _slope_windows(rewards, args.episodes)
        print(
            f"SLOPE seed={seed} first={first:.4f} last={last:.4f} POSITIVE={positive}",
            flush=True,
        )
        all_positive.append(positive)

        # Track best fitness across seeds (lower is better)
        if best_chromo is not None:
            f = float(best_chromo.fitness)
            s = float(best_chromo.total_setup_time)
            if best_ppo_fitness is None or f < best_ppo_fitness:
                best_ppo_fitness = f
                best_ppo_setup = s

    # Fitness summary line
    if best_ppo_fitness is not None:
        fit_line = f"FIT ppo={best_ppo_fitness:.2f} setup_ppo={best_ppo_setup:.1f}"
        if ga_kpis is not None:
            ga_fit, ga_setup = ga_kpis
            fit_line += f" ga={ga_fit:.2f} setup_ga={ga_setup:.1f}"
        print(fit_line, flush=True)
    else:
        print("FIT ppo=N/A setup_ppo=N/A", flush=True)

    # Exit code
    all_ok = all(all_positive)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
