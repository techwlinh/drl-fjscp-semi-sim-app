"""
Module and CLI script to extract convergence history and plot objective curves
over iterations for algorithm experiment folders in the FJSCP simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# Styling configuration for publication-quality 300 DPI plots
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "lines.linewidth": 1.75,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

PRIMARY_OBJECTIVES = [
    ("fitness", "Fitness Score (Weighted Cost)", "Cost (Lower is better)"),
    ("makespan", "Makespan ($C_{max}$)", "Time (Lower is better)"),
    ("tardiness", "Total Weighted Tardiness", "Tardiness (Lower is better)"),
    ("setup_time", "Total Setup Time", "Setup Time (Lower is better)"),
]

COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"
]

DEFAULT_COMPARISON_ALGOS = [
    "ga",
    "pso",
    "sso",
    "ppo_baseline_baseline",
    "ppo_baseline_continuous_tardiness",
    "ppo_baseline_workload_balance",
]


def load_manifest(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    """Loads algorithm metadata mapped by algorithm ID."""
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {item["algorithm"]: item for item in data.get("experiments", [])}
    except Exception:
        return {}


def load_experiment_history(exp_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extracts iteration history from schedule.json inside an experiment folder.

    Returns:
        (DataFrame of history, Dictionary of KPIs/metadata)
    """
    schedule_file = exp_dir / "schedule.json"
    if not schedule_file.exists():
        raise FileNotFoundError(f"No schedule.json found in {exp_dir}")

    with open(schedule_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history", [])
    kpis = data.get("kpis", {})

    if not history:
        raise ValueError(f"No 'history' records found in {schedule_file}")

    df = pd.DataFrame(history)

    # Standardize iteration column: metaheuristics use 'generation', PPO uses 'episode'
    if "generation" in df.columns:
        df["iteration"] = df["generation"]
    elif "episode" in df.columns:
        df["iteration"] = df["episode"]
    else:
        df["iteration"] = range(len(df))

    return df, kpis


def load_all_experiments(base_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Loads iteration history for all experiment folders under base_dir.
    """
    manifest_info = load_manifest(base_dir / "manifest.json")
    results = {}

    for sub_dir in sorted(base_dir.iterdir()):
        if not sub_dir.is_dir() or sub_dir.name.startswith("."):
            continue

        schedule_path = sub_dir / "schedule.json"
        if schedule_path.exists():
            try:
                df, kpis = load_experiment_history(sub_dir)
                algo_id = sub_dir.name
                title = manifest_info.get(algo_id, {}).get("title", algo_id.replace("_", " ").upper())
                results[algo_id] = {
                    "dir": sub_dir,
                    "title": title,
                    "history": df,
                    "kpis": kpis,
                    "manifest": manifest_info.get(algo_id, {}),
                }
            except Exception as e:
                print(f"Warning: Failed to load {sub_dir.name}: {e}")

    return results


def plot_single_algorithm(
    algo_id: str,
    data: Dict[str, Any],
    save_path: Optional[Path] = None,
    show: bool = False,
    dpi: int = 300
) -> plt.Figure:
    """
    Plots a 4-panel figure showing the convergence of primary objectives
    (Fitness, Makespan, Tardiness, Setup Time) over iterations for one algorithm.
    """
    df = data["history"]
    title = data.get("title", algo_id)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=dpi)
    fig.suptitle(f"Convergence History: {title}", fontsize=14, fontweight="bold", y=0.98)

    iterations = df["iteration"]

    for ax, (obj_key, obj_title, y_label) in zip(axes.flat, PRIMARY_OBJECTIVES):
        if obj_key in df.columns:
            series = df[obj_key]
            ax.plot(iterations, series, color="#1f77b4", label=f"Current {obj_title}")

            # Plot running best (min-so-far) curve if applicable
            cummin = series.cummin()
            ax.plot(iterations, cummin, color="#d62728", linestyle="--", alpha=0.85, label="Best-so-far")

            ax.set_title(obj_title, fontweight="bold")
            ax.set_xlabel("Iteration (Generation / Episode)")
            ax.set_ylabel(y_label)
            ax.grid(True)
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, f"'{obj_key}' not recorded", ha="center", va="center")
            ax.set_title(obj_title)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved plot: {save_path}")

    if show:
        plt.show()

    return fig


def plot_comparative_objectives(
    all_experiments: Dict[str, Dict[str, Any]],
    include_algos: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
    use_best_so_far: bool = True,
    dpi: int = 300
) -> plt.Figure:
    """
    Plots comparative objective curves across selected algorithms in a 2x2 grid
    with a unified legend placed cleanly at the bottom.
    """
    if include_algos is None:
        target_algos = [k for k in DEFAULT_COMPARISON_ALGOS if k in all_experiments]
    else:
        target_algos = [k for k in include_algos if k in all_experiments]

    if not target_algos:
        target_algos = list(all_experiments.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=dpi)
    fig.suptitle(
        f"Algorithm Convergence Comparison ({'Best-So-Far' if use_best_so_far else 'Iteration Values'})",
        fontsize=14,
        fontweight="bold",
        y=0.97
    )

    algo_labels = {
        "ga": "GA",
        "pso": "PSO",
        "sso": "SSO",
        "ppo_baseline_baseline": "PPO (Baseline)",
        "ppo_baseline_continuous_tardiness": "PPO (Continuous Tardiness)",
        "ppo_baseline_workload_balance": "PPO (Workload Balance)",
    }

    for ax, (obj_key, obj_title, y_label) in zip(axes.flat, PRIMARY_OBJECTIVES):
        ax.set_title(obj_title, fontweight="bold")
        ax.set_xlabel("Iteration (Generation / Episode)")
        ax.set_ylabel(y_label)
        ax.grid(True)

        for color_idx, algo_id in enumerate(target_algos):
            exp_info = all_experiments[algo_id]
            df = exp_info["history"]
            if obj_key not in df.columns:
                continue

            color = COLOR_PALETTE[color_idx % len(COLOR_PALETTE)]
            x = df["iteration"]
            y = df[obj_key].cummin() if use_best_so_far else df[obj_key]
            
            # Label prioritizing clean name or manifest title
            label = algo_labels.get(algo_id, exp_info.get("title", algo_id))

            ax.plot(x, y, label=label, color=color, linewidth=2, alpha=0.9)

    # Extract handles & labels from the first subplot to place a single unified bottom legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(len(labels), 3),
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=9.5,
        handlelength=2.5,
    )

    # Adjust layout to leave space for the bottom legend and top title
    plt.tight_layout(rect=[0, 0.09, 1, 0.95])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved comparative plot: {save_path}")

    if show:
        plt.show()

    return fig


def plot_grouped_comparison(
    all_experiments: Dict[str, Dict[str, Any]],
    output_dir: Path,
    show: bool = False,
    dpi: int = 300
) -> List[plt.Figure]:
    """
    Generates comparison plots categorized into:
    1. Metaheuristics (GA, PSO, SSO)
    2. PPO Baselines
    3. PPO Enhanced
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    groups = {
        "Metaheuristics (GA, PSO, SSO)": [k for k in all_experiments if k in ["ga", "pso", "sso"]],
        "PPO Baseline Variants": [k for k in all_experiments if k.startswith("ppo_baseline_")],
        "PPO Enhanced Variants": [k for k in all_experiments if k.startswith("ppo_enhanced_")],
    }

    for group_name, algo_keys in groups.items():
        subset = {k: all_experiments[k] for k in algo_keys if k in all_experiments}
        if not subset:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=dpi)
        fig.suptitle(f"Convergence Comparison: {group_name}", fontsize=14, fontweight="bold", y=0.98)

        for ax, (obj_key, obj_title, y_label) in zip(axes.flat, PRIMARY_OBJECTIVES):
            ax.set_title(obj_title, fontweight="bold")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(y_label)
            ax.grid(True)

            color_idx = 0
            for algo_id, exp_info in subset.items():
                df = exp_info["history"]
                if obj_key not in df.columns:
                    continue

                color = COLOR_PALETTE[color_idx % len(COLOR_PALETTE)]
                color_idx += 1

                x = df["iteration"]
                y_best = df[obj_key].cummin()
                label = exp_info.get("title", algo_id)

                ax.plot(x, y_best, label=label, color=color, linewidth=2)

            ax.legend(loc="best", fontsize=8.5)

        plt.tight_layout()
        slug_map = {
            "Metaheuristics (GA, PSO, SSO)": "metaheuristics",
            "PPO Baseline Variants": "ppo_baselines",
            "PPO Enhanced Variants": "ppo_enhanced",
        }
        file_slug = slug_map.get(group_name, group_name.lower().replace(" ", "_"))
        save_path = output_dir / f"comparison_{file_slug}.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved group plot: {save_path}")
        figures.append(fig)

        if show:
            plt.show()

    return figures


def main():
    parser = argparse.ArgumentParser(
        description="Plot objective convergence across iterations for experiment algorithm folders."
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default=str(Path(__file__).parent),
        help="Path to the experiments root directory containing algorithm subfolders."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Plot only a specific algorithm folder (e.g., experiments/ga)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated plot images (defaults to experiments/plots)."
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="List of algorithm IDs to compare (e.g. --algorithms ga pso sso ppo_baseline_baseline). Default is ga, pso, sso, ppo_baseline_baseline."
    )
    parser.add_argument(
        "--all-algorithms",
        action="store_true",
        help="Compare all algorithms in the experiment directory instead of the default subset."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution DPI for generated plots (default: 300)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively using plt.show()."
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving plot images to disk."
    )

    args = parser.parse_args()

    exp_dir = Path(args.experiments_dir).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (exp_dir / "plots")

    if args.folder:
        folder_path = Path(args.folder).resolve()
        df, kpis = load_experiment_history(folder_path)
        algo_id = folder_path.name
        manifest_info = load_manifest(exp_dir / "manifest.json")
        title = manifest_info.get(algo_id, {}).get("title", algo_id.upper())

        save_path = None if args.no_save else (out_dir / f"convergence_{algo_id}.png")
        plot_single_algorithm(
            algo_id,
            {"history": df, "title": title, "kpis": kpis},
            save_path=save_path,
            show=args.show,
            dpi=args.dpi
        )
        return

    print(f"Loading all experiment folders from: {exp_dir}")
    all_experiments = load_all_experiments(exp_dir)
    print(f"Loaded {len(all_experiments)} algorithm experiments: {list(all_experiments.keys())}")

    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot individual convergence for each algorithm
    print("\n--- Generating individual algorithm convergence plots ---")
    for algo_id, exp_data in all_experiments.items():
        save_path = None if args.no_save else (out_dir / f"convergence_{algo_id}.png")
        plot_single_algorithm(algo_id, exp_data, save_path=save_path, show=args.show, dpi=args.dpi)
        plt.close()

    # 2. Plot comparative curves for selected algorithms (GA, PSO, SSO, PPO Baseline)
    print("\n--- Generating comparative plots ---")
    include_algos = list(all_experiments.keys()) if args.all_algorithms else args.algorithms
    compare_save_path = None if args.no_save else (out_dir / "comparison_algorithms.png")
    plot_comparative_objectives(
        all_experiments,
        include_algos=include_algos,
        save_path=compare_save_path,
        show=args.show,
        dpi=args.dpi
    )
    plt.close()

    # 3. Plot grouped comparative curves
    print("\n--- Generating grouped comparative plots ---")
    if not args.no_save:
        plot_grouped_comparison(all_experiments, output_dir=out_dir, show=args.show, dpi=args.dpi)
        plt.close("all")

    print(f"\n[DONE] All plots have been generated at {args.dpi} DPI and saved to: {out_dir}")


if __name__ == "__main__":
    main()
