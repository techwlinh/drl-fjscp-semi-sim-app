"""
viz_ppo.py — Comprehensive PPO Training History Visualization Suite.

Plots and analyzes training curves and multi-objective performance for DRL PPO
scheduling models stored under the 'experiments/' directory.

Generates:
1. Individual Model Diagnostic Dashboards (6-panel: Reward, Fitness, Makespan,
   Tardiness, Policy Loss, Value Loss + KPI summary box).
2. Reward Strategy Comparison (Comparing 5 reward designs for Enhanced & Baseline states).
3. State Representation Ablation Study (Enhanced vs Baseline state convergence).
4. Actor-Critic Loss Convergence Dynamics (Policy Loss and Critic MSE stability).
5. KPI Multi-Objective Benchmark Bar Charts (Fitness, Makespan, Tardiness, On-Time %, Util %).
6. Multi-Criteria Radar / Spider Chart (Trade-off across factory operational metrics).

Usage:
    # Run full visualization suite (all plots + individual dashboards)
    uv run src/scripts/viz_ppo.py

    # Plot only a specific model dashboard
    uv run src/scripts/viz_ppo.py --model ppo_enhanced_continuous_tardiness

    # Generate comparative plots without individual model dashboards
    uv run src/scripts/viz_ppo.py --no-individual

    # Custom output directory and smoothing window
    uv run src/scripts/viz_ppo.py --output-dir experiments/plots/ppo --window 30 --dpi 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ==============================================================================
# GLOBAL STYLING & COLOR SCHEMES (Publication Quality)
# ==============================================================================

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "normal",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.titleweight": "bold",
    "lines.linewidth": 1.8,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "axes.grid": True,
})

# Curated Palette for Reward Strategies
REWARD_COLORS: Dict[str, str] = {
    "baseline": "#1f77b4",              # Classic Blue
    "continuous_tardiness": "#d62728",  # Crimson Red
    "milestone_progress": "#ff7f0e",    # Bright Orange
    "pbrs": "#9467bd",                  # Muted Purple
    "workload_balance": "#2ca02c",      # Forest Green
    "vanilla": "#7f7f7f",               # Neutral Gray
}

# Curated Palette for State Designs
STATE_COLORS: Dict[str, str] = {
    "enhanced": "#2980b9",  # Strong Blue
    "baseline": "#e67e22",  # Vivid Orange
    "raw": "#95a5a6",       # Cool Gray
}

REWARD_DISPLAY_NAMES: Dict[str, str] = {
    "baseline": "Terminal / Sparse Reward",
    "continuous_tardiness": "Continuous Tardiness Penalty",
    "milestone_progress": "Milestone Progress Reward",
    "pbrs": "Potential-Based Shaping (PBRS)",
    "workload_balance": "Workload Balance Reward",
    "vanilla": "Vanilla Reward",
}

STATE_DISPLAY_NAMES: Dict[str, str] = {
    "enhanced": "Enhanced State (Domain Engineered)",
    "baseline": "Baseline State",
    "raw": "Raw State",
}


# ==============================================================================
# DATA PARSING & PREPARATION
# ==============================================================================

def parse_model_id(algo_id: str) -> Tuple[str, str]:
    """
    Deconstructs an experiment algorithm ID into (state_strategy, reward_strategy).
    Examples:
        'ppo_baseline_continuous_tardiness' -> ('baseline', 'continuous_tardiness')
        'ppo_enhanced_pbrs' -> ('enhanced', 'pbrs')
        'ppo_raw_vanilla' -> ('raw', 'vanilla')
        'ppo_continuous_tardiness' -> ('enhanced', 'continuous_tardiness')
    """
    clean = algo_id
    if clean.startswith("ppo_"):
        clean = clean[4:]

    known_states = ["baseline", "enhanced", "raw"]
    for state in known_states:
        if clean.startswith(f"{state}_"):
            reward = clean[len(state) + 1:]
            return state, reward

    return "enhanced", clean


def smooth_series(series: pd.Series, window: int = 25) -> Tuple[pd.Series, pd.Series]:
    """
    Calculates moving average and rolling standard deviation for noisy RL metrics.
    """
    n = len(series)
    if n <= 1:
        return series, pd.Series(np.zeros(n), index=series.index)
    actual_window = max(2, min(window, n // 3 if n >= 6 else n))
    mean_val = series.rolling(window=actual_window, min_periods=1).mean()
    std_val = series.rolling(window=actual_window, min_periods=1).std().fillna(0.0)
    return mean_val, std_val


def load_ppo_experiments(experiments_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Scans the experiments directory for all PPO subfolders containing schedule.json.
    Loads iteration history, KPIs, and precomputes best-so-far series.
    """
    manifest_path = experiments_dir / "manifest.json"
    manifest_map: Dict[str, Dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                manifest_map = {item["algorithm"]: item for item in data.get("experiments", [])}
        except Exception as e:
            print(f"Warning: Could not read manifest.json: {e}")

    models: Dict[str, Dict[str, Any]] = {}

    for sub_dir in sorted(experiments_dir.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("ppo_"):
            continue

        schedule_path = sub_dir / "schedule.json"
        if not schedule_path.exists():
            continue

        try:
            with open(schedule_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_hist = data.get("history", [])
            kpis = data.get("kpis", {})

            if not raw_hist:
                continue

            df = pd.DataFrame(raw_hist)
            if "episode" not in df.columns:
                df["episode"] = range(1, len(df) + 1)

            # Ensure numeric conversion
            numeric_cols = [
                "fitness", "best_fitness", "makespan", "tardiness",
                "setup_time", "total_reward", "policy_loss", "value_loss"
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Precalculate cumulative best (min-so-far)
            if "fitness" in df.columns:
                df["best_fitness_cum"] = df["fitness"].cummin()
            if "makespan" in df.columns:
                df["best_makespan_cum"] = df["makespan"].cummin()
            if "tardiness" in df.columns:
                df["best_tardiness_cum"] = df["tardiness"].cummin()
            if "setup_time" in df.columns:
                df["best_setup_time_cum"] = df["setup_time"].cummin()

            algo_id = sub_dir.name
            state_strat, reward_strat = parse_model_id(algo_id)

            manifest_entry = manifest_map.get(algo_id, {})
            title = manifest_entry.get("title", f"PPO ({state_strat.title()} {reward_strat.replace('_', ' ').title()})")

            models[algo_id] = {
                "id": algo_id,
                "dir": sub_dir,
                "state_strategy": state_strat,
                "reward_strategy": reward_strat,
                "title": title,
                "history": df,
                "kpis": kpis,
                "manifest": manifest_entry,
                "num_episodes": len(df),
            }
        except Exception as err:
            print(f"Warning: Failed to load PPO experiment from {sub_dir.name}: {err}")

    return models


# ==============================================================================
# 1. INDIVIDUAL PPO DASHBOARD (6-Panel Diagnostic)
# ==============================================================================

def plot_single_ppo_dashboard(
    model_data: Dict[str, Any],
    output_dir: Path,
    window: int = 25,
    dpi: int = 300,
    fmt: str = "png",
    show: bool = False,
) -> Path:
    """
    Plots a comprehensive 6-panel training diagnostic dashboard for a single PPO model:
    (1) Episode Total Reward (raw + smoothed mean + ±1 std band)
    (2) Deterministic Evaluation Fitness vs Best-so-Far Curve
    (3) Makespan (C_max) Evolution & Best-so-Far
    (4) Total Weighted Tardiness Evolution & Best-so-Far
    (5) Actor Policy Loss (L_CLIP)
    (6) Critic Value Function Loss (L_VF)
    """
    df: pd.DataFrame = model_data["history"]
    algo_id: str = model_data["id"]
    state_strat: str = model_data["state_strategy"]
    reward_strat: str = model_data["reward_strategy"]
    kpis: Dict[str, Any] = model_data["kpis"]
    episodes = df["episode"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), dpi=dpi)

    reward_color = REWARD_COLORS.get(reward_strat, "#1f77b4")
    best_color = "#d62728"  # Strong red for best-so-far markers

    # Header title & KPI Summary Banner
    title_text = (
        f"PPO Training Diagnostic: {STATE_DISPLAY_NAMES.get(state_strat, state_strat.title())} "
        f"— {REWARD_DISPLAY_NAMES.get(reward_strat, reward_strat.title())}"
    )
    kpi_banner = (
        f"KPIs: Fitness={kpis.get('fitness', 'N/A')}  |  "
        f"Makespan={kpis.get('makespan', 'N/A')}m  |  "
        f"Tardiness={kpis.get('total_weighted_tardiness', 'N/A')}  |  "
        f"Setup={kpis.get('total_setup_time', 'N/A')}m  |  "
        f"On-Time={kpis.get('on_time_rate_percent', 'N/A')}%  |  "
        f"Tool Util={kpis.get('avg_tool_utilization_percent', 'N/A')}%"
    )
    fig.suptitle(f"{title_text}\n{kpi_banner}", fontsize=12, fontweight="bold", y=0.98)

    # 1. Total Reward Progression
    ax1 = axes[0, 0]
    if "total_reward" in df.columns:
        raw_r = df["total_reward"]
        ax1.plot(episodes, raw_r, color=reward_color, alpha=0.22, linewidth=1.0, label="Raw Episode Reward")
        smooth_r, std_r = smooth_series(raw_r, window=window)
        ax1.plot(episodes, smooth_r, color=reward_color, linewidth=2.0, label=f"Smoothed (MA-{window})")
        ax1.fill_between(episodes, smooth_r - std_r, smooth_r + std_r, color=reward_color, alpha=0.12, label="±1 Std Dev")
        ax1.set_title("1. Cumulative Episode Reward (Return)", pad=6)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel(r"Total Return ($\sum r_t$)")
        ax1.legend(loc="lower right", framealpha=0.85)
    else:
        ax1.text(0.5, 0.5, "No total_reward data", ha="center", va="center")

    # 2. Fitness Progression
    ax2 = axes[0, 1]
    if "fitness" in df.columns:
        ax2.plot(episodes, df["fitness"], color="#34495e", alpha=0.25, linewidth=1.0, label="Eval Fitness")
        cum_fit = df["best_fitness_cum"] if "best_fitness_cum" in df.columns else df["best_fitness"]
        ax2.plot(episodes, cum_fit, color=best_color, linewidth=2.0, label="Best-so-Far Fitness")
        best_val = cum_fit.min()
        best_ep = episodes.iloc[cum_fit.argmin()]
        ax2.scatter([best_ep], [best_val], color=best_color, s=70, zorder=5, marker="*")
        ax2.annotate(
            f"Min: {best_val:.1f}\n(Ep {best_ep})",
            xy=(best_ep, best_val),
            xytext=(best_ep + len(df) * 0.05, best_val * 1.05),
            arrowprops=dict(facecolor=best_color, shrink=0.08, width=1, headwidth=5),
            fontsize=8.5,
            fontweight="bold",
            color=best_color,
        )
        ax2.set_title("2. Fitness Score Convergence (Lower is Better)", pad=6)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Weighted Cost Score")
        ax2.legend(loc="upper right", framealpha=0.85)
    else:
        ax2.text(0.5, 0.5, "No fitness data", ha="center", va="center")

    # 3. Makespan Evolution
    ax3 = axes[1, 0]
    if "makespan" in df.columns:
        ax3.plot(episodes, df["makespan"], color="#2980b9", alpha=0.25, linewidth=1.0, label="Episode Makespan")
        cum_mk = df["best_makespan_cum"] if "best_makespan_cum" in df.columns else df["makespan"].cummin()
        ax3.plot(episodes, cum_mk, color="#154360", linewidth=2.0, label="Best-so-Far Makespan")
        best_mk = cum_mk.min()
        ax3.set_title(f"3. Makespan ($C_{{max}}$) — Final Best: {best_mk:.1f}m", pad=6)
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Makespan (minutes)")
        ax3.legend(loc="upper right", framealpha=0.85)
    else:
        ax3.text(0.5, 0.5, "No makespan data", ha="center", va="center")

    # 4. Total Weighted Tardiness
    ax4 = axes[1, 1]
    if "tardiness" in df.columns:
        ax4.plot(episodes, df["tardiness"], color="#e74c3c", alpha=0.25, linewidth=1.0, label="Episode Tardiness")
        cum_tard = df["best_tardiness_cum"] if "best_tardiness_cum" in df.columns else df["tardiness"].cummin()
        ax4.plot(episodes, cum_tard, color="#922b21", linewidth=2.0, label="Best-so-Far Tardiness")
        best_tard = cum_tard.min()
        ax4.set_title(f"4. Weighted Tardiness — Final Best: {best_tard:,.0f}m", pad=6)
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Weighted Tardiness (minutes)")
        ax4.legend(loc="upper right", framealpha=0.85)
    else:
        ax4.text(0.5, 0.5, "No tardiness data", ha="center", va="center")

    # 5. Actor Policy Loss
    ax5 = axes[2, 0]
    if "policy_loss" in df.columns:
        p_loss = df["policy_loss"]
        ax5.plot(episodes, p_loss, color="#8e44ad", alpha=0.35, linewidth=1.0, label="Policy Loss")
        smooth_p, _ = smooth_series(p_loss, window=window)
        ax5.plot(episodes, smooth_p, color="#5b2c6f", linewidth=2.0, label=f"Smoothed (MA-{window})")
        ax5.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax5.set_title(r"5. Actor Policy Loss ($\mathcal{L}_{CLIP}$)", pad=6)
        ax5.set_xlabel("Episode")
        ax5.set_ylabel("Policy Loss")
        ax5.legend(loc="best", framealpha=0.85)
    else:
        ax5.text(0.5, 0.5, "No policy_loss data", ha="center", va="center")

    # 6. Critic Value Function Loss
    ax6 = axes[2, 1]
    if "value_loss" in df.columns:
        v_loss = df["value_loss"]
        ax6.plot(episodes, v_loss, color="#16a085", alpha=0.35, linewidth=1.0, label="Value Loss")
        smooth_v, _ = smooth_series(v_loss, window=window)
        ax6.plot(episodes, smooth_v, color="#0e6251", linewidth=2.0, label=f"Smoothed (MA-{window})")
        ax6.set_title(r"6. Critic Value Loss ($\mathcal{L}_{VF}$)", pad=6)
        ax6.set_xlabel("Episode")
        ax6.set_ylabel("Value Loss (MSE)")
        ax6.legend(loc="best", framealpha=0.85)
    else:
        ax6.text(0.5, 0.5, "No value_loss data", ha="center", va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = output_dir / f"dashboard_{algo_id}.{fmt}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return save_path


# ==============================================================================
# 2. REWARD STRATEGY COMPARISON (Enhanced & Baseline)
# ==============================================================================

def plot_reward_strategies_comparison(
    models: Dict[str, Dict[str, Any]],
    state_strategy: str,
    output_dir: Path,
    window: int = 25,
    dpi: int = 300,
    fmt: str = "png",
    show: bool = False,
) -> Optional[Path]:
    """
    Plots a 4-panel comparative figure contrasting the 5 reward strategies
    for a chosen state representation (typically 'enhanced' or 'baseline'):
    - Best Fitness Convergence
    - Smoothed Episode Reward (Return)
    - Makespan Convergence
    - Tardiness Convergence
    """
    subset = {
        k: v for k, v in models.items()
        if v["state_strategy"] == state_strategy and v["num_episodes"] > 1
    }

    if not subset:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=dpi)
    state_title = STATE_DISPLAY_NAMES.get(state_strategy, state_strategy.title())
    fig.suptitle(
        f"Reward Strategy Convergence Comparison — [{state_title}]",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for algo_id, m in sorted(subset.items()):
        df = m["history"]
        reward_strat = m["reward_strategy"]
        color = REWARD_COLORS.get(reward_strat, "#333333")
        label = REWARD_DISPLAY_NAMES.get(reward_strat, reward_strat.replace("_", " ").title())
        episodes = df["episode"]

        # Panel 1: Best-so-Far Fitness
        ax1 = axes[0, 0]
        if "best_fitness_cum" in df.columns:
            ax1.plot(episodes, df["best_fitness_cum"], label=label, color=color, linewidth=2.0)

        # Panel 2: Smoothed Episode Return
        ax2 = axes[0, 1]
        if "total_reward" in df.columns:
            smooth_r, _ = smooth_series(df["total_reward"], window=window)
            ax2.plot(episodes, smooth_r, label=label, color=color, linewidth=1.9)

        # Panel 3: Makespan Convergence
        ax3 = axes[1, 0]
        if "best_makespan_cum" in df.columns:
            ax3.plot(episodes, df["best_makespan_cum"], label=label, color=color, linewidth=2.0)

        # Panel 4: Weighted Tardiness Convergence
        ax4 = axes[1, 1]
        if "best_tardiness_cum" in df.columns:
            ax4.plot(episodes, df["best_tardiness_cum"], label=label, color=color, linewidth=2.0)

    # Subplot titles and labels
    axes[0, 0].set_title("Best-so-Far Fitness (Lower is Better)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Weighted Cost")

    axes[0, 1].set_title(f"Episode Return Dynamics (Smoothed MA-{window})")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Episode Total Reward")

    axes[1, 0].set_title("Makespan ($C_{max}$) Convergence (Lower is Better)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Makespan (minutes)")

    axes[1, 1].set_title("Total Weighted Tardiness Convergence (Lower is Better)")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Tardiness (minutes)")

    # Extract handles & labels for unified bottom legend
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
        title="Reward Strategies",
        title_fontsize=10,
    )

    plt.tight_layout(rect=[0, 0.09, 1, 0.95])

    save_path = output_dir / f"reward_comparison_{state_strategy}.{fmt}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return save_path


# ==============================================================================
# 3. STATE REPRESENTATION ABLATION STUDY (Enhanced vs Baseline)
# ==============================================================================

def plot_state_ablation_comparison(
    models: Dict[str, Dict[str, Any]],
    output_dir: Path,
    window: int = 25,
    dpi: int = 300,
    fmt: str = "png",
    show: bool = False,
) -> Optional[Path]:
    """
    Plots an ablation study directly contrasting Enhanced State vs Baseline State
    across common reward strategies (5 subplots in a 2x3 grid, plus 1 summary plot).
    """
    # Identify unique reward strategies present in both enhanced and baseline
    reward_strats = sorted(list({
        m["reward_strategy"] for m in models.values()
        if m["num_episodes"] > 1 and m["state_strategy"] in ["enhanced", "baseline"]
    }))

    if not reward_strats:
        return None

    num_plots = len(reward_strats)
    cols = 3
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.2 * rows), dpi=dpi)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    fig.suptitle(
        "State Representation Ablation Study: Enhanced State vs Baseline State\n"
        "(Fitness Convergence across Reward Functions — Lower is Better)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for idx, strat in enumerate(reward_strats):
        ax = axes_flat[idx]
        strat_title = REWARD_DISPLAY_NAMES.get(strat, strat.replace("_", " ").title())
        ax.set_title(strat_title, pad=6)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Best Fitness Score")

        enhanced_id = f"ppo_enhanced_{strat}"
        baseline_id = f"ppo_baseline_{strat}"

        if enhanced_id in models:
            df_e = models[enhanced_id]["history"]
            ax.plot(
                df_e["episode"],
                df_e["best_fitness_cum"],
                color=STATE_COLORS["enhanced"],
                linewidth=2.2,
                label=f"Enhanced (Final: {df_e['best_fitness_cum'].iloc[-1]:.1f})",
            )

        if baseline_id in models:
            df_b = models[baseline_id]["history"]
            ax.plot(
                df_b["episode"],
                df_b["best_fitness_cum"],
                color=STATE_COLORS["baseline"],
                linewidth=2.0,
                linestyle="--",
                label=f"Baseline (Final: {df_b['best_fitness_cum'].iloc[-1]:.1f})",
            )

        ax.legend(loc="upper right", framealpha=0.85, fontsize=8.5)

    # Turn off any extra unused subplots
    for j in range(num_plots, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = output_dir / f"state_ablation_comparison.{fmt}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return save_path


# ==============================================================================
# 4. ACTOR-CRITIC LOSS DYNAMICS COMPARISON
# ==============================================================================

def plot_loss_dynamics_comparison(
    models: Dict[str, Dict[str, Any]],
    output_dir: Path,
    window: int = 25,
    dpi: int = 300,
    fmt: str = "png",
    show: bool = False,
) -> Optional[Path]:
    """
    Plots Actor Policy Loss (L_CLIP) and Critic Value Loss (L_VF) across all PPO models
    to analyze training stability and variance.
    """
    valid_models = {k: v for k, v in models.items() if v["num_episodes"] > 1}
    if not valid_models:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=dpi)
    fig.suptitle("PPO Training Stability: Actor & Critic Loss Dynamics Across Models", fontsize=13, fontweight="bold", y=0.98)

    for algo_id, m in sorted(valid_models.items()):
        df = m["history"]
        reward_strat = m["reward_strategy"]
        state_strat = m["state_strategy"]
        color = REWARD_COLORS.get(reward_strat, "#333333")
        linestyle = "-" if state_strat == "enhanced" else "--"
        label = f"{state_strat[:3].upper()}: {REWARD_DISPLAY_NAMES.get(reward_strat, reward_strat)}"

        episodes = df["episode"]

        # Policy Loss
        if "policy_loss" in df.columns:
            smooth_p, _ = smooth_series(df["policy_loss"], window=window)
            ax1.plot(episodes, smooth_p, color=color, linestyle=linestyle, linewidth=1.6, alpha=0.85, label=label)

        # Value Loss
        if "value_loss" in df.columns:
            smooth_v, _ = smooth_series(df["value_loss"], window=window)
            ax2.plot(episodes, smooth_v, color=color, linestyle=linestyle, linewidth=1.6, alpha=0.85, label=label)

    ax1.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax1.set_title(rf"Actor Policy Loss ($\mathcal{{L}}_{{CLIP}}$, MA-{window})", pad=6)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Loss")

    ax2.set_title(rf"Critic Value Loss ($\mathcal{{L}}_{{VF}}$ MSE, MA-{window})", pad=6)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Value Loss")

    # Legend for Loss plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(len(labels), 4),
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=8.5,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.94])

    save_path = output_dir / f"loss_dynamics_comparison.{fmt}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return save_path


# ==============================================================================
# 5. MULTI-OBJECTIVE KPI SUMMARY BAR CHARTS
# ==============================================================================

def plot_kpi_summary_bars(
    models: Dict[str, Dict[str, Any]],
    output_dir: Path,
    dpi: int = 300,
    fmt: str = "png",
    show: bool = False,
) -> Optional[Path]:
    """
    Plots publication-ready grouped bar charts comparing final KPIs for all PPO models:
    - Makespan (C_max)
    - Total Weighted Tardiness
    - Total Setup Time
    - Best Fitness Score
    - On-Time Delivery Rate & Tool Utilization %
    """
    valid_models = [m for m in models.values() if m["num_episodes"] > 1]
    if not valid_models:
        return None

    # Sort models by fitness (best first)
    valid_models.sort(key=lambda m: float(m["kpis"].get("fitness", float("inf"))))

    labels = [
        f"{m['state_strategy'][:3].upper()}-{m['reward_strategy'].replace('_', ' ').title()}"
        for m in valid_models
    ]
    colors = [REWARD_COLORS.get(m["reward_strategy"], "#1f77b4") for m in valid_models]

    fitness_vals = [float(m["kpis"].get("fitness", 0.0)) for m in valid_models]
    makespan_vals = [float(m["kpis"].get("makespan", 0.0)) for m in valid_models]
    tardiness_vals = [float(m["kpis"].get("total_weighted_tardiness", 0.0)) for m in valid_models]
    ontime_vals = [float(m["kpis"].get("on_time_rate_percent", 0.0)) for m in valid_models]
    util_vals = [float(m["kpis"].get("avg_tool_utilization_percent", 0.0)) for m in valid_models]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=dpi)
    fig.suptitle("PPO Models Benchmark: Final KPI Comparison (Sorted by Fitness)", fontsize=13, fontweight="bold", y=0.98)

    y_pos = np.arange(len(valid_models))

    # 1. Fitness Score
    ax1 = axes[0, 0]
    bars1 = ax1.barh(y_pos, fitness_vals, color=colors, alpha=0.85, edgecolor="#333333", linewidth=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(fitness_vals) * 1.15)
    ax1.set_xlabel("Weighted Fitness (Lower is Better)")
    ax1.set_title("1. Overall Objective Fitness Score")
    for bar, val in zip(bars1, fitness_vals):
        ax1.text(val + max(fitness_vals) * 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", fontsize=8)

    # 2. Makespan
    ax2 = axes[0, 1]
    bars2 = ax2.barh(y_pos, makespan_vals, color=colors, alpha=0.85, edgecolor="#333333", linewidth=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(makespan_vals) * 1.16)
    ax2.set_xlabel("Makespan (minutes, Lower is Better)")
    ax2.set_title("2. Makespan ($C_{max}$)")
    for bar, val in zip(bars2, makespan_vals):
        ax2.text(val + max(makespan_vals) * 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.1f}m", va="center", fontsize=8)

    # 3. Tardiness
    ax3 = axes[1, 0]
    bars3 = ax3.barh(y_pos, tardiness_vals, color=colors, alpha=0.85, edgecolor="#333333", linewidth=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, fontsize=8.5)
    ax3.invert_yaxis()
    ax3.set_xlim(0, max(tardiness_vals) * 1.16)
    ax3.set_xlabel("Weighted Tardiness (minutes, Lower is Better)")
    ax3.set_title("3. Total Weighted Tardiness ($T$)")
    for bar, val in zip(bars3, tardiness_vals):
        ax3.text(val + max(tardiness_vals) * 0.015, bar.get_y() + bar.get_height() / 2, f"{val:,.0f}", va="center", fontsize=8)

    # 4. On-Time Delivery Rate & Utilization
    ax4 = axes[1, 1]
    height = 0.38
    b_ontime = ax4.barh(y_pos - height / 2, ontime_vals, height=height, color="#27ae60", alpha=0.85, label="On-Time Rate (%)")
    b_util = ax4.barh(y_pos + height / 2, util_vals, height=height, color="#2980b9", alpha=0.85, label="Tool Utilization (%)")
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(labels, fontsize=8.5)
    ax4.invert_yaxis()
    ax4.set_xlim(0, 100)
    ax4.set_xlabel("Percentage (%) (Higher is Better)")
    ax4.set_title("4. Operational Delivery & Tool Utilization Rates")
    ax4.legend(loc="lower right", framealpha=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = output_dir / f"kpi_benchmark_bars.{fmt}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return save_path


# ==============================================================================
# 6. MULTI-CRITERIA RADAR / SPIDER CHART
# ==============================================================================

def plot_radar_tradeoff(
    models: Dict[str, Dict[str, Any]],
    output_dir: Path,
    dpi: int = 300,
    fmt: str = "png",
    show: bool = False,
) -> Optional[Path]:
    """
    Plots a Spider / Radar chart comparing top-performing PPO models
    across 5 normalized operational dimensions:
    - Makespan Efficiency (inverted normalized C_max)
    - Tardiness Minimization (inverted normalized Tardiness)
    - Setup Time Efficiency (inverted normalized Setup)
    - On-Time Job Rate (%)
    - Tool Utilization Rate (%)
    """
    valid_models = [m for m in models.values() if m["num_episodes"] > 1]
    if len(valid_models) < 2:
        return None

    # Pick top 5 models by fitness
    valid_models.sort(key=lambda m: float(m["kpis"].get("fitness", float("inf"))))
    top_models = valid_models[:5]

    categories = [
        "Makespan\nEfficiency",
        "Tardiness\nMinimization",
        "Setup Time\nEfficiency",
        "On-Time\nRate",
        "Tool\nUtilization",
    ]
    num_cats = len(categories)

    # Compute normalization bounds across all valid models
    all_mk = [float(m["kpis"].get("makespan", 0.0)) for m in valid_models]
    all_tard = [float(m["kpis"].get("total_weighted_tardiness", 0.0)) for m in valid_models]
    all_setup = [float(m["kpis"].get("total_setup_time", 0.0)) for m in valid_models]
    all_ontime = [float(m["kpis"].get("on_time_rate_percent", 0.0)) for m in valid_models]
    all_util = [float(m["kpis"].get("avg_tool_utilization_percent", 0.0)) for m in valid_models]

    min_mk, max_mk = min(all_mk), max(all_mk)
    min_tard, max_tard = min(all_tard), max(all_tard)
    min_setup, max_setup = min(all_setup), max(all_setup)

    def normalize(val: float, min_v: float, max_v: float, invert: bool = False) -> float:
        if max_v == min_v:
            return 1.0
        norm = (val - min_v) / (max_v - min_v)
        return float(np.clip(1.0 - norm if invert else norm, 0.05, 1.0))

    angles = [n / float(num_cats) * 2 * np.pi for n in range(num_cats)]
    angles += angles[:1]  # Complete circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=dpi)
    fig.suptitle(
        "PPO Top-5 Models: Multi-Objective Trade-Off Spider Chart\n(Higher radius is better)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    for m in top_models:
        kpis = m["kpis"]
        mk_eff = normalize(float(kpis.get("makespan", 0.0)), min_mk, max_mk, invert=True)
        tard_eff = normalize(float(kpis.get("total_weighted_tardiness", 0.0)), min_tard, max_tard, invert=True)
        setup_eff = normalize(float(kpis.get("total_setup_time", 0.0)), min_setup, max_setup, invert=True)
        ontime_eff = float(kpis.get("on_time_rate_percent", 0.0)) / 100.0
        util_eff = float(kpis.get("avg_tool_utilization_percent", 0.0)) / 100.0

        values = [mk_eff, tard_eff, setup_eff, ontime_eff, util_eff]
        values += values[:1]

        reward_strat = m["reward_strategy"]
        state_strat = m["state_strategy"]
        color = REWARD_COLORS.get(reward_strat, "#1f77b4")
        linestyle = "-" if state_strat == "enhanced" else "--"
        label = f"{state_strat[:3].upper()} - {REWARD_DISPLAY_NAMES.get(reward_strat, reward_strat)}"

        ax.plot(angles, values, color=color, linestyle=linestyle, linewidth=2.0, label=label)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9.5, fontweight="bold")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], color="gray", fontsize=8)
    ax.set_ylim(0, 1.05)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.05),
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=8.5,
    )

    plt.tight_layout()

    save_path = output_dir / f"radar_tradeoff_comparison.{fmt}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return save_path


# ==============================================================================
# 7. SUMMARY REPORT & CLI ORCHESTRATION
# ==============================================================================

def print_summary_table(models: Dict[str, Dict[str, Any]]) -> None:
    """Prints a clean ASCII summary table of all loaded PPO models."""
    sorted_models = sorted(
        models.values(),
        key=lambda m: float(m["kpis"].get("fitness", float("inf")))
    )

    print("\n" + "=" * 105)
    print(f"{'PPO Model ID':<35} | {'State':<9} | {'Episodes':<8} | {'Makespan (m)':<12} | {'Tardiness (m)':<13} | {'On-Time':<8} | {'Fitness':<8}")
    print("-" * 105)

    for m in sorted_models:
        kpis = m["kpis"]
        algo_id = m["id"]
        state_strat = m["state_strategy"]
        episodes = m["num_episodes"]
        mk = float(kpis.get("makespan", 0.0))
        tard = float(kpis.get("total_weighted_tardiness", 0.0))
        ontime = float(kpis.get("on_time_rate_percent", 0.0))
        fit = float(kpis.get("fitness", 0.0))

        print(
            f"{algo_id:<35} | {state_strat:<9} | {episodes:<8} | "
            f"{mk:<12.1f} | {tard:<13.1f} | {ontime:<7.1f}% | {fit:<8.2f}"
        )

    print("=" * 105 + "\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive PPO Training History Visualization Suite for FJSCP simulation."
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="experiments",
        help="Path to the experiments directory containing PPO subfolders (default: 'experiments').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/plots/ppo",
        help="Directory to save generated plot images (default: 'experiments/plots/ppo').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Plot only a specific PPO model ID (e.g., 'ppo_enhanced_continuous_tardiness').",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=25,
        help="Rolling window size for smoothing noisy RL curves (default: 25).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution DPI for high-quality export (default: 300).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output image format: 'png', 'pdf', or 'svg' (default: 'png').",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively using plt.show().",
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Skip generating individual 6-panel dashboards, only generate comparative plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    exp_dir = Path(args.experiments_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("=== PPO TRAINING HISTORY VISUALIZATION SUITE ===")
    print("=" * 75)
    print(f"Scanning for PPO experiments in: {exp_dir}")
    print(f"Output plots directory:          {out_dir}")
    print(f"Smoothing window (MA):           {args.window}")
    print(f"Plot DPI:                        {args.dpi}")
    print(f"Output Format:                   {args.format.upper()}")

    models = load_ppo_experiments(exp_dir)
    if not models:
        print(f"\n❌ No valid PPO experiments found with schedule.json history in '{exp_dir}'.")
        return

    print(f"Found {len(models)} PPO experiment models.")
    print_summary_table(models)

    # 1. Single Model Mode
    if args.model:
        target_model = args.model
        if target_model not in models:
            # Try fuzzy match if prefixed
            candidates = [k for k in models if target_model in k]
            if candidates:
                target_model = candidates[0]
            else:
                print(f"❌ Model '{args.model}' not found. Available models: {list(models.keys())}")
                return

        print(f"\n---> Generating single dashboard for: {target_model}")
        saved = plot_single_ppo_dashboard(
            models[target_model],
            output_dir=out_dir,
            window=args.window,
            dpi=args.dpi,
            fmt=args.format,
            show=args.show,
        )
        print(f"✅ Dashboard saved: {saved}")
        return

    # 2. Individual 6-Panel Diagnostic Dashboards
    if not args.no_individual:
        print("\n---> Generating individual PPO model diagnostic dashboards...")
        count = 0
        for algo_id, m in models.items():
            if m["num_episodes"] <= 1:
                print(f"   [Skipping {algo_id}: only {m['num_episodes']} episode]")
                continue
            saved = plot_single_ppo_dashboard(
                m,
                output_dir=out_dir,
                window=args.window,
                dpi=args.dpi,
                fmt=args.format,
                show=args.show,
            )
            count += 1
            print(f"   [{count}/{len(models)}] Saved dashboard: {saved.name}")

    # 3. Reward Strategy Comparison (Enhanced & Baseline)
    print("\n---> Generating Reward Strategy comparison plots...")
    p1 = plot_reward_strategies_comparison(
        models, state_strategy="enhanced", output_dir=out_dir,
        window=args.window, dpi=args.dpi, fmt=args.format, show=args.show
    )
    if p1:
        print(f"✅ Saved Enhanced reward comparison: {p1.name}")

    p2 = plot_reward_strategies_comparison(
        models, state_strategy="baseline", output_dir=out_dir,
        window=args.window, dpi=args.dpi, fmt=args.format, show=args.show
    )
    if p2:
        print(f"✅ Saved Baseline reward comparison: {p2.name}")

    # 4. State Representation Ablation Study
    print("\n---> Generating State Representation ablation comparison...")
    p3 = plot_state_ablation_comparison(
        models, output_dir=out_dir, window=args.window,
        dpi=args.dpi, fmt=args.format, show=args.show
    )
    if p3:
        print(f"✅ Saved State ablation comparison: {p3.name}")

    # 5. Actor-Critic Loss Dynamics Comparison
    print("\n---> Generating Actor-Critic loss convergence comparison...")
    p4 = plot_loss_dynamics_comparison(
        models, output_dir=out_dir, window=args.window,
        dpi=args.dpi, fmt=args.format, show=args.show
    )
    if p4:
        print(f"✅ Saved Loss dynamics comparison: {p4.name}")

    # 6. Multi-Objective KPI Summary Bar Charts
    print("\n---> Generating KPI Benchmark summary bar charts...")
    p5 = plot_kpi_summary_bars(
        models, output_dir=out_dir, dpi=args.dpi, fmt=args.format, show=args.show
    )
    if p5:
        print(f"✅ Saved KPI benchmark bar charts: {p5.name}")

    # 7. Multi-Criteria Radar / Spider Chart
    print("\n---> Generating Multi-Criteria trade-off spider chart...")
    p6 = plot_radar_tradeoff(
        models, output_dir=out_dir, dpi=args.dpi, fmt=args.format, show=args.show
    )
    if p6:
        print(f"✅ Saved Radar trade-off chart: {p6.name}")

    print("\n" + "=" * 75)
    print(f"🎉 All PPO plots successfully generated and saved to: {out_dir}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
