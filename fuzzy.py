"""
Fuzzy Inference System (Mamdani FIS) for Adaptive Genetic Algorithm Parameters.

Controls:
    - Crossover Rate (pc)
    - Mutation Rate (pm)

Based on:
    - Input 1: Population Diversity (D in [0, 1]) -> {ZE, PS, PL}
    - Input 2: Fitness Convergence Trend (C in [-1, 1]) -> {NL, NS, ZE, PS, PL}

Rules:
    - Table 3.1: Fuzzy rule base for crossover rate (Delta pc)
    - Table 3.2: Fuzzy rule base for mutation rate (Delta pm)
"""

from __future__ import annotations

import os
import shutil
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# 1. MEMBERSHIP FUNCTIONS
# ==============================================================================

def trimf(x: np.ndarray, params: List[float] | Tuple[float, float, float]) -> np.ndarray:
    """Triangular membership function params = (a, b, c)."""
    a, b, c = params
    y = np.zeros_like(x, dtype=float)
    if a < b:
        idx1 = np.logical_and(a <= x, x < b)
        y[idx1] = (x[idx1] - a) / (b - a)
    if b < c:
        idx2 = np.logical_and(b <= x, x <= c)
        y[idx2] = (c - x[idx2]) / (c - b)
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


def trapmf(x: np.ndarray, params: List[float] | Tuple[float, float, float, float]) -> np.ndarray:
    """
    Trapezoidal membership function params = (a, b, c, d).
    - If a == b: Left shoulder (1 for x <= b, then linearly decays to 0 at d).
    - If c == d: Right shoulder (0 for x <= a, rises to 1 at b, 1 for x >= b).
    """
    a, b, c, d = params
    y = np.zeros_like(x, dtype=float)

    # Rising slope
    if a < b:
        idx = np.logical_and(a <= x, x < b)
        y[idx] = (x[idx] - a) / (b - a)
    elif a == b:
        y[x <= b] = 1.0

    # Flat top plateau
    idx = np.logical_and(b <= x, x <= c)
    y[idx] = 1.0

    # Falling slope
    if c < d:
        idx = np.logical_and(c < x, x <= d)
        y[idx] = (d - x[idx]) / (d - c)
    elif c == d:
        y[x >= c] = 1.0

    return np.clip(y, 0.0, 1.0)


class FuzzyVariable:
    """Represents a linguistic variable with its universe and fuzzy sets."""

    def __init__(self, name: str, universe: np.ndarray):
        self.name = name
        self.universe = universe
        self.mfs: Dict[str, np.ndarray] = {}
        self.mf_params: Dict[str, Tuple[str, list]] = {}

    def add_triangle(self, label: str, a: float, b: float, c: float):
        mf_vals = trimf(self.universe, (a, b, c))
        self.mfs[label] = mf_vals
        self.mf_params[label] = ("triangle", [a, b, c])

    def add_trapezoid(self, label: str, a: float, b: float, c: float, d: float):
        mf_vals = trapmf(self.universe, (a, b, c, d))
        self.mfs[label] = mf_vals
        self.mf_params[label] = ("trapezoid", [a, b, c, d])

    def evaluate_point(self, label: str, crisp_val: float) -> float:
        """Evaluate membership degree of a single crisp value."""
        shape, p = self.mf_params[label]
        if shape == "triangle":
            val = trimf(np.array([crisp_val], dtype=float), p)[0]
        else:
            val = trapmf(np.array([crisp_val], dtype=float), p)[0]
        return float(val)


# ==============================================================================
# 2. ADAPTIVE GA FUZZY CONTROLLER
# ==============================================================================

class AdaptiveGAFuzzyController:
    """
    Fuzzy Logic Controller implementing Table 3.1 and Table 3.2
    for adaptive genetic algorithm crossover and mutation rates.
    """

    def __init__(
        self,
        delta_pc_max: float = 0.10,
        delta_pm_max: float = 0.05,
        resolution: int = 500,
    ):
        self.delta_pc_max = delta_pc_max
        self.delta_pm_max = delta_pm_max
        self.resolution = resolution

        self._build_variables()
        self._build_rules()

    def _build_variables(self):
        # -------------------------------------------------------------
        # Input 1: Population Diversity (D in [0, 1]) -> {ZE, PS, PL}
        # -------------------------------------------------------------
        self.u_d = np.linspace(0.0, 1.0, self.resolution)
        self.var_d = FuzzyVariable("Population Diversity", self.u_d)
        # ZE: peak at 0.0, falls to 0 at 0.50
        self.var_d.add_trapezoid("ZE", 0.0, 0.0, 0.05, 0.50)
        # PS: peak at 0.50, triangle over [0.0, 1.0]
        self.var_d.add_triangle("PS", 0.0, 0.50, 1.0)
        # PL: rises from 0.50, plateau to 1.0
        self.var_d.add_trapezoid("PL", 0.50, 0.95, 1.0, 1.0)

        # -------------------------------------------------------------
        # Input 2: Convergence Trend (C in [-1, 1]) -> {NL, NS, ZE, PS, PL}
        # -------------------------------------------------------------
        self.u_c = np.linspace(-1.0, 1.0, self.resolution)
        self.var_c = FuzzyVariable("Convergence Trend", self.u_c)
        self.var_c.add_trapezoid("NL", -1.0, -1.0, -0.75, -0.40)
        self.var_c.add_triangle("NS", -0.70, -0.35, 0.0)
        self.var_c.add_triangle("ZE", -0.30, 0.0, 0.30)
        self.var_c.add_triangle("PS", 0.0, 0.35, 0.70)
        self.var_c.add_trapezoid("PL", 0.40, 0.75, 1.0, 1.0)

        # -------------------------------------------------------------
        # Output 1: Delta pc in [-delta_pc_max, delta_pc_max]
        # -------------------------------------------------------------
        dpc = self.delta_pc_max
        self.u_pc = np.linspace(-dpc, dpc, self.resolution)
        self.var_pc = FuzzyVariable("Delta Crossover Rate", self.u_pc)
        self.var_pc.add_trapezoid("NL", -dpc, -dpc, -0.75 * dpc, -0.40 * dpc)
        self.var_pc.add_triangle("NS", -0.70 * dpc, -0.35 * dpc, 0.0)
        self.var_pc.add_triangle("ZE", -0.30 * dpc, 0.0, 0.30 * dpc)
        self.var_pc.add_triangle("PS", 0.0, 0.35 * dpc, 0.70 * dpc)
        self.var_pc.add_trapezoid("PL", 0.40 * dpc, 0.75 * dpc, dpc, dpc)

        # -------------------------------------------------------------
        # Output 2: Delta pm in [-delta_pm_max, delta_pm_max]
        # -------------------------------------------------------------
        dpm = self.delta_pm_max
        self.u_pm = np.linspace(-dpm, dpm, self.resolution)
        self.var_pm = FuzzyVariable("Delta Mutation Rate", self.u_pm)
        self.var_pm.add_trapezoid("NL", -dpm, -dpm, -0.75 * dpm, -0.40 * dpm)
        self.var_pm.add_triangle("NS", -0.70 * dpm, -0.35 * dpm, 0.0)
        self.var_pm.add_triangle("ZE", -0.30 * dpm, 0.0, 0.30 * dpm)
        self.var_pm.add_triangle("PS", 0.0, 0.35 * dpm, 0.70 * dpm)
        self.var_pm.add_trapezoid("PL", 0.40 * dpm, 0.75 * dpm, dpm, dpm)

    def _build_rules(self):
        """
        Defines the 3x5 rule matrix according to:
        - Table 3.1 for Crossover Rate (Delta pc)
        - Table 3.2 for Mutation Rate (Delta pm)
        """
        # Table 3.1: Crossover Rate
        # Rows: Input1 in [PL, PS, ZE], Cols: Input2 in [PL, PS, ZE, NS, NL]
        self.rule_base_pc = {
            ("PL", "PL"): "PL",
            ("PL", "PS"): "PL",
            ("PL", "ZE"): "PS",
            ("PL", "NS"): "ZE",
            ("PL", "NL"): "ZE",

            ("PS", "PL"): "PS",
            ("PS", "PS"): "PS",
            ("PS", "ZE"): "NL",
            ("PS", "NS"): "ZE",
            ("PS", "NL"): "ZE",

            ("ZE", "PL"): "NL",
            ("ZE", "PS"): "NS",
            ("ZE", "ZE"): "NS",
            ("ZE", "NS"): "NL",
            ("ZE", "NL"): "NL",
        }

        # Table 3.2: Mutation Rate
        # Rows: Input1 in [PL, PS, ZE], Cols: Input2 in [PL, PS, ZE, NS, NL]
        self.rule_base_pm = {
            ("PL", "PL"): "NL",
            ("PL", "PS"): "NL",
            ("PL", "ZE"): "NS",
            ("PL", "NS"): "ZE",
            ("PL", "NL"): "ZE",

            ("PS", "PL"): "NS",
            ("PS", "PS"): "NS",
            ("PS", "ZE"): "ZE",
            ("PS", "NS"): "ZE",
            ("PS", "NL"): "ZE",

            ("ZE", "PL"): "NL",
            ("ZE", "PS"): "PS",
            ("ZE", "ZE"): "PS",
            ("ZE", "NS"): "PS",
            ("ZE", "NL"): "PL",
        }

    # ==========================================================================
    # STEP 1: FUZZIFICATION
    # ==========================================================================
    def fuzzify(self, input1: float, input2: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Computes membership degrees for crisp inputs."""
        c_in1 = np.clip(input1, 0.0, 1.0)
        c_in2 = np.clip(input2, -1.0, 1.0)

        mu_in1 = {lbl: self.var_d.evaluate_point(lbl, c_in1) for lbl in ["ZE", "PS", "PL"]}
        mu_in2 = {lbl: self.var_c.evaluate_point(lbl, c_in2) for lbl in ["NL", "NS", "ZE", "PS", "PL"]}
        return mu_in1, mu_in2

    # ==========================================================================
    # STEP 2: RULE EVALUATION (MAMDANI IMPLICATION)
    # ==========================================================================
    def evaluate_rules(
        self, mu_in1: Dict[str, float], mu_in2: Dict[str, float]
    ) -> List[Dict]:
        """
        Calculates firing strengths w = min(mu_in1, mu_in2)
        and truncates the consequent fuzzy sets (Mamdani min implication).
        """
        fired_rules = []
        rule_idx = 1

        for in1_lbl in ["PL", "PS", "ZE"]:
            for in2_lbl in ["PL", "PS", "ZE", "NS", "NL"]:
                w = min(mu_in1[in1_lbl], mu_in2[in2_lbl])
                pc_consequent = self.rule_base_pc[(in1_lbl, in2_lbl)]
                pm_consequent = self.rule_base_pm[(in1_lbl, in2_lbl)]

                # Implication: Clip output membership function by firing strength w
                pc_clipped = np.minimum(w, self.var_pc.mfs[pc_consequent])
                pm_clipped = np.minimum(w, self.var_pm.mfs[pm_consequent])

                fired_rules.append({
                    "id": rule_idx,
                    "in1_label": in1_lbl,
                    "in2_label": in2_lbl,
                    "weight": w,
                    "pc_label": pc_consequent,
                    "pm_label": pm_consequent,
                    "pc_clipped": pc_clipped,
                    "pm_clipped": pm_clipped,
                })
                rule_idx += 1

        return fired_rules

    # ==========================================================================
    # STEP 3: AGGREGATION
    # ==========================================================================
    def aggregate(self, fired_rules: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Combines all truncated consequents using fuzzy max operator."""
        agg_pc = np.zeros_like(self.u_pc)
        agg_pm = np.zeros_like(self.u_pm)

        for rule in fired_rules:
            agg_pc = np.maximum(agg_pc, rule["pc_clipped"])
            agg_pm = np.maximum(agg_pm, rule["pm_clipped"])

        return agg_pc, agg_pm

    # ==========================================================================
    # STEP 4: DEFUZZIFICATION (CENTROID METHOD)
    # ==========================================================================
    @staticmethod
    def defuzzify_centroid(universe: np.ndarray, aggregated_mf: np.ndarray) -> float:
        """Calculates Center of Gravity (Centroid) defuzzified crisp value."""
        area = np.sum(aggregated_mf)
        if area <= 1e-9:
            # Fallback to middle of universe if no rules fired
            return float((universe[0] + universe[-1]) / 2.0)
        centroid = np.sum(universe * aggregated_mf) / area
        return float(centroid)

    # ==========================================================================
    # STEP 5: FULL INFERENCE & GA PARAMETER UPDATE
    # ==========================================================================
    def compute(
        self,
        diversity: float,
        convergence: float,
        current_pc: float = 0.80,
        current_pm: float = 0.05,
        pc_range: Tuple[float, float] = (0.40, 0.95),
        pm_range: Tuple[float, float] = (0.01, 0.25),
    ) -> Dict:
        """
        Executes the complete fuzzy reasoning cycle and updates GA parameters.
        """
        # Step 1: Fuzzify
        mu_in1, mu_in2 = self.fuzzify(diversity, convergence)

        # Step 2: Evaluate rules & Mamdani implication
        rules = self.evaluate_rules(mu_in1, mu_in2)

        # Step 3: Aggregate
        agg_pc, agg_pm = self.aggregate(rules)

        # Step 4: Defuzzify
        delta_pc = self.defuzzify_centroid(self.u_pc, agg_pc)
        delta_pm = self.defuzzify_centroid(self.u_pm, agg_pm)

        # Step 5: Update GA parameters with clipping
        new_pc = float(np.clip(current_pc + delta_pc, pc_range[0], pc_range[1]))
        new_pm = float(np.clip(current_pm + delta_pm, pm_range[0], pm_range[1]))

        return {
            "inputs": {"diversity": diversity, "convergence": convergence},
            "mu_in1": mu_in1,
            "mu_in2": mu_in2,
            "rules": rules,
            "agg_pc": agg_pc,
            "agg_pm": agg_pm,
            "delta_pc": delta_pc,
            "delta_pm": delta_pm,
            "current_pc": current_pc,
            "current_pm": current_pm,
            "new_pc": new_pc,
            "new_pm": new_pm,
            "pc_range": pc_range,
            "pm_range": pm_range,
        }


# ==============================================================================
# 3. VISUALIZATION MODULE (STEP-BY-STEP)
# ==============================================================================

class FuzzyVisualizer:
    """Renders comprehensive, publication-quality visualizations for each step."""

    def __init__(self, controller: AdaptiveGAFuzzyController, output_dir: str = "./fuzzy_plots"):
        self.fc = controller
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Set styling
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
        plt.rcParams["axes.edgecolor"] = "#cccccc"
        plt.rcParams["axes.linewidth"] = 0.8

    def plot_step1_fuzzification(self, res: Dict) -> str:
        """Visualizes Step 1: Input Membership Functions with Crisp Value Intersections."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
        d_val = res["inputs"]["diversity"]
        c_val = res["inputs"]["convergence"]

        # 1. Population Diversity
        colors_d = {"ZE": "#e74c3c", "PS": "#f39c12", "PL": "#2ecc71"}
        for lbl, mf in self.fc.var_d.mfs.items():
            ax1.plot(self.fc.u_d, mf, label=f"{lbl}", color=colors_d[lbl], linewidth=2.2)
            mu_val = res["mu_in1"][lbl]
            if mu_val > 0.001:
                ax1.plot([d_val], [mu_val], "o", color=colors_d[lbl], markersize=8)
                ax1.annotate(
                    rf"$\mu_{{{lbl}}}={mu_val:.2f}$",
                    xy=(d_val, mu_val),
                    xytext=(d_val + 0.03, min(1.0, mu_val + 0.05)),
                    fontsize=14,
                    fontweight="bold",
                    color=colors_d[lbl],
                    arrowprops=dict(arrowstyle="->", color=colors_d[lbl], lw=1.2),
                )

        ax1.axvline(d_val, color="#2c3e50", linestyle="--", linewidth=1.8, label=f"Crisp Input D={d_val:.2f}")
        ax1.set_title("Input 1 (Population Diversity)", fontsize=14, fontweight="bold", pad=12)
        ax1.set_xlabel("Diversity (D)", fontsize=11)
        ax1.set_ylabel(r"Membership Degree ($\mu$)", fontsize=11)
        ax1.set_ylim(-0.05, 1.15)
        ax1.legend(loc="upper right", frameon=True)

        # 2. Convergence Trend
        colors_c = {"NL": "#c0392b", "NS": "#e67e22", "ZE": "#f1c40f", "PS": "#3498db", "PL": "#27ae60"}
        for lbl, mf in self.fc.var_c.mfs.items():
            ax2.plot(self.fc.u_c, mf, label=f"{lbl}", color=colors_c[lbl], linewidth=2.2)
            mu_val = res["mu_in2"][lbl]
            if mu_val > 0.001:
                ax2.plot([c_val], [mu_val], "o", color=colors_c[lbl], markersize=8)
                ax2.annotate(
                    rf"$\mu_{{{lbl}}}={mu_val:.2f}$",
                    xy=(c_val, mu_val),
                    xytext=(c_val + 0.04, min(1.0, mu_val + 0.05)),
                    fontsize=14,
                    fontweight="bold",
                    color=colors_c[lbl],
                    arrowprops=dict(arrowstyle="->", color=colors_c[lbl], lw=1.2),
                )

        ax2.axvline(c_val, color="#2c3e50", linestyle="--", linewidth=1.8, label=f"Crisp Input C={c_val:.2f}")
        ax2.set_title("Input 2 (Convergence Trend)", fontsize=14, fontweight="bold", pad=12)
        ax2.set_xlabel("Convergence Trend (C)", fontsize=11)
        ax2.set_ylabel(r"Membership Degree ($\mu$)", fontsize=11)
        ax2.set_ylim(-0.05, 1.15)
        ax2.legend(loc="upper right", frameon=True)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "step1_fuzzification.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_step2_rule_evaluation(self, res: Dict) -> str:
        """Visualizes Step 2: Fired Rules & Mamdani Min Implication."""
        fired = [r for r in res["rules"] if r["weight"] > 0.001]
        fired.sort(key=lambda x: x["weight"], reverse=True)

        fig, (ax_bar, ax_sub) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.4]})

        # Bar chart of firing strengths
        labels = [f"R{r['id']}: ({r['in1_label']}, {r['in2_label']})\n-> pc:{r['pc_label']}, pm:{r['pm_label']}" for r in fired]
        weights = [r["weight"] for r in fired]
        colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(fired)))

        bars = ax_bar.barh(labels, weights, color=colors, edgecolor="black", height=0.55)
        ax_bar.set_xlim(0, 1.05)
        ax_bar.set_xlabel(r"Firing Strength $w = \min(\mu_{In1}, \mu_{In2})$", fontsize=11)
        ax_bar.set_title(f"Step 2a: Active Fired Rules ({len(fired)} of 15 rules)", fontsize=12, fontweight="bold", pad=12)
        ax_bar.invert_yaxis()

        for bar in bars:
            w = bar.get_width()
            ax_bar.text(w + 0.02, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=10, fontweight="bold")

        # Consequent truncation preview for top rules
        top_rules = fired[:4]
        for r in top_rules:
            lbl = rf"R{r['id']} (w={r['weight']:.2f}) -> $\Delta p_c$:{r['pc_label']}"
            ax_sub.plot(self.fc.u_pc, r["pc_clipped"], label=lbl, linewidth=2.0)
            ax_sub.fill_between(self.fc.u_pc, 0, r["pc_clipped"], alpha=0.25)

        ax_sub.set_title(r"Step 2b: Mamdani Implication Truncation (Top $\Delta p_c$ Consequents)", fontsize=12, fontweight="bold", pad=12)
        ax_sub.set_xlabel(r"$\Delta p_c$", fontsize=11)
        ax_sub.set_ylabel(r"Clipped Membership $\mu_{cut}(z)$", fontsize=11)
        ax_sub.set_ylim(-0.05, 1.1)
        ax_sub.legend(loc="upper right", frameon=True)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "step2_rule_evaluation.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_step3_step4_defuzzification(self, res: Dict) -> str:
        """Visualizes Step 3 (Aggregation) & Step 4 (Centroid Defuzzification)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

        colors_out = {"NL": "#c0392b", "NS": "#e67e22", "ZE": "#f1c40f", "PS": "#3498db", "PL": "#27ae60"}

        # -------------------------------------------------------------
        # 1. Delta pc
        # -------------------------------------------------------------
        # Vẽ full toàn bộ các hàm thuộc gốc (background)
        for lbl, mf in self.fc.var_pc.mfs.items():
            ax1.plot(self.fc.u_pc, mf, label=lbl, color=colors_out[lbl], linestyle="--", linewidth=1.2, alpha=0.65)

        # Vẽ đường biên tập mờ hợp thành và CHỈ FILL phần diện tích giải mờ
        ax1.plot(self.fc.u_pc, res["agg_pc"], color="#1f618d", linewidth=2.5, label=r"$\mu_{agg}(z)$")
        ax1.fill_between(self.fc.u_pc, 0, res["agg_pc"], color="#2980b9", alpha=0.40, hatch="//", label="Defuzz Area")
        ax1.axvline(
            res["delta_pc"],
            color="#e74c3c",
            linestyle="-",
            linewidth=2.2,
            label=rf"Centroid $\Delta p_c^* = {res['delta_pc']:+.4f}$",
        )
        ax1.set_title(r"Defuzzification ($\Delta p_c$)", fontsize=12, fontweight="bold", pad=12)
        ax1.set_xlabel(r"$\Delta p_c$", fontsize=11)
        ax1.set_ylabel("Membership Degree", fontsize=11)
        ax1.set_ylim(-0.05, 1.15)
        ax1.legend(loc="upper right", fontsize=8, ncol=2, frameon=True, framealpha=0.9)

        # -------------------------------------------------------------
        # 2. Delta pm
        # -------------------------------------------------------------
        # Vẽ full toàn bộ các hàm thuộc gốc (background)
        for lbl, mf in self.fc.var_pm.mfs.items():
            ax2.plot(self.fc.u_pm, mf, label=lbl, color=colors_out[lbl], linestyle="--", linewidth=1.2, alpha=0.65)

        # Vẽ đường biên tập mờ hợp thành và CHỈ FILL phần diện tích giải mờ
        ax2.plot(self.fc.u_pm, res["agg_pm"], color="#6c3483", linewidth=2.5, label=r"$\mu_{agg}(z)$")
        ax2.fill_between(self.fc.u_pm, 0, res["agg_pm"], color="#8e44ad", alpha=0.40, hatch=r"\\", label="Defuzz Area")
        ax2.axvline(
            res["delta_pm"],
            color="#e74c3c",
            linestyle="-",
            linewidth=2.2,
            label=rf"Centroid $\Delta p_m^* = {res['delta_pm']:+.4f}$",
        )
        ax2.set_title(r"Defuzzification ($\Delta p_m$)", fontsize=12, fontweight="bold", pad=12)
        ax2.set_xlabel(r"$\Delta p_m$", fontsize=11)
        ax2.set_ylabel("Membership Degree", fontsize=11)
        ax2.set_ylim(-0.05, 1.15)
        ax2.legend(loc="upper right", fontsize=8, ncol=2, frameon=True, framealpha=0.9)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "step3_step4_defuzzification.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_step5_parameter_update(self, res: Dict) -> str:
        """Visualizes Step 5: GA Parameter Update Comparison."""
        fig, ax = plt.subplots(figsize=(8, 5))

        categories = ["Crossover Rate ($p_c$)", "Mutation Rate ($p_m$)"]
        old_vals = [res["current_pc"], res["current_pm"]]
        new_vals = [res["new_pc"], res["new_pm"]]
        deltas = [res["delta_pc"], res["delta_pm"]]

        x = np.arange(len(categories))
        width = 0.32

        b1 = ax.bar(x - width / 2, old_vals, width, label="Current $(t)$", color="#7f8c8d", edgecolor="black")
        b2 = ax.bar(x + width / 2, new_vals, width, label="Updated $(t+1)$", color="#27ae60", edgecolor="black")

        ax.set_ylabel("Parameter Value", fontsize=11)
        ax.set_title("Step 5: Genetic Algorithm Parameter Update Summary", fontsize=12, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper right", frameon=True)

        for i, (bar1, bar2, delta) in enumerate(zip(b1, b2, deltas)):
            h1 = bar1.get_height()
            h2 = bar2.get_height()
            ax.text(bar1.get_x() + bar1.get_width() / 2, h1 + 0.02, f"{h1:.3f}", ha="center", fontsize=10)
            sign = "+" if delta >= 0 else ""
            ax.text(bar2.get_x() + bar2.get_width() / 2, h2 + 0.02, f"{h2:.3f}\n({sign}{delta:.4f})", ha="center", fontsize=10, fontweight="bold", color="#196f3d")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "step5_parameter_update.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_3d_control_surfaces(self, grid_res: int = 25) -> str:
        """Computes and renders 3D Control Surfaces for Table 3.1 and 3.2 across all (D, C)."""
        d_vals = np.linspace(0.0, 1.0, grid_res)
        c_vals = np.linspace(-1.0, 1.0, grid_res)
        D_grid, C_grid = np.meshgrid(d_vals, c_vals)

        Z_pc = np.zeros_like(D_grid)
        Z_pm = np.zeros_like(D_grid)

        for i in range(grid_res):
            for j in range(grid_res):
                out = self.fc.compute(D_grid[i, j], C_grid[i, j])
                Z_pc[i, j] = out["delta_pc"]
                Z_pm[i, j] = out["delta_pm"]

        fig = plt.figure(figsize=(16, 6))

        # 3D Surface for Crossover (Table 3.1)
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        surf1 = ax1.plot_surface(D_grid, C_grid, Z_pc, cmap="coolwarm", edgecolor="none", alpha=0.9)
        ax1.set_title(r"3D Control Surface: $\Delta p_c$ (Table 3.1)", fontsize=12, fontweight="bold", pad=12)
        ax1.set_xlabel("Diversity (D)", fontsize=10, labelpad=8)
        ax1.set_ylabel("Convergence (C)", fontsize=10, labelpad=8)
        ax1.set_zlabel(r"$\Delta p_c$", fontsize=10, labelpad=8)
        fig.colorbar(surf1, ax=ax1, shrink=0.55, aspect=10, label=r"$\Delta p_c$")
        ax1.view_init(elev=28, azim=-125)

        # 3D Surface for Mutation (Table 3.2)
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        surf2 = ax2.plot_surface(D_grid, C_grid, Z_pm, cmap="plasma", edgecolor="none", alpha=0.9)
        ax2.set_title(r"3D Control Surface: $\Delta p_m$ (Table 3.2)", fontsize=12, fontweight="bold", pad=12)
        ax2.set_xlabel("Diversity (D)", fontsize=10, labelpad=8)
        ax2.set_ylabel("Convergence (C)", fontsize=10, labelpad=8)
        ax2.set_zlabel(r"$\Delta p_m$", fontsize=10, labelpad=8)
        fig.colorbar(surf2, ax=ax2, shrink=0.55, aspect=10, label=r"$\Delta p_m$")
        ax2.view_init(elev=28, azim=-125)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "step5_control_surfaces_3d.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path


# ==============================================================================
# 4. DEMO RUNNER
# ==============================================================================

def print_step_by_step_summary(res: Dict):
    """Outputs a rich text log of the mathematical reasoning at each step."""
    d = res["inputs"]["diversity"]
    c = res["inputs"]["convergence"]
    print("=" * 78)
    print(f"FUZZY INFERENCE SYSTEM - STEP-BY-STEP REASONING")
    print(f"Inputs: Diversity D = {d:.3f}, Convergence Trend C = {c:.3f}")
    print("=" * 78)

    print("\n[STEP 1: FUZZIFICATION]")
    print(f"  Input 1 (Diversity D = {d:.3f}):")
    for lbl, val in res["mu_in1"].items():
        print(f"    - mu_{lbl:<2}(D) = {val:.4f}")

    print(f"  Input 2 (Convergence Trend C = {c:.3f}):")
    for lbl, val in res["mu_in2"].items():
        print(f"    - mu_{lbl:<2}(C) = {val:.4f}")

    print("\n[STEP 2: RULE EVALUATION (MAMDANI MIN IMPLICATION)]")
    fired = [r for r in res["rules"] if r["weight"] > 0.0001]
    print(f"  Total rules active: {len(fired)} / 15 rules")
    for r in fired:
        print(
            f"  - Rule {r['id']:02d}: IF (D is {r['in1_label']}) AND (C is {r['in2_label']}) "
            f"-> Firing Weight w = {r['weight']:.4f} "
            f"-> pc is {r['pc_label']}, pm is {r['pm_label']}"
        )

    print("\n[STEP 3 & 4: AGGREGATION & DEFUZZIFICATION (CENTROID METHOD)]")
    print(f"  Delta Crossover Rate (Delta pc) : {res['delta_pc']:+.5f}")
    print(f"  Delta Mutation Rate  (Delta pm) : {res['delta_pm']:+.5f}")

    print("\n[STEP 5: GA PARAMETER UPDATE]")
    print(f"  Current: pc = {res['current_pc']:.4f}, pm = {res['current_pm']:.4f}")
    print(
        f"  Formula: pc(t+1) = clip(pc + Delta pc, {res['pc_range'][0]}, {res['pc_range'][1]}) = {res['new_pc']:.4f}"
    )
    print(
        f"  Formula: pm(t+1) = clip(pm + Delta pm, {res['pm_range'][0]}, {res['pm_range'][1]}) = {res['new_pm']:.4f}"
    )
    print("=" * 78 + "\n")


def run_full_demo():
    """Demonstrates fuzzy reasoning and exports step-by-step visualizations."""
    controller = AdaptiveGAFuzzyController(delta_pc_max=0.10, delta_pm_max=0.05)
    visualizer = FuzzyVisualizer(controller, output_dir="./fuzzy_plots")

    # Example Scenario: Low Diversity & Stagnant/Negative Convergence
    # D = 0.20 (mostly ZE and slight PS), C = -0.45 (mostly NS and slight NL)
    test_diversity = 0.20
    test_convergence = -0.45
    init_pc = 0.80
    init_pm = 0.05

    print("Running demo calculation...")
    result = controller.compute(
        diversity=test_diversity,
        convergence=test_convergence,
        current_pc=init_pc,
        current_pm=init_pm,
    )

    # Print mathematical explanation
    print_step_by_step_summary(result)

    # Generate step-by-step charts
    print("Generating visualizations...")
    p1 = visualizer.plot_step1_fuzzification(result)
    p2 = visualizer.plot_step2_rule_evaluation(result)
    p3 = visualizer.plot_step3_step4_defuzzification(result)
    p5a = visualizer.plot_step5_parameter_update(result)
    p5b = visualizer.plot_3d_control_surfaces(grid_res=30)

    # Copy plots to artifact directory for convenient embedding if artifact directory exists
    artifact_dir = "/Users/hlinhtru/.gemini/antigravity-ide/brain/34e1481d-fe88-4f4c-8676-418f3a7fffac"
    if os.path.exists(artifact_dir):
        for p in [p1, p2, p3, p5a, p5b]:
            shutil.copy(p, artifact_dir)

    print("All visualizations saved successfully:")
    for p in [p1, p2, p3, p5a, p5b]:
        print(f"  -> {p}")


if __name__ == "__main__":
    run_full_demo()
