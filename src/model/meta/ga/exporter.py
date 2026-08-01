import json
from pathlib import Path
from typing import Dict, List

from src.schema.data import DatasetOutputModel
from src.model.meta.ga.types import Chromosome, ScheduledTask


def update_experiments_manifest(workspace_root: Path) -> None:
    """Scan experiments/ directory and generate manifest for web_viz selector."""
    exp_dir = workspace_root / "experiments"
    if not exp_dir.exists():
        return

    manifest = []
    for exp_file in sorted(exp_dir.glob("*.json"), reverse=True):
        if exp_file.name == "manifest.json":
            continue
        try:
            with open(exp_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            kpis = data.get("kpis", {})
            ga_kpi = data.get("heuristic_comparisons", {}).get("ga", {})
            manifest.append(
                {
                    "filename": exp_file.name,
                    "makespan": kpis.get("makespan", 0.0),
                    "tardiness": kpis.get("total_weighted_tardiness", 0.0),
                    "setup_time": kpis.get("total_setup_time", 0.0),
                    "fitness": ga_kpi.get("fitness", 0.0),
                    "scheduled_tasks": kpis.get("total_scheduled_tasks", 0),
                }
            )
        except Exception:
            continue

    manifest_payload = {"experiments": manifest}
    with open(exp_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    web_viz_public = workspace_root / "web_viz" / "public"
    if web_viz_public.exists():
        web_viz_exp = web_viz_public / "experiments"
        web_viz_exp.mkdir(parents=True, exist_ok=True)

        for exp_file in exp_dir.glob("*.json"):
            target_path = web_viz_exp / exp_file.name
            with open(exp_file, "r", encoding="utf-8") as src, open(target_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())

        with open(web_viz_public / "experiments_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)


def export_schedule_results(
    best_chromo: Chromosome,
    tasks: List[ScheduledTask],
    dataset: DatasetOutputModel,
    output_path: str,
    history: List[dict] = None,
    heuristic_comparisons: dict = None,
) -> None:
    """Export schedule KPI and task timeline details to JSON for web visualization."""
    total_tasks = len(tasks)
    total_tools = sum(
        len(ws.tools)
        for area in dataset.factory_infrastructure.areas
        for wsg in area.workstation_groups
        for ws in wsg.workstations
    )

    # Compute tool utilization
    tool_busy_time: Dict[str, float] = {}
    for task in tasks:
        proc_len = task.proc_end - task.proc_start
        setup_len = task.setup_end - task.setup_start
        tool_busy_time[task.tool_id] = (
            tool_busy_time.get(task.tool_id, 0.0) + proc_len + setup_len
        )

    avg_utilization = (
        sum(tool_busy_time.values()) / (total_tools * best_chromo.makespan)
        if (total_tools * best_chromo.makespan) > 0
        else 0.0
    )

    # Count tardy jobs
    job_tardiness_map: Dict[str, float] = {}
    for task in tasks:
        job_tardiness_map[task.job_id] = task.tardiness

    tardy_jobs = sum(1 for t in job_tardiness_map.values() if t > 0)
    on_time_rate = round((len(job_tardiness_map) - tardy_jobs) / len(job_tardiness_map) * 100, 1)

    ga_kpi = {
        "name": "GA Metaheuristic (Proposed)",
        "makespan": best_chromo.makespan,
        "total_weighted_tardiness": best_chromo.total_tardiness,
        "total_setup_time": best_chromo.total_setup_time,
        "fitness": best_chromo.fitness,
        "tardy_jobs": tardy_jobs,
        "on_time_rate_percent": on_time_rate,
        "avg_tool_utilization_percent": round(avg_utilization * 100, 1),
    }

    all_comparisons = {"ga": ga_kpi}
    if heuristic_comparisons:
        all_comparisons.update(heuristic_comparisons)

    export_payload = {
        "kpis": {
            "makespan": best_chromo.makespan,
            "total_weighted_tardiness": best_chromo.total_tardiness,
            "total_setup_time": best_chromo.total_setup_time,
            "total_jobs": len(dataset.job_list),
            "tardy_jobs": tardy_jobs,
            "on_time_rate_percent": on_time_rate,
            "avg_tool_utilization_percent": round(avg_utilization * 100, 1),
            "total_scheduled_tasks": total_tasks,
        },
        "history": history or [],
        "heuristic_comparisons": all_comparisons,
        "factory_hierarchy": [
            {
                "area_id": area.area_id,
                "workstation_groups": [
                    {
                        "wsg_id": wsg.wsg_id,
                        "workstations": [
                            {
                                "ws_id": ws.ws_id,
                                "tools": [t.tool_id for t in ws.tools],
                            }
                            for ws in wsg.workstations
                        ],
                    }
                    for wsg in area.workstation_groups
                ],
            }
            for area in dataset.factory_infrastructure.areas
        ],
        "tasks": [
            {
                "job_id": t.job_id,
                "product_type": t.product_type,
                "priority": t.priority,
                "priority_weight": t.priority_weight,
                "step_id": t.step_id,
                "wsg_id": t.wsg_id,
                "ws_id": t.ws_id,
                "tool_id": t.tool_id,
                "area_id": t.area_id,
                "from_location": t.from_location,
                "to_location": t.to_location,
                "transport_start": t.transport_start,
                "transport_end": t.transport_end,
                "setup_start": t.setup_start,
                "setup_end": t.setup_end,
                "proc_start": t.proc_start,
                "proc_end": t.proc_end,
                "due_date": t.due_date,
                "tardiness": t.tardiness,
            }
            for t in tasks
        ],
    }

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    print(f"Schedule optimization results exported to: {output_path}")

    # Locate workspace root by walking up to find web_viz or pyproject.toml
    workspace_root = out_file.parent
    while workspace_root.parent != workspace_root:
        if (workspace_root / "web_viz").exists() or (workspace_root / "pyproject.toml").exists():
            break
        workspace_root = workspace_root.parent

    # Sync default dataset for web viz
    web_viz_target = workspace_root / "web_viz" / "public" / "ga_schedule_results.json"
    if web_viz_target.parent.exists():
        with open(web_viz_target, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2)
        print(f"Web visualization dataset updated at: {web_viz_target}")

    # Update manifest for experiment selector dropdown in web_viz
    update_experiments_manifest(workspace_root)
