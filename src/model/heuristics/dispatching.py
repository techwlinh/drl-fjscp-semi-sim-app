from typing import Dict, List, Tuple

from src.config.ga import GAConfig
from src.fab.decoder.fjsp import FJSPDecoder
from src.fab.objective import compute_weighted_fitness
from src.schema.data import DatasetOutputModel
from src.model.meta.ga.types import Chromosome, ScheduledTask
from src.config.experiment import ObjectiveConfig


class HeuristicScheduler:
    """Evaluates 4 classic dispatching rules for FJSP dataset comparison."""

    def __init__(self, dataset: DatasetOutputModel, config: GAConfig = None):
        self.dataset = dataset
        self.config = config or GAConfig()
        self.obj_config = ObjectiveConfig()
        self.decoder = FJSPDecoder(dataset)
        self.num_jobs = len(dataset.job_list)
        self.op_info = self.decoder.op_info

    def _build_chromosome(self, sorted_job_indices: List[int]) -> Chromosome:
        """Construct OS sequence from sorted job indices and greedy MS."""
        # OS sequence: repeat each job_idx as many times as it has steps
        os: List[int] = []
        # Count remaining steps needed per job
        job_step_counts = {
            j_idx: len(self.dataset.product_recipes[self.dataset.job_list[j_idx].product_type].steps)
            for j_idx in range(self.num_jobs)
        }

        # Interleave operations based on sorted job priority order
        max_steps = max(job_step_counts.values()) if job_step_counts else 0
        for step in range(max_steps):
            for j_idx in sorted_job_indices:
                if step < job_step_counts[j_idx]:
                    os.append(j_idx)

        # MS: Greedy assignment (select tool index 0)
        ms = [0] * len(self.op_info)

        return Chromosome(os=os, ms=ms)

    def run_fifo(self) -> Tuple[Chromosome, List[ScheduledTask], dict]:
        """First-In First-Out (FIFO): Jobs in original order of arrival/creation."""
        sorted_jobs = list(range(self.num_jobs))
        return self._evaluate_rule("FIFO (First In First Out)", sorted_jobs)

    def run_edd(self) -> Tuple[Chromosome, List[ScheduledTask], dict]:
        """Earliest Due Date (EDD): Sort jobs ascending by due_date."""
        jobs_with_idx = [(j_idx, job.due_date) for j_idx, job in enumerate(self.dataset.job_list)]
        jobs_with_idx.sort(key=lambda x: x[1])
        sorted_jobs = [j_idx for j_idx, _ in jobs_with_idx]
        return self._evaluate_rule("EDD (Earliest Due Date)", sorted_jobs)

    def run_spt(self) -> Tuple[Chromosome, List[ScheduledTask], dict]:
        """Shortest Processing Time (SPT): Sort jobs ascending by raw processing time."""
        jobs_with_idx = [
            (j_idx, job.total_raw_processing_time)
            for j_idx, job in enumerate(self.dataset.job_list)
        ]
        jobs_with_idx.sort(key=lambda x: x[1])
        sorted_jobs = [j_idx for j_idx, _ in jobs_with_idx]
        return self._evaluate_rule("SPT (Shortest Processing Time)", sorted_jobs)

    def run_cr_priority(self) -> Tuple[Chromosome, List[ScheduledTask], dict]:
        """Critical Ratio & Priority (CR/Priority): Sort jobs descending by priority_weight / due_date."""
        jobs_with_idx = [
            (j_idx, job.priority_weight / max(1.0, job.due_date))
            for j_idx, job in enumerate(self.dataset.job_list)
        ]
        jobs_with_idx.sort(key=lambda x: x[1], reverse=True)
        sorted_jobs = [j_idx for j_idx, _ in jobs_with_idx]
        return self._evaluate_rule("CR / Priority Weight", sorted_jobs)

    def _evaluate_rule(self, name: str, sorted_jobs: List[int]) -> Tuple[Chromosome, List[ScheduledTask], dict]:
        chromo = self._build_chromosome(sorted_jobs)
        tasks, makespan, tardiness, setup_time, _ = self.decoder.decode(chromo)

        chromo.makespan = makespan
        chromo.total_tardiness = tardiness
        chromo.total_setup_time = setup_time
        chromo.fitness = compute_weighted_fitness(
            makespan,
            tardiness,
            setup_time,
            weight_makespan=self.obj_config.weight_makespan,
            weight_tardiness=self.obj_config.weight_tardiness,
            weight_setup=self.obj_config.weight_setup,
            num_jobs=self.num_jobs,
        )


        # Compute utilization & tardy count
        total_tools = sum(
            len(ws.tools)
            for area in self.dataset.factory_infrastructure.areas
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

        avg_utilization = (
            sum(tool_busy.values()) / (total_tools * makespan)
            if (total_tools * makespan) > 0
            else 0.0
        )
        tardy_jobs = sum(1 for t in tardy_map.values() if t > 0)
        on_time_rate = round((len(tardy_map) - tardy_jobs) / len(tardy_map) * 100, 1)

        metrics = {
            "name": name,
            "makespan": makespan,
            "total_weighted_tardiness": tardiness,
            "total_setup_time": setup_time,
            "fitness": chromo.fitness,
            "tardy_jobs": tardy_jobs,
            "on_time_rate_percent": on_time_rate,
            "avg_tool_utilization_percent": round(avg_utilization * 100, 1),
        }

        return chromo, tasks, metrics

    def run_all(self) -> Dict[str, dict]:
        """Run all 4 heuristic dispatching rules and return comparison dict."""
        _, _, fifo_kpi = self.run_fifo()
        _, _, edd_kpi = self.run_edd()
        _, _, spt_kpi = self.run_spt()
        _, _, cr_kpi = self.run_cr_priority()

        return {
            "fifo": fifo_kpi,
            "edd": edd_kpi,
            "spt": spt_kpi,
            "cr_priority": cr_kpi,
        }
