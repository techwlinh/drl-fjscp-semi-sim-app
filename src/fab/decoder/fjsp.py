from typing import Dict, List, Tuple
from src.schema.data import DatasetOutputModel
from src.schema.types import Chromosome, ScheduledTask


class FJSPDecoder:
    def __init__(self, dataset: DatasetOutputModel):
        self.dataset = dataset
        self.jobs_list = dataset.job_list
        self.job_map = {j.job_id: j for j in dataset.job_list}
        self.recipes = dataset.product_recipes
        self.transport_matrices = dataset.transport_matrices
        self.setup_matrices = dataset.setup_matrices

        # Extract factory tools map & WS hierarchy
        self.tool_info: Dict[str, dict] = {}  # tool_id -> {ws_id, area_id, wsg_id, dedicated}
        self.wsg_tools: Dict[str, List[str]] = {}  # wsg_id -> list of tool_ids

        for area in dataset.factory_infrastructure.areas:
            for wsg in area.workstation_groups:
                wsg_tools_list = []
                for ws in wsg.workstations:
                    for tool in ws.tools:
                        self.tool_info[tool.tool_id] = {
                            "ws_id": ws.ws_id,
                            "wsg_id": wsg.wsg_id,
                            "area_id": area.area_id,
                            "dedication": ws.dedicated_product,
                            "initial_setup": tool.initial_setup_state,
                        }
                        wsg_tools_list.append(tool.tool_id)
                self.wsg_tools[wsg.wsg_id] = wsg_tools_list

        # Flat operation index mapping: (job_index, step_index) -> global_op_idx
        self.op_info: List[dict] = []
        self.job_op_indices: Dict[int, List[int]] = {}

        for j_idx, job in enumerate(self.jobs_list):
            self.job_op_indices[j_idx] = []
            route = self.recipes[job.product_type]
            for step in route.steps:
                op_idx = len(self.op_info)
                valid_tools = self.get_valid_tools_for_step(step.target_wsg, job.product_type)
                self.op_info.append(
                    {
                        "op_idx": op_idx,
                        "job_idx": j_idx,
                        "job_id": job.job_id,
                        "product_type": job.product_type,
                        "priority": job.priority,
                        "priority_weight": job.priority_weight,
                        "step_id": step.step_id,
                        "wsg_id": step.target_wsg,
                        "nominal_proc_time": step.nominal_processing_time,
                        "due_date": job.due_date,
                        "valid_tools": valid_tools,
                    }
                )
                self.job_op_indices[j_idx].append(op_idx)

    def get_valid_tools_for_step(self, wsg_id: str, product_type: str) -> List[str]:
        """Return list of valid tools under target WSG (matching dedication if specified)."""
        all_tools = self.wsg_tools.get(wsg_id, [])
        valid = [
            t
            for t in all_tools
            if self.tool_info[t]["dedication"] in ("Universal", product_type)
        ]
        # Fallback to all tools if product dedication is restrictive
        return valid if valid else all_tools

    def get_travel_time(self, loc1: str, loc2: str) -> float:
        """Query transport matrix between 2 locations."""
        if loc1 == loc2:
            return 0.0
        return self.transport_matrices.get(loc1, {}).get(loc2, 10.0)

    def get_setup_time(self, wsg_id: str, prev_recipe: str, next_recipe: str) -> float:
        """Query sequence-dependent setup time."""
        if prev_recipe == next_recipe or not prev_recipe:
            return 0.0
        wsg_matrix = self.setup_matrices.get(wsg_id, {})
        return wsg_matrix.get(prev_recipe, {}).get(next_recipe, 0.0)

    def decode(
        self, chromo: Chromosome, use_greedy_ms: bool = True
    ) -> Tuple[List[ScheduledTask], float, float, float, float]:
        """
        Decode Chromosome (OS + optional MS) into complete schedule using Active Insertion Decoding.
        If chromo.ms is empty or use_greedy_ms is True, uses Earliest-Finish-Time (EFT) Machine Selection.
        Returns: (scheduled_tasks, makespan, total_weighted_tardiness, total_setup_time, fitness)
        """
        job_op_step_count: Dict[int, int] = {j_idx: 0 for j_idx in range(len(self.jobs_list))}
        job_current_time: Dict[int, float] = {j_idx: 0.0 for j_idx in range(len(self.jobs_list))}
        job_current_location: Dict[int, str] = {
            j_idx: "Central_Stockroom" for j_idx in range(len(self.jobs_list))
        }

        # Track active intervals per tool: list of [start_time, end_time, recipe_state]
        tool_intervals: Dict[str, List[List]] = {
            t_id: [[0.0, 0.0, info["initial_setup"]]]
            for t_id, info in self.tool_info.items()
        }

        scheduled_tasks: List[ScheduledTask] = []
        total_setup_time = 0.0
        job_completion_times: Dict[int, float] = {}

        has_explicit_ms = bool(chromo.ms and len(chromo.ms) == len(self.op_info) and not use_greedy_ms)

        # Iterate over OS sequence
        for op_count_idx, job_idx in enumerate(chromo.os):
            step_seq_idx = job_op_step_count[job_idx]
            global_op_idx = self.job_op_indices[job_idx][step_seq_idx]
            op_data = self.op_info[global_op_idx]

            valid_tools = op_data["valid_tools"]

            if has_explicit_ms:
                tool_selection_val = chromo.ms[global_op_idx]
                selected_tool_id = valid_tools[tool_selection_val % len(valid_tools)]
                eval_tools = [selected_tool_id]
            else:
                eval_tools = valid_tools

            best_tool_id = eval_tools[0]
            best_finish_time = float("inf")
            best_setup_time = 0.0
            best_insert_idx = -1
            best_transport_start = 0.0
            best_transport_end = 0.0
            best_setup_start = 0.0
            best_setup_end = 0.0
            best_proc_start = 0.0

            curr_recipe = op_data["product_type"]
            proc_duration = op_data["nominal_proc_time"]
            curr_job_time = job_current_time[job_idx]
            curr_loc = job_current_location[job_idx]

            # Evaluate tools & idle gaps for earliest completion time (Active Insertion)
            for t_id in eval_tools:
                tool_meta = self.tool_info[t_id]
                target_ws_id = tool_meta["ws_id"]
                travel_time = self.get_travel_time(curr_loc, target_ws_id)
                arrival_time = curr_job_time + travel_time

                intervals = tool_intervals[t_id]
                num_intervals = len(intervals)

                for idx in range(num_intervals):
                    prev_end = intervals[idx][1]
                    prev_rec = intervals[idx][2]

                    setup_dur = self.get_setup_time(op_data["wsg_id"], prev_rec, curr_recipe)
                    s_start = max(arrival_time, prev_end)
                    s_end = s_start + setup_dur
                    p_start = s_end
                    p_end = p_start + proc_duration

                    # Check if fits in gap before next interval
                    if idx + 1 < num_intervals:
                        next_start = intervals[idx + 1][0]
                        next_rec = intervals[idx + 1][2]
                        next_setup_dur = self.get_setup_time(op_data["wsg_id"], curr_recipe, next_rec)
                        if p_end + next_setup_dur > next_start:
                            continue  # Does not fit in gap

                    if p_end < best_finish_time:
                        best_finish_time = p_end
                        best_tool_id = t_id
                        best_setup_time = setup_dur
                        best_insert_idx = idx + 1
                        best_transport_start = curr_job_time
                        best_transport_end = arrival_time
                        best_setup_start = s_start
                        best_setup_end = s_end
                        best_proc_start = p_start

            # Apply state updates & insert interval
            tool_meta = self.tool_info[best_tool_id]
            target_ws_id = tool_meta["ws_id"]

            intervals = tool_intervals[best_tool_id]
            intervals.insert(best_insert_idx, [best_setup_start, best_finish_time, curr_recipe])

            job_current_time[job_idx] = max(job_current_time[job_idx], best_finish_time)
            job_current_location[job_idx] = target_ws_id
            job_op_step_count[job_idx] += 1
            job_completion_times[job_idx] = max(
                job_completion_times.get(job_idx, 0.0), best_finish_time
            )
            total_setup_time += best_setup_time

            # Tardiness at step (final step)
            is_final_step = step_seq_idx == (len(self.job_op_indices[job_idx]) - 1)
            tardiness = max(0.0, best_finish_time - op_data["due_date"]) if is_final_step else 0.0

            scheduled_tasks.append(
                ScheduledTask(
                    job_id=op_data["job_id"],
                    product_type=op_data["product_type"],
                    priority=op_data["priority"],
                    priority_weight=op_data["priority_weight"],
                    step_id=op_data["step_id"],
                    wsg_id=op_data["wsg_id"],
                    ws_id=target_ws_id,
                    tool_id=best_tool_id,
                    area_id=tool_meta["area_id"],
                    from_location=curr_loc,
                    to_location=target_ws_id,
                    transport_start=round(best_transport_start, 2),
                    transport_end=round(best_transport_end, 2),
                    setup_start=round(best_setup_start, 2),
                    setup_end=round(best_setup_end, 2),
                    proc_start=round(best_proc_start, 2),
                    proc_end=round(best_finish_time, 2),
                    due_date=op_data["due_date"],
                    tardiness=round(tardiness, 2),
                )
            )

        makespan = max(job_completion_times.values()) if job_completion_times else 0.0

        # Calculate Total Weighted Tardiness
        total_weighted_tardiness = 0.0
        for j_idx, job in enumerate(self.jobs_list):
            c_i = job_completion_times.get(j_idx, 0.0)
            tardiness_i = max(0.0, c_i - job.due_date)
            total_weighted_tardiness += job.priority_weight * tardiness_i

        return (
            scheduled_tasks,
            round(makespan, 2),
            round(total_weighted_tardiness, 2),
            round(total_setup_time, 2),
            0.0,  # Fitness calculated externally with weights
        )
