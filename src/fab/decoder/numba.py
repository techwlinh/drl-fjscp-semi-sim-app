from typing import Dict, List, Tuple
import numpy as np
from numba import njit

from src.fab.objective import (
    calculate_numba_makespan,
    calculate_numba_weighted_tardiness,
)
from src.schema.data import DatasetOutputModel
from src.model.meta.ga.types import Chromosome



@njit(fastmath=True)
def numba_decode_fitness(
    os: np.ndarray,
    ms: np.ndarray,
    num_jobs: int,
    num_tools: int,
    job_due_dates: np.ndarray,
    job_priority_weights: np.ndarray,
    job_recipe_idx: np.ndarray,
    job_op_start_idx: np.ndarray,
    op_nominal_proc_time: np.ndarray,
    op_wsg_idx: np.ndarray,
    valid_tools_flat: np.ndarray,
    valid_tools_offsets: np.ndarray,
    valid_tools_counts: np.ndarray,
    tool_ws_idx: np.ndarray,
    tool_initial_recipe: np.ndarray,
    transport_mat: np.ndarray,
    setup_mat: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Numba JIT compiled FJSP decoder.
    Calculates makespan, total_weighted_tardiness, and total_setup_time at C-speed.
    """
    job_step_count = np.zeros(num_jobs, dtype=np.int32)
    job_current_time = np.zeros(num_jobs, dtype=np.float64)
    job_current_loc = np.zeros(num_jobs, dtype=np.int32)  # Index 0 is Central_Stockroom

    tool_avail_time = np.zeros(num_tools, dtype=np.float64)
    tool_curr_recipe = tool_initial_recipe.copy()

    total_setup_time = 0.0

    for i in range(len(os)):
        job_idx = os[i]
        step_seq_idx = job_step_count[job_idx]
        global_op_idx = job_op_start_idx[job_idx] + step_seq_idx

        # 0. Tool selection from MS
        offset = valid_tools_offsets[global_op_idx]
        num_valid = valid_tools_counts[global_op_idx]
        tool_choice_val = ms[global_op_idx]
        selected_tool_idx = valid_tools_flat[offset + (tool_choice_val % num_valid)]

        # 1. Transport Calculation
        curr_loc = job_current_loc[job_idx]
        target_ws_loc = tool_ws_idx[selected_tool_idx]

        travel_time = 0.0
        if curr_loc != target_ws_loc:
            travel_time = transport_mat[curr_loc, target_ws_loc]

        transport_start = job_current_time[job_idx]
        transport_end = transport_start + travel_time

        # 2. Tool Availability & Setup Calculation
        arrival_time = transport_end
        tool_ready = tool_avail_time[selected_tool_idx]
        tool_ready_time = arrival_time if arrival_time > tool_ready else tool_ready

        prev_recipe = tool_curr_recipe[selected_tool_idx]
        curr_recipe = job_recipe_idx[job_idx]
        wsg_i = op_wsg_idx[global_op_idx]

        setup_duration = 0.0
        if prev_recipe != curr_recipe and prev_recipe >= 0 and curr_recipe >= 0:
            setup_duration = setup_mat[wsg_i, prev_recipe, curr_recipe]

        setup_start = tool_ready_time
        setup_end = setup_start + setup_duration
        total_setup_time += setup_duration

        # 3. Processing Calculation
        proc_start = setup_end
        proc_end = proc_start + op_nominal_proc_time[global_op_idx]

        # 4. State Updates
        tool_avail_time[selected_tool_idx] = proc_end
        tool_curr_recipe[selected_tool_idx] = curr_recipe
        job_current_time[job_idx] = proc_end
        job_current_loc[job_idx] = target_ws_loc
        job_step_count[job_idx] += 1

    # Calculate Objectives using src.fab.objective functions
    makespan = calculate_numba_makespan(job_current_time)
    total_weighted_tardiness = calculate_numba_weighted_tardiness(
        job_current_time, job_due_dates, job_priority_weights
    )

    return makespan, total_weighted_tardiness, total_setup_time



class NumbaFJSPDecoder:
    """Pre-encodes FJSP dataset into flat NumPy arrays for Numba JIT fast decoding."""

    def __init__(self, dataset: DatasetOutputModel):
        self.dataset = dataset
        self.jobs_list = dataset.job_list
        self.num_jobs = len(self.jobs_list)

        # 1. Location indexing (Central_Stockroom -> 0, Workstations -> 1..)
        self.loc_to_idx: Dict[str, int] = {"Central_Stockroom": 0}
        for area in dataset.factory_infrastructure.areas:
            for wsg in area.workstation_groups:
                for ws in wsg.workstations:
                    if ws.ws_id not in self.loc_to_idx:
                        self.loc_to_idx[ws.ws_id] = len(self.loc_to_idx)
        num_locs = len(self.loc_to_idx)

        # 2. Recipe indexing
        self.recipe_to_idx: Dict[str, int] = {}
        for recipe_id in dataset.product_recipes:
            if recipe_id not in self.recipe_to_idx:
                self.recipe_to_idx[recipe_id] = len(self.recipe_to_idx)
        num_recipes = len(self.recipe_to_idx)

        # 3. WSG indexing
        self.wsg_to_idx: Dict[str, int] = {}
        for area in dataset.factory_infrastructure.areas:
            for wsg in area.workstation_groups:
                if wsg.wsg_id not in self.wsg_to_idx:
                    self.wsg_to_idx[wsg.wsg_id] = len(self.wsg_to_idx)
        num_wsgs = len(self.wsg_to_idx)

        # 4. Tool indexing & mapping
        self.tool_to_idx: Dict[str, int] = {}
        tool_ws_list: List[int] = []
        tool_initial_recipe_list: List[int] = []
        self.tool_info_meta: Dict[str, dict] = {}
        self.wsg_tools: Dict[str, List[str]] = {}

        for area in dataset.factory_infrastructure.areas:
            for wsg in area.workstation_groups:
                wsg_tools_list = []
                for ws in wsg.workstations:
                    for tool in ws.tools:
                        t_idx = len(self.tool_to_idx)
                        self.tool_to_idx[tool.tool_id] = t_idx
                        self.tool_info_meta[tool.tool_id] = {
                            "ws_id": ws.ws_id,
                            "wsg_id": wsg.wsg_id,
                            "area_id": area.area_id,
                            "dedication": ws.dedicated_product,
                            "initial_setup": tool.initial_setup_state,
                        }
                        tool_ws_list.append(self.loc_to_idx[ws.ws_id])
                        initial_rec = self.recipe_to_idx.get(tool.initial_setup_state, -1)
                        tool_initial_recipe_list.append(initial_rec)
                        wsg_tools_list.append(tool.tool_id)
                self.wsg_tools[wsg.wsg_id] = wsg_tools_list

        self.num_tools = len(self.tool_to_idx)
        self.tool_ws_idx = np.array(tool_ws_list, dtype=np.int32)
        self.tool_initial_recipe = np.array(tool_initial_recipe_list, dtype=np.int32)

        # 5. Transport Matrix Array (num_locs x num_locs)
        self.transport_mat = np.full((num_locs, num_locs), 10.0, dtype=np.float64)
        np.fill_diagonal(self.transport_mat, 0.0)
        for loc1, targets in dataset.transport_matrices.items():
            if loc1 in self.loc_to_idx:
                l1_i = self.loc_to_idx[loc1]
                for loc2, t_time in targets.items():
                    if loc2 in self.loc_to_idx:
                        l2_i = self.loc_to_idx[loc2]
                        self.transport_mat[l1_i, l2_i] = float(t_time)

        # 6. Setup Matrix Array (num_wsgs x num_recipes x num_recipes)
        self.setup_mat = np.zeros((max(1, num_wsgs), max(1, num_recipes), max(1, num_recipes)), dtype=np.float64)
        for wsg_id, matrix in dataset.setup_matrices.items():
            if wsg_id in self.wsg_to_idx:
                wsg_i = self.wsg_to_idx[wsg_id]
                for r1, r2_dict in matrix.items():
                    if r1 in self.recipe_to_idx:
                        r1_i = self.recipe_to_idx[r1]
                        for r2, st in r2_dict.items():
                            if r2 in self.recipe_to_idx:
                                r2_i = self.recipe_to_idx[r2]
                                self.setup_mat[wsg_i, r1_i, r2_i] = float(st)

        # 7. Job & Operations encoding
        job_due_dates_list: List[float] = []
        job_priority_weights_list: List[float] = []
        job_recipe_idx_list: List[int] = []
        job_op_start_idx_list: List[int] = []

        op_proc_time_list: List[float] = []
        op_wsg_idx_list: List[int] = []

        valid_tools_flat_list: List[int] = []
        valid_tools_offsets_list: List[int] = []
        valid_tools_counts_list: List[int] = []

        current_op_idx = 0
        for j_idx, job in enumerate(self.jobs_list):
            job_due_dates_list.append(float(job.due_date))
            job_priority_weights_list.append(float(job.priority_weight))
            job_recipe_idx_list.append(self.recipe_to_idx[job.product_type])
            job_op_start_idx_list.append(current_op_idx)

            route = dataset.product_recipes[job.product_type]
            for step in route.steps:
                op_proc_time_list.append(float(step.nominal_processing_time))
                op_wsg_idx_list.append(self.wsg_to_idx[step.target_wsg])

                # Get valid tools
                all_tools = self.wsg_tools.get(step.target_wsg, [])
                valid_str = [
                    t for t in all_tools
                    if self.tool_info_meta[t]["dedication"] in ("Universal", job.product_type)
                ]
                if not valid_str:
                    valid_str = all_tools

                valid_indices = [self.tool_to_idx[t] for t in valid_str]
                valid_tools_offsets_list.append(len(valid_tools_flat_list))
                valid_tools_counts_list.append(len(valid_indices))
                valid_tools_flat_list.extend(valid_indices)

                current_op_idx += 1

        self.job_due_dates = np.array(job_due_dates_list, dtype=np.float64)
        self.job_priority_weights = np.array(job_priority_weights_list, dtype=np.float64)
        self.job_recipe_idx = np.array(job_recipe_idx_list, dtype=np.int32)
        self.job_op_start_idx = np.array(job_op_start_idx_list, dtype=np.int32)

        self.op_nominal_proc_time = np.array(op_proc_time_list, dtype=np.float64)
        self.op_wsg_idx = np.array(op_wsg_idx_list, dtype=np.int32)

        self.valid_tools_flat = np.array(valid_tools_flat_list, dtype=np.int32)
        self.valid_tools_offsets = np.array(valid_tools_offsets_list, dtype=np.int32)
        self.valid_tools_counts = np.array(valid_tools_counts_list, dtype=np.int32)

        # Warmup Numba JIT compilation
        dummy_os = np.zeros(current_op_idx, dtype=np.int32)
        dummy_ms = np.zeros(current_op_idx, dtype=np.int32)
        numba_decode_fitness(
            dummy_os,
            dummy_ms,
            self.num_jobs,
            self.num_tools,
            self.job_due_dates,
            self.job_priority_weights,
            self.job_recipe_idx,
            self.job_op_start_idx,
            self.op_nominal_proc_time,
            self.op_wsg_idx,
            self.valid_tools_flat,
            self.valid_tools_offsets,
            self.valid_tools_counts,
            self.tool_ws_idx,
            self.tool_initial_recipe,
            self.transport_mat,
            self.setup_mat,
        )

    def decode_fitness(self, chromo: Chromosome) -> Tuple[float, float, float]:
        """Decode chromosome and return (makespan, tardiness, setup_time)."""
        os_arr = np.array(chromo.os, dtype=np.int32)
        ms_arr = np.array(chromo.ms, dtype=np.int32)

        makespan, tardiness, setup_time = numba_decode_fitness(
            os_arr,
            ms_arr,
            self.num_jobs,
            self.num_tools,
            self.job_due_dates,
            self.job_priority_weights,
            self.job_recipe_idx,
            self.job_op_start_idx,
            self.op_nominal_proc_time,
            self.op_wsg_idx,
            self.valid_tools_flat,
            self.valid_tools_offsets,
            self.valid_tools_counts,
            self.tool_ws_idx,
            self.tool_initial_recipe,
            self.transport_mat,
            self.setup_mat,
        )
        return round(makespan, 2), round(tardiness, 2), round(setup_time, 2)
