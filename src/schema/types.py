from dataclasses import dataclass, field
from typing import List


@dataclass
class Chromosome:
    os: List[int]  # Job indices sequence (each job index appears len(steps) times)
    ms: List[int] = field(default_factory=list)  # Selected tool index for each operation (optional if using greedy MS)
    fitness: float = float("inf")
    makespan: float = 0.0
    total_tardiness: float = 0.0
    total_setup_time: float = 0.0


@dataclass
class ScheduledTask:
    job_id: str
    product_type: str
    priority: str
    priority_weight: float
    step_id: int
    wsg_id: str
    ws_id: str
    tool_id: str
    area_id: str

    # Timing breakdown (in minutes from t=0)
    from_location: str
    to_location: str
    transport_start: float
    transport_end: float
    setup_start: float
    setup_end: float
    proc_start: float
    proc_end: float
    due_date: float
    tardiness: float
