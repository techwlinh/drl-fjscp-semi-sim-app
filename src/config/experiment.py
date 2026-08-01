import string
from typing import Dict, List, Tuple
from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# Data Generator Configuration Model
# ----------------------------------------------------------------------
class DataGenConfig(BaseModel):
    seed: int = Field(default=42, description="Random seed for reproducibility")
    num_jobs: int = Field(default=100, description="Total number of jobs at t=0")
    num_products: int = Field(default=4, description="Number of products; names auto-generated")
    num_steps_per_product: Tuple[int, int] = Field(
        default=(50, 80), description="Range (min, max) of route steps per product"
    )
    areas: List[str] = Field(
        default=["LITH", "ETCH", "IMPL", "FILM", "METL", "CVD"],
        description="Fab areas in the infrastructure",
    )
    priority_ratios: Dict[str, float] = Field(
        default={"Super_Hot": 0.1, "Hot": 0.2, "Regular": 0.7},
        description="Probability distribution for lot priorities",
    )
    priority_weights: Dict[str, float] = Field(
        default={"Super_Hot": 10.0, "Hot": 3.0, "Regular": 1.0},
        description="Weighting factor for tardiness calculation",
    )
    flow_factors: Dict[str, Tuple[float, float]] = Field(
        default={
            "Super_Hot": (1.1, 1.4),
            "Hot": (1.5, 2.2),
            "Regular": (2.0, 3.5),
        },
        description="Flow Factor (cycle time / raw processing time) ranges per priority",
    )

    # AMHS travel time ranges in minutes
    amhs_stockroom_to_area: Tuple[float, float] = Field(default=(10.0, 20.0))
    amhs_inter_area: Tuple[float, float] = Field(default=(15.0, 30.0))
    amhs_inter_ws: Tuple[float, float] = Field(default=(5.0, 10.0))
    amhs_intra_ws: Tuple[float, float] = Field(default=(1.0, 3.0))

    # Processing time ranges per step in minutes
    step_processing_time_range: Tuple[float, float] = Field(default=(15.0, 60.0))

    # Setup time specs (min, max) in minutes for sequence-dependent setup sensitive WSGs
    setup_time_ranges: Dict[str, Tuple[float, float]] = Field(
        default={
            "LITH": (5.0, 20.0),
            "IMPL": (10.0, 15.0),
            "ETCH": (10.0, 15.0),
            "CVD": (8.0, 18.0),
        }
    )

    def get_product_names(self) -> List[str]:
        """Auto-generate product names like Product_A, Product_B, ... Product_Z, Product_AA, etc."""
        names = []
        for i in range(self.num_products):
            if i < 26:
                label = string.ascii_uppercase[i]
            else:
                label = f"{string.ascii_uppercase[i // 26 - 1]}{string.ascii_uppercase[i % 26]}"
            names.append(f"Product_{label}")
        return names
