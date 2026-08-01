from typing import Dict, List, Literal
from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# JSON Output Schema Models
# ----------------------------------------------------------------------
class ToolModel(BaseModel):
    tool_id: str = Field(description="Unique tool ID, e.g., LITH_STEPPER_A_WS1_T1")
    initial_setup_state: str = Field(description="Recipe state configured on tool at t=0")


class WorkstationModel(BaseModel):
    ws_id: str = Field(description="Unique Workstation ID")
    dedicated_product: str = Field(description="Product dedication or 'Universal'")
    tools: List[ToolModel] = Field(description="List of physical tools in this workstation")


class WorkstationGroupModel(BaseModel):
    wsg_id: str = Field(description="Unique Workstation Group ID, e.g., LITH_STEPPER")
    area: str = Field(description="Parent Area ID")
    workstations: List[WorkstationModel] = Field(description="List of workstations under this WSG")


class AreaModel(BaseModel):
    area_id: str = Field(description="Area ID, e.g., LITH, ETCH")
    workstation_groups: List[WorkstationGroupModel] = Field(description="WSGs in this area")


class FactoryInfrastructureModel(BaseModel):
    areas: List[AreaModel] = Field(description="List of factory areas")


class RouteStepModel(BaseModel):
    step_id: int = Field(description="Step sequence number (1, 2, ...)")
    target_wsg: str = Field(description="Required Workstation Group for this step")
    nominal_processing_time: float = Field(description="Nominal processing time in minutes")


class ProductRouteModel(BaseModel):
    product_id: str = Field(description="Product identifier, e.g., Product_A")
    steps: List[RouteStepModel] = Field(description="Sequence of route steps")


class JobModel(BaseModel):
    job_id: str = Field(description="Unique Job/Lot identifier, e.g., JOB_0001")
    product_type: str = Field(description="Product type name, e.g., Product_A")
    priority: Literal["Super_Hot", "Hot", "Regular"] = Field(description="Lot priority level")
    priority_weight: float = Field(description="Weight used in total weighted tardiness objective")
    total_raw_processing_time: float = Field(description="Sum of nominal processing times (mins)")
    est_transport_time: float = Field(description="Estimated AMHS transport time overhead (mins)")
    flow_factor: float = Field(description="Assigned flow factor multiplier")
    due_date: float = Field(description="Calculated due date in minutes from t=0")
    current_location: str = Field(default="Central_Stockroom", description="Location at t=0")


class DatasetOutputModel(BaseModel):
    factory_infrastructure: FactoryInfrastructureModel
    transport_matrices: Dict[str, Dict[str, float]] = Field(
        description="Distance/time matrix between Stockroom, Areas, and WSs"
    )
    product_recipes: Dict[str, ProductRouteModel] = Field(
        description="Routing recipes per product type"
    )
    setup_matrices: Dict[str, Dict[str, Dict[str, float]]] = Field(
        description="Sequence-dependent setup matrices per WSG (WSG -> Recipe_from -> Recipe_to -> time)"
    )
    job_list: List[JobModel] = Field(description="List of initial jobs/lots at t=0")
