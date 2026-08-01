import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DataGenConfig
from src.schema import (
    AreaModel,
    DatasetOutputModel,
    FactoryInfrastructureModel,
    JobModel,
    ProductRouteModel,
    RouteStepModel,
    ToolModel,
    WorkstationGroupModel,
    WorkstationModel,
)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)


def generate_factory_infrastructure(
    config: DataGenConfig,
) -> FactoryInfrastructureModel:
    """Generate 4-level hierarchy: Area -> WSG -> WS -> Tool."""
    areas_list: List[AreaModel] = []

    # Map area to WSG subtypes
    wsg_templates: Dict[str, List[str]] = {
        "LITH": ["LITH_STEPPER", "LITH_SCANNER", "LITH_TRACK"],
        "ETCH": ["ETCH_DRY", "ETCH_WET"],
        "IMPL": ["IMPL_HIGH_ENERGY", "IMPL_MEDIUM_ENERGY"],
        "FILM": ["FILM_PVD", "FILM_ALD"],
        "METL": ["METL_PLATING", "METL_CMP"],
        "CVD": ["CVD_PECVD", "CVD_LPCVD"],
    }

    products = config.get_product_names()

    for area_id in config.areas:
        wsg_names = wsg_templates.get(area_id, [f"{area_id}_DEFAULT_WSG"])
        wsg_models: List[WorkstationGroupModel] = []

        for wsg_id in wsg_names:
            ws_models: List[WorkstationModel] = []

            # Create 1-2 Workstations under each WSG
            num_ws = 2 if len(products) >= 2 else 1
            for ws_idx in range(1, num_ws + 1):
                ws_id = f"{wsg_id}_WS{ws_idx}"
                dedication = (
                    products[ws_idx - 1] if ws_idx <= len(products) else "Universal"
                )

                # Create 2-3 Tools per Workstation
                num_tools = random.randint(2, 3)
                tools: List[ToolModel] = []
                for tool_idx in range(1, num_tools + 1):
                    tool_id = f"{ws_id}_T{tool_idx}"
                    # Assign a random initial setup recipe state at t=0
                    initial_recipe = random.choice(products)
                    tools.append(
                        ToolModel(tool_id=tool_id, initial_setup_state=initial_recipe)
                    )

                ws_models.append(
                    WorkstationModel(
                        ws_id=ws_id,
                        dedicated_product=dedication,
                        tools=tools,
                    )
                )

            wsg_models.append(
                WorkstationGroupModel(
                    wsg_id=wsg_id,
                    area=area_id,
                    workstations=ws_models,
                )
            )

        areas_list.append(AreaModel(area_id=area_id, workstation_groups=wsg_models))

    return FactoryInfrastructureModel(areas=areas_list)


def generate_transport_matrix(
    factory: FactoryInfrastructureModel, config: DataGenConfig
) -> Dict[str, Dict[str, float]]:
    """
    Generate travel time matrix between locations (Central Stockroom, Areas, Workstations).
    Values in minutes.
    """
    # Collect all location keys
    locations: List[str] = ["Central_Stockroom"]
    area_map: Dict[str, str] = {}  # ws_id -> area_id

    for area in factory.areas:
        locations.append(area.area_id)
        for wsg in area.workstation_groups:
            for ws in wsg.workstations:
                locations.append(ws.ws_id)
                area_map[ws.ws_id] = area.area_id

    matrix: Dict[str, Dict[str, float]] = {loc1: {} for loc1 in locations}

    # Pre-generate area-to-area travel times
    area_ids = [area.area_id for area in factory.areas]
    area_travel: Dict[Tuple[str, str], float] = {}
    for a1 in area_ids:
        for a2 in area_ids:
            if a1 == a2:
                area_travel[(a1, a2)] = 0.0
            else:
                if (a2, a1) in area_travel:
                    area_travel[(a1, a2)] = area_travel[(a2, a1)]
                else:
                    area_travel[(a1, a2)] = round(
                        random.uniform(*config.amhs_inter_area), 2
                    )

    # Pre-generate Stockroom-to-Area travel times
    stockroom_to_area: Dict[str, float] = {}
    for a in area_ids:
        stockroom_to_area[a] = round(
            random.uniform(*config.amhs_stockroom_to_area), 2
        )

    for loc1 in locations:
        for loc2 in locations:
            if loc1 == loc2:
                matrix[loc1][loc2] = 0.0
                continue

            # Stockroom <-> Area
            if loc1 == "Central_Stockroom" and loc2 in area_ids:
                matrix[loc1][loc2] = stockroom_to_area[loc2]
            elif loc2 == "Central_Stockroom" and loc1 in area_ids:
                matrix[loc1][loc2] = stockroom_to_area[loc1]

            # Stockroom <-> WS
            elif loc1 == "Central_Stockroom" and loc2 in area_map:
                parent_area = area_map[loc2]
                matrix[loc1][loc2] = round(
                    stockroom_to_area[parent_area]
                    + random.uniform(*config.amhs_inter_ws),
                    2,
                )
            elif loc2 == "Central_Stockroom" and loc1 in area_map:
                parent_area = area_map[loc1]
                matrix[loc1][loc2] = round(
                    stockroom_to_area[parent_area]
                    + random.uniform(*config.amhs_inter_ws),
                    2,
                )

            # Area <-> Area
            elif loc1 in area_ids and loc2 in area_ids:
                matrix[loc1][loc2] = area_travel[(loc1, loc2)]

            # Area <-> WS
            elif loc1 in area_ids and loc2 in area_map:
                target_area = area_map[loc2]
                if loc1 == target_area:
                    matrix[loc1][loc2] = round(
                        random.uniform(*config.amhs_inter_ws), 2
                    )
                else:
                    matrix[loc1][loc2] = round(
                        area_travel[(loc1, target_area)]
                        + random.uniform(*config.amhs_inter_ws),
                        2,
                    )
            elif loc2 in area_ids and loc1 in area_map:
                target_area = area_map[loc1]
                if loc2 == target_area:
                    matrix[loc1][loc2] = round(
                        random.uniform(*config.amhs_inter_ws), 2
                    )
                else:
                    matrix[loc1][loc2] = round(
                        area_travel[(target_area, loc2)]
                        + random.uniform(*config.amhs_inter_ws),
                        2,
                    )

            # WS <-> WS
            elif loc1 in area_map and loc2 in area_map:
                a1 = area_map[loc1]
                a2 = area_map[loc2]
                if a1 == a2:
                    # Inter-WS within same area
                    matrix[loc1][loc2] = round(
                        random.uniform(*config.amhs_inter_ws), 2
                    )
                else:
                    # Inter-WS across different areas
                    matrix[loc1][loc2] = round(
                        area_travel[(a1, a2)]
                        + random.uniform(*config.amhs_inter_ws),
                        2,
                    )

    return matrix


def generate_product_routes(
    factory: FactoryInfrastructureModel, config: DataGenConfig
) -> Tuple[Dict[str, ProductRouteModel], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Generate product routing recipes and WSG sequence-dependent setup matrices.
    """
    products = config.get_product_names()
    all_wsgs: List[str] = []
    wsg_area_map: Dict[str, str] = {}

    for area in factory.areas:
        for wsg in area.workstation_groups:
            all_wsgs.append(wsg.wsg_id)
            wsg_area_map[wsg.wsg_id] = area.area_id

    product_recipes: Dict[str, ProductRouteModel] = {}

    for p_name in products:
        num_steps = random.randint(*config.num_steps_per_product)
        steps: List[RouteStepModel] = []
        for step_idx in range(1, num_steps + 1):
            target_wsg = random.choice(all_wsgs)
            proc_time = round(random.uniform(*config.step_processing_time_range), 2)
            steps.append(
                RouteStepModel(
                    step_id=step_idx,
                    target_wsg=target_wsg,
                    nominal_processing_time=proc_time,
                )
            )

        product_recipes[p_name] = ProductRouteModel(
            product_id=p_name, steps=steps
        )

    # Generate setup matrices for setup-sensitive WSGs
    setup_matrices: Dict[str, Dict[str, Dict[str, float]]] = {}
    for wsg_id, area_id in wsg_area_map.items():
        if area_id in config.setup_time_ranges:
            min_setup, max_setup = config.setup_time_ranges[area_id]
            wsg_matrix: Dict[str, Dict[str, float]] = {}

            for p_from in products:
                wsg_matrix[p_from] = {}
                for p_to in products:
                    if p_from == p_to:
                        wsg_matrix[p_from][p_to] = 0.0
                    else:
                        wsg_matrix[p_from][p_to] = round(
                            random.uniform(min_setup, max_setup), 2
                        )

            setup_matrices[wsg_id] = wsg_matrix

    return product_recipes, setup_matrices


def generate_job_list(
    num_jobs: int,
    recipes: Dict[str, ProductRouteModel],
    config: DataGenConfig,
) -> List[JobModel]:
    """
    Generate initial list of N jobs/lots with priorities and realistic due dates.
    """
    jobs: List[JobModel] = []
    product_names = list(recipes.keys())

    # Build weighted priority sampling list
    priority_choices = list(config.priority_ratios.keys())
    priority_weights_list = [config.priority_ratios[p] for p in priority_choices]

    for i in range(1, num_jobs + 1):
        job_id = f"JOB_{i:04d}"
        p_name = random.choice(product_names)
        priority = random.choices(
            priority_choices, weights=priority_weights_list, k=1
        )[0]

        route = recipes[p_name]
        total_raw_proc_time = sum(s.nominal_processing_time for s in route.steps)

        # Estimate transport time: ~10 minutes per step
        est_transport_time = len(route.steps) * 10.0

        # Sample flow factor for priority
        ff_min, ff_max = config.flow_factors[priority]
        flow_factor = round(random.uniform(ff_min, ff_max), 2)

        # Due date formula
        due_date = round(
            (total_raw_proc_time + est_transport_time) * flow_factor, 2
        )

        p_weight = config.priority_weights[priority]

        jobs.append(
            JobModel(
                job_id=job_id,
                product_type=p_name,
                priority=priority,
                priority_weight=p_weight,
                total_raw_processing_time=total_raw_proc_time,
                est_transport_time=est_transport_time,
                flow_factor=flow_factor,
                due_date=due_date,
                current_location="Central_Stockroom",
            )
        )

    return jobs


def generate_dataset(config: DataGenConfig) -> DatasetOutputModel:
    """Execute complete dataset generation pipeline."""
    set_seed(config.seed)
    factory = generate_factory_infrastructure(config)
    amhs_matrix = generate_transport_matrix(factory, config)
    recipes, setup_matrices = generate_product_routes(factory, config)
    jobs = generate_job_list(config.num_jobs, recipes, config)

    return DatasetOutputModel(
        factory_infrastructure=factory,
        transport_matrices=amhs_matrix,
        product_recipes=recipes,
        setup_matrices=setup_matrices,
        job_list=jobs,
    )


def save_dataset_json(dataset: DatasetOutputModel, output_path: str) -> None:
    """Save dataset Pydantic model to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(dataset.model_dump_json(indent=2))


def main() -> None:
    config = DataGenConfig()
    output_path = f"data/fjsp_dataset_seed{config.seed}.json"

    print(
        f"Generating FJSP Dataset with seed={config.seed}, "
        f"num_jobs={config.num_jobs}, num_products={config.num_products}..."
    )

    dataset = generate_dataset(config)
    save_dataset_json(dataset, output_path)

    print(f"Dataset successfully generated and saved to: {output_path}")


if __name__ == "__main__":
    main()
