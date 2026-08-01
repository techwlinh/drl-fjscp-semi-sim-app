from pydantic import BaseModel, Field


class GAConfig(BaseModel):
    algorithm_name: str = Field(default="ga", description="Algorithm identifier for exports")
    dataset_path: str = Field(default="data/fjsp_dataset_seed42.json", description="Input dataset JSON path")
    output_path: str = Field(default="experiments/ga/schedule.json", description="Output schedule JSON path")
    experiments_dir: str = Field(default="experiments", description="Experiments directory")

    pop_size: int = Field(default=60, description="Population size")
    generations: int = Field(default=500, description="Number of generations")
    crossover_rate: float = Field(default=0.85, description="Crossover probability")
    mutation_rate: float = Field(default=0.15, description="Mutation probability")
    tournament_size: int = Field(default=3, description="Tournament selection size")
    elitism_count: int = Field(default=2, description="Top elite individuals to preserve")

    # Numba Acceleration
    use_numba: bool = Field(default=True, description="Enable Numba JIT acceleration for GA decoding")
