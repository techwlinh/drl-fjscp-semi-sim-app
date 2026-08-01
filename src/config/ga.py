from pydantic import BaseModel, Field


class GAConfig(BaseModel):
    pop_size: int = Field(default=60, description="Population size")
    generations: int = Field(default=50, description="Number of generations")
    crossover_rate: float = Field(default=0.85, description="Crossover probability")
    mutation_rate: float = Field(default=0.15, description="Mutation probability")
    tournament_size: int = Field(default=3, description="Tournament selection size")
    elitism_count: int = Field(default=2, description="Top elite individuals to preserve")

    # Fitness weight objectives
    weight_makespan: float = Field(default=1.0)
    weight_tardiness: float = Field(default=2.0)
    weight_setup: float = Field(default=0.1)
