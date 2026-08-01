import random
from typing import List, Tuple

from src.schema.data import DatasetOutputModel
from src.model.meta.ga.config import GAConfig
from src.model.meta.ga.decoder import FJSPDecoder
from src.model.meta.ga.types import Chromosome, ScheduledTask


class GAOptimizer:
    def __init__(self, dataset: DatasetOutputModel, config: GAConfig):
        self.dataset = dataset
        self.config = config
        self.decoder = FJSPDecoder(dataset)

        # Total operations count
        self.num_jobs = len(dataset.job_list)
        self.op_info = self.decoder.op_info
        self.total_ops = len(self.op_info)

        # OS template: each job index appears as many times as its route steps
        self.os_template: List[int] = []
        for j_idx, job in enumerate(dataset.job_list):
            num_steps = len(dataset.product_recipes[job.product_type].steps)
            self.os_template.extend([j_idx] * num_steps)

    def create_individual(self) -> Chromosome:
        """Create a random feasible individual (OS + MS)."""
        os = self.os_template.copy()
        random.shuffle(os)

        ms = [
            random.randint(0, len(op["valid_tools"]) - 1)
            if op["valid_tools"]
            else 0
            for op in self.op_info
        ]

        return Chromosome(os=os, ms=ms)

    def evaluate(self, chromo: Chromosome) -> float:
        """Evaluate chromosome and compute weighted fitness score."""
        _, makespan, tardiness, setup_time, _ = self.decoder.decode(chromo)
        chromo.makespan = makespan
        chromo.total_tardiness = tardiness
        chromo.total_setup_time = setup_time

        fitness = (
            self.config.weight_makespan * makespan
            + self.config.weight_tardiness * tardiness
            + self.config.weight_setup * setup_time
        )
        chromo.fitness = round(fitness, 2)
        return chromo.fitness

    def tournament_selection(self, pop: List[Chromosome]) -> Chromosome:
        """Tournament selection."""
        candidates = random.sample(pop, self.config.tournament_size)
        return min(candidates, key=lambda c: c.fitness)

    def pox_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[List[int], List[int]]:
        """Precedence Preserving Order Crossover (POX) for OS."""
        job_indices = list(range(self.num_jobs))
        set1 = set(random.sample(job_indices, k=max(1, self.num_jobs // 2)))

        off1_os = [-1] * len(parent1.os)
        off2_os = [-1] * len(parent2.os)

        # Preserve jobs in set1 at same positions
        for i in range(len(parent1.os)):
            if parent1.os[i] in set1:
                off1_os[i] = parent1.os[i]
            if parent2.os[i] in set1:
                off2_os[i] = parent2.os[i]

        # Fill remaining positions in order from other parent
        p2_idx = 0
        for i in range(len(off1_os)):
            if off1_os[i] == -1:
                while parent2.os[p2_idx] in set1:
                    p2_idx += 1
                off1_os[i] = parent2.os[p2_idx]
                p2_idx += 1

        p1_idx = 0
        for i in range(len(off2_os)):
            if off2_os[i] == -1:
                while parent1.os[p1_idx] in set1:
                    p1_idx += 1
                off2_os[i] = parent1.os[p1_idx]
                p1_idx += 1

        return off1_os, off2_os

    def ms_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[List[int], List[int]]:
        """Uniform Crossover for Machine Selection MS."""
        off1_ms = []
        off2_ms = []

        for i in range(len(parent1.ms)):
            if random.random() < 0.5:
                off1_ms.append(parent1.ms[i])
                off2_ms.append(parent2.ms[i])
            else:
                off1_ms.append(parent2.ms[i])
                off2_ms.append(parent1.ms[i])

        return off1_ms, off2_ms

    def mutate_os(self, os: List[int]) -> List[int]:
        """Swap mutation for OS."""
        mutated = os.copy()
        if random.random() < self.config.mutation_rate and len(mutated) >= 2:
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    def mutate_ms(self, ms: List[int]) -> List[int]:
        """Random re-assignment mutation for MS."""
        mutated = ms.copy()
        for i in range(len(mutated)):
            if random.random() < self.config.mutation_rate:
                num_valid = len(self.op_info[i]["valid_tools"])
                if num_valid > 1:
                    mutated[i] = random.randint(0, num_valid - 1)
        return mutated

    def run(self) -> Tuple[Chromosome, List[ScheduledTask], List[dict]]:
        """Run GA optimization loop and return (best_chromo, tasks, generation_history)."""
        history: List[dict] = []

        # Initial Population
        population = [self.create_individual() for _ in range(self.config.pop_size)]
        for chromo in population:
            self.evaluate(chromo)

        population.sort(key=lambda c: c.fitness)
        best_chromo = population[0]

        history.append(
            {
                "generation": 0,
                "fitness": best_chromo.fitness,
                "makespan": best_chromo.makespan,
                "tardiness": best_chromo.total_tardiness,
                "setup_time": best_chromo.total_setup_time,
            }
        )

        print(
            f"Gen 0/{self.config.generations} - Best Fitness: {best_chromo.fitness} "
            f"(Makespan: {best_chromo.makespan}m, Tardiness: {best_chromo.total_tardiness}m, Setup: {best_chromo.total_setup_time}m)"
        )

        for gen in range(1, self.config.generations + 1):
            next_population: List[Chromosome] = []

            # Preserve Elites
            next_population.extend(population[: self.config.elitism_count])

            while len(next_population) < self.config.pop_size:
                p1 = self.tournament_selection(population)
                p2 = self.tournament_selection(population)

                if random.random() < self.config.crossover_rate:
                    off1_os, off2_os = self.pox_crossover(p1, p2)
                    off1_ms, off2_ms = self.ms_crossover(p1, p2)
                else:
                    off1_os, off1_ms = p1.os.copy(), p1.ms.copy()
                    off2_os, off2_ms = p2.os.copy(), p2.ms.copy()

                off1_os = self.mutate_os(off1_os)
                off1_ms = self.mutate_ms(off1_ms)
                off2_os = self.mutate_os(off2_os)
                off2_ms = self.mutate_ms(off2_ms)

                child1 = Chromosome(os=off1_os, ms=off1_ms)
                child2 = Chromosome(os=off2_os, ms=off2_ms)

                self.evaluate(child1)
                self.evaluate(child2)

                next_population.append(child1)
                if len(next_population) < self.config.pop_size:
                    next_population.append(child2)

            next_population.sort(key=lambda c: c.fitness)
            population = next_population

            if population[0].fitness < best_chromo.fitness:
                best_chromo = population[0]

            history.append(
                {
                    "generation": gen,
                    "fitness": best_chromo.fitness,
                    "makespan": best_chromo.makespan,
                    "tardiness": best_chromo.total_tardiness,
                    "setup_time": best_chromo.total_setup_time,
                }
            )

            if gen % 10 == 0 or gen == self.config.generations:
                print(
                    f"Gen {gen}/{self.config.generations} - Best Fitness: {best_chromo.fitness} "
                    f"(Makespan: {best_chromo.makespan}m, Tardiness: {best_chromo.total_tardiness}m, Setup: {best_chromo.total_setup_time}m)"
                )

        tasks, makespan, tardiness, setup_time, _ = self.decoder.decode(best_chromo)
        return best_chromo, tasks, history
