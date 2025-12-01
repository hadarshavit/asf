from asf.pre_selector.abstract_pre_selector import AbstractPreSelector
import pandas as pd
import numpy as np
from typing import Callable


class GeneticAlgorithmPreSelector(AbstractPreSelector):
    """
    Genetic Algorithm-based pre-selector for finding optimal algorithm subsets.

    Uses evolutionary principles to search for the best subset of algorithms:
    1. Initialize a population of random subsets
    2. Evaluate fitness of each individual
    3. Select parents based on fitness (tournament selection)
    4. Create offspring through crossover
    5. Apply mutation
    6. Repeat until convergence or max generations

    Attributes:
        metric (Callable): A function to evaluate the performance of the selected algorithms.
        n_algorithms (int): The number of algorithms to select.
        maximize (bool): Whether to maximize or minimize the performance metric.
        population_size (int): Number of individuals in the population.
        n_generations (int): Maximum number of generations.
        crossover_rate (float): Probability of crossover (0-1).
        mutation_rate (float): Probability of mutation per gene (0-1).
        tournament_size (int): Number of individuals in tournament selection.
        elitism (int): Number of best individuals to preserve each generation.
        seed (int | None): Random seed for reproducibility.
    """

    def __init__(
        self,
        metric: Callable,
        n_algorithms: int,
        maximize: bool = False,
        population_size: int = 50,
        n_generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        tournament_size: int = 3,
        elitism: int = 2,
        seed: int | None = None,
        **kwargs,
    ):
        """
        Initializes the GeneticAlgorithmPreSelector.

        Args:
            metric (Callable): A function to evaluate the performance of the selected algorithms.
            n_algorithms (int): The number of algorithms to select.
            maximize (bool, optional): Whether to maximize the performance metric. Defaults to False.
            population_size (int, optional): Population size. Defaults to 50.
            n_generations (int, optional): Maximum generations. Defaults to 100.
            crossover_rate (float, optional): Crossover probability. Defaults to 0.8.
            mutation_rate (float, optional): Mutation probability per gene. Defaults to 0.1.
            tournament_size (int, optional): Tournament selection size. Defaults to 3.
            elitism (int, optional): Number of elite individuals to preserve. Defaults to 2.
            seed (int | None, optional): Random seed. Defaults to None.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.metric = metric
        self.n_algorithms = n_algorithms
        self.maximize = maximize
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.seed = seed

    def _create_individual(self, n_total: int, rng: np.random.Generator) -> np.ndarray:
        """Create a random individual (binary array with exactly n_algorithms 1s)."""
        individual = np.zeros(n_total, dtype=np.int8)
        selected_indices = rng.choice(n_total, size=self.n_algorithms, replace=False)
        individual[selected_indices] = 1
        return individual

    def _repair_individual(
        self, individual: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Repair an individual to have exactly n_algorithms selected."""
        individual = individual.copy()
        n_selected = individual.sum()

        if n_selected == self.n_algorithms:
            return individual

        if n_selected > self.n_algorithms:
            # Remove excess selections
            selected_indices = np.where(individual == 1)[0]
            to_remove = rng.choice(
                selected_indices,
                size=n_selected - self.n_algorithms,
                replace=False,
            )
            individual[to_remove] = 0
        else:
            # Add missing selections
            unselected_indices = np.where(individual == 0)[0]
            to_add = rng.choice(
                unselected_indices,
                size=self.n_algorithms - n_selected,
                replace=False,
            )
            individual[to_add] = 1

        return individual

    def _evaluate_fitness(
        self, individual: np.ndarray, performance_frame: pd.DataFrame
    ) -> float:
        """Evaluate the fitness of an individual."""
        selected_cols = [
            col
            for col, selected in zip(performance_frame.columns, individual)
            if selected
        ]
        return self.metric(performance_frame[selected_cols])

    def _tournament_selection(
        self,
        population: list[np.ndarray],
        fitness_scores: list[float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Select an individual using tournament selection."""
        tournament_indices = rng.choice(
            len(population), size=self.tournament_size, replace=False
        )
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]

        if self.maximize:
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        else:
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]

        return population[winner_idx].copy()

    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform uniform crossover between two parents."""
        if rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        # Uniform crossover
        mask = rng.random(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)

        return child1, child2

    def _mutate(self, individual: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply mutation by flipping bits with mutation_rate probability."""
        individual = individual.copy()
        mutation_mask = rng.random(len(individual)) < self.mutation_rate

        # Flip selected bits
        individual[mutation_mask] = 1 - individual[mutation_mask]

        return individual

    def fit_transform(
        self, performance: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """
        Selects the best subset of algorithms using a genetic algorithm.

        Args:
            performance (pd.DataFrame | np.ndarray): A DataFrame or NumPy array containing
                the performance data of algorithms. Rows represent instances, and columns
                represent algorithms.

        Returns:
            pd.DataFrame | np.ndarray: A DataFrame or NumPy array containing the performance
                data of the selected algorithms.
        """
        if isinstance(performance, np.ndarray):
            performance_frame = pd.DataFrame(
                performance,
                columns=[f"Algorithm_{i}" for i in range(performance.shape[1])],
            )
            numpy = True
        else:
            performance_frame = performance
            numpy = False

        n_total = len(performance_frame.columns)

        # Handle edge case
        if self.n_algorithms >= n_total:
            if numpy:
                return performance_frame.values
            return performance_frame.reset_index(drop=True)

        rng = np.random.default_rng(self.seed)

        # Initialize population
        population = [
            self._create_individual(n_total, rng) for _ in range(self.population_size)
        ]

        # Track best solution
        best_individual = None
        best_fitness = float("-inf") if self.maximize else float("inf")

        for generation in range(self.n_generations):
            # Evaluate fitness
            fitness_scores = [
                self._evaluate_fitness(ind, performance_frame) for ind in population
            ]

            # Update best solution
            for ind, fitness in zip(population, fitness_scores):
                if self.maximize:
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_individual = ind.copy()
                else:
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_individual = ind.copy()

            # Elitism: preserve best individuals
            sorted_indices = np.argsort(fitness_scores)
            if self.maximize:
                sorted_indices = sorted_indices[::-1]
            elite = [population[i].copy() for i in sorted_indices[: self.elitism]]

            # Create new population
            new_population = elite.copy()

            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_scores, rng)
                parent2 = self._tournament_selection(population, fitness_scores, rng)

                # Crossover
                child1, child2 = self._crossover(parent1, parent2, rng)

                # Mutation
                child1 = self._mutate(child1, rng)
                child2 = self._mutate(child2, rng)

                # Repair to ensure exactly n_algorithms are selected
                child1 = self._repair_individual(child1, rng)
                child2 = self._repair_individual(child2, rng)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = new_population

        # Extract selected algorithms from best individual
        selected_cols = [
            col
            for col, selected in zip(performance_frame.columns, best_individual)
            if selected
        ]

        selected_performance = performance_frame[selected_cols]

        if numpy:
            selected_performance = selected_performance.values
        else:
            selected_performance = selected_performance.reset_index(drop=True)

        return selected_performance
