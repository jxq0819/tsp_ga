"""
This is the code to solve Travelling Salesman Problem by using genetic algorithm.

Before running this python programme, please install python, and packages numpy and pandas.

To run this programme, open the terminal and run command:'python tsp_ga.py'.
"""

import numpy as np
import pandas as pd


class TSP:
    """TSP class for solving the problem"""

    def __init__(self, population_size=100, generation_number=100,
                 crossover_probability=0.0,
                 mutation_probability=0.0):
        """Initialise some key attributes of TSP"""
        self.population_size = population_size
        self.generation_number = generation_number
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.city_coordinates = None
        self.city_names = None
        self.city_size = None
        self.city_pair_distance = np.zeros(100).reshape(10, 10)
        self.population = None
        self.population_fitness = None
        self.best_chromosome = None
        self.least_distance = None

    def initialise(self):
        """
        Initialise TSP setup for genetic algorithm.
        Load the city data.
        Set the first generation of chromosome
        """
        self.load_data()
        # Create first generation of chromosomes
        self.population = self.create_population(self.population_size)
        # Calculate the fitness of this generation
        self.population_fitness = self.get_fitness(self.population)

    def load_data(self, file='city.csv', delimiter=','):
        """Load the city data for preparation"""
        # Use pandas to read city data
        data = pd.read_csv(filepath_or_buffer=file, delimiter=delimiter,
                           header=None).values
        print(f'City data loaded:\n{data}')

        self.city_coordinates = data[:, 1:]

        self.city_names = data[:, 0]
        self.city_size = data.shape[0]

        # Calculate the distance between a pair of cities
        for i in range(self.city_size):
            for j in range(self.city_size):
                self.city_pair_distance[i][j] = \
                    np.sqrt((self.city_coordinates[i][0] -
                             self.city_coordinates[j][0]) ** 2 +
                            (self.city_coordinates[i][1] -
                             self.city_coordinates[j][1]) ** 2)

    def create_population(self, size):
        """Create initial population with defined population size"""
        population = []
        for _ in range(size):
            chromosome = np.arange(self.city_size)
            np.random.shuffle(chromosome)
            population.append(chromosome)
        return np.array(population)

    def get_chromosome_distance(self, chromosome):
        """Calculate the distance of a single chromosome"""
        distance = 0.0
        for i in range(-1, len(self.city_names) - 1):
            index1, index2 = chromosome[i], chromosome[i + 1]
            distance += self.city_pair_distance[index1][index2]
        return distance

    def get_fitness(self, population):
        """Get the population_fitness for each chromosome in population."""
        # Use population_fitness to store the population fitness array
        population_fitness = np.array([])

        # Iterate over the population
        for i in range(self.population_size):
            # Pick a chromosome
            chromosome = population[i]
            # Calculate distance for the chromosome
            chromosome_distance = self.get_chromosome_distance(chromosome)
            # Calculate population_fitness as reciprocal of the distance
            single_fitness = 1 / chromosome_distance
            population_fitness = np.append(population_fitness, single_fitness)
        return population_fitness

    def select_population(self, population):
        # Calculate the selection probability distribution based on the
        # weighting population_fitness
        probability = self.population_fitness / self.population_fitness.sum()
        ids_selected = np.random.choice(a=np.arange(self.population_size),
                                        size=self.population_size,
                                        replace=True,
                                        p=probability)
        selected_population = population[ids_selected, :]
        return selected_population

    def crossover(self, parent1, parent2):
        """Perform genes crossover"""
        if np.random.random() > self.crossover_probability:
            return parent1
        # Create tow random indexes to slice gene piece
        index1 = np.random.randint(0, self.city_size - 1)
        index2 = np.random.randint(index1, self.city_size - 1)
        gene_piece = parent2[index1:index2]
        new_chromosome = []
        pointer = 0
        # Rebuild the new chromosome
        for gene in parent1:
            if pointer == index1:
                new_chromosome.extend(gene_piece)
            if gene not in gene_piece:
                new_chromosome.append(gene)
            pointer += 1
        new_chromosome = np.array(new_chromosome)
        return new_chromosome

    def mutate(self, chromosome):
        """Mutate the chromosome by randomly swapping two genes"""
        if np.random.random() > self.mutation_probability:
            return chromosome
        index1 = np.random.randint(0, self.city_size - 1)
        index2 = np.random.randint(0, self.city_size - 1)
        # Create mutated_chromosome for gene mutation in the chromosome
        mutated_chromosome = chromosome.copy()
        mutated_chromosome[index1], mutated_chromosome[index2] = \
            mutated_chromosome[index2], \
            mutated_chromosome[index1]
        return mutated_chromosome

    def evolve(self):
        for generation in range(1, self.generation_number + 1):
            print(f'Generation {generation}:')
            # Find the index of chromosome with the best population_fitness in
            # the current generation
            best_fitness_index = np.argmax(self.population_fitness)

            # Pick out the chromosome with the best population_fitness in the
            # current generation
            local_best_chromosome = self.population[best_fitness_index]
            print(f'Local best chromosome: {local_best_chromosome}')
            local_least_distance = self.get_chromosome_distance(
                local_best_chromosome)
            print(f'Local least distance: {local_least_distance}')
            if generation == 1:
                self.best_chromosome = local_best_chromosome
                self.least_distance = local_least_distance

            if local_least_distance < self.least_distance:
                self.least_distance = local_least_distance
                self.best_chromosome = local_best_chromosome

            print(f'Global least distance: {self.least_distance}')
            path = '->'.join(
                ['city' + str(city_index + 1) for city_index in
                 list(self.best_chromosome)])
            print(f'Path: {path}')
            print('\n------------------------------------------------\n')
            # Start breeding the next generation chromosomes and perform
            # crossover and mutation
            self.population = self.select_population(self.population)
            # Get the population_fitness of each chromosome in the population
            self.population_fitness = self.get_fitness(self.population)
            # Perform crossover and mutation
            # Iterate the population in the current generation
            for selected in range(self.population_size):
                # Randomly select another chromosome for crossover
                random = np.random.randint(0, self.population_size - 1)
                if selected != random:
                    self.population[selected] = self.crossover(
                        self.population[selected],
                        self.population[random])
                    self.population[selected] = self.mutate(
                        self.population[selected])
            # Update the global least distance
            self.least_distance = self.get_chromosome_distance(
                self.best_chromosome)


def main():
    """The execution point of this programme"""
    tsp = TSP(population_size=150, generation_number=800,
              crossover_probability=0.7,
              mutation_probability=0.3)
    tsp.initialise()
    print('\nStart evolution!\n')
    print('================================================')
    tsp.evolve()
    print('Evolution complete!')


if __name__ == '__main__':
    main()
