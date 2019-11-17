#
# Copyright (c) 2019. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from ga_abstract import GAAbstract
from population import Population
from individual import Individual
import numpy as np


class GAEngine(GAAbstract):
    """
    GAEngine drives the whole algorithm that finds optimal solution for the problem.
    :param search_space: search domain to limit the search for optimal hyperparameters for Neural Net. Example can be
    found in main.py
    :param kwargs: list of optional arguments:
    population_size: number individuals in the population to choose parents (default = 5)
    mutation_probability: probability for mutation to occur. Should be low (default = 0.2)
    func_should_exit: function to check exit condition and return True to stop the search
    on_generation_end: called on end of each generation
    """

    def __init__(self, search_space, **kwargs):
        self.population_size = kwargs['population_size'] if 'population_size' in kwargs else 5
        self.mutation_probability = kwargs['mutation_probability'] if 'mutation_probability' in kwargs else 0.2
        self.func_should_exit = kwargs['func_should_exit'] if 'func_should_exit' in kwargs else self.should_exit()
        self.search_space = search_space
        if self.population_size < 2:
            raise Exception("Need at least 2 individuals to compare")
        self._population = Population(self.search_space, kwargs['func_eval'], self.population_size)
        self.on_generation_end = kwargs['on_generation_end'] if 'on_generation_end' in kwargs else None

    @property
    def target(self):
        pass

    @target.setter
    def target(self, target):
        pass

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, population):
        self._population = population

    def selection(self):
        second_parent_rank = np.random.randint(1, min(5, self.population_size))
        return self.population.get_n_best_individual(1), self.population.get_n_best_individual(second_parent_rank)

    def mutation(self, individual):
        params = individual.get_nn_params()
        mutation_key = list(params.keys())[np.random.randint(0, len(params.keys()))]
        mutation_index = np.random.randint(0, len(self.target.get_value()))
        value = individual.get_value()
        value[mutation_index] = (int) (not value[mutation_index])
        individual.set_value(value)
        return individual

    def cross_over(self, individual1, individual2, individual1_part=None, individual2_part=None):
        if not individual1_part:
            individual1_part = sorted(np.random.randint(0, self.individual_size, 2))
        part_size = individual1_part[1] - individual1_part[0]
        if not individual2_part:
            individual2_part = [None, None]
            individual2_part[0] = np.random.randint(0, self.individual_size-part_size, 1)[0]
            individual2_part[1] = individual2_part[0] + part_size
        value1 = individual1.get_value()
        value2 = individual2.get_value()
        value1[individual1_part[0]:individual1_part[1]] = value2[individual2_part[0]:individual2_part[1]]
        individual1.set_value(value1)
        return individual1

    def run(self, only_mutation=False):
        count = 0
        while True:
            # selection
            (parent1, best_score), (parent2, parent2_score) = self.selection()
            cross_over_prob, mutation_prob = np.random.uniform(0, 1, 2)
            child = None
            # cross over
            if not only_mutation:
                child = self.cross_over(parent1, parent2)
                fitness = self.population.calc_fitness_score(self.target, child)

            # mutate
            if mutation_prob < self.mutation_probability or only_mutation:
                child = self.mutation(parent1 if not child else child)
                fitness = self.population.calc_fitness_score(self.target, child)

            if fitness > best_score:
                self.population.add_individual(child, fitness)
                # print("new best found: {}, {}".format(child.get_value(), fitness))

            if fitness == len(self.target.get_value()):
                break
            count = count + 1
            if count%500 == 0:
                print("Generation :", count)
        print("Best individual is {} and target is {}; generations = {}".format(child.get_value(),
                                                                                self.target.get_value(), count))
        return count

    def should_exit(self):
        pass






