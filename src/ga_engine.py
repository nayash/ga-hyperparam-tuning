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
    def __init__(self, length=5, population_size=10, mutation_probability=0.2):
        if population_size < 2:
            raise Exception("Need at least 2 individuals to compare")
        self.population_size = population_size
        self.individual_size = length
        self.mutation_probability = mutation_probability
        self._target = Individual(length, list(np.ones((length, ), dtype=int)))
        self._population = Population(self.target, population_size)

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        self._target = target

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






