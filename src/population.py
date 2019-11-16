#
# Copyright (c) 2019. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import numpy as np
from individual import Individual

# TODO accept function as param for evaluate_model


class Population:
    def __init__(self, target, population_size=5, individuals=None):
        self.gene_set = [0, 1]
        self.population_size = population_size
        if individuals is None:
            self.individuals = [Individual(len(target.get_value())) for i in range(population_size)]
        else:
            if not (population_size == len(individuals)):
                raise Exception("population size and length of individuals passed are different")
            self.individuals = individuals
        self.fitness_scores = []
        self.target = target
        self.calc_fitness_scores()

    def get_individual_values_as_list(self):
        return [individual.value for individual in self.individuals]

    @staticmethod
    def calc_fitness_score(individual1, individual2):
        score = 0
        for i1, i2 in zip(individual1.get_value(), individual2.get_value()):
            if i1 == i2:
                score = score + 1
        return score

    def calc_fitness_scores(self):
        self.fitness_scores.clear()
        for individual in self.individuals:
            self.fitness_scores.append(self.calc_fitness_score(self.target, individual))

    def get_n_best_individual(self, n):
        best_index = np.argsort(self.fitness_scores)[-n]
        # print("get_n_best", best_index, self.get_individual_values_as_list(), self.fitness_scores)
        return self.individuals[best_index], self.fitness_scores[best_index]

    def add_individual(self, new_individual, fitness_score=-np.inf):
        worst_index = np.argsort(self.fitness_scores)[0]  # least fit individual
        if fitness_score == -np.inf:
            new_individual.set_fitness_score(self.calc_fitness_score(self.target, new_individual))
        else:
            new_individual.set_fitness_score(fitness_score)
        self.fitness_scores[worst_index] = new_individual.get_fitness_score()
        self.individuals[worst_index] = new_individual
        # print("add_individual", worst_index, self.get_individual_values_as_list(), self.fitness_scores)
