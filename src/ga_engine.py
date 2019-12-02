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
from utils import *
import numpy as np
from operator import itemgetter
from copy import deepcopy


class GAEngine(GAAbstract):
    """
    GAEngine drives the whole algorithm that finds optimal solution for the problem.
    :param search_space: search domain to limit the search for optimal hyperparameters for Neural Net. Example can be
    found in main.py
    :param kwargs: list of optional arguments:
    population_size: number individuals in the population to choose parents (default = 5)
    mutation_probability: probability for mutation to occur. Should be low (default = 0.2)
    exit_check: function to check exit condition and return True to stop the search
    on_generation_end: called on end of each generation
    opt_mode: optimization mode ['min', 'max'] default 'min'
    """

    def __init__(self, search_space, **kwargs):
        self.population_size = kwargs['population_size'] if 'population_size' in kwargs else 5
        self.mutation_probability = kwargs['mutation_probability'] if 'mutation_probability' in kwargs else 0.2
        self.func_should_exit = kwargs['exit_check'] if 'exit_check' in kwargs else self.should_exit
        self.search_space = search_space
        if self.population_size < 2:
            raise Exception("Need at least 2 individuals to compare")
        self.opt_mode = kwargs['opt_mode'] if 'opt_mode' in kwargs else 'min'
        self._population = Population(self.search_space, kwargs['func_eval'], self.opt_mode, self.population_size)
        self.on_generation_end = kwargs['on_generation_end'] if 'on_generation_end' in kwargs else \
            self.on_generation_end_dummy()
        self.param_importance = {}
        self.current_generation_updated_params = set()

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
        second_parent_rank = np.random.randint(1, min(5, self.population_size + 1))
        first_parent = self.population.get_n_best_individual(1)
        print("second_rank", second_parent_rank)
        while first_parent.get_nn_params() == self.population.get_n_best_individual(
                second_parent_rank).get_nn_params():  # if rank of both parents same, choose another
            log("calling selection recursive", second_parent_rank, min(5, self.population_size))
            second_parent_rank = np.random.randint(1, min(5, self.population_size + 1))
        return deepcopy(first_parent), deepcopy(self.population.get_n_best_individual(second_parent_rank)), \
               second_parent_rank

    def on_generation_end_dummy(self, *args):
        pass

    def mutation(self, individual):
        # TODO change random number of genes instead of single gene
        params = individual.get_nn_params()
        keys = filter_list_by_prefix(list(params.keys()), ("input", "output"), True)
        mutation_key = list(keys)[np.random.randint(0, len(keys))]
        log("mutate_key", mutation_key)
        # TODO if secondary mutation prob < 0.5 and mutation_key == 'layer type' completely mutate layer params
        if np.random.uniform(0, 1) < 0.5 and "layer_" in mutation_key:
            layer_params = choose_from_search_space(get_key_in_nested_dict(self.search_space, "layers"))
            params.update(layer_params)
            individual.set_nn_params(params)
            self.current_generation_updated_params.update(layer_params.keys())
            log("complete_layer_mutate", params)
        else:
            values = get_key_in_nested_dict(self.search_space, mutation_key)
            if not isinstance(values, str):
                mutation_value_index = np.random.randint(0, len(values))
                params[mutation_key] = values[mutation_value_index]
            else:
                params[mutation_key] = values
            log("post mutation params", params)
            individual.set_nn_params(params)
            self.current_generation_updated_params.add(mutation_key)
        return individual

    def cross_over(self, individual1: Individual, individual2: Individual, individual1_part=None,
                   individual2_part=None):
        # TODO choose important genes for cross overs
        ind1_params = individual1.get_nn_params()
        ind2_params = individual2.get_nn_params()
        l1 = filter_list_by_prefix(list(ind1_params.keys()), ("input", "output"), True)
        l2 = filter_list_by_prefix(list(ind2_params.keys()), ("input", "output"), True)
        portion1 = itemgetter(*np.random.randint(0, len(l1), 5))(l1)
        portion2 = itemgetter(*np.random.randint(0, len(l2), 5))(l2)
        sorted_param_imp = self.get_sorted_param_importance()
        log("sorted params:", sorted_param_imp)
        if len(sorted_param_imp) == 0:
            common_portion = list(set(portion1).intersection(l2))
        else:
            co_size = min(np.random.randint(1, 5), len(sorted_param_imp))
            common_portion = []
            count = 0
            while len(common_portion) < co_size and count < len(sorted_param_imp):
                if sorted_param_imp[count][0] in l1 and sorted_param_imp[count][0] in l2:
                    common_portion.append(sorted_param_imp[count][0])
                count = count + 1
            log("selected imp_params", common_portion)
        if len(common_portion) == 0:
            self.cross_over(individual1, individual2)
        # swap parent attributes
        log("properties being crossed over:", str(common_portion))
        for key in common_portion:
            temp = ind1_params[key]
            ind1_params[key] = ind2_params[key]
            ind2_params[key] = temp
            self.current_generation_updated_params.add(key)

        log("post cross-over individual1 params", ind1_params)
        log("post cross-over individual2 params", ind2_params)
        individual1.set_nn_params(ind1_params)
        individual2.set_nn_params(ind2_params)
        return individual1, individual2

    def run(self, only_mutation=False):
        count = 0
        # mul = get_mode_multiplier(self.opt_mode)
        best_score = -np.inf
        start_time = time.time()
        while True:
            log("Generation Start:", count)
            prev_best_score = self.population.get_n_best_individual(1).get_fitness_score()
            # selection
            parent1, parent2, second_parent_rank = self.selection()
            mutation_prob = np.random.uniform(0, 1)

            # cross over
            if not only_mutation:
                child1, child2 = self.cross_over(parent1, parent2)

            # mutate
            if mutation_prob < self.mutation_probability or only_mutation:
                child1 = self.mutation(parent1 if not child1 else child1)
                child2 = self.mutation(parent2 if not child2 else child2)

            log("Evaluating params:\n{}\nand\n{}".format(child1.get_nn_params(), child2.get_nn_params()))
            fitness1 = self.population.calc_fitness_score(child1)
            fitness2 = self.population.calc_fitness_score(child2)
            log("fitness1 = {}, fitness2 = {} and prev_best = {}".format(fitness1, fitness2, prev_best_score))
            if fitness1 > prev_best_score:
                log("added child1 to {} index".format(self.population.add_individual(child1, fitness1)))
            if fitness2 > prev_best_score:
                log("added child2 to {} index".format(self.population.add_individual(child2, fitness2)))

            best_score = max(fitness1, fitness2)
            log("current generation score: {}".format(best_score))
            log("All scores", self.population.get_fitness_scores())
            self.on_generation_end(best_score, count)
            if self.func_should_exit(best_score):
                break
            self.update_param_importance(self.current_generation_updated_params, best_score, prev_best_score)
            self.current_generation_updated_params.clear()
            log("curr gen updated param after clear", self.current_generation_updated_params)
            log("Generation End:", count)
            count = count + 1
        log("Best individual is {} and target is {}; generations = {}".format(child1.get_fitness_score(),
                                                                              child2.get_fitness_score(),
                                                                              count))
        log("Best parameter: ", child1.get_nn_params(), "\n", child2.get_nn_params())
        log("Total run duration:", seconds_to_minutes(time.time() - start_time))
        log_flush()
        return count

    def should_exit(self, best_score):
        return np.abs(best_score) < 0.1

    def update_param_importance(self, params, best_score, prev_best_score):
        log("current gen update params:", params)
        delta = best_score - prev_best_score
        for param in params:
            if param in self.param_importance:
                self.param_importance[param] = self.param_importance[param] + delta
            else:
                self.param_importance[param] = delta

    def get_sorted_param_importance(self):
        """
        :return: list of tuples [(key, value)]
        """

        return sorted(self.param_importance.items(), key=lambda kv: kv[1], reverse=True)
