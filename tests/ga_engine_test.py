#
# Copyright (c) 2019. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#


import sys

sys.path.append('../src')
import unittest
import numpy as np
from src.ga_engine import GAEngine
from src.population import Population
from src.individual import Individual


class GAEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.search_space = {
            'input_size': 784,
            'batch_size': [80, 100, 120],
            'layers': [
                {
                    'nodes_layer_1': [50, 100, 200, 300, 500, 700, 900],
                    'do_layer_1': [0.0, 0.1, 0.2, 0.3, 0.4],
                    'activation_layer_1': ['relu', 'sigmoid']
                },
                {
                    'nodes_layer_1': [300, 500, 700, 900],
                    'do_layer_1': [0.0, 0.1, 0.2, 0.3, 0.4],
                    'activation_layer_1': ['relu', 'sigmoid'],

                    'nodes_layer_2': [100, 300, 500, 700, 900],
                    'do_layer_2': [0.0, 0.1, 0.2, 0.3, 0.4],
                    'activation_layer_2': ['relu', 'sigmoid']
                },
                {
                    'nodes_layer_1': [300, 500, 700, 900],
                    'do_layer_1': [0.0, 0.1, 0.2, 0.3, 0.4],
                    'activation_layer_1': ['relu', 'sigmoid'],

                    'nodes_layer_2': [100, 300, 500, 700, 900],
                    'do_layer_2': [0.0, 0.1, 0.2, 0.3, 0.4],
                    'activation_layer_2': ['relu', 'sigmoid'],

                    'nodes_layer_3': [100, 300, 500, 700, 900],
                    'do_layer_3': [0.0, 0.1, 0.2, 0.3, 0.4],
                    'activation_layer_3': ['relu', 'sigmoid']
                }
            ],
            'lr': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
            'epochs': [3000],
            'optimizer': ['rmsprop', 'sgd', 'adam'],
            'output_nodes': 10,
            'output_activation': 'softmax',
            'loss': 'categorical_crossentropy'
        }
        self.ga_engine = GAEngine(self.search_space, func_eval=self.dummy_func_eval)
        # self.scores = [-0.1, -0.01, -0.005, -0.2, -0.03]
        # self.ga_engine.population.set_fitness_scores(self.scores)

    def test_selection(self):
        self.assertEqual(self.ga_engine.selection()[0].get_fitness_score(),
                         self.ga_engine.population.get_fitness_scores()[np.argsort(self.ga_engine.population.
                                                                                   get_fitness_scores())[-1]])
        second_parent_rank = self.ga_engine.selection()[2]
        self.assertEqual(self.ga_engine.population.individuals[second_parent_rank].get_fitness_score(),
                         self.ga_engine.population.get_fitness_scores()[second_parent_rank])

    def test_mutation(self):
        print("individual 0:", self.ga_engine.population.individuals[0].get_nn_params())
        print(self.ga_engine.mutation(self.ga_engine.population.individuals[0]).get_nn_params())

        print("individual 2:", self.ga_engine.population.individuals[2].get_nn_params())
        print(self.ga_engine.mutation(self.ga_engine.population.individuals[2]).get_nn_params())

    def test_cross_over(self):
        i1 = 0
        i2 = 3
        print("test_cross_over")
        print(self.ga_engine.population.individuals[i1].get_nn_params(), "\n\n",
              self.ga_engine.population.individuals[i2].get_nn_params())
        ind1, ind2 = self.ga_engine.cross_over(self.ga_engine.population.individuals[i1],
                                               self.ga_engine.population.individuals[i2])
        print("-------------------------------------------------------------------------------------------------------")
        print(ind1.get_nn_params(), "\n\n", ind2.get_nn_params())
        print("test_cross_over end")

    def dummy_func_eval(self, model):
        return np.random.uniform(0, 1)
