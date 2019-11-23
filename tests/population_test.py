import sys
sys.path.append('../src')
import unittest
import numpy as np
from src.ga_engine import GAEngine
from src.population import Population
from src.individual import Individual
from utils import *


class PopulationTest(unittest.TestCase):
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
        # self.ga_engine = GAEngine(self.search_space, func_eval=self.dummy_func_eval)
        self.population = Population(self.search_space, self.dummy_func_eval, 'min')
        # self.scores = [-0.1, -0.01, -0.005, -0.2, -0.03]
        # self.population.set_fitness_scores(self.scores)

    def test_get_n_best_individual(self):
        print("scores", self.population.get_fitness_scores())
        temp = [score * get_mode_multiplier('min') for score in self.population.get_fitness_scores()]
        print('temp', temp)
        self.assertEqual(-self.population.get_n_best_individual(1).get_fitness_score(), temp[np.argmin(temp)])
        self.assertEqual(-self.population.get_n_best_individual(3).get_fitness_score(), np.partition(temp, 2)[2])
        self.assertEqual(-self.population.get_n_best_individual(len(temp)).get_fitness_score(),
                         np.partition(temp, len(temp)-1)[len(temp)-1])
        self.assertRaises(AssertionError, self.population.get_n_best_individual, -1)
        self.assertRaises(AssertionError, self.population.get_n_best_individual, 0)

    def test_add_individual(self):
        print("scores_b4_adding", self.population.get_fitness_scores())
        new_individual = Individual(choose_from_search_space(self.search_space))
        new_individual.set_fitness_score(0.01978)
        worst_index = np.argmin(self.population.get_fitness_scores())
        self.population.add_individual(new_individual, new_individual.get_fitness_score())
        self.assertEqual(self.population.individuals[worst_index], new_individual)
        print("scores_after_adding", self.population.get_fitness_scores())

    def dummy_func_eval(self, model):
        return np.random.uniform(0, 1)

    def tearDown(self) -> None:
        pass
