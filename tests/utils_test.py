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
from copy import deepcopy
import time

from src.individual import Individual

sys.path.append('../src')

from src.utils import *
import unittest
import timeit


class UtilsTest(unittest.TestCase):
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

    def test_get_key_in_nested_dict(self):
        self.assertEqual(get_key_in_nested_dict(self.search_space, 'batch_size'), [80, 100, 120])
        self.assertEqual(get_key_in_nested_dict(self.search_space, 'nodes_layer_1'), [50, 100, 200, 300, 500, 700, 900])
        self.assertEqual(get_key_in_nested_dict(self.search_space, 'activation_layer_3'), ['relu', 'sigmoid'])
        self.assertEqual(choose_from_search_space(get_key_in_nested_dict(self.search_space, 'loss')),
                         'categorical_crossentropy')
        for i in range(10000):
            # print("calling with layers------------------------")
            d = choose_from_search_space(self.search_space['layers'], None, {})
            # print(i)
            self.assertTrue('output_nodes' not in d.keys())

    def test_choose_from_search_space(self):
        print(choose_from_search_space(self.search_space))
        print("layers choose", choose_from_search_space(self.search_space['layers'], 'layers'))
        self.assertTrue(type(choose_from_search_space(self.search_space['lr'])) is float)
        self.assertTrue(type(choose_from_search_space(self.search_space['loss'])) is str)
        print("choose_space_timeit", timeit.timeit(lambda: choose_from_search_space(self.search_space), number=10000))
        # print("choose_space_list_timeit",
        #       timeit.timeit(lambda: [Individual(choose_from_search_space(self.search_space)) for i in
        #                              range(100)], number=5))
        # print("choose_space_list_timeit_deepcopy",
        #       timeit.timeit(lambda: [Individual(choose_from_search_space(self.search_space)).__deepcopy__() for i in
        #                              range(100)], number=5))

    def test_filter_list_by_prefix(self):
        d = {'input_size': 784, 'batch_size': 120, 'nodes_layer_1': 300, 'do_layer_1': 0.0, 'activation_layer_1':
            'sigmoid', 'nodes_layer_2': 500, 'do_layer_2': 0.3, 'activation_layer_2': 'sigmoid', 'lr': 1e-07, 'epochs':
                 3000, 'optimizer': 'sgd', 'output_nodes': 10, 'output_activation': 'softmax', 'loss':
                 'categorical_crossentropy'}
        l = list(d.keys())
        self.assertEqual(sorted(filter_list_by_prefix(l, ('nodes_', 'do_'))), sorted(['nodes_layer_1', 'nodes_layer_2',
                                                                                      'do_layer_1', 'do_layer_2']))
        self.assertTrue(len(filter_list_by_prefix(l, ('do_', 'output_'), True)) == len(l) - 4)

    def test_get_mode_multiplier(self):
        self.assertEqual(get_mode_multiplier('min'), -1, "min mode multiplier wrong")
        self.assertEqual(get_mode_multiplier('max'), 1, "max mode multiplier wrong")

    def test_log(self):
        log("on_generation_end: best_score=", -0.0176, "generation_count=", 10)

    def test_find_replace_index(self):
        num = 5
        list_nums = [1, 2, 6, 3]
        self.assertTrue(find_replace_index(num, list_nums) == 0)

        num = 2
        list_nums = [3, 5, 1, 3]
        self.assertTrue(find_replace_index(num, list_nums) == 2)

        num = 2
        list_nums = [3, 5, 6, 3]
        self.assertTrue(find_replace_index(num, list_nums) == -1)

        num = 0.2
        list_nums = [0.3, 0.1, 0.6, 0.3]
        self.assertTrue(find_replace_index(num, list_nums) == 1)

    def test_plot_iterable(self):
        l1 = list(np.random.uniform(0, 100, 30))
        plot_iterable(list1=l1)
        time.sleep(2)
        plot_iterable(list2=list(np.random.uniform(0, 100, 30)))

    def tearDown(self) -> None:
        pass
