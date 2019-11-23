import sys
sys.path.append('../src')

from src.utils import *
import unittest


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
        # print(choose_from_search_space(get_key_in_nested_dict(self.search_space, 'layers')))

    def test_choose_from_search_space(self):
        print(choose_from_search_space(self.search_space['layers'], 'layers', {}))
        self.assertTrue(type(choose_from_search_space(self.search_space['lr'])) is float)
        self.assertTrue(type(choose_from_search_space(self.search_space['loss'])) is str)
        print(choose_from_search_space(self.search_space))

    def test_filter_list_by_prefix(self):
        d = {'input_size': 784, 'batch_size': 120, 'nodes_layer_1': 300, 'do_layer_1': 0.0, 'activation_layer_1':
             'sigmoid', 'nodes_layer_2': 500, 'do_layer_2': 0.3, 'activation_layer_2': 'sigmoid', 'lr': 1e-07, 'epochs':
             3000, 'optimizer': 'sgd', 'output_nodes': 10, 'output_activation': 'softmax', 'loss':
             'categorical_crossentropy'}
        l = list(d.keys())
        self.assertEqual(sorted(filter_list_by_prefix(l, ('nodes_', 'do_'))), sorted(['nodes_layer_1', 'nodes_layer_2',
                                                                                      'do_layer_1', 'do_layer_2']))
        self.assertTrue(len(filter_list_by_prefix(l, ('do_', 'output_'), True)) == len(l)-4)

    def test_get_mode_multiplier(self):
        self.assertEqual(get_mode_multiplier('min'), -1, "min mode multiplier wrong")
        self.assertEqual(get_mode_multiplier('max'), 1, "max mode multiplier wrong")

    def tearDown(self) -> None:
        pass
