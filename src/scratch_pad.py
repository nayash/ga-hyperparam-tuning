import numpy as np
from utils import *

search_space = {
            'input_size': 784,
            'batch_size': [80, 100, 120],
            'layers': [
                {
                    'nodes_1': [50, 100, 200, 300, 500, 700, 900],
                    'do_1': [0.0, 0.1, 0.2, 0.3, 0.4],
                    'activation_1': ['relu', 'sigmoid']
                },
                {
                    'nodes_1': [300, 500, 700, 900],
                    'do_1': [0.0, 0.1, 0.2, 0.3, 0.4],
                    'activation_1': ['relu', 'sigmoid'],

                    'nodes_2': [100, 300, 500, 700, 900],
                    'do_2': [0.0, 0.1, 0.2, 0.3, 0.4],
                    'activation_2': ['relu', 'sigmoid']
                },
                {
                    'nodes_1': [300, 500, 700, 900],
                    'do_1': [0.0, 0.1, 0.2, 0.3, 0.4],
                    'activation_1': ['relu', 'sigmoid'],

                    'nodes_2': [100, 300, 500, 700, 900],
                    'do_2': [0.0, 0.1, 0.2, 0.3, 0.4],
                    'activation_2': ['relu', 'sigmoid'],

                    'nodes_3': [100, 300, 500, 700, 900],
                    'do_3': [0.0, 0.1, 0.2, 0.3, 0.4],
                    'activation_3': ['relu', 'sigmoid']
                }
            ],
            'lr': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
            'epochs': [3000],
            'optimizer': ['rmsprop', 'sgd', 'adam'],
            'output_nodes': 10,
            'output_activation': 'softmax',
            'loss': 'categorical_crossentropy'
        }

assert get_key_in_nested_dict(search_space, 'batch_size') == [80, 100, 120]
assert get_key_in_nested_dict(search_space, 'nodes_3') == [100, 300, 500, 700, 900]
# print(get_key_in_nested_dict(search_space, 'activation_3'))
assert get_key_in_nested_dict(search_space, 'nodes_1') == [50, 100, 200, 300, 500, 700, 900]
assert get_key_in_nested_dict(search_space, 'activation_3') == ['relu', 'sigmoid']
