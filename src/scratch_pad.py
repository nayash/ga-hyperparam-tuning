import numpy as np
from individual import Individual

search_space_mlp = {
            'input_size': 784,
            'batch_size': [80, 100, 120],
            'num_layers': [
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

# search_space_mlp = {
#             'batch_size': [80, 100, 120],
#             'num_layers': {
#                 'one': {
#                     'nodes_1': [50, 100, 200, 300, 500, 700, 900],
#                     'do_1': [0.0, 0.1, 0.2, 0.3, 0.4],
#                     'activation_1': ['relu', 'sigmoid']
#                 },
#                 'two': {
#                     'nodes_1': [300, 500, 700, 900],
#                     'do_1': [0.0, 0.1, 0.2, 0.3, 0.4],
#                     'activation_1': ['relu', 'sigmoid'],
#
#                     'nodes_2': [100, 300, 500, 700, 900],
#                     'do_2': [0.0, 0.1, 0.2, 0.3, 0.4],
#                     'activation_2': ['relu', 'sigmoid']
#                 },
#                 'three': {
#                     'nodes_1': [300, 500, 700, 900],
#                     'do_1': [0.0, 0.1, 0.2, 0.3, 0.4],
#                     'activation_1': ['relu', 'sigmoid'],
#
#                     'nodes_2': [100, 300, 500, 700, 900],
#                     'do_2': [0.0, 0.1, 0.2, 0.3, 0.4],
#                     'activation_2': ['relu', 'sigmoid'],
#
#                     'nodes_3': [100, 300, 500, 700, 900],
#                     'do_3': [0.0, 0.1, 0.2, 0.3, 0.4],
#                     'activation_3': ['relu', 'sigmoid']
#                 }
#             },
#             "lr": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
#             "epochs": [3000],
#             "optimizer": ["rmsprop", "sgd", "adam"]
#         }


def choose_from_search_space(search_space_mlp, key="params", params={}):
    if type(search_space_mlp) is dict:
        keys = search_space_mlp.keys()
        for key in keys:
            choose_from_search_space(search_space_mlp[key], key, params)
    elif type(search_space_mlp) is list:  # or type(search_space_mlp) is tuple:
        choose_from_search_space(search_space_mlp[np.random.randint(0, len(search_space_mlp))], key, params)
    else:
        params[key] = search_space_mlp
    return params


result = choose_from_search_space(search_space_mlp)
print("chosen param", result)
individual = Individual(result)
individual.get_model().summary()

# search_space_mlp = {
#             'batch_size': [80, 100, 120],
#             'num_layers': {
#                 'one': {
#                     'nodes_1': [50, 100, 200, 300, 500, 700, 900],
#                     'do_1': [0.0, 0.1, 0.2, 0.3, 0.4],
#                     'activation_1': ['relu', 'sigmoid']
#                 },
#                 'two': {
#                     'nodes_1': [300, 500, 700, 900],
#                     'do_1': [0.0, 0.1, 0.2, 0.3, 0.4],
#                     'activation_1': ['relu', 'sigmoid'],
#
#                     'nodes_2': [100, 300, 500, 700, 900],
#                     'do_2': [0.0, 0.1, 0.2, 0.3, 0.4],
#                     'activation_1': ['relu', 'sigmoid']
#                 },
#                 'three': {
#                     'nodes_1': [300, 500, 700, 900],
#                     'do_1': [0.0, 0.1, 0.2, 0.3, 0.4],
#                     'activation_1': ['relu', 'sigmoid'],
#
#                     'nodes_2': [100, 300, 500, 700, 900],
#                     'do_2': [0.0, 0.1, 0.2, 0.3, 0.4],
#                     'activation_1': ['relu', 'sigmoid'],
#
#                     'nodes_3': [100, 300, 500, 700, 900],
#                     'do_3': [0.0, 0.1, 0.2, 0.3, 0.4],
#                     'activation_1': ['relu', 'sigmoid']
#                 }
#             },
#             "lr": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
#             "epochs": [3000],
#             "optimizer": ["rmsprop", "sgd", "adam"]
#         }