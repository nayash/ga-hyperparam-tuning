from hyperopt import Trials, STATUS_OK, tpe, fmin, hp, rand
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from keras import optimizers

# iris

search_space_mlp = {
    'input_size': 4,
    'batch_size': hp.choice('bs', [10, 50, 100]),
    'layers': hp.choice('layers_', [
        {
            'nodes_layer_1': hp.choice('nl11', [10, 50, 60, 80, 100, 500]),
            'do_layer_1': hp.choice('do11', [0.0, 0.1, 0.2, 0.3, 0.4]),
            'activation_layer_1': hp.choice('al11', ['relu', 'sigmoid'])
        },
        {
            'nodes_layer_1': [10, 50, 60, 80, 100, 500],
            'do_layer_1': [0.0, 0.1, 0.2, 0.3, 0.4],
            'activation_layer_1': ['relu', 'sigmoid'],

            'nodes_layer_2': [10, 50, 60, 80, 100, 500],
            'do_layer_2': [0.0, 0.1, 0.2, 0.3, 0.4],
            'activation_layer_2': ['relu', 'sigmoid']
        },
        {
            'nodes_layer_1': [10, 50, 60, 80, 100, 500],
            'do_layer_1': [0.0, 0.1, 0.2, 0.3, 0.4],
            'activation_layer_1': ['relu', 'sigmoid'],

            'nodes_layer_2': [10, 50, 60, 80, 100, 500],
            'do_layer_2': [0.0, 0.1, 0.2, 0.3, 0.4],
            'activation_layer_2': ['relu', 'sigmoid'],

            'nodes_layer_3': [10, 50, 60, 80, 100, 500],
            'do_layer_3': [0.0, 0.1, 0.2, 0.3, 0.4],
            'activation_layer_3': ['relu', 'sigmoid']
        }
    ]),
    'lr': [1e-2, 1e-3, 1e-4, 1e-5],
    'epochs': [3000],
    'optimizer': ['rmsprop', 'sgd', 'adam'],
    'output_nodes': 3,
    'output_activation': 'softmax',
    'loss': 'categorical_crossentropy'
}