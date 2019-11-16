#
# Copyright (c) 2019. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#


"""
Individual class represents each individual in a population that participate in reproduction.
In this project each Neural Network model is represented as Individual.
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout


class Individual:
    def __init__(self, nn_prams, **kwargs):
        self.nn_params = nn_prams
        self.func_model = kwargs['func_model'] if 'func_model' in kwargs else self.create_model
        self.__model = self.func_model(nn_prams)
        self.__fitness_score = {}

    def create_model(self, params: dict = None):
        """
        Creates a NN model using values in params dict.
        :param params: expects a dict with hyper parameters for NN.
        format of 'params' {'batch_size': 100, 'nodes_1': 500, 'do_1': 0.0, 'activation_1': 'sigmoid', 'lr': 1e-07,
        'epochs': 3000, 'optimizer': 'sgd', 'output_nodes': 10, 'output_activation: 'softmax',
        'loss': ['categorical_crossentropy']}
        :return: Keras Sequential model
        """

        if not params:
            params = self.nn_params

        num_layers = 0
        for key in params.keys():
            print("key", key)
            if key.startswith('nodes_'):
                num_layers = num_layers + 1
        if num_layers == 0:
            raise Exception("Please specify number of nodes for each layer using format nodes_{layer_number}")
        model = Sequential()
        for i in range(num_layers):
            i = str(i + 1)
            if i == '1':
                model.add(
                    Dense(params["nodes_" + i], input_shape=(params['input_size'], ), activation=params['activation_' + i] if ('activation_' + i) in params
                          else None))
            else:
                model.add(
                    Dense(params["nodes_" + i], activation=params['activation_' + i] if ('activation_' + i) in params
                          else None))
            if 'do_' + i in params:
                model.add(Dropout(params['do_' + i]))

        model.add(Dense(params['output_nodes'], activation=params['output_activation']))
        model.compile(loss=params['loss'], optimizer=params["optimizer"])  # TODO metric
        return model

    def set_fitness_score(self, score):
        self.__fitness_score = score

    def get_fitness_score(self):
        return self.__fitness_score

    def get_model(self):
        return self.__model

    def set_model(self, params):
        self.__model = self.func_model(params)
