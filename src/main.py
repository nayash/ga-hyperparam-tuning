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
Entry point of the project. This can be used as an example of how to use this project.
"""

import os
import sys

import pandas as pd
from hyperopt import hp
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.utils import compute_class_weight
from tqdm import tqdm
from stockstats import StockDataFrame as sdf

from ga_engine import GAEngine
import keras
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle

from individual import Individual
from utils import *
import datetime
import numpy as np
from sklearn.datasets import make_classification
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp, rand

# mnist
# search_space_mlp = {
#     'input_size': 784,
#     'batch_size': [80, 100, 120],
#     'layers': [
#         {
#             'nodes_layer_1': list(np.arange(10, 501)),
#             'do_layer_1': list(np.linspace(0, 0.5)),
#             'activation_layer_1': ['relu', 'sigmoid']
#         },
#         {
#             'nodes_layer_1': list(np.arange(10, 501)),
#             'do_layer_1': list(np.linspace(0, 0.5)),
#             'activation_layer_1': ['relu', 'sigmoid'],
#
#             'nodes_layer_2': list(np.arange(10, 501)),
#             'do_layer_2': list(np.linspace(0, 0.5)),
#             'activation_layer_2': ['relu', 'sigmoid']
#         },
#         {
#             'nodes_layer_1': list(np.arange(10, 501)),
#             'do_layer_1': list(np.linspace(0, 0.5)),
#             'activation_layer_1': ['relu', 'sigmoid'],
#
#             'nodes_layer_2': list(np.arange(10, 501)),
#             'do_layer_2': list(np.linspace(0, 0.5)),
#             'activation_layer_2': ['relu', 'sigmoid'],
#
#             'nodes_layer_3': list(np.arange(10, 501)),
#             'do_layer_3': list(np.linspace(0, 0.5)),
#             'activation_layer_3': ['relu', 'sigmoid']
#         }
#     ],
#     'lr': [1e-2, 1e-3, 1e-4],
#     'epochs': [3000],
#     'optimizer': ['rmsprop', 'sgd', 'adam'],
#     'output_nodes': 10,
#     'output_activation': 'softmax',
#     'loss': 'categorical_crossentropy'
# }

# synthetic data params
search_space_mlp = {
    'input_size': 200,
    'batch_size': [40, 60, 80, 100, 120, 150],
    'layers': [
        {
            'nodes_layer_1': list(np.arange(10, 501)),
            'do_layer_1': list(np.linspace(0, 0.5, dtype=np.float32)),
            'activation_layer_1': ['relu', 'sigmoid']
        },
        {
            'nodes_layer_1': list(np.arange(10, 501)),
            'do_layer_1': list(np.linspace(0, 0.5, dtype=np.float32)),
            'activation_layer_1': ['relu', 'sigmoid'],

            'nodes_layer_2': list(np.arange(10, 501)),
            'do_layer_2': list(np.linspace(0, 0.5, dtype=np.float32)),
            'activation_layer_2': ['relu', 'sigmoid']
        },
        {
            'nodes_layer_1': list(np.arange(10, 501)),
            'do_layer_1': list(np.linspace(0, 0.5, dtype=np.float32)),
            'activation_layer_1': ['relu', 'sigmoid'],

            'nodes_layer_2': list(np.arange(10, 501)),
            'do_layer_2': list(np.linspace(0, 0.5, dtype=np.float32)),
            'activation_layer_2': ['relu', 'sigmoid'],

            'nodes_layer_3': list(np.arange(10, 501)),
            'do_layer_3': list(np.linspace(0, 0.5, dtype=np.float32)),
            'activation_layer_3': ['relu', 'sigmoid']
        }
    ],
    "lr": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    "epochs": [3000],
    "optimizer": ["rmsprop", "sgd", "adam"],
    'output_nodes': 3,
    'output_activation': 'softmax',
    'loss': 'categorical_crossentropy'
}

search_space_mlp_hyperopt = {
    'input_size': 200,
    'batch_size': hp.choice('bs', [40, 60, 80, 100, 120, 150]),
    'layers': hp.choice('layers_', [
        {
            'nodes_layer_1': hp.choice('nl11', list(np.arange(10, 501))),
            'do_layer_1': hp.choice('dol11', list(np.linspace(0, 0.5, dtype=np.float32))),
            'activation_layer_1': hp.choice('al11', ['relu', 'sigmoid'])
        },
        {
            'nodes_layer_1': hp.choice('nl21', list(np.arange(10, 501))),
            'do_layer_1': hp.choice('dol21', list(np.linspace(0, 0.5, dtype=np.float32))),
            'activation_layer_1': hp.choice('al21', ['relu', 'sigmoid']),

            'nodes_layer_2': hp.choice('nl22', list(np.arange(10, 501))),
            'do_layer_2': hp.choice('dol22', list(np.linspace(0, 0.5, dtype=np.float32))),
            'activation_layer_2': hp.choice('al22', ['relu', 'sigmoid'])
        },
        {
            'nodes_layer_1': hp.choice('nl31', list(np.arange(10, 501))),
            'do_layer_1': hp.choice('dol31', list(np.linspace(0, 0.5, dtype=np.float32))),
            'activation_layer_1': hp.choice('al31', ['relu', 'sigmoid']),

            'nodes_layer_2': hp.choice('nl32', list(np.arange(10, 501))),
            'do_layer_2': hp.choice('dol32', list(np.linspace(0, 0.5, dtype=np.float32))),
            'activation_layer_2': hp.choice('al32', ['relu', 'sigmoid']),

            'nodes_layer_3': hp.choice('nl33', list(np.arange(10, 501))),
            'do_layer_3': hp.choice('dol33', list(np.linspace(0, 0.5, dtype=np.float32))),
            'activation_layer_3': hp.choice('al33', ['relu', 'sigmoid'])
        }
    ]),
    "lr": hp.choice('lr', [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]),
    "epochs": hp.choice('epochs', [3000]),
    "optimizer": hp.choice('opt', ["rmsprop", "sgd", "adam"]),
    'output_nodes': 3,
    'output_activation': 'softmax',
    'loss': 'categorical_crossentropy'
}

OUTPUT_PATH = 'outputs'

# exit conditions
test_loss = 0.01
test_acc = 1.0
time_limit = 180  # minutes
gen_count = 60
patience = 15

mode = 'max'
ga_history_list = []
ga_history_dict = {}
best_scores = []
avg_scores = []
start_time = get_readable_ctime()
start_time_epoch = datetime.datetime.now()
patience_count = 0
prev_population_avg = 0
run_id = 'ga_hp_mnist'  # change utils log prefix


def get_data_mnist():
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test


def get_synthetic_data():
    x, y = make_classification(n_samples=10000, n_features=200, n_informative=200, n_redundant=0, n_repeated=0,
                               n_classes=3, n_clusters_per_class=2, weights=None, flip_y=0.02, class_sep=0.4,
                               hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=2)
    log("class weights", compute_class_weight('balanced', np.unique(y), y))
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    y = np_utils.to_categorical(encoded_y)
    mm_scaler = MinMaxScaler()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_train = mm_scaler.fit_transform(x_train)
    x_test = mm_scaler.transform(x_test)
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = get_synthetic_data()


def func_eval(model, **kwargs):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, min_delta=0.0001)
    history = model.fit(x_train, y_train, batch_size=kwargs['batch_size'], epochs=kwargs['epochs'], verbose=2,
                        validation_split=0.3, callbacks=[es])
    val_error = np.amin(history.history['val_loss'])
    train_error = np.amin(history.history['loss'])
    log("train/val errors:", train_error, val_error)
    score = model.evaluate(x_test, y_test, verbose=0)  # (loss, accuracy)
    log('eval test keys:', model.metrics_names)
    log('eval test res:', score)
    res_dict = {'train_error': train_error, 'val_error': val_error}
    for i, key in enumerate(model.metrics_names):
        res_dict[key] = score[i]
    ga_history_list.append(res_dict)
    return score[0] if mode == 'min' else score[1]


def exit_check(**kwargs):
    # return False
    global prev_population_avg
    global patience_count
    best_score = kwargs['best_score']
    generation_count = kwargs['generation_count']
    population_avg = kwargs['population_avg']
    if not (population_avg > prev_population_avg):
        patience_count = patience_count + 1
    else:
        patience_count = 0
    prev_population_avg = population_avg
    return np.abs(best_score) >= test_acc or ((datetime.datetime.now() - start_time_epoch).seconds / 60) >= time_limit \
           or generation_count >= gen_count or patience_count >= patience


def on_generation_end(**kwargs):
    best_score = kwargs['best_score']
    generation_count = kwargs['generation_count']
    avg = kwargs['avg_score']
    log("on_generation_end: best_score=", best_score, "generation_count=", generation_count)
    ga_history_dict[generation_count] = ga_history_list.copy()
    ga_history_list.clear()
    best_scores.append(best_score)
    avg_scores.append(avg)
    # if generation_count % 5:
    pickle.dump(ga_history_dict, open(os.path.join(OUTPUT_PATH, run_id + '_history_' + start_time), 'wb'))
    pickle.dump(best_scores, open(os.path.join(OUTPUT_PATH, run_id + '_best_scores_' + start_time), 'wb'))
    pickle.dump(avg_scores, open(os.path.join(OUTPUT_PATH, run_id + '_avg_scores_' + start_time), 'wb'))
    log('history dumped...')


def func_eval_dummy(model, **kwargs):
    ga_history_list.append({})
    return np.random.uniform(0, 1, 2)


def create_model(params: dict = None):
    """
    Creates a NN model using values in params dict.
    :param params: expects a dict with hyper parameters for NN.
    format of 'params' {'batch_size': 100, 'nodes_1': 500, 'do_1': 0.0, 'activation_1': 'sigmoid', 'lr': 1e-07,
    'epochs': 3000, 'optimizer': 'sgd', 'output_nodes': 10, 'output_activation: 'softmax',
    'loss': ['categorical_crossentropy']}
    :return: Keras Sequential model
    """
    print(params)
    for key in params['layers']:
        params[key] = params['layers'][key]
    params.pop('layers')

    log("hyperopt trail", params)

    num_layers = 0
    for key in params.keys():
        if key.startswith('nodes_'):
            num_layers = num_layers + 1

    if num_layers == 0:
        raise Exception("Please specify number of nodes for each layer using format nodes_{layer_number}")

    try:
        model = Sequential()
        for i in range(num_layers):
            i = str(i + 1)
            if i == '1':
                model.add(
                    Dense(params["nodes_layer_" + i], input_shape=(params['input_size'],),
                          activation=params['activation_layer_' + i] if ('activation_layer_' + i) in params
                          else None))
            else:
                model.add(
                    Dense(params["nodes_layer_" + i],
                          activation=params['activation_layer_' + i] if ('activation_layer_' + i) in params
                          else None))
            if 'do_layer_' + i in params:
                model.add(Dropout(params['do_layer_' + i]))

        model.add(Dense(params['output_nodes'], activation=params['output_activation']))
        model.compile(loss=params['loss'], optimizer=params["optimizer"], metrics=['accuracy'])  # TODO metric
    except Exception as ex:
        log("problem params", params)
        log("exception:", str(ex))
        log_flush()
        sys.exit()
    score = func_eval(model, batch_size=params['batch_size'], epochs=params['epochs'])
    score = -score if mode == 'max' else score
    return {'loss': score, 'params': params, 'status': STATUS_OK}


num_gen = 5
log("Running " + run_id)

# GA search
log("********************** GA Search **********************")
num_gen, score, param = GAEngine(search_space_mlp, mutation_probability=0.4, on_generation_end=on_generation_end,
                                 func_eval=func_eval, population_size=5, opt_mode=mode).ga_search(time_limit=5)

# plot_iterable(best_scores=best_scores, avg_scores=avg_scores)
# plot_history(pickle.load(open(os.path.join('history_' + start_time), 'rb')))


# hyperopt search
# log("********************** Hyperopt Search **********************")
# htime = time.time()
# trials = Trials()
# best = fmin(create_model, space=search_space_mlp_hyperopt, algo=tpe.suggest, max_evals=num_gen * 2,
#             trials=trials)
#
# scores = []
# for result in trials.results:
#     scores.append(result['loss'])
#
# # print(scores)
# min_idx = np.argmin(scores)
# log("hyperopt best", scores[min_idx], '--', trials.results[min_idx]['params'])
# plot_iterable(best_scores=scores)

# Random search
# log("********************** Random Search **********************")
# stime = time.time()
# log("population size:", num_gen * 2)
# best = GAEngine(search_space_mlp, exit_check=exit_check, on_generation_end=on_generation_end, func_eval=func_eval,
#                 population_size=num_gen * 2, opt_mode=mode).random_search()
# log("Random search best individual: {}, {}".format(best.get_fitness_score(), best.get_nn_params()))
# log("Random search time:", seconds_to_minutes(time.time() - stime))
# log("Total duration:", seconds_to_minutes((datetime.datetime.now() - start_time_epoch).seconds))
# log_flush()
# get_data_wisconsin()


# http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html
