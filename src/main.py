#
# Copyright (c) 2019. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import os
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from ga_engine import GAEngine
import keras
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
from utils import *
import datetime
import numpy as np

# mnist
# search_space_mlp = {
#     'input_size': 784,
#     'batch_size': [80, 100, 120],
#     'layers': [
#         {
#             'nodes_layer_1': [10, 50, 60, 80, 500],
#             'do_layer_1': [0.0, 0.1, 0.2, 0.3, 0.4],
#             'activation_layer_1': ['relu', 'sigmoid']
#         },
#         {
#             'nodes_layer_1': [10, 50, 60, 80, 500],
#             'do_layer_1': [0.0, 0.1, 0.2, 0.3, 0.4],
#             'activation_layer_1': ['relu', 'sigmoid'],
#
#             'nodes_layer_2': [10, 50, 60, 80, 500],
#             'do_layer_2': [0.0, 0.1, 0.2, 0.3, 0.4],
#             'activation_layer_2': ['relu', 'sigmoid']
#         },
#         {
#             'nodes_layer_1': [10, 50, 60, 80, 500],
#             'do_layer_1': [0.0, 0.1, 0.2, 0.3, 0.4],
#             'activation_layer_1': ['relu', 'sigmoid'],
#
#             'nodes_layer_2': [10, 50, 60, 80, 500],
#             'do_layer_2': [0.0, 0.1, 0.2, 0.3, 0.4],
#             'activation_layer_2': ['relu', 'sigmoid'],
#
#             'nodes_layer_3': [10, 50, 60, 80, 500],
#             'do_layer_3': [0.0, 0.1, 0.2, 0.3, 0.4],
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

# iris
# search_space_mlp = {
#     'input_size': 4,
#     'batch_size': [10, 50, 100],
#     'layers': [
#         {
#             'nodes_layer_1': [10, 50, 60, 80, 100, 500],
#             'do_layer_1': [0.0, 0.1, 0.2, 0.3, 0.4],
#             'activation_layer_1': ['relu', 'sigmoid']
#         },
#         {
#             'nodes_layer_1': [10, 50, 60, 80, 100, 500],
#             'do_layer_1': [0.0, 0.1, 0.2, 0.3, 0.4],
#             'activation_layer_1': ['relu', 'sigmoid'],
#
#             'nodes_layer_2': [10, 50, 60, 80, 100, 500],
#             'do_layer_2': [0.0, 0.1, 0.2, 0.3, 0.4],
#             'activation_layer_2': ['relu', 'sigmoid']
#         },
#         {
#             'nodes_layer_1': [10, 50, 60, 80, 100, 500],
#             'do_layer_1': [0.0, 0.1, 0.2, 0.3, 0.4],
#             'activation_layer_1': ['relu', 'sigmoid'],
#
#             'nodes_layer_2': [10, 50, 60, 80, 100, 500],
#             'do_layer_2': [0.0, 0.1, 0.2, 0.3, 0.4],
#             'activation_layer_2': ['relu', 'sigmoid'],
#
#             'nodes_layer_3': [10, 50, 60, 80, 100, 500],
#             'do_layer_3': [0.0, 0.1, 0.2, 0.3, 0.4],
#             'activation_layer_3': ['relu', 'sigmoid']
#         }
#     ],
#     'lr': [1e-2, 1e-3, 1e-4, 1e-5],
#     'epochs': [3000],
#     'optimizer': ['rmsprop', 'sgd', 'adam'],
#     'output_nodes': 3,
#     'output_activation': 'softmax',
#     'loss': 'categorical_crossentropy'
# }

# wisconsin
search_space_mlp = {
    'input_size': 30,
    'batch_size': [10, 50, 100],
    'layers': [
        {
            'nodes_layer_1': [10, 50, 60, 80, 100, 500],
            'do_layer_1': [0.0, 0.1, 0.2, 0.3, 0.4],
            'activation_layer_1': ['relu', 'sigmoid']
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
    ],
    'lr': [1e-2, 1e-3, 1e-4, 1e-5],
    'epochs': [3000],
    'optimizer': ['rmsprop', 'sgd', 'adam'],
    'output_nodes': 2,
    'output_activation': 'softmax',
    'loss': 'categorical_crossentropy'
}

OUTPUT_PATH = 'outputs'

# exit conditions
test_loss = 0.07
test_acc = 1.0
time_limit = 300  # minutes
gen_count = 100

mode = 'max'
ga_history_list = []
ga_history_dict = {}
best_scores = []
avg_scores = []
start_time = get_readable_ctime()
start_time_epoch = datetime.datetime.now()


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


def get_data_iris():
    dataframe = pd.read_csv("./inputs/Iris.csv")
    dataset = dataframe.values
    X = dataset[:, 1:5].astype(float)
    Y = dataset[:, 5]
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_y)
    x_train, x_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.3)
    return x_train, y_train, x_test, y_test


def get_data_wisconsin():
    dataframe = pd.read_csv("./inputs/breast_cancer.csv")
    temp = dataframe.iloc[:, 2:]
    temp.drop('Unnamed: 32', axis=1, inplace=True)
    X = temp.values
    series = dataframe.iloc[:, 1]
    print("value_counts", series.value_counts())
    # Y = np.zeros(len(series))
    # Y[series == 'M'] = 1
    encoder = LabelEncoder()
    encoder.fit(series)
    encoded_y = encoder.transform(series)
    # convert integers to dummy variables (i.e. one hot encoded)
    Y = np_utils.to_categorical(encoded_y)
    # print(Y[20:30])
    # print(X.shape, Y.shape)
    mm_scaler = MinMaxScaler()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    x_train = mm_scaler.fit_transform(x_train)
    x_test = mm_scaler.transform(x_test)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def func_eval(model, **kwargs):
    x_train, y_train, x_test, y_test = get_data_wisconsin()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, min_delta=0.0001)
    history = model.fit(x_train, y_train, batch_size=kwargs['batch_size'], epochs=500, verbose=2,
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
    best_score = kwargs['best_score']
    generation_count = kwargs['generation_count']
    return np.abs(best_score) >= test_acc or (
            (datetime.datetime.now() - start_time_epoch).seconds / 60) >= time_limit or generation_count >= gen_count


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
    pickle.dump(ga_history_dict, open(os.path.join(OUTPUT_PATH, 'history_' + start_time), 'wb'))
    pickle.dump(best_scores, open(os.path.join(OUTPUT_PATH, 'best_scores_' + start_time), 'wb'))
    log('history dumped...')


def func_eval_dummy(model, **kwargs):
    ga_history_list.append({})
    return np.random.uniform(0, 1, 2)


log("Running for Wisconsin data set...")
ga_engine_ = GAEngine(search_space_mlp, mutation_probability=0.3, exit_check=exit_check,
                      on_generation_end=on_generation_end, func_eval=func_eval, population_size=3, opt_mode=mode).run()

plot_iterable(best_scores=best_scores, avg_scores=avg_scores)
plot_history(pickle.load(open(os.path.join('history_' + start_time), 'rb')))

# get_data_wisconsin()
