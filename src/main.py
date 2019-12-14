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
import sys

import pandas as pd
from hyperopt import hp
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
from utils import *
import datetime
import numpy as np
from sklearn.datasets import make_classification

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
# search_space_mlp = {
#     'input_size': 30,
#     'batch_size': [10, 50, 100],
#     'layers': [
#         {
#             'nodes_layer_1': list(np.arange(0, 501)),
#             'do_layer_1': list(np.linspace(0, 1.0)),
#             'activation_layer_1': ['relu', 'sigmoid']
#         },
#         {
#             'nodes_layer_1': list(np.arange(0, 501)),
#             'do_layer_1': list(np.linspace(0, 1.0)),
#             'activation_layer_1': ['relu', 'sigmoid'],
#
#             'nodes_layer_2': list(np.arange(0, 501)),
#             'do_layer_2': list(np.linspace(0, 1.0)),
#             'activation_layer_2': ['relu', 'sigmoid']
#         },
#         {
#             'nodes_layer_1': list(np.arange(0, 501)),
#             'do_layer_1': list(np.linspace(0, 1.0)),
#             'activation_layer_1': ['relu', 'sigmoid'],
#
#             'nodes_layer_2': list(np.arange(0, 501)),
#             'do_layer_2': list(np.linspace(0, 1.0)),
#             'activation_layer_2': ['relu', 'sigmoid'],
#
#             'nodes_layer_3': list(np.arange(0, 501)),
#             'do_layer_3': list(np.linspace(0, 1.0)),
#             'activation_layer_3': ['relu', 'sigmoid']
#         }
#     ],
#     'lr': [1e-2, 1e-3, 1e-4, 1e-5],
#     'epochs': [3000],
#     'optimizer': ['rmsprop', 'sgd', 'adam'],
#     'output_nodes': 2,
#     'output_activation': 'softmax',
#     'loss': 'categorical_crossentropy'
# }

# stock_price
# search_space_mlp = {
#             'input_size': 225,
#             'batch_size': [40, 60, 80, 100, 120, 150],
#             'layers': [
#                 {
#                     'nodes_layer_1': [100, 300, 500, 700, 900],
#                     'do_layer_1': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
#                     'activation_layer_1': ['relu', 'sigmoid']
#                 },
#                 {
#                     'nodes_layer_1': [100, 300, 500, 700, 900],
#                     'do_layer_1': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
#                     'activation_layer_1': ['relu', 'sigmoid'],
#
#                     'nodes_layer_2': [100, 300, 500, 700, 900],
#                     'do_layer_2': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
#                     'activation_layer_2': ['relu', 'sigmoid']
#                 },
#                 {
#                     'nodes_layer_1': [100, 300, 500, 700, 900],
#                     'do_layer_1': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
#                     'activation_layer_1': ['relu', 'sigmoid'],
#
#                     'nodes_layer_2': [100, 300, 500, 700, 900],
#                     'do_layer_2': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
#                     'activation_layer_2': ['relu', 'sigmoid'],
#
#                     'nodes_layer_3': [100, 300, 500, 700, 900],
#                     'do_layer_3': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
#                     'activation_layer_3': ['relu', 'sigmoid']
#                 }
#                 ],
#             "lr": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
#             "epochs": [3000],
#             "optimizer": ["rmsprop", "sgd", "adam"],
#             'output_nodes': 3,
#             'output_activation': 'softmax',
#             'loss': 'categorical_crossentropy'
#         }

# magic04
# search_space_mlp = {
#     'input_size': 10,
#     'batch_size': [50, 100, 200, 300],
#     'layers': [
#         {
#             'nodes_layer_1': [10, 50, 60, 80, 100, 200, 500],
#             'do_layer_1': [0.0, 0.1, 0.2, 0.3, 0.4],
#             'activation_layer_1': ['relu', 'sigmoid']
#         },
#         {
#             'nodes_layer_1': [10, 50, 60, 80, 100, 200, 500],
#             'do_layer_1': [0.0, 0.1, 0.2, 0.3, 0.4],
#             'activation_layer_1': ['relu', 'sigmoid'],
#
#             'nodes_layer_2': [10, 50, 60, 80, 100, 200, 500],
#             'do_layer_2': [0.0, 0.1, 0.2, 0.3, 0.4],
#             'activation_layer_2': ['relu', 'sigmoid']
#         },
#         {
#             'nodes_layer_1': [10, 50, 60, 80, 100, 200, 500],
#             'do_layer_1': [0.0, 0.1, 0.2, 0.3, 0.4],
#             'activation_layer_1': ['relu', 'sigmoid'],
#
#             'nodes_layer_2': [10, 50, 60, 80, 100, 200, 500],
#             'do_layer_2': [0.0, 0.1, 0.2, 0.3, 0.4],
#             'activation_layer_2': ['relu', 'sigmoid'],
#
#             'nodes_layer_3': [10, 50, 60, 80, 100, 200, 500],
#             'do_layer_3': [0.0, 0.1, 0.2, 0.3, 0.4],
#             'activation_layer_3': ['relu', 'sigmoid']
#         }
#     ],
#     'lr': [1e-2, 1e-3, 1e-4, 1e-5],
#     'epochs': [3000],
#     'optimizer': ['rmsprop', 'sgd', 'adam'],
#     'output_nodes': 2,
#     'output_activation': 'softmax',
#     'loss': 'categorical_crossentropy'
# }

# Anuran calls
# search_space_mlp = {
#     'input_size': 22,
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
#     'output_nodes': 4,
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
    "lr": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    "epochs": [3000],
    "optimizer": ["rmsprop", "sgd", "adam"],
    'output_nodes': 3,
    'output_activation': 'softmax',
    'loss': 'categorical_crossentropy'
}

OUTPUT_PATH = 'outputs'

# exit conditions
test_loss = 0.01
test_acc = 0.95
time_limit = 180  # minutes
gen_count = 100

mode = 'max'
ga_history_list = []
ga_history_dict = {}
best_scores = []
avg_scores = []
start_time = get_readable_ctime()
start_time_epoch = datetime.datetime.now()
run_id = 'ga_rs_synthetic_2'  # change utils log prefix

# last one good => 93%
bad_params_wisconsin = [{'input_size': 30, 'batch_size': 100, 'nodes_layer_1': 435, 'do_layer_1': 0.02040816326530612,
                         'activation_layer_1': 'sigmoid', 'nodes_layer_2': 248, 'do_layer_2': 0.061224489795918366,
                         'activation_layer_2': 'relu', 'lr': 0.001, 'epochs': 3000, 'optimizer': 'sgd',
                         'output_nodes': 2, 'output_activation': 'softmax', 'loss': 'categorical_crossentropy',
                         'nodes_layer_3': 411, 'do_layer_3': 0.6938775510204082, 'activation_layer_3': 'sigmoid'},
                        {'input_size': 30, 'batch_size': 101, 'nodes_layer_1': 434, 'do_layer_1': 0.02040816326530612,
                         'activation_layer_1': 'sigmoid', 'nodes_layer_2': 247, 'do_layer_2': 0.061224489795918366,
                         'activation_layer_2': 'relu', 'lr': 0.001, 'epochs': 3000, 'optimizer': 'sgd',
                         'output_nodes': 2, 'output_activation': 'softmax', 'loss': 'categorical_crossentropy',
                         'nodes_layer_3': 411, 'do_layer_3': 0.6938775510204082, 'activation_layer_3': 'sigmoid'},
                        {'input_size': 30, 'batch_size': 100, 'nodes_layer_1': 434, 'do_layer_1': 0.01100000000000000,
                         'activation_layer_1': 'sigmoid', 'nodes_layer_2': 247, 'do_layer_2': 0.061224489795918366,
                         'activation_layer_2': 'relu', 'lr': 0.001, 'epochs': 3000, 'optimizer': 'sgd',
                         'output_nodes': 2, 'output_activation': 'softmax', 'loss': 'categorical_crossentropy',
                         'nodes_layer_3': 411, 'do_layer_3': 0.6938775510204082, 'activation_layer_3': 'sigmoid'},
                        {'input_size': 30, 'batch_size': 100, 'nodes_layer_1': 434, 'do_layer_1': 0.02040816326530612,
                         'activation_layer_1': 'sigmoid', 'nodes_layer_2': 247, 'do_layer_2': 0.061224489795918366,
                         'activation_layer_2': 'relu', 'lr': 0.001, 'epochs': 3000, 'optimizer': 'sgd',
                         'output_nodes': 2, 'output_activation': 'softmax', 'loss': 'categorical_crossentropy',
                         'nodes_layer_3': 410, 'do_layer_3': 0.5938775510204082, 'activation_layer_3': 'sigmoid'},
                        {'input_size': 30, 'batch_size': 100, 'nodes_layer_1': 434, 'do_layer_1': 0.03040816326530612,
                         'activation_layer_1': 'sigmoid', 'nodes_layer_2': 247, 'do_layer_2': 0.051224489795918366,
                         'activation_layer_2': 'relu', 'lr': 0.001, 'epochs': 3000, 'optimizer': 'sgd',
                         'output_nodes': 2, 'output_activation': 'softmax', 'loss': 'categorical_crossentropy',
                         'nodes_layer_3': 410, 'do_layer_3': 0.6938775510204082, 'activation_layer_3': 'sigmoid'}]


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


def get_data_magic():
    df = pd.read_csv('inputs\magic04.data', names=['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym',
                                                   'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class'])
    x = df.iloc[:, 0:10].values
    y = df['class'].values
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    y = np_utils.to_categorical(encoded_y)
    mm_scaler = MinMaxScaler()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_train = mm_scaler.fit_transform(x_train)
    x_test = mm_scaler.transform(x_test)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def get_data_anuran():
    df = pd.read_csv('inputs\Frogs_MFCCs.csv')
    x = df.iloc[:, :22].values
    y = df['Family']
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    y = np_utils.to_categorical(encoded_y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def get_data_stock():
    def get_sample_weights(y):
        y = y.astype(int)  # compute_class_weight needs int labels
        class_weights = compute_class_weight('balanced', np.unique(y), y)
        log("class weights are {}".format(class_weights), np.unique(y))
        log("value_counts", np.unique(y, return_counts=True))
        sample_weights = y.copy().astype(float)
        for i in np.unique(y):
            sample_weights[sample_weights == i] = class_weights[i]
            # sample_weights = np.where(sample_weights == i, class_weights[int(i)], y_)

        return sample_weights

    def get_SMA(df, col_name, intervals):
        """
        Momentum indicator
        """
        stime = time.time()
        log("Calculating SMA")
        df_ss = sdf.retype(df)
        for i in tqdm(intervals):
            df['sma_' + str(i)] = df_ss[col_name + '_' + str(i) + '_sma']
            del df['close_' + str(i) + '_sma']

        log("Calculation of SMA Done", stime)

    def create_label_30_150_MA(df, col_name):
        log("creating label with create_label_30_150_MA")

        def detect_crossover(diff_prev, diff):
            if diff_prev >= 0 and diff < 0:
                # buy
                return 1
            elif diff_prev <= 0 and diff > 0:
                return 0
            else:
                return 2

        get_SMA(df, 'close', [30, 150])
        labels = np.zeros((len(df)))
        labels[:] = np.nan
        diff = df['sma_30'] - df['sma_150']
        diff_prev = diff.shift()
        df['diff_prev'] = diff_prev
        df['diff'] = diff

        res = df.apply(lambda row: detect_crossover(row['diff_prev'], row['diff']), axis=1)
        log("labels count", np.unique(res, return_counts=True))
        df.drop(columns=['diff_prev', 'diff'], inplace=True)
        return res

    # generate train batch
    df = pickle.load(open('inputs\df_MSFT', 'rb'))
    prev_len = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    log("Dropped {0} nan rows before label calculation".format(prev_len - len(df)))

    if 'labels' not in df.columns:
        df['labels'] = create_label_30_150_MA(df, 'close')
        pickle.dump(df, open(os.path.join("inputs", "df_MSFT"), 'wb'))
    else:
        log("labels already calculated")

    prev_len = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    log("Dropped {0} nan rows after label calculation".format(prev_len - len(df)))

    train_duration_years = 6
    test_duration_years = 2
    start_date = df.iloc[0]['timestamp']
    end_date = start_date + pd.offsets.DateOffset(years=train_duration_years)
    df_batch = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

    mm_scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    x_train = mm_scaler.fit_transform(df_batch.loc[:, 'rsi_6':'eom_20'])
    y_ = np.asarray(df_batch['labels'])

    sample_weights = get_sample_weights(y_)

    # OHE can be fit once outside
    one_hot_enc = OneHotEncoder(sparse=False, categories='auto')  # , categories='auto'
    y_train = one_hot_enc.fit_transform(y_.reshape(-1, 1))
    if len(np.unique(y_)) != 3:
        log('Number of labels ({}) wrong for batch {} to {}. Labels={}'.
            format(len(np.unique(y_train)), start_date, end_date, np.unique(y_, return_counts=True)))
        sys.exit()

    # generate test batch
    test_start_date = end_date + pd.offsets.DateOffset(days=1)
    test_end_date = test_start_date + pd.offsets.DateOffset(years=test_duration_years)

    is_last_batch = False
    if (df.tail(1).iloc[0]["timestamp"] - test_end_date).days < 180:  # 6 months
        is_last_batch = True

    df_batch_test = df[(df["timestamp"] >= test_start_date) & (df["timestamp"] <= test_end_date)]
    x_test = mm_scaler.transform(df_batch_test.loc[:, 'rsi_6':'eom_20'])
    y_ = np.asarray(df_batch_test['labels'])
    y_test = one_hot_enc.transform(y_.reshape(-1, 1))

    log("stock train/test size", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # white_noise_check(["close_train", "close_test"], logger, df_batch["close"], df_batch_test["close"])
    return x_train, y_train, x_test, y_test


def get_synthetic_data():
    x, y = make_classification(n_samples=10000, n_features=200, n_informative=200, n_redundant=0, n_repeated=0,
                               n_classes=3, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0,
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
    pickle.dump(ga_history_dict, open(os.path.join(OUTPUT_PATH, run_id + '_history_' + start_time), 'wb'))
    pickle.dump(best_scores, open(os.path.join(OUTPUT_PATH, run_id + '_best_scores_' + start_time), 'wb'))
    pickle.dump(avg_scores, open(os.path.join(OUTPUT_PATH, run_id + '_avg_scores_' + start_time), 'wb'))
    log('history dumped...')


def func_eval_dummy(model, **kwargs):
    ga_history_list.append({})
    return np.random.uniform(0, 1, 2)


log("Running " + run_id)
# GA search
log("********************** GA Search **********************")
num_gen, score, param = GAEngine(search_space_mlp, mutation_probability=0.3, exit_check=exit_check,
                                 on_generation_end=on_generation_end, func_eval=func_eval,
                                 population_size=5, opt_mode=mode).run()

plot_iterable(best_scores=best_scores, avg_scores=avg_scores)
# plot_history(pickle.load(open(os.path.join('history_' + start_time), 'rb')))

# Random search
log("********************** Random Search **********************")
stime = time.time()
log("population size:", num_gen * 2)
best = GAEngine(search_space_mlp, exit_check=exit_check, on_generation_end=on_generation_end, func_eval=func_eval,
                population_size=num_gen * 2, opt_mode=mode).random_search()
log("Random search best individual: {}, {}".format(best.get_fitness_score(), best.get_nn_params()))
log("Random search time:", seconds_to_minutes(time.time()-stime))
log("Total duration:", seconds_to_minutes(time.time()-start_time_epoch))
# get_data_wisconsin()
