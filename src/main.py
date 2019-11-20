#
# Copyright (c) 2019. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from ga_engine import GAEngine
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

search_space_mlp = {
            'input_size': 784,
            'batch_size': [80, 100, 120],
            'layers': [
                {
                    'nodes_1': [200, 300, 500, 700, 900],
                    'do_1': [0.0, 0.1, 0.2, 0.3, 0.4],
                    'activation_1': ['relu', 'sigmoid']
                },
                {
                    'nodes_1': [200, 300, 500, 700, 900],
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
val_loss = 0.08
val_acc = 0.97


def get_data():
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # print(x_train[1][5])
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def func_eval(model, **kwargs):
    x_train, y_train, x_test, y_test = get_data()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, min_delta=0.0001)
    history = model.fit(x_train, y_train, batch_size=kwargs['batch_size'], epochs=500, verbose=1,
                        validation_split=0.3, callbacks=[es])
    val_error = np.amin(history.history['val_loss'])
    train_error = np.amin(history.history['loss'])
    print("errors:", train_error, val_error)
    score = model.evaluate(x_test, y_test, verbose=0)  # (loss, accuracy)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score[0], score[1]


def exit_check(best_score):
    return best_score <= val_loss


def on_generation_end(best_score, generation_count):
    print("on_generation_end: best_score=", best_score, "generation_count=", generation_count)


ga_engine_ = GAEngine(search_space_mlp, exit_check=exit_check, on_generation_end=on_generation_end, func_eval=func_eval)
