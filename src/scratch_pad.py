import pickle
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from utils import *
import os
import pandas as pd

print(os.getcwd())
df = pd.read_csv('inputs\magic04.data', names=['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym',
                                                               'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class'])
print(df.head())
print(df.iloc[:, 0:10].columns)
print(df.isna().any().any())
x = df.iloc[:, 0:10].values
y = df['class'].values
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
y = np_utils.to_categorical(encoded_y)
print(y[0:10])

