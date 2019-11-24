import pickle
import numpy as np
from utils import *
import os

print(os.getcwd())
history = pickle.load(open('history_24-11-2019 18_48_45', 'rb'))
plot_history(history)

