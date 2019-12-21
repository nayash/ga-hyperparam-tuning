import pickle
import numpy as np
# from keras.utils import np_utils
# from sklearn.preprocessing import LabelEncoder

from utils import *
import os
import pandas as pd
import threading
from multiprocessing import Process
import time

print(os.getcwd())
print(get_readable_ctime())
def rand_sleep(t=None):
	time.sleep(np.random.randint(3, 5) if not t else t)
	print(get_readable_ctime())

def t1_func():
	for i in range(1000):
		print(i, ", t1")

def t2_func():
	for i in range(1000):
		print(i, ", t2")

t1 = Process(target=t1_func)
t1.start()

t2 = Process(target=t2_func)
t2.start()

t3 = threading.Thread(target=rand_sleep(2))
# t3.start()

#t1.join()
#t2.join()
#t3.join()