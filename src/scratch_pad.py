import pickle
import numpy as np
# from keras.utils import np_utils
# from sklearn.preprocessing import LabelEncoder

from utils import get_readable_ctime
import os
import pandas as pd
import threading
import multiprocessing as mp
import time

def rand_sleep(t=None):
	time.sleep(np.random.randint(3, 5) if not t else t)
	print(get_readable_ctime())

def t1_func():
	for i in range(1000):
		print(i, ", t1")

def t2_func():
	for i in range(1000):
		print(i, ", t2")

if __name__ == '__main__':
	print(os.getcwd())
	print(get_readable_ctime(), mp.cpu_count())
	pool = mp.Pool(mp.cpu_count())
	pool.apply(rand_sleep, args=(3, ))
	pool.apply(rand_sleep, args=(5, ))

# t3 = threading.Thread(target=rand_sleep(2))
# t3.start()

#t1.join()
#t2.join()
#t3.join()