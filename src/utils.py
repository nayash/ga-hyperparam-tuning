import time
import numpy as np
from logger import Logger
from matplotlib import pyplot as plt

_logger = Logger('outputs', 'ga', 20)


def get_key_in_nested_dict(nested_dict, target_key):
    for key in nested_dict:
        if key == target_key:
            return nested_dict[key]
        elif type(nested_dict[key]) is dict:
            return get_key_in_nested_dict(nested_dict[key], target_key)
        elif type(nested_dict[key]) is list:
            if type(nested_dict[key][0]) is dict:
                for item in nested_dict[key]:
                    res = get_key_in_nested_dict(item, target_key)
                    if res:
                        return res


def choose_from_search_space(search_space_mlp, key=None, params={}):
    # print("choose_from_ss", '---', search_space_mlp, '---', key, '---', params)
    if type(search_space_mlp) is dict:
        keys = search_space_mlp.keys()
        for _key in keys:
            params = choose_from_search_space(search_space_mlp[_key], _key, params)
    elif type(search_space_mlp) is list:  # or type(search_space_mlp) is tuple:
        params = choose_from_search_space(search_space_mlp[np.random.randint(0, len(search_space_mlp))], key, params)
    else:
        if key:
            params[key] = search_space_mlp
        else:  # the search_space passed is not dict
            params = search_space_mlp  # result is not dict
    # print("choose_from_search: ",params)
    return params


def filter_list_by_prefix(_list, prefix, negation: bool = False):
    if negation:
        return [item for item in _list if not item.startswith(prefix)]
    else:
        return [item for item in _list if item.startswith(prefix)]


def log(*args):
    _logger.append_log(" ".join([str(arg) for arg in args]))


def get_mode_multiplier(mode):
    return -1 if mode == 'min' else 1


def seconds_to_minutes(seconds):
    return str(seconds // 60) + " minutes " + str(np.round(seconds % 60)) + " seconds"


def print_time(text, stime):
    seconds = (time.time() - stime)
    print(text, seconds_to_minutes(seconds))


def log_flush():
    _logger.flush()


def get_readable_ctime():
    return time.strftime("%d-%m-%Y %H_%M_%S")


def plot_iterable(**kwargs):
    for key in kwargs.keys():
        plt.plot(kwargs[key])
    plt.legend(list(kwargs.keys()))
    plt.show()


def smooth_coordinates(x, y):
    poly = np.polyfit(x, y, 5)
    y_ = np.poly1d(poly)(x)
    return x, y_


def plot_history(history):
    c1 = []
    c2 = []
    for key in list(history.keys())[1:]:
        gen_list = history[key]
        c1.append(gen_list[0]['accuracy'])
        c2.append(gen_list[1]['accuracy'])
    x = np.arange(0, len(c1))
    _, y1 = smooth_coordinates(x, c1)
    _, y2 = smooth_coordinates(x, c2)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.legend(['child1', 'child2'], loc=1)
    plt.show()


def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not (window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']):
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len:-window_len + 1]


def find_replace_index(num, list_nums):
    for idx, n in enumerate(list_nums):
        if num > n:
            return idx
    return -1
