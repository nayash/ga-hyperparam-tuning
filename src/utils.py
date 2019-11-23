import time
import numpy as np
from logger import Logger

_logger = Logger('.', 'ga', 10)


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


def choose_from_search_space(search_space_mlp: dict, key=None, params={}):
    if type(search_space_mlp) is dict:
        keys = search_space_mlp.keys()
        for key in keys:
            params = choose_from_search_space(search_space_mlp[key], key, params)
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
    return str(seconds//60) + " minutes " + str(np.round(seconds%60)) + " seconds"


def print_time(text, stime):
    seconds = (time.time()-stime)
    print(text, seconds_to_minutes(seconds))


def get_readable_ctime():
    return time.strftime("%d-%m-%Y %H_%M_%S")

