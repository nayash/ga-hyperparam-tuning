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
