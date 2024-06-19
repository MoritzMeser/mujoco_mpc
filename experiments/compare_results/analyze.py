import json

import matplotlib.pyplot as plt


def list_fields(json_file_path, depth=1, current_depth=0, key_path=''):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    for key, value in data.items():
        new_key_path = f"{key_path}.{key}" if key_path else key
        print(new_key_path)
        if isinstance(value, dict) and current_depth < depth:
            list_fields_from_dict(value, depth, current_depth + 1, new_key_path)


def list_fields_from_dict(data, depth, current_depth, key_path):
    for key, value in data.items():
        new_key_path = f"{key_path}.{key}"
        print(new_key_path)
        if isinstance(value, dict) and current_depth < depth:
            list_fields_from_dict(value, depth, current_depth + 1, new_key_path)


if __name__ == "__main__":
    list_fields('hb_original.json', depth=5)
    with open('hb_original.json', 'r') as f:
        data = json.load(f)
    x = data['walk']['TD-MPC2']['seed_2']['million_steps']
    y = data['walk']['TD-MPC2']['seed_2']['return']
    plt.plot(x, y)
    plt.show()
