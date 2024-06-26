import json

import matplotlib.pyplot as plt
import numpy as np


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


def get_data(task_name):
    with open('/Users/moritzmeser/lokal/Code_Local/MJPC/mujoco_mpc/experiments/compare_results/hb_original.json',
              'r') as f:
        data = json.load(f)
    data = data[task_name]

    algo_names = list(data.keys())
    seeds = list(data[algo_names[0]].keys())
    all_data = []
    mean_data = []
    std_data = []
    for algo_name in algo_names:
        algo_data = []
        for seed in seeds:
            algo_data.append(np.max(data[algo_name][seed]['return']))  # use max, as this is training over time
        all_data.append(algo_data)
        mean_data.append(np.mean(algo_data, axis=0))
        std_data.append(np.std(algo_data, axis=0))
    return mean_data, std_data, algo_names, all_data


if __name__ == "__main__":
    # list_fields('hb_original.json', depth=5)
    # with open('hb_original.json', 'r') as f:
    #     data = json.load(f)
    # x = data['push']['TD-MPC2']['seed_2']['million_steps']
    # y = data['push']['TD-MPC2']['seed_2']['return']
    # plt.plot(x, y)
    # plt.show()
    means, stds, algo_names = get_data('push')

    sum_of_rewards = means
    rewards_std = stds
    experiment_labels = algo_names

    # Create the bar plot
    x_pos = np.arange(len(experiment_labels))
    plt.figure(figsize=(10, 6))
    plt.bar(x_pos, sum_of_rewards, yerr=rewards_std, align='center', alpha=0.7, capsize=10, width=0.5)
    plt.xticks(x_pos, experiment_labels, rotation=45, ha="right")
    plt.ylabel('Sum of Rewards')
    plt.title(f"Sum of Rewards over Time for {'push'} Task")

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

