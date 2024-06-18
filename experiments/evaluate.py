import json
import os

import numpy as np
from matplotlib import pyplot as plt


def evaluate(path, n_experiments):
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    # ------ Sanity checks ------ #
    assert n_experiments == len(directories)
    for directory in directories:
        assert directory.startswith('experiment_'), f"Directory {directory} does not start with 'experiment_'"

    # Extract experiment numbers and sort them
    experiment_numbers = sorted([int(d.split('_')[-1]) for d in directories])

    # Check if all experiments are done and have ascending indices
    if experiment_numbers == list(range(min(experiment_numbers), max(experiment_numbers) + 1)):
        print("All experiments are done and have ascending indices.")
    else:
        print("Some experiments are missing or the indices are not in ascending order.")

    # ------Load Cost Terms ------ #
    # Load the config.json file from the first directory
    with open(f"{path}/{directories[0]}/config.json", 'r') as f:
        config = json.load(f)

    cost_terms = config['cost_terms']
    print(f"Cost Terms: {cost_terms}")
    total_time = config['total_time']
    print(f"Total Time: {total_time}")

    # ------ Load experiment data ------ #
    all_hb_rewards = []
    all_total_costs = []

    # Load experiment data
    for i, directory in enumerate(directories):
        print(f"Loading experiment {i} from {directory}")

        # ------ Humanoid Bench Reward ------ #
        with open(f"{path}/{directory}/config.json", 'r') as f:
            config = json.load(f)
            cost_terms = config['cost_terms']

        costs = np.load(f"{path}/{directory}/costs_individual.npy")

        humanoid_bench_index = cost_terms.index('humanoid_bench')
        humanoid_bench_costs = costs[humanoid_bench_index, :]
        humanoid_bench_reward = 1.0 - humanoid_bench_costs

        # Append the reward to the list
        all_hb_rewards.append(humanoid_bench_reward)

        # ------ Total Costs ------ #
        costs_total = np.load(f"{path}/{directory}/costs_total.npy")
        all_total_costs.append(costs_total)

    # ------ Plot the results ------ #

    # Convert the list to a numpy array for easier calculations
    all_hb_rewards = np.array(all_hb_rewards)
    all_total_costs = np.array(all_total_costs)

    # Calculate the average and standard deviation
    avg_reward = np.mean(all_hb_rewards, axis=0)
    std_reward = np.std(all_hb_rewards, axis=0)
    avg_total_costs = np.mean(all_total_costs, axis=0)
    std_total_costs = np.std(all_total_costs, axis=0)

    # Create a time array for the x-axis
    time = np.linspace(0, total_time, len(avg_reward))

    # Plot the average and standard deviation
    plt.figure()
    plt.plot(time, avg_reward, label='HB Reward', color='b')
    plt.fill_between(time, avg_reward - std_reward, avg_reward + std_reward, color='b', alpha=0.2,
                     label='Standard Deviation')
    plt.plot(time, avg_total_costs, label='Total Costs', color='r')
    plt.fill_between(time, avg_total_costs - std_total_costs, avg_total_costs + std_total_costs, color='r', alpha=0.2,
                     label='Standard Deviation')
    plt.xlabel("Time (s)")
    plt.ylabel("Reward/Costs")
    plt.title(f"{config['task_id']},{n_experiments} runs")
    plt.legend()

    # Set y-axis limits
    plt.ylim(0, 1)

    plt.savefig(f"{path}/humanoid_bench_reward.png")
    plt.show()


if __name__ == "__main__":
    evaluate("/Users/moritzmeser/Desktop/experiments/2024_06_18/Walk_H1_18_26_55", 8)
