import json
import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from Enums import Experiment
from compare_results.analyze import get_data


def new_evaluation_method(folder_paths: List[pathlib.Path]):
    experiments = []
    for path in folder_paths:
        experiment = Experiment.load_from_json(path / "experiment_info.json")

        # sanity check -- all folder exist
        for i in range(experiment.num_runs):
            assert (path / f"experiment_{str(i).zfill(3)}").exists(), f"Folder {i} does not exist"

        # sanity check -- all log files exist and end with "Experiment finished"
        for i in range(experiment.num_runs):
            with open(path / f"experiment_{str(i).zfill(3)}/log.txt", "r") as f:
                lines = f.readlines()
                assert lines[-1].strip().split()[-1] == "Experiment_finished", f"Experiment {i} did not finish"

        # read log files and extract rewards and costs
        time_list = None
        experiment.rewards = []
        experiment.costs = []
        experiment.qpos = []
        for i in range(experiment.num_runs):
            with open(path / f"experiment_{str(i).zfill(3)}/config.json", 'r') as f:
                config = json.load(f)
                experiment.cost_terms = config['cost_terms']

            # load costs, rewards and qpos from log files
            costs = np.load(path / f"experiment_{str(i).zfill(3)}/costs_individual.npy")
            rewards = experiment.task_name.max_reward - costs[experiment.cost_terms.index('humanoid_bench'), :]
            qpso = np.load(path / f"experiment_{str(i).zfill(3)}/qpso.npy")

            # append to experiment
            experiment.costs.append(costs)
            experiment.rewards.append(rewards)
            experiment.qpos.append(qpso)

            # check if time arrays match
            if time_list is None:
                time_list = np.load(path / f"experiment_{str(i).zfill(3)}/time.npy")
            else:
                assert np.array_equal(time_list, np.load(
                    path / f"experiment_{str(i).zfill(3)}/time.npy")), "Time arrays do not match"

        # plot rewards
        plt.figure(figsize=(13, 6))  # Increase width to create space for text
        plt.axes([0.1, 0.1, 0.65, 0.8])  # Adjust axes to leave space on the right

        for i, r in enumerate(experiment.rewards):
            plt.plot(time_list[:-1], r, label=f"Run {i}")
        plt.xlabel("Time (s)")
        plt.ylabel("Reward")
        plt.ylim(0, experiment.task_name.max_reward + 0.1)
        plt.title("Humanoid Bench Reward over Time")
        # Adding additional information outside the plot on the right side
        labels_and_values = [
            ("Task:", experiment.task_name.name),
            ("Robot:", experiment.robot_name.name),
            ("Planner:", experiment.planner.name),
            ("Reward Function:", experiment.reward_function.name),
            ("Agent Horizon:", str(experiment.agent_horizon)),
            ("Planner Iterations:", str(experiment.planner_iterations)),
            ("Total Time:", str(experiment.total_time)),
            ("Number of Runs:", str(experiment.num_runs)),
            ("Render Video:", str(experiment.render_video)),
            ("Max Possible Reward:", str(experiment.task_name.max_reward))
        ]
        plt.figtext(0.78, 0.1, tabulate(labels_and_values, tablefmt="plain"), fontsize=12, verticalalignment='bottom',
                    bbox=dict(boxstyle="round", facecolor='white', alpha=1.0))
        plt.show()
        experiments.append(experiment)

    # make violin plot
    violin_data = [np.sum(exp.rewards, axis=1) for exp in experiments]
    violin_names = [f"{exp.reward_function.name}" for exp in experiments]

    # include also Algorithms from Humanoid Bench
    means, stds, algo_names, all_data = get_data(experiments[0].task_name.name.lower())
    violin_data += all_data
    violin_names += algo_names

    plt.figure(figsize=(10, 6))
    plt.axes([0.1, 0.2, 0.8, 0.7])  # Adjust axes to leave space on the right
    plt.violinplot(violin_data, showmeans=True, showextrema=True, showmedians=True)


    # Adding labels and title
    plt.xticks(np.arange(1, len(violin_names) + 1), violin_names, rotation=45, ha="right")
    plt.ylabel('Sum of Rewards')
    plt.title(f"Sum of Rewards for {experiments[0].task_name.name} Task")

    plt.show()
