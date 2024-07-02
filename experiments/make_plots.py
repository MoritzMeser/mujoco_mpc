import json
import pathlib
from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import numpy as np
from tabulate import tabulate

from Enums import Experiment, Robot, TaskName
from compare_results.analyze import get_data
from datetime import timedelta
import re

def parse_time_string(time_string):
    """Parse a time string formatted as 'H:MM:SS.mmmmmm' into a timedelta object."""
    parts = re.split('[:.]', time_string)
    hours, minutes, seconds, microseconds = parts[0], parts[1], parts[2], parts[3]
    return timedelta(hours=int(hours), minutes=int(minutes), seconds=int(seconds), microseconds=int(microseconds))

def average_time(time_strings):
    """Calculate the average of a list of time strings."""
    total_duration = sum((parse_time_string(time) for time in time_strings), timedelta())
    average_duration = total_duration / len(time_strings)
    # Convert average_duration to desired string format if necessary
    total_seconds = int(average_duration.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    microseconds = average_duration.microseconds
    return f'{hours}:{minutes:02d}:{seconds:02d}.{microseconds}'


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
        experiment.qvel = []
        experiment.ctrl = []
        experiment.inference_times = []
        for i in range(experiment.num_runs):
            with open(path / f"experiment_{str(i).zfill(3)}/config.json", 'r') as f:
                config = json.load(f)
                experiment.cost_terms = config['cost_terms']
                experiment.inference_times.append(config['time']['duration'])

            # load costs, rewards and qpos from log files
            costs = np.load(path / f"experiment_{str(i).zfill(3)}/costs_individual.npy")
            rewards = experiment.task_name.max_reward - costs[experiment.cost_terms.index('humanoid_bench'), :]
            qpso = np.load(path / f"experiment_{str(i).zfill(3)}/qpso.npy")
            qvel = np.load(path / f"experiment_{str(i).zfill(3)}/qvel.npy")
            ctrl = np.load(path / f"experiment_{str(i).zfill(3)}/ctrl.npy")

            # append to experiment
            experiment.costs.append(costs)
            experiment.rewards.append(rewards)
            experiment.qpos.append(qpso)
            experiment.qvel.append(qvel)
            experiment.ctrl.append(ctrl)

            # check if time arrays match
            if time_list is None:
                time_list = np.load(path / f"experiment_{str(i).zfill(3)}/time.npy")
            else:
                assert np.array_equal(time_list, np.load(
                    path / f"experiment_{str(i).zfill(3)}/time.npy")), "Time arrays do not match"

        # # plot rewards
        # plt.figure(figsize=(13, 6))  # Increase width to create space for text
        # plt.axes([0.1, 0.1, 0.65, 0.8])  # Adjust axes to leave space on the right
        #
        # for i, r in enumerate(experiment.rewards):
        #     plt.plot(time_list[:-1], r, label=f"Run {i}")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Reward")
        # plt.ylim(0, experiment.task_name.max_reward + 0.1)
        # plt.title("Humanoid Bench Reward over Time")
        # # Adding additional information outside the plot on the right side
        # labels_and_values = [
        #     ("Task:", experiment.task_name.name),
        #     ("Robot:", experiment.robot_name.name),
        #     ("Planner:", experiment.planner.name),
        #     ("Reward Function:", experiment.reward_function.name),
        #     ("Agent Horizon:", str(experiment.agent_horizon)),
        #     ("Planner Iterations:", str(experiment.planner_iterations)),
        #     ("Total Time:", str(experiment.total_time)),
        #     ("Number of Runs:", str(experiment.num_runs)),
        #     ("Render Video:", str(experiment.render_video)),
        #     ("Max Possible Reward:", str(experiment.task_name.max_reward))
        # ]
        # plt.figtext(0.78, 0.1, tabulate(labels_and_values, tablefmt="plain"), fontsize=12, verticalalignment='bottom',
        #             bbox=dict(boxstyle="round", facecolor='white', alpha=1.0))
        # plt.show()
        experiments.append(experiment)

    # show plot of qpos[0] and qpos[1]
    # Initialize a color palette, one color for each experiment
    # colors = ['red', 'blue', 'green', 'orange', 'purple',
    #           'brown']  # Extend this list based on the number of experiments
    #
    # plt.figure(figsize=(10, 6))
    #
    # # Iterate through each experiment and plot qpos[0] vs qpos[1]
    # for i, exp in enumerate(experiments):
    #     qpos_0_all = [qpos[0, :] for qpos in exp.qpos]
    #     qpos_1_all = [qpos[1, :] for qpos in exp.qpos]
    #     # Flatten the lists
    #     qpos_0_all = [item for sublist in qpos_0_all for item in sublist]
    #     qpos_1_all = [item for sublist in qpos_1_all for item in sublist]
    #     plt.scatter(qpos_0_all, qpos_1_all, color=colors[i], label=f'{exp.reward_function.name}', s=2)
    #
    # # Customize the plot
    # plt.xlabel('x position')
    # plt.ylabel('y position')
    # plt.title('Scatter Plot of qpos[0] (x position) vs qpos[1] (y position)')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("/Users/moritzmeser/Desktop/qpos0_vs_qpos1.pdf")
    # plt.show()
    #
    # # plot qpos[0,:] vs time
    # plt.figure(figsize=(10, 6))
    # for i, exp in enumerate(experiments):
    #     for j in range(exp.num_runs):
    #         if j == 0:
    #             plt.plot(time_list, exp.qpos[j][1, :], color=colors[i], label=f'{exp.reward_function.name}')
    #         else:
    #             plt.plot(time_list, exp.qpos[j][1, :], color=colors[i])
    # plt.xlabel('Time (s)')
    # plt.ylabel('qpos[1]')
    # plt.title('qpos[1] vs Time for Different Experiments')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    # # plot all qvel [i,:] vs time
    #
    # # Create a figure with 3 subplots vertically aligned
    # fig, axs = plt.subplots(3, 1, figsize=(10, 18))
    #
    # # Determine the global y-axis limits
    # global_min = min([exp.qvel[j][k, :].min() for exp in experiments for j in range(exp.num_runs) for k in range(exp.qvel[j].shape[0])])
    # global_max = max([exp.qvel[j][k, :].max() for exp in experiments for j in range(exp.num_runs) for k in range(exp.qvel[j].shape[0])])
    #
    # # Iterate through each experiment to plot its qvel data
    # for i, exp in enumerate(experiments):
    #     for j in range(exp.num_runs):
    #         for k in range(exp.qvel[j].shape[0]):
    #             if j == 0 and k == 0:
    #                 axs[i].plot(time_list, exp.qvel[j][k, :], color=colors[i], label=f'{exp.reward_function.name}')
    #             else:
    #                 axs[i].plot(time_list, exp.qvel[j][k, :], color=colors[i])
    #     axs[i].set_xlabel('Time (s)')
    #     axs[i].set_ylabel('qvel')
    #     axs[i].set_title(f'qvel vs Time for {exp.task_name.name}')
    #     axs[i].legend()
    #     axs[i].grid(True)
    #     # Set the same y-axis limits for all subplots
    #     axs[i].set_ylim(global_min, global_max)
    #
    # # Adjust layout to prevent overlap
    # plt.tight_layout()
    # plt.savefig("/Users/moritzmeser/Desktop/qvel_vs_time.pdf")
    # # Display the plot
    # plt.show()
    #
    # # Create a figure with 3 subplots vertically aligned
    # fig, axs = plt.subplots(3, 1, figsize=(10, 18))
    #
    # # Determine the global y-axis limits
    # global_min = min([exp.ctrl[j][k, :].min() for exp in experiments for j in range(exp.num_runs) for k in range(exp.ctrl[j].shape[0])])
    # global_max = max([exp.ctrl[j][k, :].max() for exp in experiments for j in range(exp.num_runs) for k in range(exp.ctrl[j].shape[0])])
    #
    # # Iterate through each experiment to plot its ctrl data
    # for i, exp in enumerate(experiments):
    #     for j in range(exp.num_runs):
    #         for k in range(exp.ctrl[j].shape[0]):
    #             if j == 0 and k == 0:
    #                 axs[i].plot(time_list[:-1], exp.ctrl[j][k, :], color=colors[i], label=f'{exp.reward_function.name}')
    #             else:
    #                 axs[i].plot(time_list[:-1], exp.ctrl[j][k, :], color=colors[i])
    #     axs[i].set_xlabel('Time (s)')
    #     axs[i].set_ylabel('ctrl')
    #     axs[i].set_title(f'ctrl vs Time for {exp.task_name.name}')
    #     axs[i].legend()
    #     axs[i].grid(True)
    #     # Set the same y-axis limits for all subplots
    #     axs[i].set_ylim(global_min, global_max)
    #
    # # Adjust layout to prevent overlap
    # plt.tight_layout()
    # plt.savefig("/Users/moritzmeser/Desktop/ctrl_vs_time.pdf")
    # # Display the plot
    # plt.show()
    #
    # order experiments by reward function

    # print("Task: ", experiments[0].task_name.name)
    # for exp in experiments:
    #     qvel = np.array(exp.qvel)
    #     # print("before qvel shape: ", qvel.shape)
    #     # remove the first 6 elements, as they are belong to the free base joint, not an actual joint
    #     qvel = qvel[:, 6:, :]
    #     if exp.task_name == TaskName.Push:  # the last 6 entries belong to the free base joint of the box
    #         qvel = qvel[:, :19, :]
    #
    #     # print("after qvel shape: ", qvel.shape)
    #     # compute v_dot^2
    #     v_dot = np.diff(qvel, axis=2)
    #     v_dot = np.linalg.norm(v_dot, axis=1)
    #     v_dot = np.mean(v_dot, axis=1)
    #     v_dot = np.mean(v_dot, axis=0)
    #     print(f"v_dot^2 for {exp.reward_function.name}: {v_dot}")


    filtered_experiments = []
    for exp in experiments:
        if exp.reward_function.name == 'ours_plus_hb':
            filtered_experiments.append(exp)
    for exp in experiments:
        if exp.reward_function.name == 'hb':
            filtered_experiments.append(exp)
    experiments = filtered_experiments

    # report inference time
    for exp in experiments:
        print(f"Average Inference Time for {exp.task_name}, Planner {exp.planner.name}:, Simulated Time: {exp.total_time}, Planner Iterations: {exp.planner_iterations}, Agent Horizon: {exp.agent_horizon} Number of Runs: {exp.num_runs}, Reward Function: {exp.reward_function.name}")
        parsed_times = [parse_time_string(time) for time in exp.inference_times]
        average_time_string = average_time([str(time) for time in parsed_times])
        print(f"Average Inference Time: {average_time_string}")





    # make one plot, showing mean and std of rewards
    plt.figure(figsize=(11, 6))  # Increase width to create space for text
    plt.axes([0.05, 0.08, 0.69, 0.84])  # Adjust axes to leave space on the right
    colors = ['b', 'g']
    for i, exp in enumerate(experiments):
        median = np.median(exp.rewards, axis=0)
        mean = np.mean(exp.rewards, axis=0)
        std = np.std(exp.rewards, axis=0)
        label = exp.reward_function.name
        if label == 'ours_plus_hb':
            label = 'ours'
        plt.plot(time_list[:-1], mean, label=label, color=colors[i])
        plt.fill_between(time_list[:-1], mean - std, mean + std, alpha=0.3, color=colors[i])
        plt.plot(time_list[:-1], median, linestyle='--', color=colors[i])
    plt.xlabel("Time (s)")
    plt.ylabel("Reward")
    plt.ylim(0, experiment.task_name.max_reward + 0.1)
    plt.title("Humanoid Bench Reward over Time Walk Task")

    # add a vertical line at 2s
    plt.axvline(x=2, color='black', linestyle=':', label='2s', linewidth=3)

    # Define colors for each reward function
    colors = {'ours': 'blue', 'hb': 'green'}

    # Create custom legend handles for each reward function
    legend_handles = []
    for label, color in colors.items():
        legend_handles.append(mlines.Line2D([], [], color=color, label=f'{label} Mean'))
        legend_handles.append(mlines.Line2D([], [], color=color, linestyle='--', label=f'{label} Median'))
        legend_handles.append(mpatches.Patch(color=color, alpha=0.3, label=f'{label} Standard Error'))

    # Add custom legend to the plot
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.005, 1), fontsize=12)
    plt.savefig("/Users/moritzmeser/Desktop/reward_over_time_walk.pdf")

    plt.show()

    # limit rewards to 1000 first steps, to make it comparable to Humanoid Bench
    for exp in experiments:
        exp.rewards = np.array([r[:1000] for r in exp.rewards])
    if experiments[0].task_name == TaskName.Push:
        for exp in experiments:
            exp.rewards = np.array([r[:500] for r in exp.rewards])
            for i, run in enumerate(exp.rewards):
                for j, r in enumerate(run):
                    if r > 0:
                        run[j + 1:] = [0] * (len(run) - (j + 1))  # Set everything after index j to zero
                        break
                exp.rewards[i] = run

    # make violin plot
    violin_data = [np.sum(exp.rewards, axis=1) for exp in experiments]
    violin_names = [f"{exp.reward_function.name}" for exp in experiments]
    violin_names = ['ours' if v == 'ours_plus_hb' else v for v in violin_names]
    violin_names = [f'MPC\n{s}' for s in violin_names]

    # include also Algorithms from Humanoid Bench
    if experiments[0].robot_name == Robot.H1:
        means, stds, algo_names, all_data = get_data(experiments[0].task_name.name.lower())
        violin_data += all_data
        violin_names += algo_names

    # plt.figure(figsize=(5, 6))
    # # plt.axes([0.1, 0.2, 0.8, 0.7])  # Adjust axes to leave space on the right
    # plt.violinplot(violin_data, showmeans=True, showextrema=True, showmedians=True)
    #
    # # Adding labels and title
    # plt.xticks(np.arange(1, len(violin_names) + 1), violin_names, rotation=45, ha="right")
    # plt.ylabel('Sum of Rewards')
    # plt.title(f"Sum of Rewards for {experiments[0].task_name.name} Task with Robot {experiments[0].robot_name.name}")
    #
    # if not experiments[0].task_name == TaskName.Push:
    #     plt.ylim(0, 1000)
    #
    # #  crop figure bevor saving
    # plt.tight_layout()
    # # plt.savefig(f"/Users/moritzmeser/Desktop/sum_of_rewards_{experiments[0].task_name.name}_{experiments[0].robot_name.name}.pdf")
    # # plt.show()

    return violin_data, violin_names
