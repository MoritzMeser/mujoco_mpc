import json
import os

import numpy as np
from matplotlib import pyplot as plt

from compare_results.analyze import get_data


def evaluate_experiments(experiments):
    time_list = None

    for experiment in experiments:
        directories = [d for d in os.listdir(experiment['path']) if os.path.isdir(os.path.join(experiment['path'], d))]
        experiment['n_runs'] = len(directories)

        # open config file and read task_id
        with open(f"{experiment['path']}/{directories[0]}/config.json", 'r') as f:
            config = json.load(f)
        experiment['task_id'] = config['task_id']
        experiment['reward_function'] = config['reward_function']
        experiment['hb_rewards'] = []
        experiment['costs_total'] = []
        experiment['goal dist'] = []
        experiment['qpos_2'] = []
        experiment['qpos_0'] = []

        # Load experiment data
        for i, directory in enumerate(directories):
            print(f"Loading experiment {i} from {directory}")

            if time_list is None:
                time_list = np.load(f"{experiment['path']}/{directory}/time.npy")[:-1]

            # ------ Humanoid Bench Reward ------ #
            costs = np.load(f"{experiment['path']}/{directory}/costs_individual.npy")
            with open(f"{experiment['path']}/{directory}/config.json", 'r') as f:
                config = json.load(f)
            cost_terms = config['cost_terms']
            humanoid_bench_index = cost_terms.index('humanoid_bench')
            humanoid_bench_costs = costs[humanoid_bench_index, :]
            humanoid_bench_reward = 1000.0 - humanoid_bench_costs

            try:
                last_index = np.where(humanoid_bench_reward > 900)[0][0] + 1
            except:
                last_index = 500
            last_index = min(last_index, 500)

            humanoid_bench_reward[last_index:] = 0
            experiment['hb_rewards'].append(humanoid_bench_reward)

            # ------ Goal Distance ------ #
            goal_dist = costs[cost_terms.index('goal dist'), :]
            experiment['goal dist'].append(goal_dist)

            # ------ Total Costs ------ #
            costs_total = np.load(f"{experiment['path']}/{directory}/costs_total.npy")
            experiment['costs_total'].append(costs_total)

            # ------ qpos ------ #
            qpos = np.load(f"{experiment['path']}/{directory}/qpso.npy")
            experiment['qpos_2'].append(qpos[2, :][:-1])  # qpos 2 is the height of the robot
            experiment['qpos_0'].append(qpos[0, :][:-1])  # qpos 0 is the x position of the robot

    for experiment in experiments:
        for i in range(len(experiment['goal dist'])):
            plt.plot(time_list, experiment['goal dist'][i], label=experiment['reward_function'])
        plt.title(experiment['reward_function'])
        plt.show()
    # Initialize figure
    plt.figure(figsize=(10, 20))

    # Subplot 1: Humanoid Bench Reward
    plt.subplot(4, 1, 1)
    for experiment in experiments:
        hb_avg = np.mean(experiment['hb_rewards'], axis=0)
        hb_std = np.std(experiment['hb_rewards'], axis=0)
        plt.plot(time_list, hb_avg, label=experiment['reward_function'])
        plt.fill_between(time_list, hb_avg - hb_std, hb_avg + hb_std, alpha=0.2)
    plt.title("Humanoid Bench Reward" + f" ({experiment['task_id']}  {experiment['title']})")
    plt.ylabel("HB Reward")
    plt.xlabel("Time (s)")
    # plt.ylim(0, 1.1)
    plt.legend()

    # Subplot 2: Total Costs
    plt.subplot(4, 1, 2)
    for experiment in experiments:
        cost_avg = np.mean(experiment['costs_total'], axis=0)
        cost_std = np.std(experiment['costs_total'], axis=0)
        plt.plot(time_list, cost_avg, label=experiment['reward_function'])
        plt.fill_between(time_list, cost_avg - cost_std, cost_avg + cost_std, alpha=0.2)
    plt.title("Total Costs" + f" ({experiment['task_id']}  {experiment['title']})")
    plt.ylabel("Costs")
    plt.xlabel("Time (s)")
    plt.legend()

    # Subplot 3: qpos[2]
    plt.subplot(4, 1, 3)
    for i, experiment in enumerate(experiments):
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        first_line = True
        for qpos2 in experiment['qpos_2']:
            if first_line:
                plt.plot(time_list, qpos2, color=color, label=experiment['reward_function'])
                first_line = False
            else:
                plt.plot(time_list, qpos2, color=color)
    plt.axhline(y=0.2, color='black', linestyle='-')
    plt.text(time_list[-1] * 0.15, 0.2, 'threshold', verticalalignment='bottom', horizontalalignment='right',
             color='black', fontsize=10)
    plt.xlabel("Time (s)")
    plt.ylabel("qpos[2]")
    plt.xlabel("Time (s)")
    plt.title("qpos[2] over time" + f" ({experiment['task_id']}  {experiment['title']})")
    plt.legend()

    # Subplot 4: qpos[0]
    plt.subplot(4, 1, 4)
    for i, experiment in enumerate(experiments):
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        first_line = True
        for qpos0 in experiment['qpos_0']:
            if first_line:
                plt.plot(time_list, qpos0, color=color, label=experiment['reward_function'])
                first_line = False
            else:
                plt.plot(time_list, qpos0, color=color)

    plt.ylabel("qpos[0]")
    plt.xlabel("Time (s)")
    plt.title("qpos[0] over time" + f" ({experiment['task_id']}  {experiment['title']})")
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save or display the figure
    # plt.savefig(f"/Users/moritzmeser/Desktop/{experiment['task_id']}.pdf")
    plt.show()

    # Assuming 'experiments' is already defined and contains all necessary data
    sum_of_rewards = []
    rewards_std = []
    experiment_labels = []

    all_rewards = []
    # my experiments
    for experiment in experiments:
        # Calculate the sum of rewards for each experiment
        total_reward = np.sum(experiment['hb_rewards'], axis=1)  # Sum over time for each run
        sum_of_rewards.append(np.mean(total_reward))  # Average over all runs
        rewards_std.append(np.std(total_reward))  # Standard deviation over all runs
        experiment_labels.append(
            f"{experiment['title']} \n {experiment['reward_function']}")  # Assuming each experiment has a 'title'
        all_rewards.append(total_reward)

    # include also the original experiments
    means, stds, algo_names, all_data = get_data(experiments[0]['task_id'].split()[0].lower())
    sum_of_rewards.extend(means)
    rewards_std.extend(stds)
    experiment_labels.extend(algo_names)
    all_rewards.extend(all_data)

    # # Create the bar plot
    # x_pos = np.arange(len(experiment_labels))
    # plt.figure(figsize=(10, 6))
    # plt.bar(x_pos, sum_of_rewards, yerr=rewards_std, align='center', alpha=0.7, capsize=10, width=0.5)
    # plt.xticks(x_pos, experiment_labels, rotation=45, ha="right")
    # plt.ylabel('Sum of Rewards')
    # plt.title(f"Sum of Rewards over Time for {experiment['task_id']} Task")

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig(f"/Users/moritzmeser/Desktop/{experiment['task_id']}_sum_of_rewards.pdf")
    plt.show()

    # Prepare data for violin plot
    data_for_violin = [np.random.normal(loc=reward, scale=std, size=100) for reward, std in
                       zip(sum_of_rewards, rewards_std)]

    # Create the violin plot
    plt.figure(figsize=(10, 6))
    violin_parts = plt.violinplot(all_rewards, showmeans=True, showmedians=True)

    # # Customizing the violin plot
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    #     vp = violin_parts[partname]
    #     vp.set_edgecolor('black')
    #     vp.set_linewidth(1)

    # Adding labels and title
    plt.xticks(np.arange(1, len(experiment_labels) + 1), experiment_labels, rotation=45, ha="right")
    plt.ylabel('Sum of Rewards')
    plt.title(f"Sum of Rewards over Time for {experiment['task_id']} Task")

    plt.tight_layout()
    plt.show()


def main():
    experiments = [
        # {'path': '/Users/moritzmeser/Desktop/experiments/2024_06_20/Walk_H1_12_15_45', 'title': 'iLQG'},
        # {'path': '/Users/moritzmeser/Desktop/old_experiments/2024_06_20/Walk_H1_09_31_38'},
        # {'path': '/Users/moritzmeser/Desktop/results/iLQG/stand_experiments/2024_06_20/Stand_H1_14_04_46',
        #  'title': 'iLQG'},
        # {'path': '/Users/moritzmeser/Desktop/results/iLQG/stand_experiments/2024_06_20/Stand_H1_14_22_25',
        #  'title': 'iLQG'},
        # {'path': '/Users/moritzmeser/Desktop/results/iLQG/stand_experiments/2024_06_20/Stand_H1_14_56_08',
        #  'title': 'iLQG'},
        # {'path': '/Users/moritzmeser/Desktop/results/iLQG/walk_experiments/2024_06_20/Walk_H1_16_51_44', 'title': 'iLQG'},
        # {'path': '/Users/moritzmeser/Desktop/results/iLQG/walk_experiments/2024_06_20/Walk_H1_17_11_47', 'title': 'iLQG'},
        # {'path': '/Users/moritzmeser/Desktop/results/iLQG/walk_experiments/2024_06_20/Walk_H1_17_34_05', 'title': 'iLQG'},
        # {'path': '/Users/moritzmeser/Desktop/iLQG/run_experiments/2024_06_20/Run_H1_18_37_52', 'title': 'iLQG'},
        # {'path': '/Users/moritzmeser/Desktop/iLQG/run_experiments/2024_06_20/Run_H1_18_59_25', 'title': 'iLQG'},
        # {'path': '/Users/moritzmeser/Desktop/iLQG/run_experiments/2024_06_20/Run_H1_19_19_06', 'title': 'iLQG'},
        # {'path': '/Users/moritzmeser/Desktop/results/walk_sampling/2024_06_21/Walk_H1_10_40_59', 'title': 'Sampling'},
        # {'path': '/Users/moritzmeser/Desktop/results/walk_sampling/2024_06_21/Walk_H1_10_44_10', 'title': 'Sampling'},
        # {'path': '/Users/moritzmeser/Desktop/results/walk_sampling/2024_06_21/Walk_H1_10_47_26', 'title': 'Sampling'},

        # {'path': '/Users/moritzmeser/Desktop/run_sampling/2024_06_21/Run_H1_11_42_57', 'title': 'Sampling'},
        # {'path': '/Users/moritzmeser/Desktop/run_sampling/2024_06_21/Run_H1_11_46_07', 'title': 'Sampling'},
        # {'path': '/Users/moritzmeser/Desktop/run_sampling/2024_06_21/Run_H1_11_49_33', 'title': 'Sampling'},

        # {'path': '/Users/moritzmeser/Desktop/results/run_sampling/2024_06_21/Run_H1_12_41_11', 'title': 'Sampling'},
        # {'path': '/Users/moritzmeser/Desktop/results/run_sampling/2024_06_21/Run_H1_12_44_35', 'title': 'Sampling'},
        # {'path': '/Users/moritzmeser/Desktop/results/run_sampling/2024_06_21/Run_H1_12_48_14', 'title': 'Sampling'},

        # {'path': '/Users/moritzmeser/Desktop/push_sampling/2024_06_22/Push_H1_18_44_21', 'title': 'Sampling'},
        # {'path': '/Users/moritzmeser/Desktop/push_sampling/2024_06_22/Push_H1_19_06_33', 'title': 'Sampling'},
        # {'path': '/Users/moritzmeser/Desktop/push_sampling/2024_06_22/Push_H1_19_29_50', 'title': 'Sampling'},
        {'path': '/Users/moritzmeser/Desktop/push_sampling/2024_06_22/Push_H1_20_11_26', 'title': 'Sampling'},
        {'path': '/Users/moritzmeser/Desktop/push_sampling/2024_06_22/Push_H1_20_22_29', 'title': 'Sampling'},
        {'path': '/Users/moritzmeser/Desktop/push_sampling/2024_06_22/Push_H1_20_33_56', 'title': 'Sampling'},
        {'path': '/Users/moritzmeser/Desktop/push_sampling/2024_06_22/Push_H1_20_46_37', 'title': 'Sampling'},
    ]
    evaluate_experiments(experiments)


if __name__ == "__main__":
    main()
