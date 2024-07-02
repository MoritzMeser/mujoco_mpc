import pathlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from time import sleep

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from Enums import Experiment, Robot, TaskName, RewardFunction, Planner
import parse_xml
from run_experiment import run_single_experiment
from make_plots import new_evaluation_method
from util import util


def set_color(violin_parts, color):
    for pc in violin_parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
    for key in violin_parts.keys():
        if not key == 'bodies':
            violin_parts[key].set_color(color)


def main():
    paths = []
    # for reward_fuction in [RewardFunction.ours, RewardFunction.hb, RewardFunction.ours_plus_hb]:
    for reward_fuction in [RewardFunction.ours, RewardFunction.ours_plus_hb, RewardFunction.hb]:
        home_path = pathlib.Path("/Users/moritzmeser/Desktop/Walk_G1/")
        curr_date = datetime.today().strftime("%Y_%m_%d")
        curr_time = datetime.today().strftime("%H_%M_%S")
        path = home_path / curr_date / curr_time

        experiment = Experiment(
            folder_path=path,
            num_runs=6,
            robot_name=Robot.G1,
            task_name=TaskName.Walk,
            total_time=2.0,
            reward_function=reward_fuction,
            planner=Planner.iLQG,
            agent_horizon=1.0,  # 0.35
            planner_iterations=5,  # 10
            render_video=True,
        )

        path.mkdir(parents=True, exist_ok=True)
        experiment.save_to_json(path / "experiment_info.json")

        # set planner and agent_horizon in xml file
        task_id = experiment.task_name.name + " " + experiment.robot_name.name
        xml_path = util.model_path_from_id(task_id)
        parse_xml.set_planner(xml_path, experiment.planner)
        parse_xml.set_agent_horizon(xml_path, experiment.agent_horizon)

        with ThreadPoolExecutor(max_workers=3) as executor:
            for i in range(experiment.num_runs):
                future = executor.submit(run_single_experiment, experiment, i, False)
                print(f"Experiment {i} started")
                future.result()
                print(f"Experiment {i} finished")
                sleep(3)  # sleep for 3 seconds to avoid problems with the initialization of the server
        print("folder path: ", path)
        paths.append(path)
    print("Paths: ", paths)
    new_evaluation_method(paths)


if __name__ == "__main__":
    # main()
    # Walk H1
    paths = [pathlib.Path("/Users/moritzmeser/lokal/Code_Local/data/morning_run/2024_06_27/10_19_40"),
             pathlib.Path("/Users/moritzmeser/lokal/Code_Local/data/morning_run/2024_06_27/08_13_29"),
             pathlib.Path("/Users/moritzmeser/lokal/Code_Local/data/morning_run/2024_06_27/12_54_04")]
    walk_data, walk_names = new_evaluation_method(paths)
    #
    # Stand H1
    paths = [pathlib.Path("/Users/moritzmeser/lokal/Code_Local/data/Nachmittag/2024_06_27/16_52_17"),
             pathlib.Path("/Users/moritzmeser/lokal/Code_Local/data/Nachmittag/2024_06_27/16_57_56"),
             pathlib.Path("/Users/moritzmeser/lokal/Code_Local/data/Nachmittag/2024_06_27/17_03_38")]
    stand_data, stand_names = new_evaluation_method(paths)
    #
    # # Stand G1
    # paths = [pathlib.Path("/Users/moritzmeser/Desktop/Stand_G1/2024_06_27/18_12_28"),
    #          pathlib.Path("/Users/moritzmeser/Desktop/Stand_G1/2024_06_27/18_31_01"),
    #          pathlib.Path("/Users/moritzmeser/Desktop/Stand_G1/2024_06_27/19_18_20")]
    # new_evaluation_method(paths)
    #
    # # Push H1
    paths = [pathlib.Path("/Users/moritzmeser/lokal/Code_Local/data/Push_H1/2024_06_27/21_03_46"),
             pathlib.Path("/Users/moritzmeser/lokal/Code_Local/data/Push_H1/2024_06_27/20_58_27"),
             pathlib.Path("/Users/moritzmeser/lokal/Code_Local/data/Push_H1/2024_06_27/20_33_01")]
    push_data, push_names = new_evaluation_method(paths)

    # # Walk G1
    # paths = [pathlib.Path("/Users/moritzmeser/Desktop/Walk_G1/2024_06_28/11_28_37")]
    # new_evaluation_method(paths)

    # plt.figure(figsize=(6, 7))
    # plt.axes([0.14, 0.12, 0.68, 0.879])
    #
    #
    # violin_parts = plt.violinplot(stand_data, showmeans=True, showextrema=True, showmedians=True)
    # set_color(violin_parts, 'red')
    #
    # violin_parts = plt.violinplot(walk_data, showmeans=True, showextrema=True, showmedians=True)
    # set_color(violin_parts, 'blue')
    #
    # violin_parts = plt.violinplot(push_data, showmeans=True, showextrema=True, showmedians=True)
    # set_color(violin_parts, 'green')
    #
    # # Create custom legend handles
    # legend_handles = [Patch(facecolor='red', edgecolor='black', label='Stand\nTask'),
    #                   Patch(facecolor='blue', edgecolor='black', label='Walk\nTask'),
    #                   Patch(facecolor='green', edgecolor='black', label='Push\nTask')]
    #
    # # Add the custom legend to the plot, positioning it outside the top right corner
    # plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1,1))
    #
    # plt.xticks(np.arange(1, len(push_names) + 1), push_names, rotation=45, ha="right")
    # plt.ylabel('Sum of Rewards')
    # plt.savefig("/Users/moritzmeser/Desktop/stand_walk_push.pdf")
    # plt.show()
