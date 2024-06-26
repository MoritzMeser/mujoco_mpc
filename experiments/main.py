import pathlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from time import sleep

from Enums import Experiment, Robot, TaskName, RewardFunction, Planner
import parse_xml
from run_experiment import run_single_experiment
from make_plots import new_evaluation_method
from util import util


def main():
    home_path = pathlib.Path("/Users/moritzmeser/Desktop/Stand_Task")
    curr_date = datetime.today().strftime("%Y_%m_%d")
    curr_time = datetime.today().strftime("%H_%M_%S")
    path = home_path / curr_date / curr_time

    experiment = Experiment(
        folder_path=path,
        num_runs=4,
        robot_name=Robot.G1,
        task_name=TaskName.Run,
        total_time=2.0,
        reward_function=RewardFunction.ours,
        planner=Planner.Sampling,
        agent_horizon=0.35,
        planner_iterations=2,  # 10
        render_video=True,
    )

    path.mkdir(parents=True, exist_ok=True)
    experiment.save_to_json(path / "experiment_info.json")

    # set planner and agent_horizon in xml file
    task_id = experiment.task_name.name + " " + experiment.robot_name.name
    xml_path = util.model_path_from_id(task_id)
    parse_xml.set_planner(xml_path, experiment.planner)
    parse_xml.set_agent_horizon(xml_path, experiment.agent_horizon)

    with ThreadPoolExecutor(max_workers=4) as executor:
        for i in range(experiment.num_runs):
            future = executor.submit(run_single_experiment, experiment, i, False)
            print(f"Experiment {i} started")

            sleep(1)  # sleep for 1 second to avoid problems with the initialization of the server
        future.result()

    print("folder path: ", path)
    new_evaluation_method([path])


if __name__ == "__main__":
    main()
    # new_evaluation_method([pathlib.Path("/Users/moritzmeser/Desktop/Stand_Task/2024_06_26/20_58_48")])
