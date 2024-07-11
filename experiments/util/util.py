import pathlib


def model_path_from_id(task_id):
    basic_locomotion = {"Walk", "Stand", "Run", "Stairs"}

    # Split the task_id into task and robot
    task, robot = task_id.split()

    # Determine the path based on whether the task is in basic_locomotion
    if task in basic_locomotion:
        model_path = (
                pathlib.Path(__file__).parent.parent.parent
                / f"mjpc/tasks/humanoid_bench/basic_locomotion/{task.lower()}/{task}_{robot}.xml"
        )
    else:
        model_path = (
                pathlib.Path(__file__).parent.parent.parent
                / f"mjpc/tasks/humanoid_bench/{task.lower()}/{task}_{robot}.xml"
        )

    return model_path
