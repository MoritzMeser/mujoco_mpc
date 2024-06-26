from enum import Enum, auto
import json
import pathlib
from dataclasses import dataclass, asdict, fields
from enum import Enum, auto
from typing import Optional, List, Union


class Robot(Enum):
    H1 = auto()
    G1 = auto()


class TaskName(Enum):
    Stand = (auto(), 1.0)
    Walk = (auto(), 1.0)
    Run = (auto(), 1.0)

    def __init__(self, _, max_reward):
        self._max_reward = max_reward

    @property
    def max_reward(self):
        return self._max_reward


class Planner(Enum):
    iLQG = auto()
    Sampling = auto()
    Derivative = auto()


class RewardFunction(Enum):
    hb = auto()
    ours = auto()
    ours_plus_hb = auto()


@dataclass
class Experiment:
    folder_path: pathlib.Path
    num_runs: int
    robot_name: Robot
    task_name: TaskName
    total_time: float
    reward_function: RewardFunction
    planner: Planner
    planner_iterations: int
    agent_horizon: float
    render_video: bool
    # fields that are set in evaluation
    rewards: Optional[List[float]] = None
    costs: Optional[List[float]] = None
    cost_names: Optional[List[str]] = None
    qpos: Optional[List[List[float]]] = None

    def to_dict(self):
        # Convert dataclass instance to dict, handling enums and pathlib.Path
        d = asdict(self)
        for field in fields(self):
            value = d[field.name]
            if isinstance(value, Enum):
                d[field.name] = value.name
            elif isinstance(value, pathlib.Path):
                d[field.name] = str(value)
        return d

    @classmethod
    def from_dict(cls, data):
        # Convert dict to dataclass instance, handling enums and pathlib.Path
        data['folder_path'] = pathlib.Path(data['folder_path'])
        for field in fields(cls):
            if field.type in [Robot, TaskName, RewardFunction, Planner] and isinstance(data[field.name], str):
                enum_class = field.type
                data[field.name] = enum_class[data[field.name]]
        return cls(**data)

    def save_to_json(self, file_path):
        # Serialize the object and save it as JSON
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_from_json(cls, file_path):
        # Load a JSON file and deserialize it into an Experiment object
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
