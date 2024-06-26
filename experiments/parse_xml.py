import pathlib
import xml.etree.ElementTree as ET
from Enums import Planner


def get_home_qpos(path):
    # Parse the XML file
    tree = ET.parse(path)
    root = tree.getroot()

    # Find the 'keyframe' element
    keyframe = root.find('keyframe')

    # Find the 'key' element with name='home'
    home_key = next(key for key in keyframe.findall('key') if key.get('name') == 'home')

    # Get the 'qpos' attribute and split it into a list of strings
    qpos_str_list = home_key.get('qpos').split()

    # Convert the list of strings to a list of floats
    qpos = [float(val) for val in qpos_str_list]

    return qpos


def set_planner(path: pathlib.Path, planner: Planner):
    tree = ET.parse(path)
    root = tree.getroot()

    # Find the <custom> element
    custom = root.find('custom')

    # Find the <numeric> element for the planner and update its 'data' attribute
    for numeric in custom.findall('numeric'):
        if numeric.get('name') == 'agent_planner':
            numeric.set('data', str(planner_to_numeric(planner)))
            break

    # Save the modified XML back to the file
    tree.write(path)


def set_agent_horizon(path: pathlib.Path, agent_horizon: float):
    tree = ET.parse(path)
    root = tree.getroot()

    # Find the <custom> element
    custom = root.find('custom')

    # Find the <numeric> element for the agent horizon and update its 'data' attribute
    for numeric in custom.findall('numeric'):
        if numeric.get('name') == 'agent_horizon':
            numeric.set('data', str(agent_horizon))
            break

    # Save the modified XML back to the file
    tree.write(path)


def planner_to_numeric(planner):
    match planner:
        case Planner.Sampling:
            return 0
        case Planner.iLQG:
            return 2
        case Planner.Derivative:
            return 1
        case _:
            return None  # Default case if planner does not match any case above
