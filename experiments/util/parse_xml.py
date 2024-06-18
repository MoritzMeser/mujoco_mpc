import xml.etree.ElementTree as ET


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
