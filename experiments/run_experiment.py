import json
from datetime import datetime
import platform
import concurrent.futures
import logging

from util import parse_xml, util

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pathlib

# set current directory: mujoco_mpc/python/mujoco_mpc
from mujoco_mpc import agent as agent_lib


def run_single_experiment(config):
    start_time = datetime.now()

    render_video = config['render_video']
    make_plots = config['make_plots']
    log_costs = config['log_costs']
    sim_time_step = config['sim_time_step']
    total_time = config['total_time']
    planer_iterations = config['planer_iterations']
    task_id = config['task_id']
    reward_function = config['reward_function']
    path = config['path']
    folder = config['folder']

    # check if the render_fps is a valid value
    if render_video:
        render_fps = 25
        sim_steps_per_frame = 1 / (render_fps * sim_time_step)
        if not (int(sim_steps_per_frame) == sim_steps_per_frame):
            raise ValueError("The choose another render_fps value.")

    # ------ Set up directory ------ #
    today_str = datetime.today().strftime("%Y_%m_%d")
    time_str = datetime.today().strftime("%H_%M_%S")

    # make directory for the results
    if folder is None:
        results_dir = pathlib.Path(path) / 'results' / today_str / time_str
    else:
        results_dir = pathlib.Path(path) / folder
    results_dir.mkdir(exist_ok=True, parents=True)

    # ------ Set up logger ------ #
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(results_dir / 'log.txt')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # ------ Load model ------ #
    model_path = util.model_path_from_id(task_id)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = sim_time_step
    logger.info(f"Model loaded from {model_path}")

    # data
    data = mujoco.MjData(model)

    # ------ Set up renderer ------ #
    if render_video:
        # set resolution to full HD
        model.vis.global_.offwidth = 1920
        model.vis.global_.offheight = 1080
        renderer = mujoco.Renderer(model, height=1080, width=1920)
        logger.info("Renderer initialized")

    # ------ Set up agent ------ #
    agent = agent_lib.Agent(task_id=task_id, model=model)
    logger.info("Agent initialized")

    # ------ Set cost weights ------ #
    if reward_function == 'humanoid_bench':
        # set all costs to zero
        agent.set_cost_weights({name: 0 for name in agent.get_cost_weights().keys()})
        # set humanoid bench to 1
        agent.set_cost_weights({"humanoid_bench": 1})
    elif reward_function == 'my_reward':
        # set humanoid bench to zero
        agent.set_cost_weights({"humanoid_bench": 0})
    else:
        raise ValueError("Unknown reward function")
    logger.info(f"Reward function set to {reward_function}")

    # rollout horizon
    T = int(total_time / sim_time_step)

    # trajectories
    qpos = np.zeros((model.nq, T))
    qvel = np.zeros((model.nv, T))
    ctrl = np.zeros((model.nu, T - 1))
    time = np.zeros(T)

    # costs
    cost_total = np.zeros(T - 1)
    cost_terms = np.zeros((len(agent.get_cost_term_values()), T - 1))

    # rollout
    mujoco.mj_resetData(model, data)

    # ------ Set initial state ------ #
    data.qpos = parse_xml.get_home_qpos(model_path)
    logger.info("home qpos set from xml file")

    # cache initial state
    qpos[:, 0] = data.qpos
    qvel[:, 0] = data.qvel
    time[0] = data.time

    if render_video:
        frames = []

    # simulation loop
    def loop():
        # set planner state
        agent.set_state(
            time=data.time,
            qpos=data.qpos,
            qvel=data.qvel,
            act=data.act,
            mocap_pos=data.mocap_pos,
            mocap_quat=data.mocap_quat,
            userdata=data.userdata,
        )

        # run planner for planer_iterations steps
        for _ in range(planer_iterations):
            agent.planner_step()

        # set ctrl from agent policy
        data.ctrl = agent.get_action()
        ctrl[:, t] = data.ctrl

        # get costs
        cost_total[t] = agent.get_total_cost()
        for i, c in enumerate(agent.get_cost_term_values().items()):
            cost_terms[i, t] = c[1]

        # step
        mujoco.mj_step(model, data)

        # cache
        qpos[:, t + 1] = data.qpos
        qvel[:, t + 1] = data.qvel
        time[t + 1] = data.time

        if render_video and t % sim_steps_per_frame == 0:
            # render and save frames
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)

    # ------ Run simulation ------ #
    for t in range(T - 1):
        try:
            loop()
        except Exception as e:
            print(e)
            logger.error(f"Exception at time step {t}")
            logger.error(e)
            return

    # reset
    agent.reset()
    logger.info("Agent reset")

    # save copy of the config file as a json file
    with open(results_dir / 'config.json', 'w') as f:
        # Add start and end time
        end_time = datetime.now()
        config['time'] = {
            'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': end_time.strftime("%Y-%m-%d %H:%M:%S"),
            'duration': str(end_time - start_time),
        }

        # Add device information
        config['device_info'] = {
            'system': platform.system(),
            'node': platform.node(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
        }
        # Add cost terms
        config['cost_terms'] = [c[0] for c in agent.get_cost_term_values().items()]
        json.dump(config, f, indent=4)
    logger.info("Config file saved")

    # ------ Save Cost Plots ------ #
    if make_plots:
        # plot costs
        fig = plt.figure()

        for i, c in enumerate(agent.get_cost_term_values().items()):
            plt.plot(time[:-1], cost_terms[i, :], label=c[0])

        plt.plot(time[:-1], cost_total, label="Total (weighted)", color="black")

        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Costs")

        plt.savefig(results_dir / "costs.png")

        # plot humaniod bench reward
        plt.figure()
        for index, c in enumerate(agent.get_cost_term_values().items()):
            if c[0] == 'humanoid_bench':
                plt.plot(time[:-1], 1.0 - cost_terms[index, :], label="humanoid_bench_reward", color="blue")
                plt.savefig(results_dir / "humanoid_bench_reward.png")
                break

    # ------ Save Cost Values ------ #
    if log_costs:
        np.save(results_dir / "costs_individual.npy", cost_terms)
        np.save(results_dir / "costs_total.npy", cost_total)
        logger.info("Costs saved")

    # ------ Save video ------ #
    if render_video:
        height, width, layers = frames[0].shape

        # make video path to my desktop
        video_path = results_dir / "video.mp4"
        video_path = str(video_path)

        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), render_fps, (width, height))

        for frame in frames:
            # Convert color from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video.write(frame_rgb)

        video.release()
        logger.info("Video saved")
    logger.info("Experiment finished")


def run_multiple_experiments(task_id, total_time, planer_iterations, n_experiments, home_path):
    date = datetime.today().strftime("%Y_%m_%d")
    time = datetime.today().strftime("%H_%M_%S")
    path = home_path + '/' + date + '/' + task_id.split(' ')[0] + '_' + task_id.split(' ')[1] + '_' + time
    config = {
        'render_video': False,
        'make_plots': False,
        'log_costs': True,
        'sim_time_step': 0.02,
        'total_time': total_time,
        'planer_iterations': planer_iterations,
        'task_id': task_id,
        'reward_function': 'my_reward',
        'path': path,
        'folder': None  # set to None to use date and time for folder name
    }
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(n_experiments):
            curr_config = config.copy()
            curr_config['folder'] = 'experiment_' + str(i).zfill(3)
            executor.submit(run_single_experiment, curr_config)

    # wait for all experiments to finish
    executor.shutdown(wait=True)

    return path
