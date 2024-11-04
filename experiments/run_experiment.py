
from Enums import Experiment, RewardFunction
import json
from datetime import datetime
import platform
import logging
from util import util
import parse_xml
import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import os

# set current directory: mujoco_mpc/python/mujoco_mpc
from mujoco_mpc import agent as agent_lib


def run_single_experiment(experiment: Experiment, index: int, show_tqdm: bool = False):
    start_time = datetime.now()
    print("running experiment ", index)

    # default values not to be changed
    sim_time_step = 0.002
    ###################################

    # make folder for experiment
    path = experiment.folder_path / f'experiment_{str(index).zfill(3)}'
    path.mkdir(parents=True, exist_ok=True)

    # check if the render_fps is a valid value
    if experiment.render_video:
        render_fps = 25
        sim_steps_per_frame = 1 / (render_fps * sim_time_step)
        if not (int(sim_steps_per_frame) == sim_steps_per_frame):
            raise ValueError("The choose another render_fps value.")

    # ------ Set up logger ------ #
    logger = logging.getLogger(__name__ + str(index))
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(path / 'log.txt')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # ------ Load model ------ #
    task_id = experiment.task_name.name + " " + experiment.robot_name.name
    print("task_id: ", task_id)
    model_path = util.model_path_from_id(task_id)
    print("model_path: ", model_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = sim_time_step
    logger.info(f"Model loaded from {model_path}")

    # data
    data = mujoco.MjData(model)

    # ------ Set up renderer ------ #
    if experiment.render_video:
        # set resolution to full HD
        model.vis.global_.offwidth = 1920
        model.vis.global_.offheight = 1080
        renderer = mujoco.Renderer(model, height=1080, width=1920)
        logger.info("Renderer initialized")

    # ------ Set up agent ------ #
    print("task_id: ", task_id)
    print("model: ", model)
    agent = agent_lib.Agent(task_id=task_id, model=model)
    logger.info("Agent initialized")

    # ------ Set cost weights ------ #
    if experiment.reward_function == RewardFunction.hb:
        # set all costs to zero
        agent.set_cost_weights({name: 0 for name in agent.get_cost_weights().keys()})
        # set humanoid bench to 1
        agent.set_cost_weights({"humanoid_bench": 1})
    elif experiment.reward_function == RewardFunction.ours:
        # set humanoid bench to zero
        agent.set_cost_weights({"humanoid_bench": 0})
    elif experiment.reward_function == RewardFunction.ours_plus_hb:
        agent.set_cost_weights({"humanoid_bench": 100})
    else:
        raise ValueError("Unknown reward function")
    logger.info(f"Reward function set to {experiment.reward_function.value}")

    # rollout horizon
    T = int(experiment.total_time / sim_time_step)
    logger.info(f"Rollout horizon set to {T}")

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

    if experiment.render_video:
        frames = []

    cost_term_names = list(agent.get_cost_term_values().keys())

    # simulation loop
    def loop():
        if t % 100 == 0:
            logger.info(f"Time step {t}")
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
        for _ in range(experiment.planner_iterations):
            agent.planner_step()

        # set ctrl from agent policy
        data.ctrl = agent.get_action()
        ctrl[:, t] = data.ctrl

        # get costs
        cost_total[t] = agent.get_total_cost()
        for i, c in enumerate(agent.get_cost_term_values().items()):
            index = cost_term_names.index(c[0])
            cost_terms[index, t] = c[1]

        # step
        mujoco.mj_step(model, data)

        # cache
        qpos[:, t + 1] = data.qpos
        qvel[:, t + 1] = data.qvel
        time[t + 1] = data.time

        if experiment.render_video and t % sim_steps_per_frame == 0:
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
    with open(path / 'config.json', 'w') as f:
        config = experiment.to_dict()
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
        config['cost_terms'] = cost_term_names
        json.dump(config, f, indent=4)
    logger.info("Config file saved")

    # ------ Save Cost Plots ------ #
    # plot costs
    plt.figure()

    for i, c in enumerate(cost_term_names):
        plt.plot(time[:-1], cost_terms[i, :], label=c)

    plt.plot(time[:-1], cost_total, label="Total (weighted)", color="black")

    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Costs")

    plt.savefig(path / "costs.png")

    # plot humaniod bench reward
    plt.figure()
    for index, c in enumerate(cost_term_names):
        if c == 'humanoid_bench':
            plt.plot(time[:-1], 1.0 - cost_terms[index, :], label="humanoid_bench_reward", color="blue")
            plt.savefig(path / "humanoid_bench_reward.png")
            break

    # ------ Save Cost Values + Trajectory ------ #

    np.save(path / "costs_individual.npy", cost_terms)
    np.save(path / "costs_total.npy", cost_total)
    logger.info("Costs saved")

    np.save(path / "qpso.npy", qpos)
    np.save(path / "qvel.npy", qvel)
    np.save(path / "ctrl.npy", ctrl)
    np.save(path / "time.npy", time)

    logger.info("Trajectory saved")

    # ------ Save video ------ #
    if experiment.render_video:
        height, width, layers = frames[0].shape

        # make video path to my desktop
        video_path = path / "video.mp4"
        video_path = str(video_path)

        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), render_fps, (width, height))

        for frame in frames:
            # Convert color from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video.write(frame_rgb)

        video.release()
        logger.info("Video saved")
    logger.info("Experiment_finished")

    # ------ Clean up logger at the end of the thread ------ #
    file_handler.close()
    logger.removeHandler(file_handler)
