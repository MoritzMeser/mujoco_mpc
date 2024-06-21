from run_experiment import run_multiple_experiments


def main():
    n_experiments = 8

    # valid Tasks: Stand H1, Walk H1, Run H1
    # valid reward functions: humanoid_bench, full_reward_no_hb, hb_posture_control

    result_paths = []
    for reward_fuction in ["humanoid_bench", "full_reward_no_hb", "hb_posture_control"]:
        result_path = run_multiple_experiments(
            task_id="Run H1",
            planer_iterations=4,
            n_experiments=n_experiments,
            home_path="/Users/moritzmeser/Desktop/run_sampling",
            reward_fuction=reward_fuction)
        result_paths.append(result_path)

    print("Results:")
    for result_path in result_paths:
        print(result_path)


if __name__ == "__main__":
    main()
