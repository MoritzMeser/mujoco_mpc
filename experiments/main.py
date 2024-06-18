from run_experiment import run_multiple_experiments
from evaluate import evaluate


def main():
    n_experiments = 8

    result_path = run_multiple_experiments(
        task_id="Walk H1",
        total_time=10,  # seconds
        planer_iterations=1,
        n_experiments=n_experiments,
        home_path="/Users/moritzmeser/Desktop/experiments")
    evaluate(result_path, n_experiments)


if __name__ == "__main__":
    main()
