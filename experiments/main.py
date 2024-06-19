from run_experiment import run_multiple_experiments
from evaluate import evaluate


def main():
    n_experiments = 2

    result_path = run_multiple_experiments(
        task_id="Push H1",
        total_time=1,  # seconds
        planer_iterations=1,
        n_experiments=n_experiments,
        home_path="/Users/moritzmeser/Desktop/experiments")
    print(result_path)
    evaluate(result_path, n_experiments)


if __name__ == "__main__":
    main()
