import os
from src.evaluate import evaluate_models


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "goemotions", "data")
    results_dir = os.path.join(base_dir, "results")

    evaluate_models(
        data_dir=data_dir,
        results_dir=results_dir,
        test_size=0.2,
        random_state=42,
        top_n=4
    )

    print(f"Evaluation complete. Confusion matrices saved in: {results_dir}")

if __name__ == '__main__':
    main()