import os
from src.load_data import load_all_data
from src.preprocess import preprocess_dataframe
from src.vectorize import vectorize_texts, encode_labels
from src.train_model import train_models


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "goemotions", "data")
    results_dir = os.path.join(base_dir, "results")

    df, emotions = load_all_data(data_dir)
    df = preprocess_dataframe(df, emotions, clean=True, balance=True)
    X, _ = vectorize_texts(df['text'])
    y, _ = encode_labels(df['label_names'])

    train_models(
        X, y,
        results_dir=results_dir,
        test_size=0.2,
        random_state=42,
        class_weight='balanced'
    )

    print(f"Training complete. Models and metrics saved in: {results_dir}")

if __name__ == '__main__':
    main()