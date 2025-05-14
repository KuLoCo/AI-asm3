import os
import joblib

from src.load_data import load_all_data
from src.preprocess import preprocess_dataframe, clean_text
from src.vectorize import vectorize_texts, encode_labels

def predict_texts(texts, data_dir, model_path):
    df, emotions = load_all_data(data_dir)
    df = preprocess_dataframe(df, emotions, clean=True)

    X_all, vectorizer = vectorize_texts(df['text'])
    _, mlb = encode_labels(df['label_names'])

    cleaned = [clean_text(t) for t in texts]
    X_new = vectorizer.transform(cleaned)

    model = joblib.load(model_path)
    y_pred = model.predict(X_new)
    label_lists = mlb.inverse_transform(y_pred)

    return list(zip(texts, label_lists))

def predict_file(input_path, data_dir, model_path):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"No such input file: {input_path}")
    texts = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    return predict_texts(texts, data_dir, model_path)