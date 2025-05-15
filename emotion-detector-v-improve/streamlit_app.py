import streamlit as st
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

def load_emotions(emotions_path):
    with open(emotions_path, 'r', encoding='utf-8') as f:
        emotions = [line.strip() for line in f.readlines()]
    return emotions

def load_split(filepath, split_name):
    df = pd.read_csv(filepath, sep='\t', header=None, names=["text", "labels", "valence"])
    df['split'] = split_name
    return df

def load_all_data(base_path):
    emotions = load_emotions(os.path.join(base_path, "emotions.txt"))
    train = load_split(os.path.join(base_path, "train.tsv"), "train")
    dev = load_split(os.path.join(base_path, "dev.tsv"), "dev")
    test = load_split(os.path.join(base_path, "test.tsv"), "test")
    full_df = pd.concat([train, dev, test], ignore_index=True)
    return full_df, emotions

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_labels(label_str):
    return [int(x) for x in label_str.split(',') if x.strip().isdigit()]

def map_labels_to_names(df, emotions_list):
    df = df.copy()
    df['label_ids'] = df['labels'].apply(parse_labels)
    df['label_names'] = df['label_ids'].apply(lambda ids: [emotions_list[i] for i in ids])
    return df

def preprocess_dataframe(df, emotions_list, clean=True, balance=False): # Added emotions_list
    df = map_labels_to_names(df, emotions_list) # Added emotions_list
    if clean:
        df['text'] = df['text'].apply(clean_text)
    return df

def vectorize_texts(texts, max_features=10000, ngram_range=(1, 2)):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def encode_labels(label_lists):
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(label_lists)
    return y, mlb

def predict_texts(texts, data_dir, model_path):
    df, emotions = load_all_data(data_dir)
    df = preprocess_dataframe(df, emotions, clean=True) # Added emotions
    X_all, vectorizer = vectorize_texts(df['text'])
    _, mlb = encode_labels(df['label_names'])
    cleaned = [clean_text(t) for t in texts]
    X_new = vectorizer.transform(cleaned)
    model = joblib.load(model_path)
    y_pred = model.predict(X_new)
    label_lists = mlb.inverse_transform(y_pred)
    return list(zip(texts, label_lists))

def main():
    st.title("Emotion Detection from Journal Entries")

    journal_text = st.text_area("Enter your journal entry:", "")

    if st.button("Analyze"):
        if journal_text:
            data_dir = "data/goemotions/data"  
            model_path = "results/logistic_regression.joblib"

            try:
                predictions = predict_texts([journal_text], data_dir, model_path)  #  Make the prediction
            except FileNotFoundError as e:
                st.error(f"Error: {e}")
                st.stop()
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.stop()
            st.subheader("Emotion Analysis:")
            for text, labels in predictions:
                st.write(f"**Text:** {text}")
                if labels:
                    st.write(f"**Emotions:** {', '.join(labels)}")
                else:
                    st.write("No emotions detected.")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
