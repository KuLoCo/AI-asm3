import pandas as pd
import re
from collections import Counter

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

def balance_dataframe(df, label_col='label_ids'):
    counts = Counter(label for labels in df[label_col] for label in labels)
    if not counts:
        return df

    max_count = max(counts.values())
    to_append = []
    for label, cnt in counts.items():
        if cnt < max_count:
            needed = max_count - cnt
            subset = df[df[label_col].apply(lambda ids: label in ids)]
            if not subset.empty:
                samples = subset.sample(n=needed, replace=True, random_state=42)
                to_append.append(samples)

    if to_append:
        df = pd.concat([df] + to_append, ignore_index=True)
    return df

def preprocess_dataframe(df, emotions_list, clean=True, balance=False):
    df = df.copy()
    if clean:
        df['text'] = df['text'].astype(str).apply(clean_text)
        
    df = map_labels_to_names(df, emotions_list)
    
    if balance:
        df = balance_dataframe(df, label_col='label_ids')
    
    return df