import os
import pandas as pd

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