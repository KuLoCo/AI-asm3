from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

def vectorize_texts(texts, max_features=10000, ngram_range=(1,2)):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def encode_labels(label_lists):
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(label_lists)
    return y, mlb