"""TF-IDF vectorization helpers, including per-class word frequencies for wordclouds."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(ngram_range=(1, 1), max_features=None) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)


def class_word_frequencies(texts: pd.Series, top_n: int = 100) -> pd.DataFrame:
    """Sum of TF-IDF weights per word across `texts`, sorted descending."""
    vectorizer = build_tfidf_vectorizer()
    matrix = vectorizer.fit_transform(texts)
    freq = pd.DataFrame(
        {
            "word": vectorizer.get_feature_names_out(),
            "frequency": np.asarray(matrix.sum(axis=0)).ravel(),
        }
    ).sort_values("frequency", ascending=False)
    return freq.head(top_n) if top_n else freq


def distinctive_words_by_sentiment(df: pd.DataFrame, text_col: str, sentiment_col: str, top_n: int = 100):
    """For each sentiment class, words most characteristic of it (i.e. not shared with the others)."""
    freqs = {
        label: class_word_frequencies(df.loc[df[sentiment_col] == label, text_col], top_n=None)
        for label in df[sentiment_col].unique()
    }

    distinctive = {}
    for label, freq in freqs.items():
        other_words = pd.concat(
            [f["word"] for other_label, f in freqs.items() if other_label != label]
        )
        distinctive[label] = freq[~freq["word"].isin(other_words)].head(top_n)
    return distinctive
