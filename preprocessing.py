"""Text cleaning, stopword removal and lemmatization.

Exposes preprocess_text()/preprocess_batch() as the single place that turns raw
review text into the normalized form the vectorizer is trained on. Both
training and inference must go through this same function — the original
notebook fit on lemmatized text but scored raw, unprocessed strings at
inference time, which silently degraded predictions.
"""
import re
from functools import lru_cache

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymorphy3 import MorphAnalyzer

_CYRILLIC_ONLY = re.compile(r"[^а-яА-ЯёЁ\s]")

_EXTRA_STOPWORDS = {
    "'", "''", '""', '"', "<", ">", "?", ")", "(", ".", "!", ",", ":", "-",
    "%", "$", "^", "@", "[", "]", "{", "}", "/", "\\", "_", "&", "№", "~", "`",
}
# Negations carry sentiment and must survive stopword removal.
_KEEP_WORDS = {"нет", "не", "ни", "всё", "все"}


def _ensure_nltk_data() -> None:
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab", "corpora/stopwords"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1], quiet=True)


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset:
    _ensure_nltk_data()
    words = set(stopwords.words("russian")) - _KEEP_WORDS
    words |= _EXTRA_STOPWORDS
    return frozenset(words)


@lru_cache(maxsize=1)
def get_morph_analyzer() -> MorphAnalyzer:
    return MorphAnalyzer()


@lru_cache(maxsize=200_000)
def _lemmatize_word(word: str) -> str:
    # Cached because the same surface forms repeat constantly across reviews,
    # and morph.parse() is the bottleneck of the whole pipeline.
    return get_morph_analyzer().parse(word)[0].normal_form


def clean_text(text: str) -> str:
    """Strip everything but Cyrillic letters (mirrors the notebook's `preprocess`)."""
    return _CYRILLIC_ONLY.sub(" ", str(text))


def preprocess_text(text: str) -> str:
    """Clean -> tokenize -> drop stopwords -> lemmatize -> join back into a string."""
    _ensure_nltk_data()
    stop_words = get_stopwords()

    cleaned = clean_text(text).lower()
    tokens = word_tokenize(cleaned, language="russian")
    lemmas = [_lemmatize_word(token) for token in tokens if token not in stop_words]
    lemmas = [lemma for lemma in lemmas if lemma not in stop_words]
    return " ".join(lemmas)


def preprocess_batch(texts) -> list[str]:
    """Vectorized-friendly entry point: accepts a list/Series, returns a list of strings.

    Used both to build the training corpus and as the first step inside the
    inference Pipeline, so raw text is always normalized the same way.
    """
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    return [preprocess_text(t) for t in texts]
