"""Plotting functions for EDA and the per-industry satisfaction comparison.

Every function returns a matplotlib Figure instead of calling plt.show(), so
callers (main.py, notebooks, tests) decide whether to display or save it.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud


def plot_rubric_counts(df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    order = df["rubric"].value_counts().nlargest(top_n).index
    sns.countplot(data=df, y="rubric", order=order, ax=ax)
    ax.bar_label(ax.containers[0])
    ax.set_title("Количество отзывов")
    ax.set_xlabel(None)
    ax.set_ylabel("Рубрика")
    fig.tight_layout()
    return fig


def plot_sentiment_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["negative", "neutral", "positive"]
    colors = {"negative": "red", "neutral": "blue", "positive": "green"}
    counts = df["sentiment"].value_counts().reindex(order)
    ax.bar(order, counts.values, color=[colors[o] for o in order])
    ax.set_title("Распределение тональности")
    ax.set_ylabel("Количество")
    fig.tight_layout()
    return fig


def plot_city_ratings(mean_ratings: pd.DataFrame, companies: list[str], title: str) -> plt.Figure:
    """mean_ratings must have columns: city, name_ru, rating (mean per city/company)."""
    fig, ax = plt.subplots(figsize=(16, 6))
    for company in companies:
        subset = mean_ratings[mean_ratings["name_ru"] == company].sort_values("city")
        ax.plot(subset["city"], subset["rating"], marker="o", label=company)
    ax.set_xlabel("Города")
    ax.set_ylabel("Средний рейтинг")
    ax.set_title(title)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_rating_boxplot(mean_ratings: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_ratings.boxplot(column="rating", by="name_ru", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Рейтинг")
    ax.set_title(title)
    fig.suptitle("")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_wordcloud(word_freq: pd.DataFrame, title: str, colormap: str = "viridis") -> plt.Figure:
    wc = WordCloud(
        background_color="black", colormap=colormap, max_words=200, width=1600, height=1600
    ).generate_from_frequencies(dict(word_freq.values))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title, fontsize=20)
    ax.axis("off")
    fig.tight_layout()
    return fig
