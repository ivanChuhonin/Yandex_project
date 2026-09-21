"""Downloading and loading the Yandex geo-reviews dataset."""
import os

import pandas as pd

import config


def download_dataset(force: bool = False) -> None:
    """Download and unzip the dataset via the Kaggle API using credentials from .env."""
    if config.RAW_CSV_PATH.exists() and not force:
        return

    if not config.KAGGLE_USERNAME or not config.KAGGLE_KEY:
        raise RuntimeError(
            "KAGGLE_USERNAME/KAGGLE_KEY не заданы. Заполните .env (см. .env.example) "
            "или положите geo-reviews-dataset-2023.csv в data/ вручную."
        )

    # The kaggle package reads credentials from the environment at import time,
    # so they must be set before importing it.
    os.environ["KAGGLE_USERNAME"] = config.KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = config.KAGGLE_KEY

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(config.KAGGLE_DATASET, path=config.DATA_DIR, unzip=True)


def load_reviews(csv_path=None) -> pd.DataFrame:
    """Load the raw CSV, drop unrated reviews and derive city/rubric/sentiment columns."""
    csv_path = csv_path or config.RAW_CSV_PATH
    df = pd.read_csv(csv_path)
    df = df[df["rating"] > 0].copy()

    df["rating"] = df["rating"].astype(int)
    df["rating_label"] = df["rating"].map(config.RATING_DISPLAY_LABELS)
    df["sentiment"] = df["rating"].map(config.SENTIMENT_LABELS)

    df["rubric"] = (
        df["rubrics"].astype(str).str.split(",").str[0].str.split(";").str[0].str.strip()
    )
    df["city"] = df["address"].astype(str).str.split(", ").str[0].str.strip()

    return df


def filter_by_rubric(df: pd.DataFrame, pattern: str = config.RUBRIC_FILTER_REGEX) -> pd.DataFrame:
    return df[df["rubric"].str.contains(pattern, na=False)].copy()
