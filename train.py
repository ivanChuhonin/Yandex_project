"""Model training: a single Pipeline that owns preprocessing + vectorization + classifier.

Bundling preprocessing into the Pipeline (rather than pre-computing a
`reviews` column, as the original notebook did) means predict() on raw,
unseen text goes through the exact same steps as training — no risk of the
train/inference mismatch the notebook had.
"""
import joblib
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import config
from preprocessing import preprocess_batch


def split_data(df: pd.DataFrame, text_col: str = "text", target_col: str = "sentiment"):
    return train_test_split(
        df[text_col],
        df[target_col],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df[target_col],
    )


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocess", FunctionTransformer(preprocess_batch)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=config.TFIDF_NGRAM_RANGE,
                    max_features=config.TFIDF_MAX_FEATURES,
                    min_df=config.TFIDF_MIN_DF,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=config.RANDOM_STATE, class_weight="balanced"
                ),
            ),
        ]
    )


def tune_and_train(
    pipeline: Pipeline, X_train, y_train, n_iter: int = 8, cv: int = 3
) -> Pipeline:
    """RandomizedSearchCV over the RandomForest hyperparameters, macro-F1 scored
    because the sentiment classes are imbalanced (positive dominates)."""
    param_distributions = {
        "classifier__n_estimators": randint(100, 400),
        "classifier__max_depth": randint(10, 60),
        "classifier__min_samples_split": randint(2, 10),
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_macro",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def save_model(pipeline: Pipeline, filename: str = "sentiment_model.joblib") -> None:
    joblib.dump(pipeline, config.MODELS_DIR / filename)


def load_model(filename: str = "sentiment_model.joblib") -> Pipeline:
    return joblib.load(config.MODELS_DIR / filename)
