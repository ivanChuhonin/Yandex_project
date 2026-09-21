"""Model evaluation and inference on new, raw text."""
from sklearn.metrics import classification_report, confusion_matrix

import config


def evaluate_model(pipeline, X_test, y_test) -> dict:
    predictions = pipeline.predict(X_test)
    report = classification_report(
        y_test, predictions, labels=config.SENTIMENT_CLASSES, output_dict=True
    )
    print(classification_report(y_test, predictions, labels=config.SENTIMENT_CLASSES))
    print(confusion_matrix(y_test, predictions, labels=config.SENTIMENT_CLASSES))
    return report


def predict_sentiment(pipeline, texts: list[str]) -> list[str]:
    """texts must be raw, unprocessed strings — the pipeline handles cleaning itself."""
    return list(pipeline.predict(texts))
