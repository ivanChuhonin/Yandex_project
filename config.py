"""Central configuration: paths, reproducibility settings and business constants."""
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

for _dir in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    _dir.mkdir(exist_ok=True)

KAGGLE_DATASET = "kyakovlev/yandex-geo-reviews-dataset-2023"
RAW_CSV_PATH = DATA_DIR / "geo-reviews-dataset-2023.csv"

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")

RANDOM_STATE = 40
TEST_SIZE = 0.2

# Bigrams help the neutral class, which unigram TF-IDF struggles to separate
# from positive/negative (e.g. "не понравилось" vs "понравилось").
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 50_000
TFIDF_MIN_DF = 2

# Rubric subset the pipeline focuses on (mirrors the original notebook's scope).
RUBRIC_FILTER_REGEX = "Ресторан|Кафе"

# rating -> display label, used only for EDA/plots, never as a model target.
RATING_DISPLAY_LABELS = {
    5: "Отлично",
    4: "Хорошо",
    3: "Удовлетворительно",
    2: "Неудовлетворительно",
    1: "Плохо",
}

# rating -> sentiment class, the single source of truth for the ML target.
# 1-2 negative, 3 neutral, 4-5 positive.
SENTIMENT_LABELS = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}
SENTIMENT_CLASSES = ["negative", "neutral", "positive"]

BIG_CITIES = [
    "Москва", "Санкт-Петербург", "Новосибирск", "Воронеж", "Республика Татарстан",
    "Самара", "Краснодар", "Республика Башкортостан", "Нижний Новгород",
    "Красноярск", "Волгоград", "Пермь",
]
FAST_FOOD = ["KFC", "Вкусно — и точка", "Бургер Кинг"]
MARKETPLACE = ["Ozon", "Wildberries", "Яндекс Маркет"]
SHOPS = ["Магнит", "О'кей", "Перекрёсток", "Пятёрочка", "Ашан"]
