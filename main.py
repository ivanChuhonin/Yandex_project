"""End-to-end pipeline: download -> EDA -> business analysis -> preprocess ->
vectorize -> train -> evaluate -> demo predictions.

Run with `python main.py`. See `python main.py --help` for options.
"""
import argparse
import sys

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import config
import data_loading
import evaluate
import train
import vectorize
import visualize


def run_business_analysis(df: pd.DataFrame) -> None:
    """Average rating per company/city, on the FULL dataset (not rubric-filtered —
    marketplaces and supermarkets don't carry a "Ресторан|Кафе" rubric)."""
    big_cities_df = df[df["city"].isin(config.BIG_CITIES)]

    groups = {
        "fast_food": (config.FAST_FOOD, "Удовлетворённость: фастфуд"),
        "marketplace": (config.MARKETPLACE, "Удовлетворённость: маркетплейсы"),
        "shops": (config.SHOPS, "Удовлетворённость: супермаркеты"),
    }
    for key, (companies, title) in groups.items():
        subset = big_cities_df[big_cities_df["name_ru"].isin(companies)]
        if subset.empty:
            print(f"[business_analysis] '{key}': нет данных для {companies}")
            continue
        mean_ratings = subset.groupby(["city", "name_ru"])["rating"].mean().reset_index()
        fig = visualize.plot_city_ratings(mean_ratings, companies, title)
        fig.savefig(config.REPORTS_DIR / f"{key}_by_city.png")

        fig = visualize.plot_rating_boxplot(mean_ratings, title)
        fig.savefig(config.REPORTS_DIR / f"{key}_boxplot.png")


def run_eda(df: pd.DataFrame) -> None:
    visualize.plot_rubric_counts(df).savefig(config.REPORTS_DIR / "rubric_counts.png")
    visualize.plot_sentiment_distribution(df).savefig(
        config.REPORTS_DIR / "sentiment_distribution.png"
    )


def run_wordclouds(df_text: pd.DataFrame) -> None:
    colors = {"negative": "Blues", "neutral": "Greens", "positive": "Oranges"}
    distinctive = vectorize.distinctive_words_by_sentiment(df_text, "clean_text", "sentiment")
    for label, freq in distinctive.items():
        if freq.empty:
            continue
        fig = visualize.plot_wordcloud(freq, f"wordcloud: {label}", colors.get(label, "viridis"))
        fig.savefig(config.REPORTS_DIR / f"wordcloud_{label}.png")


def run_training(df_text: pd.DataFrame, n_iter: int):
    X_train, X_test, y_train, y_test = train.split_data(df_text, text_col="text", target_col="sentiment")
    pipeline = train.build_pipeline()
    pipeline = train.tune_and_train(pipeline, X_train, y_train, n_iter=n_iter)
    evaluate.evaluate_model(pipeline, X_test, y_test)
    train.save_model(pipeline)
    return pipeline


def demo_predictions(pipeline) -> None:
    examples = [
        "Этот продукт просто потрясающий! Очень доволен покупкой.",
        "Упаковка пошарпанная, вкус так себе. Больше сюда не приду.",
        "Кафе на четвёрку: обслужили быстро, но было шумно.",
    ]
    for text, label in zip(examples, evaluate.predict_sentiment(pipeline, examples)):
        print(f"[{label}] {text}")


def main():
    parser = argparse.ArgumentParser(description="Sentiment analysis pipeline (Yandex geo-reviews)")
    parser.add_argument("--skip-download", action="store_true", help="use the CSV already in data/")
    parser.add_argument("--sample-frac", type=float, default=1.0, help="subsample the modeling subset for a quick run")
    parser.add_argument("--search-iter", type=int, default=8, help="RandomizedSearchCV n_iter")
    args = parser.parse_args()

    if not args.skip_download:
        data_loading.download_dataset()

    df = data_loading.load_reviews()
    print(f"Loaded {len(df)} rated reviews")

    run_eda(df)
    run_business_analysis(df)

    df_rest = data_loading.filter_by_rubric(df)
    if args.sample_frac < 1.0:
        df_rest = df_rest.sample(frac=args.sample_frac, random_state=config.RANDOM_STATE)
    print(f"Modeling subset (Ресторан|Кафе): {len(df_rest)} reviews")

    from preprocessing import preprocess_batch

    df_rest = df_rest.copy()
    df_rest["clean_text"] = preprocess_batch(df_rest["text"])

    run_wordclouds(df_rest)

    pipeline = run_training(df_rest, n_iter=args.search_iter)
    demo_predictions(pipeline)


if __name__ == "__main__":
    main()
