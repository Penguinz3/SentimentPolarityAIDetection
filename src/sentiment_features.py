from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from utils import categorize_sentiment, ensure_parent_dir, safe_read_csv, safe_text


def build_vader_analyzer():
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency for VADER sentiment: nltk") from exc

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


def run_hf_sentiment(
    texts: list[str],
    model_name: str,
    batch_size: int,
) -> list[float]:
    try:
        import torch
        from transformers import pipeline
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency for transformer sentiment: torch/transformers") from exc

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,
    )

    scores: list[float] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="hf_sentiment", unit="batch"):
        batch = texts[start : start + batch_size]
        predictions = sentiment_pipeline(batch, truncation=True, max_length=512, batch_size=batch_size)
        for prediction in predictions:
            label = str(prediction.get("label", "")).upper()
            score = float(prediction.get("score", 0.0))
            scores.append(score if label == "POSITIVE" else -score)
    return scores


def add_sentiment_features(
    df: pd.DataFrame,
    text_col: str,
    skip_hf: bool,
    hf_model: str,
    batch_size: int,
) -> pd.DataFrame:
    if text_col not in df.columns:
        raise SystemExit(f"Text column '{text_col}' not found.")

    result = df.copy()
    texts = [safe_text(value) for value in result[text_col]]

    vader = build_vader_analyzer()
    result["vader_compound"] = [vader.polarity_scores(text)["compound"] for text in texts]

    if skip_hf:
        result["hf_sentiment"] = result["vader_compound"]
    else:
        result["hf_sentiment"] = run_hf_sentiment(texts, hf_model, batch_size)

    result["hybrid_sentiment"] = 0.5 * result["vader_compound"] + 0.5 * result["hf_sentiment"]
    result["sentiment_category"] = result["hybrid_sentiment"].apply(categorize_sentiment)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add VADER and optional transformer sentiment features.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--text-col", default="text", help="Text column name.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Use VADER only and set hf_sentiment equal to vader_compound.",
    )
    parser.add_argument(
        "--hf-model",
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Hugging Face sentiment model.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Transformer inference batch size.")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row limit for testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = safe_read_csv(args.input)
    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()
    result = add_sentiment_features(df, args.text_col, args.skip_hf, args.hf_model, args.batch_size)
    output_path = ensure_parent_dir(args.output)
    result.to_csv(output_path, index=False)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
