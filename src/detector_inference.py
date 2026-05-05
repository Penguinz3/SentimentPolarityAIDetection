from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from utils import (
    categorize_sentiment,
    ensure_dir,
    ensure_parent_dir,
    safe_read_csv,
    safe_text,
)


DEFAULT_DETECTOR_MODELS = ["SuperAnnotate/ai-detector", "roberta-base-openai-detector"]


def detector_score_column(model_name: str, existing_columns: set[str]) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    base = f"ai_prob_{slug}" if slug else "ai_prob_detector"
    column = base
    counter = 2
    while column in existing_columns:
        column = f"{base}_{counter}"
        counter += 1
    existing_columns.add(column)
    return column


def ai_score_from_prediction(prediction: dict) -> float:
    label = str(prediction.get("label", "")).lower()
    score = float(prediction.get("score", 0.0))
    ai_like_labels = {"ai", "generated", "fake", "label_1", "1"}
    if label in ai_like_labels or "ai" in label or "generated" in label:
        return score
    return 1.0 - score


def build_ai_detector(model_name: str):
    try:
        import torch
        from transformers import pipeline
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency for detector inference: torch/transformers") from exc

    return pipeline(
        "text-classification",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,
    )


def run_detector(texts: list[str], model_name: str, batch_size: int) -> list[float]:
    detector = build_ai_detector(model_name)
    scores: list[float] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"detector:{model_name}", unit="batch"):
        batch = texts[start : start + batch_size]
        predictions = detector(batch, truncation=True, max_length=512, batch_size=batch_size)
        scores.extend(ai_score_from_prediction(prediction) for prediction in predictions)
    return scores


def aggregate_detector_columns(df: pd.DataFrame, score_columns: list[str], method: str) -> pd.Series:
    scores = df[score_columns].apply(pd.to_numeric, errors="coerce")
    if method == "mean":
        return scores.mean(axis=1)
    if method == "median":
        return scores.median(axis=1)
    if method == "max":
        return scores.max(axis=1)
    if method == "min":
        return scores.min(axis=1)
    raise ValueError(f"Unsupported aggregation method: {method}")


def run_detectors(
    result: pd.DataFrame,
    texts: list[str],
    detector_models: list[str],
    batch_size: int,
    aggregation: str,
    first_available: bool,
) -> tuple[pd.DataFrame, list[str]]:
    score_columns: list[str] = []
    existing_columns = set(result.columns)
    attempted = 0

    for model_name in detector_models:
        attempted += 1
        try:
            print(f"Running AI detector model: {model_name}")
            column = detector_score_column(model_name, existing_columns)
            result[column] = run_detector(texts, model_name, batch_size)
            score_columns.append(column)
            if first_available:
                break
        except Exception as exc:
            print(f"Warning: failed to run AI detector model '{model_name}': {exc}")

    if not score_columns:
        raise SystemExit(
            f"No detector model completed successfully after {attempted} attempt(s). "
            "Check model names, local cache, or network access."
        )

    result["ai_prob"] = aggregate_detector_columns(result, score_columns, aggregation)
    return result, score_columns


def add_detector_outputs(
    df: pd.DataFrame,
    text_col: str,
    detector_models: list[str],
    batch_size: int,
    threshold: float,
    skip_detector: bool,
    aggregation: str,
    first_available: bool,
) -> pd.DataFrame:
    if text_col not in df.columns:
        raise SystemExit(f"Text column '{text_col}' not found.")

    result = df.copy()
    if skip_detector:
        if "ai_prob" not in result.columns and "ai_probability" in result.columns:
            result["ai_prob"] = result["ai_probability"]
        if "ai_prob" not in result.columns:
            raise SystemExit("--skip-detector requires an existing ai_prob or ai_probability column.")
    else:
        texts = [safe_text(value) for value in result[text_col]]
        result, score_columns = run_detectors(
            result=result,
            texts=texts,
            detector_models=detector_models,
            batch_size=batch_size,
            aggregation=aggregation,
            first_available=first_available,
        )
        result["detector_models_used"] = ",".join(score_columns)

    result["ai_positive"] = (pd.to_numeric(result["ai_prob"], errors="coerce") >= threshold).astype(int)
    result["is_ai"] = result["ai_positive"]

    if "hybrid_sentiment" in result.columns and "sentiment_category" not in result.columns:
        result["sentiment_category"] = result["hybrid_sentiment"].apply(categorize_sentiment)
    return result


def write_detection_outputs(
    df: pd.DataFrame,
    output_path: Path,
    summary_path: Path,
    positive_path: Path,
    plot_dir: Path,
) -> None:
    output_path = ensure_parent_dir(output_path)
    summary_path = ensure_parent_dir(summary_path)
    positive_path = ensure_parent_dir(positive_path)
    plot_dir = ensure_dir(plot_dir)

    df.to_csv(output_path, index=False)

    ai_positive = pd.to_numeric(df["ai_positive"], errors="coerce").fillna(0).astype(int)
    ai_prob = pd.to_numeric(df["ai_prob"], errors="coerce")
    summary = {
        "total_rows": int(len(df)),
        "ai_positive_rows": int(ai_positive.sum()),
        "ai_positive_rate": float(ai_positive.mean()) if len(df) else np.nan,
        "avg_ai_prob": float(ai_prob.mean()) if len(df) else np.nan,
    }
    if "hybrid_sentiment" in df.columns:
        summary["avg_hybrid_sentiment"] = float(pd.to_numeric(df["hybrid_sentiment"], errors="coerce").mean())
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    keep_cols = [
        col
        for col in [
            "chunk_id",
            "doc_id",
            "source",
            "year",
            "section",
            "ai_prob",
            "ai_positive",
            "is_ai",
            "hybrid_sentiment",
            "text",
        ]
        if col in df.columns
    ]
    positive_df = df.loc[ai_positive == 1, keep_cols].sort_values("ai_prob", ascending=False)
    positive_df.to_csv(positive_path, index=False)

    write_detector_plots(df, plot_dir)


def write_detector_plots(df: pd.DataFrame, plot_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ai_prob = pd.to_numeric(df["ai_prob"], errors="coerce").dropna()
    if not ai_prob.empty:
        plt.figure(figsize=(8, 5))
        ai_prob.hist(bins=40)
        plt.title("AI Probability Distribution")
        plt.xlabel("AI probability")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(plot_dir / "ai_probability_distribution.png", dpi=160)
        plt.close()

    if "hybrid_sentiment" in df.columns:
        sentiment = pd.to_numeric(df["hybrid_sentiment"], errors="coerce").dropna()
        if not sentiment.empty:
            plt.figure(figsize=(8, 5))
            sentiment.hist(bins=40)
            plt.title("Sentiment Distribution")
            plt.xlabel("Hybrid sentiment")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(plot_dir / "sentiment_distribution.png", dpi=160)
            plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run probabilistic AI-detector inference on text chunks.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--text-col", default="text", help="Text column name.")
    parser.add_argument("--output", default="outputs/corpus_with_results.csv", help="Output CSV path.")
    parser.add_argument("--summary-output", default="outputs/detection_summary.csv", help="Summary CSV path.")
    parser.add_argument(
        "--positive-output",
        default="outputs/positive_detections.csv",
        help="Rows with positive detector labels.",
    )
    parser.add_argument("--plot-dir", default="plots", help="Plot output directory.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Positive detector threshold.")
    parser.add_argument("--batch-size", type=int, default=32, help="Detector inference batch size.")
    parser.add_argument(
        "--aggregation",
        choices=["mean", "median", "max", "min"],
        default="mean",
        help="How to aggregate multiple detector probability columns into ai_prob.",
    )
    parser.add_argument(
        "--first-available",
        action="store_true",
        help="Use the first detector that runs successfully instead of aggregating all configured detectors.",
    )
    parser.add_argument(
        "--detector-models",
        nargs="*",
        default=DEFAULT_DETECTOR_MODELS,
        help="Candidate Hugging Face detector models, tried in order.",
    )
    parser.add_argument(
        "--skip-detector",
        action="store_true",
        help="Use an existing detector probability column instead of running a model.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row limit for testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = safe_read_csv(args.input)
    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()
    result = add_detector_outputs(
        df=df,
        text_col=args.text_col,
        detector_models=args.detector_models,
        batch_size=args.batch_size,
        threshold=args.threshold,
        skip_detector=args.skip_detector,
        aggregation=args.aggregation,
        first_available=args.first_available,
    )
    write_detection_outputs(
        result,
        Path(args.output),
        Path(args.summary_output),
        Path(args.positive_output),
        Path(args.plot_dir),
    )
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
