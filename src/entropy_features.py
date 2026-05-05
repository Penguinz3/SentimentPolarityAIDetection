from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import (
    char_shannon_entropy,
    ensure_dir,
    ensure_parent_dir,
    ngram_transition_entropy,
    safe_read_csv,
    sentence_entropy_stats,
    tokenize_words,
    word_shannon_entropy,
)


ENTROPY_COLUMNS = [
    "char_shannon_entropy",
    "word_shannon_entropy",
    "bigram_transition_entropy",
    "trigram_transition_entropy",
    "sentence_entropy_mean",
    "sentence_entropy_std",
]


def entropy_features_for_text(text: object) -> dict[str, float]:
    tokens = tokenize_words(text)
    sentence_mean, sentence_std = sentence_entropy_stats(text)
    return {
        "char_shannon_entropy": char_shannon_entropy(text),
        "word_shannon_entropy": word_shannon_entropy(text),
        "bigram_transition_entropy": ngram_transition_entropy(tokens, order=2),
        "trigram_transition_entropy": ngram_transition_entropy(tokens, order=3),
        "sentence_entropy_mean": sentence_mean,
        "sentence_entropy_std": sentence_std,
    }


def add_entropy_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    if text_col not in df.columns:
        raise SystemExit(f"Text column '{text_col}' not found.")
    feature_rows = [entropy_features_for_text(text) for text in df[text_col]]
    feature_df = pd.DataFrame(feature_rows, index=df.index)
    return pd.concat([df.copy(), feature_df], axis=1)


def detector_probability_column(df: pd.DataFrame) -> str | None:
    for column in ["ai_prob", "ai_probability", "detector_score", "detector_probability"]:
        if column in df.columns:
            return column
    return None


def write_entropy_plot(df: pd.DataFrame, plot_dir: str | Path) -> None:
    probability_col = detector_probability_column(df)
    if probability_col is None or "char_shannon_entropy" not in df.columns:
        return

    plot_dir = ensure_dir(plot_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = df[["char_shannon_entropy", probability_col]].copy()
    color_col = "ai_positive" if "ai_positive" in df.columns else "is_ai" if "is_ai" in df.columns else None
    if color_col:
        plot_df[color_col] = df[color_col]
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce").dropna()
    if plot_df.empty:
        return

    plt.figure(figsize=(8, 5))
    if color_col and plot_df[color_col].nunique() > 1:
        scatter = plt.scatter(
            plot_df["char_shannon_entropy"],
            plot_df[probability_col],
            c=plot_df[color_col],
            s=12,
            alpha=0.45,
            cmap="coolwarm",
        )
        plt.colorbar(scatter, label=color_col)
    else:
        plt.scatter(plot_df["char_shannon_entropy"], plot_df[probability_col], s=12, alpha=0.45)
    plt.title("Character Entropy vs AI Probability")
    plt.xlabel("Character-level Shannon entropy")
    plt.ylabel(probability_col)
    plt.tight_layout()
    plt.savefig(plot_dir / "entropy_vs_ai_probability.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add entropy-based text features to a CSV.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--text-col", default="text", help="Text column name.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--plot-dir", default="plots", help="Optional plot output directory.")
    parser.add_argument("--no-plot", action="store_true", help="Do not write entropy plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = safe_read_csv(args.input)
    result = add_entropy_features(df, args.text_col)
    output_path = ensure_parent_dir(args.output)
    result.to_csv(output_path, index=False)
    if not args.no_plot:
        write_entropy_plot(result, args.plot_dir)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()

