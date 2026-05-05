from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from utils import (
    count_punctuation,
    ensure_parent_dir,
    safe_read_csv,
    safe_text,
    split_sentences,
    tokenize_words,
)


STYLOMETRIC_COLUMNS = [
    "word_count",
    "sentence_count",
    "avg_sentence_length",
    "sentence_length_std",
    "avg_word_length",
    "type_token_ratio",
    "unique_word_ratio",
    "repetition_rate",
    "punctuation_density",
]


def stylometric_features_for_text(text: object) -> dict[str, float]:
    clean_text = safe_text(text)
    words = tokenize_words(clean_text)
    sentences = split_sentences(clean_text)
    sentence_lengths = [len(tokenize_words(sentence)) for sentence in sentences]
    word_lengths = [len(word) for word in words]

    word_count = len(words)
    sentence_count = len(sentences)
    word_counts = Counter(words)
    type_count = len(word_counts)
    once_count = sum(1 for count in word_counts.values() if count == 1)

    return {
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "avg_sentence_length": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
        "sentence_length_std": float(np.std(sentence_lengths, ddof=0)) if sentence_lengths else 0.0,
        "avg_word_length": float(np.mean(word_lengths)) if word_lengths else 0.0,
        "type_token_ratio": (type_count / word_count) if word_count else 0.0,
        "unique_word_ratio": (once_count / word_count) if word_count else 0.0,
        "repetition_rate": ((word_count - type_count) / word_count) if word_count else 0.0,
        "punctuation_density": (count_punctuation(clean_text) / len(clean_text)) if clean_text else 0.0,
    }


def add_stylometric_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    if text_col not in df.columns:
        raise SystemExit(f"Text column '{text_col}' not found.")
    feature_rows = [stylometric_features_for_text(text) for text in df[text_col]]
    feature_df = pd.DataFrame(feature_rows, index=df.index)
    result = df.copy()
    for column in STYLOMETRIC_COLUMNS:
        result[column] = feature_df[column]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add stylometric text features to a CSV.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--text-col", default="text", help="Text column name.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = safe_read_csv(args.input)
    result = add_stylometric_features(df, args.text_col)
    output_path = ensure_parent_dir(args.output)
    result.to_csv(output_path, index=False)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
