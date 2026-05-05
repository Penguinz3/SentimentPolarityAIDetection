from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


WORD_RE = re.compile(r"\b[\w']+\b")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
PUNCTUATION_RE = re.compile(r"[^\w\s]")

DEFAULT_LEAKAGE_COLUMNS = {
    "ai_positive",
    "is_ai",
    "ai_prob",
    "ai_probability",
    "ai_score",
    "ai_label",
    "ai_generated",
    "ai_generated_score",
    "ai_generated_probability",
    "prob_ai",
    "probability_ai",
    "score_ai",
    "detector_score",
    "detector_probability",
    "detector_prob",
    "detector_output",
    "detector_confidence",
    "normalized_detector_score",
    "normalized_detector_probability",
    "normalized_ai_score",
    "normalized_ai_probability",
    "generated_score",
    "generated_probability",
    "human_score",
    "human_probability",
    "fake_score",
    "fake_probability",
    "real_score",
    "real_probability",
    "prediction_score",
    "prediction_probability",
    "classification_score",
    "classification_probability",
    "detector_label",
    "predicted_label",
    "prediction",
    "label",
    "target",
}

DEFAULT_NON_FEATURE_COLUMNS = {
    "chunk_id",
    "doc_id",
    "source",
    "journal",
    "section",
    "raw_doc_path",
    "text",
    "text_hash",
    "title",
    "abstract",
    "year",
    "chunk_index",
    "start_sentence",
    "n_sentences",
}


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    parent = path.parent
    if parent and str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def safe_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def tokenize_words(text: object) -> list[str]:
    return WORD_RE.findall(safe_text(text).lower())


def split_sentences(text: object) -> list[str]:
    text = safe_text(text)
    if not text:
        return []
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences or [text]


def count_punctuation(text: object) -> int:
    return len(PUNCTUATION_RE.findall(safe_text(text)))


def shannon_entropy(items: Iterable[object]) -> float:
    values = list(items)
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def char_shannon_entropy(text: object) -> float:
    return shannon_entropy(safe_text(text))


def word_shannon_entropy(text: object) -> float:
    return shannon_entropy(tokenize_words(text))


def ngram_transition_entropy(tokens: Sequence[str], order: int) -> float:
    """Conditional entropy H(next token | prefix) for word n-gram transitions."""
    if order < 2:
        raise ValueError("order must be at least 2")
    if len(tokens) < order:
        return 0.0

    prefix_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    total_transitions = 0
    for index in range(len(tokens) - order + 1):
        prefix = tuple(tokens[index : index + order - 1])
        next_token = tokens[index + order - 1]
        prefix_counts[prefix][next_token] += 1
        total_transitions += 1

    if total_transitions == 0:
        return 0.0

    conditional_entropy = 0.0
    for next_counts in prefix_counts.values():
        prefix_total = sum(next_counts.values())
        weight = prefix_total / total_transitions
        conditional_entropy += weight * shannon_entropy(next_counts.elements())
    return conditional_entropy


def sentence_entropy_stats(text: object) -> tuple[float, float]:
    entropies = [word_shannon_entropy(sentence) for sentence in split_sentences(text)]
    if not entropies:
        return 0.0, 0.0
    return float(np.mean(entropies)), float(np.std(entropies, ddof=0))


def categorize_sentiment(score: float) -> str:
    if score > 0.05:
        return "Positive"
    if score < -0.05:
        return "Negative"
    return "Neutral"


def resolve_target_column(df: pd.DataFrame, target_col: str) -> str:
    if target_col in df.columns:
        return target_col
    if target_col == "ai_positive" and "is_ai" in df.columns:
        df["ai_positive"] = df["is_ai"]
        return "ai_positive"
    raise SystemExit(
        f"Target column '{target_col}' not found. Available columns: {', '.join(df.columns)}"
    )


def should_exclude_feature_column(column: str, target_col: str | None = None) -> bool:
    lower = column.lower()
    target_lower = target_col.lower() if target_col else None
    if target_lower and lower == target_lower:
        return True
    if lower in DEFAULT_LEAKAGE_COLUMNS or lower in DEFAULT_NON_FEATURE_COLUMNS:
        return True
    score_tokens = ("prob", "probability", "score", "confidence", "positive", "label", "prediction", "pred", "normalized")
    source_tokens = ("ai", "detector", "generated", "human", "fake", "real", "prediction", "classification")
    if any(source in lower for source in source_tokens) and any(
        token in lower
        for token in score_tokens
    ):
        return True
    if lower.startswith("normalized") and any(token in lower for token in ("prob", "score", "output", "confidence")):
        return True
    return False


def select_numeric_feature_columns(
    df: pd.DataFrame,
    target_col: str | None = None,
    exclude_cols: Sequence[str] | None = None,
) -> list[str]:
    exclude = {col.lower() for col in (exclude_cols or [])}
    numeric_cols = list(df.select_dtypes(include=[np.number, "bool"]).columns)
    feature_cols = []
    has_new_char_entropy = "char_shannon_entropy" in df.columns
    for col in numeric_cols:
        lower = col.lower()
        if has_new_char_entropy and lower == "shannon_entropy":
            continue
        if lower in exclude or should_exclude_feature_column(col, target_col):
            continue
        feature_cols.append(col)
    return feature_cols


def identifier_columns(df: pd.DataFrame) -> list[str]:
    candidates = ["chunk_id", "doc_id", "source", "year", "section"]
    return [col for col in candidates if col in df.columns]
