from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from utils import ensure_dir, resolve_target_column, safe_read_csv


def cramers_v(contingency: pd.DataFrame, chi2: float) -> float:
    n = contingency.to_numpy().sum()
    if n == 0:
        return np.nan
    rows, cols = contingency.shape
    denom = n * (min(rows, cols) - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt(chi2 / denom))


def run_sentiment_chi_square(
    df: pd.DataFrame,
    sentiment_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if sentiment_col not in df.columns:
        raise SystemExit(f"Sentiment column '{sentiment_col}' not found.")
    resolved_target = resolve_target_column(df, target_col)

    test_df = df[[sentiment_col, resolved_target]].dropna().copy()
    if test_df.empty:
        raise SystemExit("No valid rows available for chi-square test.")

    contingency = pd.crosstab(test_df[sentiment_col], test_df[resolved_target])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        raise SystemExit(
            "Chi-square test requires at least two sentiment groups and two target classes."
        )

    chi2, p_value, dof, expected = chi2_contingency(contingency)
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
    stats_df = pd.DataFrame(
        [
            {
                "test": "chi_square",
                "sentiment_col": sentiment_col,
                "target_col": resolved_target,
                "chi2": float(chi2),
                "p_value": float(p_value),
                "dof": int(dof),
                "n": int(contingency.to_numpy().sum()),
                "cramers_v": cramers_v(contingency, float(chi2)),
            }
        ]
    )
    return contingency, expected_df, stats_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run legacy sentiment-category association tests against detector positives."
    )
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument(
        "--sentiment-col",
        default="sentiment_category",
        help="Categorical sentiment column.",
    )
    parser.add_argument("--target-col", default="ai_positive", help="Binary detector-positive column.")
    parser.add_argument("--output-dir", default="outputs", help="CSV output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = safe_read_csv(args.input)
    contingency, expected, stats = run_sentiment_chi_square(
        df=df,
        sentiment_col=args.sentiment_col,
        target_col=args.target_col,
    )

    output_dir = ensure_dir(args.output_dir)
    contingency.to_csv(output_dir / "sentiment_ai_contingency.csv")
    expected.to_csv(output_dir / "sentiment_ai_expected.csv")
    stats.to_csv(output_dir / "sentiment_ai_chi_square.csv", index=False)
    print(f"Wrote sentiment chi-square outputs to: {output_dir}")


if __name__ == "__main__":
    main()

