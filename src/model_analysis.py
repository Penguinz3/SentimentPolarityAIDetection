from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    ensure_dir,
    resolve_target_column,
    safe_read_csv,
    select_numeric_feature_columns,
)


def prepare_model_data(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    resolved_target = resolve_target_column(df, target_col)
    feature_cols = select_numeric_feature_columns(df, resolved_target, exclude_cols)
    if not feature_cols:
        raise SystemExit("No numeric non-leaking feature columns were found.")

    y = pd.to_numeric(df[resolved_target], errors="coerce")
    valid_mask = y.notna()
    X = df.loc[valid_mask, feature_cols].replace([np.inf, -np.inf], np.nan)
    y = y.loc[valid_mask].astype(int)

    classes = sorted(y.unique())
    if classes != [0, 1]:
        raise SystemExit(f"Target column must be binary 0/1 after coercion; found {classes}.")
    return X, y, feature_cols


def metric_row(model_name: str, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict:
    row = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": np.nan,
        "n_test": int(len(y_true)),
    }
    if y_prob is not None and y_true.nunique() == 2:
        try:
            row["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            row["roc_auc"] = np.nan
    return row


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    class_weight: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=class_weight,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    forest = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=random_state,
                    class_weight=class_weight,
                    min_samples_leaf=2,
                    n_jobs=1,
                ),
            ),
        ]
    )

    models = {
        "logistic_regression": logistic,
        "random_forest": forest,
    }

    metric_rows = []
    importance_rows = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        metric_rows.append(metric_row(model_name, y_test, y_pred, y_prob))

        if model_name == "logistic_regression":
            coefficients = model.named_steps["model"].coef_[0]
            for feature, coefficient in zip(X.columns, coefficients):
                importance_rows.append(
                    {
                        "model": model_name,
                        "feature": feature,
                        "importance_type": "standardized_coefficient",
                        "importance": float(coefficient),
                        "importance_abs": float(abs(coefficient)),
                    }
                )
        else:
            importances = model.named_steps["model"].feature_importances_
            for feature, importance in zip(X.columns, importances):
                importance_rows.append(
                    {
                        "model": model_name,
                        "feature": feature,
                        "importance_type": "random_forest_importance",
                        "importance": float(importance),
                        "importance_abs": float(abs(importance)),
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(importance_rows)


def write_feature_importance_plot(importance_df: pd.DataFrame, plot_dir: Path, top_n: int) -> None:
    plot_dir = ensure_dir(plot_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = importance_df[importance_df["importance_type"] == "random_forest_importance"].copy()
    if plot_df.empty:
        plot_df = importance_df.copy()
    plot_df = plot_df.sort_values("importance_abs", ascending=False).head(top_n)
    plot_df = plot_df.sort_values("importance_abs", ascending=True)

    plt.figure(figsize=(9, max(5, 0.35 * len(plot_df))))
    plt.barh(plot_df["feature"], plot_df["importance_abs"])
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(plot_dir / "feature_importance.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict detector false positives from linguistic features.")
    parser.add_argument("--input", required=True, help="Input feature CSV path.")
    parser.add_argument("--target-col", default="ai_positive", help="Binary target column.")
    parser.add_argument("--output-dir", default="outputs", help="CSV output directory.")
    parser.add_argument("--plot-dir", default="plots", help="Plot output directory.")
    parser.add_argument("--exclude-cols", nargs="*", default=[], help="Additional feature columns to exclude.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="balanced",
        help="Class weighting for imbalanced detector positives.",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Number of features to show in the plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = safe_read_csv(args.input)
    X, y, feature_cols = prepare_model_data(df, args.target_col, args.exclude_cols)
    class_weight = None if args.class_weight == "none" else args.class_weight
    metrics_df, importance_df = train_models(
        X=X,
        y=y,
        test_size=args.test_size,
        random_state=args.random_state,
        class_weight=class_weight,
    )

    output_dir = ensure_dir(args.output_dir)
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    write_feature_importance_plot(importance_df, Path(args.plot_dir), args.top_n)
    print(f"Model feature columns used: {len(feature_cols)}")
    print(f"Wrote model outputs to: {output_dir}")


if __name__ == "__main__":
    main()
