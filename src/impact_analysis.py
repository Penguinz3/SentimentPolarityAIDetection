from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import ensure_dir, resolve_target_column, safe_read_csv, select_numeric_feature_columns


FEATURE_GROUP_PATTERNS = {
    "sentiment": ("sentiment", "vader", "hf"),
    "entropy": ("entropy", "shannon"),
    "ngram_transition": ("bigram", "trigram", "transition", "ngram"),
    "stylometric_structure": (
        "sentence",
        "word_count",
        "avg_word",
        "punctuation",
        "repetition",
        "type_token",
        "unique_word",
    ),
}


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, list[str], str]:
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
    return X, y, feature_cols, resolved_target


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
):
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def build_logistic_model(class_weight: str | None) -> Pipeline:
    return Pipeline(
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


def build_random_forest_model(class_weight: str | None, random_state: int, n_estimators: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    class_weight=class_weight,
                    min_samples_leaf=2,
                    n_jobs=1,
                ),
            ),
        ]
    )


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


def evaluate_model(model_name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return metric_row(model_name, y_test, y_pred, y_prob)


def fit_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight: str | None,
    random_state: int,
    n_estimators: int,
) -> dict[str, Pipeline]:
    models = {
        "logistic_regression": build_logistic_model(class_weight),
        "random_forest": build_random_forest_model(class_weight, random_state, n_estimators),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def write_metrics(
    models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> pd.DataFrame:
    metrics_df = pd.DataFrame(
        [evaluate_model(model_name, model, X_test, y_test) for model_name, model in models.items()]
    )
    metrics_df.to_csv(output_dir / "impact_model_metrics.csv", index=False)
    return metrics_df


def plot_horizontal_bars(
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
    color: str = "#4c78a8",
    signed: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = df.copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values(value_col, ascending=True)
    colors = color
    if signed:
        colors = np.where(plot_df[value_col] >= 0, "#4c78a8", "#d35f5f")

    plt.figure(figsize=(9, max(5, 0.35 * len(plot_df))))
    plt.barh(plot_df[label_col], plot_df[value_col], color=colors)
    plt.axvline(0, color="0.55", linewidth=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_permutation_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    plot_dir: Path,
    n_repeats: int,
    random_state: int,
    top_n: int,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(output_dir / "permutation_importance.csv", index=False)

    plot_df = importance_df.head(top_n).sort_values("importance_mean", ascending=True)
    plot_horizontal_bars(
        plot_df,
        "importance_mean",
        "feature",
        plot_dir / "permutation_importance.png",
        "Top Features by Permutation Importance",
        "Mean decrease in ROC-AUC after permutation",
    )
    return importance_df


def logistic_coefficients(
    logistic_model: Pipeline,
    feature_cols: list[str],
    output_dir: Path,
    plot_dir: Path,
    top_n: int,
) -> pd.DataFrame:
    coefficients = logistic_model.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": coefficients,
            "coefficient_abs": np.abs(coefficients),
        }
    ).sort_values("coefficient", ascending=False)
    coef_df.to_csv(output_dir / "logistic_coefficients.csv", index=False)

    positive = coef_df.sort_values("coefficient", ascending=False).head(top_n)
    negative = coef_df.sort_values("coefficient", ascending=True).head(top_n)
    plot_df = pd.concat([negative, positive], axis=0).drop_duplicates("feature")
    plot_horizontal_bars(
        plot_df,
        "coefficient",
        "feature",
        plot_dir / "logistic_coefficients.png",
        "Standardized Logistic Regression Coefficients",
        "Coefficient: negative values push away, positive values push toward ai_positive",
        signed=True,
    )
    return coef_df


def false_positive_feature_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    plot_dir: Path,
    top_n: int,
) -> pd.DataFrame:
    comparison_rows = []
    for feature in X.columns:
        positive = pd.to_numeric(X.loc[y == 1, feature], errors="coerce").dropna()
        negative = pd.to_numeric(X.loc[y == 0, feature], errors="coerce").dropna()
        mean_positive = float(positive.mean()) if not positive.empty else np.nan
        mean_negative = float(negative.mean()) if not negative.empty else np.nan
        difference = mean_positive - mean_negative

        var_positive = positive.var(ddof=1) if len(positive) > 1 else 0.0
        var_negative = negative.var(ddof=1) if len(negative) > 1 else 0.0
        denom_n = max(len(positive) + len(negative) - 2, 0)
        pooled_std = np.sqrt(
            (((len(positive) - 1) * var_positive) + ((len(negative) - 1) * var_negative)) / denom_n
        ) if denom_n > 0 else np.nan
        standardized_difference = difference / pooled_std if pooled_std and pooled_std > 0 else np.nan

        comparison_rows.append(
            {
                "feature": feature,
                "mean_ai_positive_1": mean_positive,
                "mean_ai_positive_0": mean_negative,
                "difference": difference,
                "standardized_difference": standardized_difference,
                "standardized_difference_abs": abs(standardized_difference)
                if pd.notna(standardized_difference)
                else np.nan,
                "n_ai_positive_1": int(len(positive)),
                "n_ai_positive_0": int(len(negative)),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        "standardized_difference_abs", ascending=False
    )
    comparison_df.to_csv(output_dir / "false_positive_feature_comparison.csv", index=False)

    plot_df = comparison_df.dropna(subset=["standardized_difference"]).head(top_n)
    plot_horizontal_bars(
        plot_df,
        "standardized_difference",
        "feature",
        plot_dir / "false_positive_feature_differences.png",
        "Largest Feature Differences by Detector Flag",
        "Standardized difference: ai_positive=1 minus ai_positive=0",
        signed=True,
    )
    return comparison_df


def feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    groups = {}
    for group_name, patterns in FEATURE_GROUP_PATTERNS.items():
        groups[group_name] = [
            feature for feature in feature_cols if any(pattern in feature.lower() for pattern in patterns)
        ]
    return groups


def train_ablation_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    class_weight: str | None,
    random_state: int,
    n_estimators: int,
) -> dict:
    if model_type == "logistic_regression":
        model = build_logistic_model(class_weight)
    elif model_type == "random_forest":
        model = build_random_forest_model(class_weight, random_state, n_estimators)
    else:
        raise ValueError(f"Unsupported ablation model type: {model_type}")
    model.fit(X_train, y_train)
    return evaluate_model(model_type, model, X_test, y_test)


def feature_group_ablation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_cols: list[str],
    output_dir: Path,
    plot_dir: Path,
    model_type: str,
    class_weight: str | None,
    random_state: int,
    n_estimators: int,
) -> pd.DataFrame:
    baseline_metrics = train_ablation_model(
        X_train,
        y_train,
        X_test,
        y_test,
        model_type,
        class_weight,
        random_state,
        n_estimators,
    )
    groups = feature_groups(feature_cols)
    rows = []

    for group_name, group_features in groups.items():
        if not group_features:
            rows.append(
                {
                    "feature_group": group_name,
                    "removed_features": "",
                    "n_removed": 0,
                    "baseline_accuracy": baseline_metrics["accuracy"],
                    "ablated_accuracy": np.nan,
                    "accuracy_drop": np.nan,
                    "baseline_f1": baseline_metrics["f1"],
                    "ablated_f1": np.nan,
                    "f1_drop": np.nan,
                    "baseline_roc_auc": baseline_metrics["roc_auc"],
                    "ablated_roc_auc": np.nan,
                    "roc_auc_drop": np.nan,
                }
            )
            continue

        keep_features = [feature for feature in feature_cols if feature not in group_features]
        if not keep_features:
            ablated_metrics = {metric: np.nan for metric in ["accuracy", "f1", "roc_auc"]}
        else:
            ablated_metrics = train_ablation_model(
                X_train[keep_features],
                y_train,
                X_test[keep_features],
                y_test,
                model_type,
                class_weight,
                random_state,
                n_estimators,
            )

        rows.append(
            {
                "feature_group": group_name,
                "removed_features": ",".join(group_features),
                "n_removed": len(group_features),
                "baseline_accuracy": baseline_metrics["accuracy"],
                "ablated_accuracy": ablated_metrics["accuracy"],
                "accuracy_drop": baseline_metrics["accuracy"] - ablated_metrics["accuracy"],
                "baseline_f1": baseline_metrics["f1"],
                "ablated_f1": ablated_metrics["f1"],
                "f1_drop": baseline_metrics["f1"] - ablated_metrics["f1"],
                "baseline_roc_auc": baseline_metrics["roc_auc"],
                "ablated_roc_auc": ablated_metrics["roc_auc"],
                "roc_auc_drop": baseline_metrics["roc_auc"] - ablated_metrics["roc_auc"],
            }
        )

    ablation_df = pd.DataFrame(rows).sort_values("roc_auc_drop", ascending=False)
    ablation_df.to_csv(output_dir / "feature_group_ablation.csv", index=False)

    plot_col = "roc_auc_drop" if ablation_df["roc_auc_drop"].notna().any() else "f1_drop"
    plot_df = ablation_df.dropna(subset=[plot_col]).copy()
    plot_horizontal_bars(
        plot_df,
        plot_col,
        "feature_group",
        plot_dir / "feature_group_ablation.png",
        "Feature Group Ablation",
        f"Performance drop after removing group ({plot_col})",
    )
    return ablation_df


def optional_shap_analysis(
    forest_model: Pipeline,
    X_test: pd.DataFrame,
    output_dir: Path,
    plot_dir: Path,
    max_rows: int,
) -> str:
    try:
        import shap
    except ModuleNotFoundError:
        return "SHAP skipped: package is not installed."
    except Exception as exc:
        return f"SHAP skipped: could not import shap ({exc})."

    try:
        sample = X_test.head(max_rows).copy()
        transformed = forest_model.named_steps["imputer"].transform(sample)
        estimator = forest_model.named_steps["model"]
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            values = shap_values
            if values.ndim == 3 and values.shape[-1] > 1:
                values = values[:, :, 1]

        mean_abs = np.abs(values).mean(axis=0)
        shap_df = pd.DataFrame(
            {"feature": sample.columns, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = shap_df.head(15).sort_values("mean_abs_shap", ascending=True)
        plt.figure(figsize=(9, max(5, 0.35 * len(top))))
        plt.barh(top["feature"], top["mean_abs_shap"], color="#4c78a8")
        plt.title("SHAP Feature Importance")
        plt.xlabel("Mean absolute SHAP value")
        plt.tight_layout()
        plt.savefig(plot_dir / "shap_bar.png", dpi=160)
        plt.close()
        return f"SHAP completed on {len(sample)} test rows."
    except Exception as exc:
        return f"SHAP skipped: analysis failed gracefully ({exc})."


def top_feature_list(df: pd.DataFrame, value_col: str, n: int = 5, ascending: bool = False) -> list[str]:
    if df.empty or value_col not in df.columns:
        return []
    top_df = df.dropna(subset=[value_col]).sort_values(value_col, ascending=ascending).head(n)
    return [f"{row['feature']} ({row[value_col]:.4f})" for _, row in top_df.iterrows()]


def write_summary(
    output_dir: Path,
    permutation_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    shap_status: str,
) -> None:
    top_perm = top_feature_list(permutation_df, "importance_mean", n=5, ascending=False)
    top_positive = top_feature_list(coef_df, "coefficient", n=5, ascending=False)
    top_negative = top_feature_list(coef_df, "coefficient", n=5, ascending=True)
    top_differences = top_feature_list(
        comparison_df,
        "standardized_difference_abs",
        n=5,
        ascending=False,
    )

    ablation_metric = "roc_auc_drop" if ablation_df["roc_auc_drop"].notna().any() else "f1_drop"
    ablation_ranked = ablation_df.dropna(subset=[ablation_metric]).sort_values(
        ablation_metric, ascending=False
    )
    top_group = None
    if not ablation_ranked.empty:
        row = ablation_ranked.iloc[0]
        top_group = f"{row['feature_group']} ({ablation_metric}={row[ablation_metric]:.4f})"

    lines = [
        "Impact Analysis Summary",
        "",
        "Interpretation note: these results describe association and predictive contribution within the fitted models. They should not be read as causal claims.",
        "",
        "Top features by permutation importance:",
        *[f"- {item}" for item in top_perm],
        "",
        "Top positive logistic regression predictors:",
        *[f"- {item}" for item in top_positive],
        "",
        "Top negative logistic regression predictors:",
        *[f"- {item}" for item in top_negative],
        "",
        "Largest false-positive vs non-false-positive standardized differences:",
        *[f"- {item}" for item in top_differences],
        "",
        "Feature group whose removal reduced performance the most:",
        f"- {top_group}" if top_group else "- Not available",
        "",
        shap_status,
    ]
    (output_dir / "impact_analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze feature impact for predicting detector-positive false positives."
    )
    parser.add_argument("--input", required=True, help="Input feature CSV path.")
    parser.add_argument("--target-col", default="ai_positive", help="Binary detector-positive column.")
    parser.add_argument("--output-dir", default="outputs", help="CSV/text output directory.")
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
    parser.add_argument("--n-estimators", type=int, default=300, help="Random forest tree count.")
    parser.add_argument("--permutation-repeats", type=int, default=10, help="Permutation repeats.")
    parser.add_argument("--top-n", type=int, default=15, help="Number of features to show in plots.")
    parser.add_argument(
        "--ablation-model",
        choices=["random_forest", "logistic_regression"],
        default="random_forest",
        help="Model used for feature-group ablation.",
    )
    parser.add_argument("--shap-max-rows", type=int, default=1000, help="Maximum test rows for optional SHAP.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip optional SHAP analysis.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    plot_dir = ensure_dir(args.plot_dir)
    class_weight = None if args.class_weight == "none" else args.class_weight

    df = safe_read_csv(args.input)
    X, y, feature_cols, resolved_target = prepare_data(df, args.target_col, args.exclude_cols)
    X_train, X_test, y_train, y_test = split_data(X, y, args.test_size, args.random_state)

    models = fit_models(X_train, y_train, class_weight, args.random_state, args.n_estimators)
    metrics_df = write_metrics(models, X_test, y_test, output_dir)
    permutation_df = run_permutation_importance(
        models["random_forest"],
        X_test,
        y_test,
        output_dir,
        plot_dir,
        args.permutation_repeats,
        args.random_state,
        args.top_n,
    )
    coef_df = logistic_coefficients(
        models["logistic_regression"],
        feature_cols,
        output_dir,
        plot_dir,
        args.top_n,
    )
    comparison_df = false_positive_feature_comparison(X, y, output_dir, plot_dir, args.top_n)
    ablation_df = feature_group_ablation(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_cols,
        output_dir,
        plot_dir,
        args.ablation_model,
        class_weight,
        args.random_state,
        args.n_estimators,
    )
    if args.skip_shap:
        shap_status = "SHAP skipped by command-line option."
    else:
        shap_status = optional_shap_analysis(
            models["random_forest"], X_test, output_dir, plot_dir, args.shap_max_rows
        )

    write_summary(output_dir, permutation_df, coef_df, comparison_df, ablation_df, shap_status)
    print(f"Impact feature columns used: {len(feature_cols)}")
    print(f"Target column: {resolved_target}")
    print(f"Wrote impact analysis outputs to: {output_dir}")
    print(shap_status)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()

