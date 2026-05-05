from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    ensure_dir,
    identifier_columns,
    resolve_target_column,
    safe_read_csv,
    select_numeric_feature_columns,
)


def fit_pca(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: list[str],
    n_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    resolved_target = resolve_target_column(df, target_col) if target_col else ""
    feature_cols = select_numeric_feature_columns(df, resolved_target, exclude_cols)
    if len(feature_cols) < 2:
        raise SystemExit(f"PCA requires at least two numeric feature columns; found {len(feature_cols)}.")

    feature_data = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    max_components = min(n_components, len(feature_cols), len(feature_data))
    if max_components < 1:
        raise SystemExit("PCA requires at least one valid row.")

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=max_components, random_state=42)),
        ]
    )
    coordinates = pipeline.fit_transform(feature_data)
    pca = pipeline.named_steps["pca"]

    pc_columns = [f"PC{i + 1}" for i in range(max_components)]
    coord_df = pd.DataFrame(coordinates, columns=pc_columns, index=df.index)
    for column in reversed(identifier_columns(df)):
        coord_df.insert(0, column, df[column].values)
    if resolved_target:
        coord_df[resolved_target] = df[resolved_target].values

    loadings_df = pd.DataFrame(pca.components_.T, columns=pc_columns, index=feature_cols)
    loadings_df.insert(0, "feature", loadings_df.index)
    component_cols = [col for col in pc_columns if col in loadings_df.columns]
    loadings_df["max_abs_loading"] = loadings_df[component_cols].abs().max(axis=1)
    loadings_df = loadings_df.reset_index(drop=True).sort_values("max_abs_loading", ascending=False)

    variance_df = pd.DataFrame(
        {
            "component": pc_columns,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    return coord_df, loadings_df, variance_df, feature_cols


def write_pca_plots(
    coordinates: pd.DataFrame,
    loadings: pd.DataFrame,
    target_col: str,
    plot_dir: Path,
    max_loading_features: int,
) -> None:
    plot_dir = ensure_dir(plot_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    y_values = coordinates["PC2"] if "PC2" in coordinates.columns else pd.Series(0, index=coordinates.index)
    if target_col in coordinates.columns and pd.Series(coordinates[target_col]).nunique(dropna=True) > 1:
        colors = pd.to_numeric(coordinates[target_col], errors="coerce")
        scatter = plt.scatter(coordinates["PC1"], y_values, c=colors, s=14, alpha=0.55, cmap="coolwarm")
        plt.colorbar(scatter, label=target_col)
    else:
        plt.scatter(coordinates["PC1"], y_values, s=14, alpha=0.55)
    plt.axhline(0, color="0.8", linewidth=0.8)
    plt.axvline(0, color="0.8", linewidth=0.8)
    plt.title("PCA Feature-Space Map")
    plt.xlabel("PC1")
    plt.ylabel("PC2" if "PC2" in coordinates.columns else "0")
    plt.tight_layout()
    plt.savefig(plot_dir / "pca_false_positive_map.png", dpi=160)
    plt.close()

    top = loadings.head(max_loading_features).copy()
    top = top.sort_values("max_abs_loading", ascending=True)
    y_pos = np.arange(len(top))
    plt.figure(figsize=(9, max(5, 0.35 * len(top))))
    if "PC2" in top.columns:
        plt.barh(y_pos - 0.18, top["PC1"], height=0.36, label="PC1")
        plt.barh(y_pos + 0.18, top["PC2"], height=0.36, label="PC2")
        plt.legend()
    else:
        plt.barh(y_pos, top["PC1"], height=0.5, label="PC1")
    plt.yticks(y_pos, top["feature"])
    plt.axvline(0, color="0.65", linewidth=0.8)
    plt.title("Strongest PCA Loadings")
    plt.xlabel("Loading")
    plt.tight_layout()
    plt.savefig(plot_dir / "pca_loadings.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PCA on non-leaking numeric linguistic features.")
    parser.add_argument("--input", required=True, help="Input feature CSV path.")
    parser.add_argument("--target-col", default="ai_positive", help="Binary target column for coloring.")
    parser.add_argument("--output-dir", default="outputs", help="CSV output directory.")
    parser.add_argument("--plot-dir", default="plots", help="Plot output directory.")
    parser.add_argument("--exclude-cols", nargs="*", default=[], help="Additional feature columns to exclude.")
    parser.add_argument("--n-components", type=int, default=5, help="Maximum number of PCA components.")
    parser.add_argument("--max-loading-features", type=int, default=15, help="Number of loading features to plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = safe_read_csv(args.input)
    coord_df, loadings_df, variance_df, feature_cols = fit_pca(
        df=df,
        target_col=args.target_col,
        exclude_cols=args.exclude_cols,
        n_components=args.n_components,
    )

    output_dir = ensure_dir(args.output_dir)
    coord_df.to_csv(output_dir / "pca_coordinates.csv", index=False)
    loadings_df.to_csv(output_dir / "pca_loadings.csv", index=False)
    variance_df.to_csv(output_dir / "pca_explained_variance.csv", index=False)
    write_pca_plots(coord_df, loadings_df, args.target_col, Path(args.plot_dir), args.max_loading_features)
    print(f"PCA feature columns used: {len(feature_cols)}")
    print(f"Wrote PCA outputs to: {output_dir}")


if __name__ == "__main__":
    main()

