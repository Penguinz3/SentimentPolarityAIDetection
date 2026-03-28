import os
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


INPUT_CSV = Path("corpus_with_results.csv")
OUT_DIR = Path("stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = [
    "ai_prob",
    "hybrid_sentiment",
    "shannon_entropy",
    "bigram_diversity",
]

# Runtime knobs (override via env vars if needed)
BAYES_MAX_ROWS = int(os.getenv("BAYES_MAX_ROWS", "6000"))  # 0 = full dataset
BAYES_DRAWS = int(os.getenv("BAYES_DRAWS", "500"))
BAYES_TUNE = int(os.getenv("BAYES_TUNE", "500"))
BAYES_CHAINS = int(os.getenv("BAYES_CHAINS", "2"))
BAYES_TARGET_ACCEPT = float(os.getenv("BAYES_TARGET_ACCEPT", "0.9"))


def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"Missing input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}")

    model_df = df[REQUIRED_COLUMNS].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(model_df) < 200:
        raise SystemExit(f"Not enough valid rows for Bayesian regression: {len(model_df)}")

    predictors = ["hybrid_sentiment", "shannon_entropy", "bigram_diversity"]
    for c in predictors:
        std = model_df[c].std(ddof=0)
        if std == 0 or np.isnan(std):
            raise SystemExit(f"Predictor has zero variance: {c}")
        model_df[f"{c}_z"] = (model_df[c] - model_df[c].mean()) / std

    if BAYES_MAX_ROWS > 0 and len(model_df) > BAYES_MAX_ROWS:
        model_df = model_df.sample(BAYES_MAX_ROWS, random_state=42).copy()

    x_cols = [f"{c}_z" for c in predictors]
    X = model_df[x_cols].to_numpy()
    y = model_df["ai_prob"].to_numpy()

    print(f"Running Bayesian regression on {len(model_df)} rows", flush=True)
    print(
        f"draws={BAYES_DRAWS}, tune={BAYES_TUNE}, chains={BAYES_CHAINS}, target_accept={BAYES_TARGET_ACCEPT}",
        flush=True,
    )

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        betas = pm.Normal("betas", mu=0, sigma=1, shape=X.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = intercept + pm.math.dot(X, betas)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(
            draws=BAYES_DRAWS,
            tune=BAYES_TUNE,
            chains=BAYES_CHAINS,
            target_accept=BAYES_TARGET_ACCEPT,
            cores=1,
            progressbar=True,
            return_inferencedata=True,
        )

    summary = az.summary(trace, var_names=["intercept", "betas", "sigma"], hdi_prob=0.95)
    summary.to_csv(OUT_DIR / "bayesian_regression_summary.csv")
    (OUT_DIR / "bayesian_regression_summary.txt").write_text(summary.to_string(), encoding="utf-8")

    beta_draws = trace.posterior["betas"].stack(sample=("chain", "draw")).values
    posterior_probs = pd.DataFrame(
        {
            "term": x_cols,
            "p_beta_gt_0": [(beta_draws[i] > 0).mean() for i in range(len(x_cols))],
            "p_beta_lt_0": [(beta_draws[i] < 0).mean() for i in range(len(x_cols))],
        }
    )
    posterior_probs.to_csv(OUT_DIR / "bayesian_direction_probs.csv", index=False)

    report_lines = [
        "Bayesian Regression Report",
        f"Rows used: {len(model_df)}",
        f"draws={BAYES_DRAWS}, tune={BAYES_TUNE}, chains={BAYES_CHAINS}",
        "",
        "Model: ai_prob ~ hybrid_sentiment_z + shannon_entropy_z + bigram_diversity_z",
        "",
        "Posterior direction probabilities:",
    ]
    for _, row in posterior_probs.iterrows():
        report_lines.append(
            f"- {row['term']}: P(beta>0)={row['p_beta_gt_0']:.3f}, P(beta<0)={row['p_beta_lt_0']:.3f}"
        )
    report_lines.append("")
    report_lines.append("Outputs:")
    report_lines.append("- stats/bayesian_regression_summary.csv")
    report_lines.append("- stats/bayesian_regression_summary.txt")
    report_lines.append("- stats/bayesian_direction_probs.csv")

    report = "\n".join(report_lines)
    (OUT_DIR / "bayesian_report.txt").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
