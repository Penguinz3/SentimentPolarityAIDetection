import math
import os
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import statsmodels.api as sm
from scipy.stats import chi2_contingency


INPUT_CSV = Path("corpus_with_results.csv")
OUT_DIR = Path("stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = [
    "ai_prob",
    "is_ai",
    "hybrid_sentiment",
    "shannon_entropy",
    "bigram_diversity",
    "sentiment_category",
]

BAYES_MAX_ROWS = int(os.getenv("BAYES_MAX_ROWS", "8000"))
BAYES_DRAWS = int(os.getenv("BAYES_DRAWS", "600"))
BAYES_TUNE = int(os.getenv("BAYES_TUNE", "600"))
BAYES_CHAINS = int(os.getenv("BAYES_CHAINS", "2"))
BAYES_TARGET_ACCEPT = float(os.getenv("BAYES_TARGET_ACCEPT", "0.9"))


def fmt_p(p_value):
    if p_value < 1e-4:
        return "<0.0001"
    return f"{p_value:.4f}"


def coefficient_table(result, logistic=False):
    ci = result.conf_int()
    ci.columns = ["ci_lower", "ci_upper"]
    table = pd.DataFrame(
        {
            "coef": result.params,
            "std_err": result.bse,
            "z_or_t": result.tvalues,
            "p_value": result.pvalues,
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
        }
    )
    if logistic:
        table["odds_ratio"] = np.exp(table["coef"])
        table["or_ci_lower"] = np.exp(table["ci_lower"])
        table["or_ci_upper"] = np.exp(table["ci_upper"])
    return table


def cramers_v(contingency):
    n = contingency.to_numpy().sum()
    if n == 0:
        return np.nan
    chi2 = chi2_contingency(contingency)[0]
    r, c = contingency.shape
    return math.sqrt(chi2 / (n * (min(r, c) - 1)))


def run_bayesian_linear(df, x_cols):
    # Subsample for tractable Bayesian sampling on large corpora.
    if len(df) > BAYES_MAX_ROWS:
        bayes_df = df.sample(BAYES_MAX_ROWS, random_state=42).copy()
    else:
        bayes_df = df.copy()

    X = bayes_df[x_cols].to_numpy()
    y = bayes_df["ai_prob"].to_numpy()

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
    summary.to_csv(OUT_DIR / "bayesian_linear_summary.csv")
    with open(OUT_DIR / "bayesian_linear_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary.to_string())

    beta_draws = trace.posterior["betas"].stack(sample=("chain", "draw")).values
    posterior_probs = pd.DataFrame(
        {
            "term": x_cols,
            "p_beta_gt_0": [(beta_draws[i] > 0).mean() for i in range(len(x_cols))],
            "p_beta_lt_0": [(beta_draws[i] < 0).mean() for i in range(len(x_cols))],
        }
    )
    posterior_probs.to_csv(OUT_DIR / "bayesian_posterior_direction_probs.csv", index=False)

    return summary, posterior_probs, len(bayes_df)


def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"Missing input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in {INPUT_CSV}: {', '.join(missing)}")

    # -------- Chi-square evidence --------
    contingency = pd.crosstab(df["sentiment_category"], df["is_ai"])
    chi2, p, dof, expected = chi2_contingency(contingency)
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
    c_v = cramers_v(contingency)

    contingency.to_csv(OUT_DIR / "chi_square_contingency.csv")
    expected_df.to_csv(OUT_DIR / "chi_square_expected.csv")
    chi_sq_stats = pd.DataFrame(
        [
            {
                "chi2": chi2,
                "p_value": p,
                "dof": dof,
                "n": int(contingency.to_numpy().sum()),
                "cramers_v": c_v,
            }
        ]
    )
    chi_sq_stats.to_csv(OUT_DIR / "chi_square_stats.csv", index=False)

    # -------- Regression evidence --------
    model_df = df[
        ["ai_prob", "is_ai", "hybrid_sentiment", "shannon_entropy", "bigram_diversity"]
    ].replace([np.inf, -np.inf], np.nan).dropna().copy()
    n_rows = len(model_df)
    if n_rows < 100:
        raise SystemExit(f"Not enough rows for stable regression after filtering: {n_rows}")

    predictors = ["hybrid_sentiment", "shannon_entropy", "bigram_diversity"]
    for col in predictors:
        std = model_df[col].std(ddof=0)
        if std == 0 or math.isnan(std):
            raise SystemExit(f"Column has zero variance and cannot be modeled: {col}")
        model_df[f"{col}_z"] = (model_df[col] - model_df[col].mean()) / std

    x_cols = [f"{c}_z" for c in predictors]
    X = sm.add_constant(model_df[x_cols], has_constant="add")

    ols_result = sm.OLS(model_df["ai_prob"], X).fit(cov_type="HC3")
    ols_table = coefficient_table(ols_result, logistic=False)
    ols_table.to_csv(OUT_DIR / "ols_coefficients.csv", index_label="term")
    with open(OUT_DIR / "ols_summary.txt", "w", encoding="utf-8") as f:
        f.write(ols_result.summary().as_text())

    glm_result = sm.GLM(model_df["is_ai"], X, family=sm.families.Binomial()).fit(cov_type="HC3")
    glm_table = coefficient_table(glm_result, logistic=True)
    glm_table.to_csv(OUT_DIR / "logit_coefficients.csv", index_label="term")
    with open(OUT_DIR / "logit_summary.txt", "w", encoding="utf-8") as f:
        f.write(glm_result.summary().as_text())

    # -------- Bayesian evidence --------
    bayes_summary, posterior_probs, bayes_n = run_bayesian_linear(model_df, x_cols)

    metrics = pd.DataFrame(
        [
            {
                "model": "OLS(ai_prob)",
                "n_rows": n_rows,
                "r_squared": float(ols_result.rsquared),
                "aic": float(ols_result.aic),
            },
            {
                "model": "GLM_Binomial(is_ai)",
                "n_rows": n_rows,
                "r_squared": np.nan,
                "aic": float(glm_result.aic),
            },
            {
                "model": "BayesianLinear(ai_prob)",
                "n_rows": bayes_n,
                "r_squared": np.nan,
                "aic": np.nan,
            },
        ]
    )
    metrics.to_csv(OUT_DIR / "model_metrics.csv", index=False)

    report_lines = [
        "Statistical Evidence Report",
        f"Rows used (frequentist): {n_rows}",
        f"Rows used (Bayesian): {bayes_n}",
        "",
        "Chi-square test: sentiment_category vs is_ai",
        f"- chi2={chi2:.4f}, p={fmt_p(float(p))}, dof={int(dof)}, Cramer's V={c_v:.4f}",
        "",
        "OLS model: ai_prob ~ hybrid_sentiment_z + shannon_entropy_z + bigram_diversity_z",
        f"- R-squared={ols_result.rsquared:.4f}, AIC={ols_result.aic:.2f}",
        "",
        "Binomial GLM: is_ai ~ hybrid_sentiment_z + shannon_entropy_z + bigram_diversity_z",
        f"- AIC={glm_result.aic:.2f}",
        "",
        "Key p-values (OLS):",
    ]
    for term in x_cols:
        report_lines.append(f"- {term}: p={fmt_p(float(ols_result.pvalues[term]))}")
    report_lines.append("")
    report_lines.append("Key p-values (GLM):")
    for term in x_cols:
        report_lines.append(f"- {term}: p={fmt_p(float(glm_result.pvalues[term]))}")
    report_lines.append("")
    report_lines.append("Bayesian posterior direction probabilities:")
    for _, row in posterior_probs.iterrows():
        report_lines.append(
            f"- {row['term']}: P(beta>0)={row['p_beta_gt_0']:.3f}, P(beta<0)={row['p_beta_lt_0']:.3f}"
        )
    report_lines.append("")
    report_lines.append("Output files:")
    report_lines.append("- stats/chi_square_stats.csv")
    report_lines.append("- stats/chi_square_contingency.csv")
    report_lines.append("- stats/chi_square_expected.csv")
    report_lines.append("- stats/ols_summary.txt")
    report_lines.append("- stats/logit_summary.txt")
    report_lines.append("- stats/ols_coefficients.csv")
    report_lines.append("- stats/logit_coefficients.csv")
    report_lines.append("- stats/bayesian_linear_summary.csv")
    report_lines.append("- stats/bayesian_posterior_direction_probs.csv")
    report_lines.append("- stats/model_metrics.csv")

    report = "\n".join(report_lines)
    (OUT_DIR / "regression_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
