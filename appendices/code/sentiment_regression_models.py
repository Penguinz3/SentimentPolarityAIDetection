from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


INPUT_CSV = Path("corpus_with_results.csv")
OUT_DIR = Path("stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = ["ai_prob", "hf_sentiment", "hybrid_sentiment"]


def fit_ols(y, X):
    X_const = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X_const).fit(cov_type="HC3")


def coef_row(model_name, result, term):
    ci = result.conf_int().loc[term]
    return {
        "model": model_name,
        "term": term,
        "coef": float(result.params[term]),
        "std_err": float(result.bse[term]),
        "t_value": float(result.tvalues[term]),
        "p_value": float(result.pvalues[term]),
        "ci_lower": float(ci.iloc[0]),
        "ci_upper": float(ci.iloc[1]),
        "r_squared": float(result.rsquared),
        "aic": float(result.aic),
    }


def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"Missing input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}")

    model_df = df[REQUIRED_COLUMNS].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(model_df) < 100:
        raise SystemExit(f"Not enough valid rows for regression: {len(model_df)}")

    y = model_df["ai_prob"]

    # Model 1: AI_Prob ~ HF_Sentiment
    m1 = fit_ols(y, model_df[["hf_sentiment"]])
    (OUT_DIR / "model1_hf_summary.txt").write_text(m1.summary().as_text(), encoding="utf-8")

    # Model 2: AI_Prob ~ Hybrid_Sentiment
    m2 = fit_ols(y, model_df[["hybrid_sentiment"]])
    (OUT_DIR / "model2_hybrid_summary.txt").write_text(m2.summary().as_text(), encoding="utf-8")

    # Model 3: AI_Prob ~ HF_Sentiment + Hybrid_Sentiment
    m3 = fit_ols(y, model_df[["hf_sentiment", "hybrid_sentiment"]])
    (OUT_DIR / "model3_combined_summary.txt").write_text(m3.summary().as_text(), encoding="utf-8")

    coef_rows = [
        coef_row("Model1_HF", m1, "hf_sentiment"),
        coef_row("Model2_Hybrid", m2, "hybrid_sentiment"),
        coef_row("Model3_Combined", m3, "hf_sentiment"),
        coef_row("Model3_Combined", m3, "hybrid_sentiment"),
    ]
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(OUT_DIR / "sentiment_regression_coefficients.csv", index=False)

    metrics_df = pd.DataFrame(
        [
            {"model": "Model1_HF", "r_squared": float(m1.rsquared), "aic": float(m1.aic), "n_rows": len(model_df)},
            {"model": "Model2_Hybrid", "r_squared": float(m2.rsquared), "aic": float(m2.aic), "n_rows": len(model_df)},
            {"model": "Model3_Combined", "r_squared": float(m3.rsquared), "aic": float(m3.aic), "n_rows": len(model_df)},
        ]
    )
    metrics_df.to_csv(OUT_DIR / "sentiment_regression_metrics.csv", index=False)

    # VIF diagnostics for combined model predictors.
    X_vif = sm.add_constant(model_df[["hf_sentiment", "hybrid_sentiment"]], has_constant="add")
    vif_df = pd.DataFrame(
        {
            "term": X_vif.columns,
            "vif": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
        }
    )
    vif_df.to_csv(OUT_DIR / "sentiment_regression_vif.csv", index=False)

    report = [
        "Sentiment Regression Report (No Bayesian Models)",
        f"Rows used: {len(model_df)}",
        "",
        "Model 1: AI_Prob ~ HF_Sentiment",
        f"- beta_hf={m1.params['hf_sentiment']:.6f}, p={m1.pvalues['hf_sentiment']:.6g}, R^2={m1.rsquared:.6f}",
        "",
        "Model 2: AI_Prob ~ Hybrid_Sentiment",
        f"- beta_hybrid={m2.params['hybrid_sentiment']:.6f}, p={m2.pvalues['hybrid_sentiment']:.6g}, R^2={m2.rsquared:.6f}",
        "",
        "Model 3: AI_Prob ~ HF_Sentiment + Hybrid_Sentiment",
        f"- beta_hf={m3.params['hf_sentiment']:.6f}, p={m3.pvalues['hf_sentiment']:.6g}",
        f"- beta_hybrid={m3.params['hybrid_sentiment']:.6f}, p={m3.pvalues['hybrid_sentiment']:.6g}",
        f"- R^2={m3.rsquared:.6f}",
        "",
        "VIF (combined model):",
    ]
    for _, row in vif_df.iterrows():
        report.append(f"- {row['term']}: {row['vif']:.4f}")
    report.append("")
    report.append("Outputs:")
    report.append("- stats/model1_hf_summary.txt")
    report.append("- stats/model2_hybrid_summary.txt")
    report.append("- stats/model3_combined_summary.txt")
    report.append("- stats/sentiment_regression_coefficients.csv")
    report.append("- stats/sentiment_regression_metrics.csv")
    report.append("- stats/sentiment_regression_vif.csv")

    report_text = "\n".join(report)
    (OUT_DIR / "sentiment_regression_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()
