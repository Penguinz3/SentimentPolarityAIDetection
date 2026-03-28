import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency


INPUT_CSV = Path("corpus_with_results.csv")
OUT_DIR = Path("stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"Missing input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = ["sentiment_category", "is_ai", "ai_prob"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}")

    total = int(len(df))
    pos = int((df["is_ai"] == 1).sum())
    pos_rate = pos / total if total else np.nan

    summary = pd.DataFrame(
        [
            {
                "total_rows": total,
                "ai_positive_rows": pos,
                "ai_positive_rate": pos_rate,
                "ai_prob_mean": float(df["ai_prob"].mean()),
                "ai_prob_median": float(df["ai_prob"].median()),
            }
        ]
    )
    summary.to_csv(OUT_DIR / "detection_summary_stats.csv", index=False)

    contingency = pd.crosstab(df["sentiment_category"], df["is_ai"])
    contingency.to_csv(OUT_DIR / "chi_square_contingency.csv")

    chi2, p, dof, expected = chi2_contingency(contingency)
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
    expected_df.to_csv(OUT_DIR / "chi_square_expected.csv")

    chi_stats = pd.DataFrame(
        [
            {
                "chi2": float(chi2),
                "p_value": float(p),
                "dof": int(dof),
                "n": int(contingency.to_numpy().sum()),
            }
        ]
    )
    chi_stats.to_csv(OUT_DIR / "chi_square_stats.csv", index=False)

    print("Wrote:")
    print(f"- {OUT_DIR / 'detection_summary_stats.csv'}")
    print(f"- {OUT_DIR / 'chi_square_contingency.csv'}")
    print(f"- {OUT_DIR / 'chi_square_expected.csv'}")
    print(f"- {OUT_DIR / 'chi_square_stats.csv'}")


if __name__ == "__main__":
    main()

