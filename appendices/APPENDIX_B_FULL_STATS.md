## APPENDIX B
## FULL STATISTICAL OUTPUT TABLES

This appendix provides complete statistical outputs referenced in the main Results section. Tables include full regression summaries, contingency table counts, and associated inferential statistics to ensure analytical transparency and reproducibility.

### B.1 Chi-Square Contingency Table

**TABLE B1**  
Observed Frequencies: Sentiment Category x AI Classification

| Sentiment Category | AI-Negative (< 0.5) | AI-Positive (>= 0.5) | Total |
|---|---:|---:|---:|
| Negative | 21471 | 3178 | 24649 |
| Neutral  | 3221  | 435  | 3656  |
| Positive | 3774  | 499  | 4273  |
| **Total** | **28466** | **4112** | **32578** |

**Chi-Square Test Output**
- \(\chi^2(df=2) = 6.8308\)
- \(p = 0.0329\)
- \(\alpha = 0.05\)

**Effect Size (Cramer's V)**  
\[
V = \sqrt{ \frac{\chi^2}{N(k-1)} }
\]
With \(N = 32578\) and \(k = \min(3,2)=2\):  
- \(V = \sqrt{6.8308 / 32578} = 0.01448\)

**Interpretation.** The association is statistically significant at \(\alpha=0.05\), but the effect size is very small (\(V \approx 0.0145\)), consistent with a weak practical association in a large sample.

### B.2 OLS Regression Model 1 (AI Probability ~ HF Sentiment)

\[
AI\_Prob = \beta_0 + \beta_1(HF\_Sentiment) + \epsilon
\]

**TABLE B2**  
OLS Regression Summary (Model 1; HC3 robust SE; N = 32,578)

| Parameter | Coefficient (β) | Std. Error | z | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| Intercept | 0.1325 | 0.0025 | 53.721 | <0.001 | [0.1277, 0.1374] |
| HF_Sentiment | -0.0063 | 0.0026 | -2.471 | 0.0135 | [-0.0113, -0.0013] |

Model statistics:
- \(R^2 = 0.000177\)
- Adjusted \(R^2 \approx 0.000146\)
- F-statistic \(= 6.108\)
- df \(= (1,\ 32576)\)

### B.3 OLS Regression Model 2 (AI Probability ~ Hybrid Sentiment)

\[
AI\_Prob = \beta_0 + \beta_1(Hybrid\_Sentiment) + \epsilon
\]

**TABLE B3**  
OLS Regression Summary (Model 2; HC3 robust SE; N = 32,578)

| Parameter | Coefficient (β) | Std. Error | z | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| Intercept | 0.1334 | 0.0018 | 73.939 | <0.001 | [0.1299, 0.1370] |
| Hybrid_Sentiment | -0.0138 | 0.0032 | -4.284 | 1.83e-05 | [-0.0201, -0.0075] |

Model statistics:
- \(R^2 = 0.000536\)
- Adjusted \(R^2 \approx 0.000505\)
- F-statistic \(= 18.35\)
- df \(= (1,\ 32576)\)

### B.4 OLS Regression Model 3 (AI Probability ~ HF Sentiment + Hybrid Sentiment)

\[
AI\_Prob = \beta_0 + \beta_1(HF\_Sentiment) + \beta_2(Hybrid\_Sentiment) + \epsilon
\]

**TABLE B4**  
OLS Regression Summary (Model 3; HC3 robust SE; N = 32,578)

| Parameter | Coefficient (β) | Std. Error | z | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| Intercept | 0.1356 | 0.0026 | 51.159 | <0.001 | [0.1304, 0.1408] |
| HF_Sentiment | 0.0045 | 0.0040 | 1.118 | 0.263 | [-0.0034, 0.0124] |
| Hybrid_Sentiment | -0.0180 | 0.0051 | -3.565 | 0.000363 | [-0.0279, -0.0081] |

Model statistics:
- \(R^2 = 0.000574\)
- Adjusted \(R^2 \approx 0.000513\)
- F-statistic \(= 9.637\)
- df \(= (2,\ 32575)\)

### B.5 Confidence Interval Summary

**TABLE B5**  
95% Confidence Intervals for Sentiment Predictors

| Model | Predictor | Lower Bound | Upper Bound |
|---|---|---:|---:|
| Model 1 | HF_Sentiment | -0.0113 | -0.0013 |
| Model 2 | Hybrid_Sentiment | -0.0201 | -0.0075 |
| Model 3 | HF_Sentiment | -0.0034 | 0.0124 |
| Model 3 | Hybrid_Sentiment | -0.0279 | -0.0081 |

### B.6 Practical Effect Size Interpretation

Across Models 1–3, all \(R^2\) values are below 0.001. This indicates that sentiment polarity explains less than 0.1% of the variance in AI detection probability. Therefore, even when coefficients are statistically significant, the practical predictive influence of sentiment on AI probability is negligible in this dataset.
