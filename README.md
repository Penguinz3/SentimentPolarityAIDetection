# Multi-Signal AI Detector Reliability Analysis

This project studies the reliability of probabilistic AI-text detectors by analyzing false positives in human-written text with interpretable linguistic and statistical features. It began as an investigation of whether sentiment polarity is associated with AI-detector false positives. The initial sentiment-focused analysis indicated statistically detectable but practically weak explanatory power, so sentiment is treated as one signal rather than a primary explanation. This motivates a broader framework that evaluates entropy, structural text features, and PCA-based feature-space analysis.

## Overview

The repository is an independent research/data science project about AI-detector reliability. The central idea is to treat detector false positives as an object of analysis: when a probabilistic detector labels human-written text as AI-like, which interpretable features are associated with that outcome?

The project keeps the original sentiment analysis work, but expands it into a broader feature pipeline and avoids claiming that any one signal explains detector behavior on its own.

## Research Question

Which interpretable text features are associated with false positives in probabilistic AI-text detection?

## Current Finding

The initial sentiment-only analysis suggests that sentiment polarity can be statistically associated with detector output, but the explanatory power is practically weak. That motivates testing additional signals such as entropy, n-gram transition structure, lexical diversity, repetition, sentence-length variation, and PCA feature-space structure.

## Pipeline

1. Build or load a chunked human-written text corpus.
2. Compute sentiment features and detector probabilities.
3. Add entropy-based features.
4. Add stylometric features.
5. Run the sentiment-category association test preserved from the original analysis.
6. Run PCA for interpretable feature-space analysis.
7. Train simple predictive models to test which extracted features are associated with detector positives.

## Features Extracted

- Sentiment: VADER compound score, optional transformer sentiment score, hybrid sentiment, sentiment category.
- Entropy: character entropy, word entropy, bigram transition entropy, trigram transition entropy, sentence entropy mean and standard deviation.
- Stylometry: word count, sentence count, average sentence length, sentence-length variation, average word length, type-token ratio, unique-word ratio, repetition rate, punctuation density.

## Methods

- Probabilistic AI-text detection with one or more configured Hugging Face transformer classifiers when available.
- Shannon entropy and n-gram transition entropy for uncertainty and transition-structure features.
- PCA on standardized non-leaking numeric linguistic features.
- Logistic regression and random forest classifiers for feature association checks.

PCA and models exclude target labels and detector-score columns by default to reduce leakage.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

If you need to rebuild detector outputs from chunked text:

```bash
python src/sentiment_features.py --input data_sample/sample_chunks.csv --text-col text --output outputs/features_sentiment.csv --skip-hf
python src/detector_inference.py --input outputs/features_sentiment.csv --text-col text --output outputs/corpus_with_results.csv
```

`detector_inference.py` attempts all configured detector models by default, writes per-model probability columns, and aggregates successful detector scores into `ai_prob` using `--aggregation mean`. Use `--first-available` if you want the older fallback behavior.

Run the multi-signal feature and analysis pipeline:

```bash
python src/entropy_features.py --input outputs/corpus_with_results.csv --text-col text --output outputs/features_entropy.csv
python src/stylometric_features.py --input outputs/features_entropy.csv --text-col text --output outputs/features_full.csv
python src/statistical_tests.py --input outputs/features_full.csv --target-col ai_positive --output-dir outputs
python src/pca_analysis.py --input outputs/features_full.csv --target-col ai_positive --output-dir outputs --plot-dir plots
python src/model_analysis.py --input outputs/features_full.csv --target-col ai_positive --output-dir outputs --plot-dir plots
python src/impact_analysis.py --input outputs/features_full.csv --target-col ai_positive --output-dir outputs --plot-dir plots
```

Large text-bearing intermediates such as `outputs/corpus_with_results.csv` and `outputs/features_full.csv` are treated as generated local artifacts and ignored by git. The rows in `data_sample/` are small illustrative smoke-test inputs, not research evidence.

## Interpreting Feature Impact

`src/impact_analysis.py` adds model-based diagnostics for understanding which extracted features are most associated with detector-positive false positives. Permutation importance shows which features matter most for model prediction when their values are disrupted, logistic coefficients show direction of association on standardized features, feature comparisons show how flagged and non-flagged chunks differ, and group ablation tests show which feature groups contribute most to predictive performance. These analyses suggest association and predictive contribution, not proof of causation.

## Repo Structure

```text
SentimentPolarityAIDetection/
  src/              # Pipeline modules
  data_sample/      # Small sample data for smoke tests
  outputs/          # Final summaries and generated analysis outputs
  plots/            # Generated figures
  docs/             # Methodology, codebook, and paper notes
  archive/          # Preserved legacy scripts, stats, and local-only large artifacts
```

## Outputs

- `outputs/detection_summary.csv`
- `outputs/model_metrics.csv`
- `outputs/feature_importance.csv`
- `outputs/impact_model_metrics.csv`
- `outputs/permutation_importance.csv`
- `outputs/logistic_coefficients.csv`
- `outputs/false_positive_feature_comparison.csv`
- `outputs/feature_group_ablation.csv`
- `outputs/sentiment_ai_chi_square.csv`
- `outputs/pca_coordinates.csv`
- `outputs/pca_loadings.csv`
- `outputs/pca_explained_variance.csv`
- `plots/ai_probability_distribution.png`
- `plots/sentiment_distribution.png`
- `plots/entropy_vs_ai_probability.png`
- `plots/pca_false_positive_map.png`
- `plots/pca_loadings.png`
- `plots/feature_importance.png`
- `plots/permutation_importance.png`
- `plots/logistic_coefficients.png`
- `plots/false_positive_feature_differences.png`
- `plots/feature_group_ablation.png`

## Future Work

- Report detector-specific behavior when multiple detector models are run.
- Add stronger validation around corpus provenance and pre-2020 filtering.
- Test feature robustness across domains, genres, and chunk lengths.
- Report model results as associations, not causal explanations.
