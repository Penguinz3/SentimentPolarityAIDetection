# Methodology

## Dataset Source

The corpus pipeline is designed for PubMed Central (PMC) / JATS XML files. `src/build_corpus.py` parses article metadata and body sections, then chunks section text into sentence-based analysis units. The repository includes `data_sample/sample_chunks.csv` for lightweight smoke tests; full local data artifacts are kept out of the public-facing top level.

## Why Pre-2020 Human-Written Text

The project uses pre-2020 human-written scientific text to reduce the likelihood that the source corpus contains modern large-language-model output. This does not make the corpus perfect, but it gives a defensible baseline for studying false positives: detector-positive labels in this setting are treated as cases where human-written text is scored as AI-like.

## False Positive Definition

The detector produces a probability-like score, stored as `ai_prob` or `ai_probability`. A false positive is defined as a human-written text chunk with a detector score at or above the chosen threshold, currently `0.5` by default. The binary target is stored as `ai_positive`; `is_ai` is retained as a legacy alias.

The active detector script can run multiple configured Hugging Face detector models. When more than one model completes successfully, it keeps the per-model probability columns and aggregates them into `ai_prob`; analysis scripts treat those detector-output columns as targets or metadata rather than predictor features.

## Sentiment Measurement

The original project measured sentiment using VADER and an optional Hugging Face sentiment model. The hybrid sentiment score is the average of `vader_compound` and `hf_sentiment`, and `sentiment_category` bins the hybrid score into Positive, Neutral, and Negative using the original `0.05` thresholds.

The active pipeline keeps the original chi-square test between `sentiment_category` and the detector-positive target. This test is reported as an association check, not as a strong explanatory claim about detector behavior.

## Why Add Entropy And Stylometric Features

The initial sentiment analysis found statistically detectable but practically weak explanatory power. The expanded framework therefore adds interpretable signals that may better capture detector behavior: Shannon entropy, word-level entropy, n-gram transition entropy, lexical diversity, repetition, punctuation density, and sentence-length variation.

## Why PCA Is Used

PCA is used for interpretable feature-space analysis. The PCA script standardizes numeric linguistic features, excludes detector outputs and target labels, saves component coordinates and loadings, and plots the strongest feature contributions. PCA is not treated as proof of class separation; it is used to inspect which feature combinations explain the largest variance in the extracted feature space.
