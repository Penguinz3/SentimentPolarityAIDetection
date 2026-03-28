## Appendix C: Codebook

This codebook defines the variables produced by the corpus pipeline and analysis scripts.

### B1. `corpus_chunks.csv` variables
- `chunk_id` (string): Unique identifier for the chunk.
- `doc_id` (string): Document identifier (e.g., PMCID).
- `source` (string): Source label (e.g., `PMC`).
- `year` (string/int): Publication year when available.
- `journal` (string): Journal title when available.
- `section` (string): Section label extracted from the article body (lowercased).
- `chunk_index` (int): Sequential chunk index within a document section.
- `start_sentence` (int): Starting sentence offset within the section text used for chunking.
- `n_sentences` (int): Number of sentences in the chunk.
- `word_count` (int): Whitespace token count in `text`.
- `text` (string): Chunk text (analysis unit).
- `raw_doc_path` (string): Path to the source document file.

### B2. Additional columns in `corpus_with_results.csv`
Lexical features:
- `shannon_entropy` (float): Character-level Shannon entropy of `text`.
- `bigram_diversity` (float): Ratio `unique_bigrams / total_bigrams` based on word bigrams in `text`.

Sentiment:
- `vader_compound` (float): VADER compound sentiment score in approximately `[-1, 1]`.
- `hf_sentiment` (float): Transformer sentiment score mapped to `[-1, 1]` (positive = positive, negative = negative).
- `hybrid_sentiment` (float): `0.5*vader_compound + 0.5*hf_sentiment`.
- `sentiment_category` (string): Discretized sentiment label:
  - `Positive` if `hybrid_sentiment > 0.05`
  - `Negative` if `hybrid_sentiment < -0.05`
  - `Neutral` otherwise

AI detection:
- `ai_prob` (float): Model output probability that the chunk is AI-generated (range `[0,1]`).
- `is_ai` (int): Binary label derived from `ai_prob`:
  - `1` if `ai_prob >= 0.5`
  - `0` otherwise

### B3. Derived outputs
Detection logs:
- `detection_summary.csv`: dataset-level totals and means (counts, AI-positive rate, mean probability).
- `positive_detections.csv`: all rows with `is_ai = 1`, sorted by `ai_prob` descending (subset of columns).

Chi-square outputs:
- `chi_square_contingency.csv`: cross-tab of `sentiment_category` by `is_ai`.
- `chi_square_expected.csv`: expected cell counts under independence.
- `stats/chi_square_stats.csv`: chi-square statistic, p-value, degrees of freedom.

Regression outputs:
- `stats/model1_hf_summary.txt`: OLS summary for `ai_prob ~ hf_sentiment`.
- `stats/model2_hybrid_summary.txt`: OLS summary for `ai_prob ~ hybrid_sentiment`.
- `stats/model3_combined_summary.txt`: OLS summary for `ai_prob ~ hf_sentiment + hybrid_sentiment`.
- `stats/sentiment_regression_coefficients.csv`: coefficients, SE, p-values, confidence intervals, R^2, AIC.
- `stats/sentiment_regression_metrics.csv`: model-level metrics (R^2, AIC, n).
- `stats/sentiment_regression_vif.csv`: VIF diagnostics for combined model.

Plots (`plots/*.png`):
- `ai_probability_hist.png`: histogram of `ai_prob`.
- `hybrid_sentiment_hist.png`: histogram of `hybrid_sentiment`.
- `chi_square_contingency_heatmap.png`: heatmap of contingency table counts.
- `hybrid_vs_ai_scatter.png`: scatter of `hybrid_sentiment` vs `ai_prob` with fitted line.
- `top_bigrams_barh.png`: most frequent bigrams in the corpus.
