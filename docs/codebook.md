# Codebook

## Corpus Columns

- `chunk_id`: Unique text-chunk identifier.
- `doc_id`: Source document identifier.
- `source`: Corpus source label, such as `PMC`.
- `year`: Publication year when available.
- `journal`: Journal title when available.
- `section`: Article section label.
- `chunk_index`: Chunk order within a section.
- `start_sentence`: Starting sentence offset within the section.
- `n_sentences`: Number of source sentences in the chunk.
- `text`: Text chunk used as the analysis unit.
- `raw_doc_path`: Local source XML path when available.

## Sentiment Features

- `vader_compound`: VADER compound sentiment score.
- `hf_sentiment`: Optional transformer sentiment score mapped to positive and negative signed values.
- `hybrid_sentiment`: Average of `vader_compound` and `hf_sentiment`.
- `sentiment_category`: Positive, Neutral, or Negative category derived from `hybrid_sentiment`.

## Entropy Features

- `char_shannon_entropy`: Shannon entropy over characters.
- `word_shannon_entropy`: Shannon entropy over word tokens.
- `bigram_transition_entropy`: Conditional entropy of the next word given the previous word.
- `trigram_transition_entropy`: Conditional entropy of the next word given the previous two words.
- `sentence_entropy_mean`: Mean word-level entropy across sentences.
- `sentence_entropy_std`: Standard deviation of word-level entropy across sentences.

## Stylometric Features

- `word_count`: Number of word tokens.
- `sentence_count`: Number of detected sentences.
- `avg_sentence_length`: Mean words per sentence.
- `sentence_length_std`: Standard deviation of sentence lengths in words.
- `avg_word_length`: Mean character length of word tokens.
- `type_token_ratio`: Unique word types divided by total word tokens.
- `unique_word_ratio`: Words appearing once divided by total word tokens.
- `repetition_rate`: Repeated-token share, computed as `(tokens - unique types) / tokens`.
- `punctuation_density`: Punctuation characters divided by total characters.

## Detector Output Columns

- `ai_prob`: Detector probability-like score for AI-generated text.
- `ai_prob_<model_slug>`: Per-detector probability-like score when multiple Hugging Face detector models are run.
- `ai_probability`: Alternate detector probability column name supported by analysis scripts.
- `detector_score`: Alternate detector score column name supported by analysis scripts.
- `detector_models_used`: Comma-separated per-model detector score columns used to compute `ai_prob`.

## Target Columns

- `ai_positive`: Binary detector-positive label, using the configured threshold.
- `is_ai`: Legacy alias for `ai_positive`.

These target and detector-output columns are excluded from PCA and model feature sets by default.

## PCA Output Columns

- `PC1`, `PC2`, ...: PCA component coordinates for each row.
- `feature`: Feature name in the PCA loadings table.
- `max_abs_loading`: Largest absolute loading for a feature across retained components.
- `explained_variance_ratio`: Share of standardized feature variance explained by a component.
- `cumulative_explained_variance_ratio`: Cumulative variance share through a component.

## Model Output Columns

- `model`: Model name, such as `logistic_regression` or `random_forest`.
- `accuracy`, `precision`, `recall`, `f1`: Classification metrics on the held-out test split.
- `roc_auc`: ROC-AUC when the test split contains both classes.
- `importance_type`: Either standardized logistic coefficient or random forest importance.
- `importance`: Signed coefficient or nonnegative feature importance.
- `importance_abs`: Absolute importance used for ranking.

## Impact Analysis Output Columns

- `importance_mean`: Mean permutation importance score on the held-out test set.
- `importance_std`: Standard deviation of permutation importance across repeats.
- `coefficient`: Standardized logistic regression coefficient.
- `coefficient_abs`: Absolute standardized logistic coefficient.
- `mean_ai_positive_1`: Feature mean among detector-positive chunks.
- `mean_ai_positive_0`: Feature mean among detector-negative chunks.
- `standardized_difference`: Difference between detector-positive and detector-negative means divided by pooled standard deviation.
- `feature_group`: Feature family removed during ablation.
- `roc_auc_drop`, `f1_drop`, `accuracy_drop`: Performance drop after removing a feature group.

## Statistical Test Output Columns

- `sentiment_col`: Categorical sentiment column used in the chi-square test.
- `target_col`: Detector-positive target column used in the chi-square test.
- `chi2`: Chi-square test statistic.
- `p_value`: Chi-square p-value.
- `dof`: Degrees of freedom.
- `n`: Number of valid rows in the contingency table.
- `cramers_v`: Effect-size estimate for the sentiment-category association.
