## Appendix A: Research Instrument (Computational)

This study used a computational research instrument consisting of a corpus construction pipeline, a sentiment scoring pipeline, an AI-detection pipeline, and statistical evaluation scripts.

### A1. Inputs

- Raw documents: PubMed Central (PMC) full-text XML files (`data/raw/PMC/*.xml`).
- Unit of analysis: text "chunks" derived from section text in each document.

### A2. Corpus Construction Instrument

- Parser: extracts article metadata (journal, year, PMC ID) and body section text from JATS XML.
- Chunking: section text is sentence-split and grouped into fixed-size sentence chunks.
- Output: `corpus_chunks.csv` containing one row per chunk with:
  - identifiers (`chunk_id`, `doc_id`)
  - metadata (`source`, `year`, `journal`, `section`)
  - chunking fields (`chunk_index`, `start_sentence`, `n_sentences`, `word_count`)
  - text (`text`)
  - provenance (`raw_doc_path`)

### A3. Sentiment Scoring Instrument

- VADER sentiment: compound score computed per chunk.
- Transformer sentiment: HF pipeline sentiment score computed per chunk.
- Hybrid sentiment: average of VADER compound and transformer score.
- Output columns written to `corpus_with_results.csv`:
  - `vader_compound`, `hf_sentiment`, `hybrid_sentiment`, `sentiment_category`

### A4. AI Detection Instrument

- AI detector: transformer text-classification pipeline producing an AI probability (`ai_prob` in [0,1]).
- Binary label: `is_ai = 1` when `ai_prob >= 0.5`, otherwise `0`.
- Detection logs:
  - `detection_summary.csv`
  - `positive_detections.csv`

### A5. Statistical Evaluation Instrument

- Chi-square test: association between `sentiment_category` and `is_ai`.
- Linear regression (OLS, robust SE): models with `ai_prob` as dependent variable and sentiment predictors.
- Collinearity diagnostics: VIF for combined predictor models.
- Outputs in `stats/` include:
  - chi-square tables/stats
  - regression summaries and coefficient tables
