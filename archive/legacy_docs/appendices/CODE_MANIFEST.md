## Appendix D: Code Manifest (Reproducibility)

This appendix lists the analysis scripts used to generate the corpus, run AI/sentiment detection, and produce statistical evidence.

### C1. Corpus construction
- `pmc_parser.py`: Parses PMC/JATS XML and extracts metadata and section text.
- `build_corpus_chunks.py`: Builds `corpus_chunks.csv` from `data/raw/PMC/*.xml`.

Run:
```powershell
python .\build_corpus_chunks.py --pmc-dir ..\raw\PMC --out corpus_chunks.csv --sentences-per-chunk 8
```

### C2. AI + sentiment detection and logging
- `anlysis.py`: Loads `corpus_chunks.csv`, computes lexical features and sentiment scores, runs AI detection, writes outputs, checkpoints, and plots.

Run:
```powershell
.\research_env\Scripts\python.exe -u .\anlysis.py .\corpus_chunks.csv
```

### C3. Statistical tests (chi-square + detection summary)
- `statistical_tests_runner.py`: Reads `corpus_with_results.csv` and writes chi-square tables/stats plus detection summary stats into `stats/`.

Run:
```powershell
.\research_env\Scripts\python.exe .\statistical_tests_runner.py
```

### C4. Linear regression models + VIF
- `sentiment_regression_models.py`: Runs three OLS models (HF-only, Hybrid-only, Combined) and writes summaries/coefficients/VIF into `stats/`.

Run:
```powershell
.\research_env\Scripts\python.exe .\sentiment_regression_models.py
```

### C5. Key outputs
- `corpus_chunks.csv`: chunked corpus (analysis input).
- `corpus_with_results.csv`: main results table (analysis output).
- `stats/*`: statistical evidence files used in Results/Discussion.
- `plots/*`: figures used in the paper.
