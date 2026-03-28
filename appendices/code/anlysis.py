import importlib
import importlib.util
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


REQUIRED_MODULES = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("nltk", "nltk"),
    ("nltk.sentiment", "nltk"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("scipy.stats", "scipy"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
]


def is_missing(module_name):
    try:
        return importlib.util.find_spec(module_name) is None
    except ModuleNotFoundError:
        return True


missing = [(module, package) for module, package in REQUIRED_MODULES if is_missing(module)]
if missing:
    missing_modules = ", ".join(module for module, _ in missing)
    install_packages = " ".join(sorted({package for _, package in missing}))
    raise SystemExit(
        f"Missing modules: {missing_modules}\n"
        f"Active interpreter: {sys.executable}\n"
        f"Install into this interpreter with:\n"
        f'  "{sys.executable}" -m pip install {install_packages}'
    )


pd = importlib.import_module("pandas")
np = importlib.import_module("numpy")
nltk = importlib.import_module("nltk")
nltk_sentiment = importlib.import_module("nltk.sentiment")
torch = importlib.import_module("torch")
transformers = importlib.import_module("transformers")
scipy_stats = importlib.import_module("scipy.stats")
tqdm_mod = importlib.import_module("tqdm.auto")

matplotlib = importlib.import_module("matplotlib")
matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")
transformers.logging.set_verbosity_error()
tqdm = tqdm_mod.tqdm

SentimentIntensityAnalyzer = nltk_sentiment.SentimentIntensityAnalyzer
pipeline = transformers.pipeline
chi2_contingency = scipy_stats.chi2_contingency


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        print(f"Warning: malformed rows detected in {path}. Retrying with bad-line skipping.")
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def shannon_entropy(text):
    text = str(text)
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


WORD_RE = re.compile(r"\b\w+\b")


def tokenize(text):
    return WORD_RE.findall(str(text).lower())


def bigrams_from_tokens(tokens):
    return list(zip(tokens, tokens[1:]))


def bigram_diversity(text):
    tokens = tokenize(text)
    bigrams = bigrams_from_tokens(tokens)
    if not bigrams:
        return 0.0
    return len(set(bigrams)) / len(bigrams)


def top_corpus_bigrams(text_series, top_n=20):
    counter = Counter()
    for text in text_series.astype(str):
        counter.update(bigrams_from_tokens(tokenize(text)))
    return counter.most_common(top_n)


def ai_score_from_prediction(pred):
    label = str(pred.get("label", "")).lower()
    score = float(pred.get("score", 0.0))
    ai_like = {"ai", "generated", "fake", "label_1", "1"}
    if label in ai_like or "ai" in label:
        return score
    return 1.0 - score


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def build_ai_detector():
    candidate_models = [
        "SuperAnnotate/ai-detector",
        "roberta-base-openai-detector",
    ]
    for model_name in candidate_models:
        try:
            detector = pipeline(
                "text-classification",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
            )
            print(f"Using AI detector model: {model_name}")
            return detector
        except Exception as exc:
            print(f"Warning: failed to load AI detector model '{model_name}': {exc}")
    raise SystemExit(
        "No compatible AI detector model could be loaded. "
        "Check network access or set a local compatible model."
    )


INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "corpus_chunks.csv"
MAX_ROWS = int(os.getenv("MAX_ROWS", "0"))
HF_BATCH_SIZE = int(os.getenv("HF_BATCH_SIZE", "32"))
SKIP_HF_MODELS = os.getenv("SKIP_HF_MODELS", "0") == "1"
VERBOSE = os.getenv("VERBOSE", "1") == "1"
CHECKPOINT_EVERY_BATCHES = int(os.getenv("CHECKPOINT_EVERY_BATCHES", "50"))
CHECKPOINT_EVERY_SECONDS = int(os.getenv("CHECKPOINT_EVERY_SECONDS", "300"))

checkpoints_dir = Path("checkpoints")
checkpoints_dir.mkdir(parents=True, exist_ok=True)
PARTIAL_RESULTS_PATH = checkpoints_dir / "corpus_with_results.partial.csv"
RUN_STATUS_PATH = checkpoints_dir / "run_status.txt"


def log(message):
    print(message, flush=True)


def format_eta(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    mins, sec = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h {mins}m {sec}s"
    return f"{mins}m {sec}s"


def save_checkpoint(df, stage_name, batch_idx, total_batches, start_time):
    df.to_csv(PARTIAL_RESULTS_PATH, index=False)
    elapsed = max(1e-9, time.time() - start_time)
    rate = batch_idx / elapsed if batch_idx > 0 else 0.0
    eta = ((total_batches - batch_idx) / rate) if rate > 0 else float("inf")
    RUN_STATUS_PATH.write_text(
        (
            f"stage={stage_name}\n"
            f"batch={batch_idx}\n"
            f"total_batches={total_batches}\n"
            f"elapsed_seconds={elapsed:.2f}\n"
            f"eta_seconds={eta if math.isfinite(eta) else -1}\n"
            f"partial_csv={PARTIAL_RESULTS_PATH.resolve()}\n"
        ),
        encoding="utf-8",
    )
    if VERBOSE:
        log(
            f"[checkpoint] stage={stage_name} batch={batch_idx}/{total_batches} "
            f"elapsed={format_eta(elapsed)} eta={format_eta(eta)}"
        )


def run_batched_inference_into_column(df, source_col, target_col, model_pipeline, map_pred_fn, batch_size, stage_name):
    texts = df[source_col].astype(str).tolist()
    total_batches = max(1, math.ceil(len(texts) / batch_size))
    start_time = time.time()
    last_checkpoint_time = start_time
    last_checkpoint_batch = 0

    pbar = tqdm(total=total_batches, desc=stage_name, unit="batch", dynamic_ncols=True)
    for batch_idx, i in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts[i : i + batch_size]
        preds = model_pipeline(batch, truncation=True, max_length=512, batch_size=batch_size)
        mapped = [map_pred_fn(pred) for pred in preds]
        df.iloc[i : i + len(mapped), df.columns.get_loc(target_col)] = mapped

        elapsed = max(1e-9, time.time() - start_time)
        rate = batch_idx / elapsed
        eta = (total_batches - batch_idx) / rate if rate > 0 else float("inf")
        pbar.update(1)
        pbar.set_postfix_str(f"eta={format_eta(eta)}")

        should_checkpoint = (
            batch_idx == total_batches
            or (batch_idx - last_checkpoint_batch) >= CHECKPOINT_EVERY_BATCHES
            or (time.time() - last_checkpoint_time) >= CHECKPOINT_EVERY_SECONDS
        )
        if should_checkpoint:
            save_checkpoint(df, stage_name, batch_idx, total_batches, start_time)
            last_checkpoint_time = time.time()
            last_checkpoint_batch = batch_idx
    pbar.close()


# ----------------------------
# LOAD DATA
# ----------------------------
df = safe_read_csv(INPUT_CSV)
if "text" not in df.columns:
    raise SystemExit(
        f"Required column 'text' not found in {INPUT_CSV}. "
        f"Available columns: {', '.join(df.columns)}"
    )
if MAX_ROWS > 0:
    df = df.head(MAX_ROWS).copy()
    log(f"Row limit enabled: using first {len(df)} rows")
log(f"Loaded: {len(df)} chunks")


# ----------------------------
# LEXICAL FEATURES
# ----------------------------
df["shannon_entropy"] = df["text"].apply(shannon_entropy)
df["bigram_diversity"] = df["text"].apply(bigram_diversity)
top_bigrams = top_corpus_bigrams(df["text"], top_n=20)
log("Computed lexical features: shannon_entropy, bigram_diversity, top bigrams")
save_checkpoint(df, "lexical_features", 1, 1, time.time())


# ----------------------------
# VADER SENTIMENT
# ----------------------------
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()
df["vader_compound"] = df["text"].astype(str).apply(lambda x: sia.polarity_scores(x)["compound"])
log("Computed VADER sentiment")
save_checkpoint(df, "vader_sentiment", 1, 1, time.time())


# ----------------------------
# HUGGING FACE SENTIMENT MODEL
# ----------------------------
if SKIP_HF_MODELS:
    log("SKIP_HF_MODELS=1: using VADER sentiment only and skipping AI detector model inference.")
    df["hf_sentiment"] = df["vader_compound"]
    save_checkpoint(df, "hf_sentiment_skipped", 1, 1, time.time())
else:
    log("Loading HF sentiment model...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1,
    )
    log("Running HF sentiment inference...")
    df["hf_sentiment"] = np.nan
    run_batched_inference_into_column(
        df=df,
        source_col="text",
        target_col="hf_sentiment",
        model_pipeline=sentiment_pipeline,
        map_pred_fn=lambda pred: pred["score"] if pred["label"] == "POSITIVE" else -pred["score"],
        batch_size=HF_BATCH_SIZE,
        stage_name="hf_sentiment",
    )


# ----------------------------
# HYBRID SENTIMENT
# ----------------------------
df["hybrid_sentiment"] = 0.5 * df["vader_compound"] + 0.5 * df["hf_sentiment"]


# ----------------------------
# SENTIMENT CATEGORY (for Chi-square)
# ----------------------------
def categorize(score):
    if score > 0.05:
        return "Positive"
    if score < -0.05:
        return "Negative"
    return "Neutral"


df["sentiment_category"] = df["hybrid_sentiment"].apply(categorize)


# ----------------------------
# AI DETECTOR
# ----------------------------
if SKIP_HF_MODELS:
    ai_source = (df["hybrid_sentiment"] - df["hybrid_sentiment"].min()) / (
        (df["hybrid_sentiment"].max() - df["hybrid_sentiment"].min()) + 1e-9
    )
    df["ai_prob"] = ai_source.clip(0, 1)
else:
    log("Loading AI detector model...")
    ai_detector = build_ai_detector()
    log("Running AI detector inference...")
    df["ai_prob"] = np.nan
    run_batched_inference_into_column(
        df=df,
        source_col="text",
        target_col="ai_prob",
        model_pipeline=ai_detector,
        map_pred_fn=ai_score_from_prediction,
        batch_size=HF_BATCH_SIZE,
        stage_name="ai_detector",
    )
df["is_ai"] = (df["ai_prob"] >= 0.5).astype(int)
log("Computed ai_prob and is_ai")
save_checkpoint(df, "ai_probability_complete", 1, 1, time.time())

# ----------------------------
# DETECTION LOGGING
# ----------------------------
positive_df = df[df["is_ai"] == 1].copy()
positive_count = int(positive_df.shape[0])
total_count = int(df.shape[0])
positive_rate = (positive_count / total_count) if total_count else 0.0
log(f"AI-positive detections: {positive_count}/{total_count} ({positive_rate:.2%})")

summary_row = pd.DataFrame(
    [
        {
            "total_rows": total_count,
            "ai_positive_rows": positive_count,
            "ai_positive_rate": positive_rate,
            "avg_ai_prob": float(df["ai_prob"].mean()),
            "avg_hybrid_sentiment": float(df["hybrid_sentiment"].mean()),
        }
    ]
)
summary_row.to_csv("detection_summary.csv", index=False)

keep_cols = [c for c in ["chunk_id", "doc_id", "source", "year", "section", "ai_prob", "is_ai", "hybrid_sentiment", "text"] if c in positive_df.columns]
positive_df = positive_df.sort_values("ai_prob", ascending=False)
positive_df[keep_cols].to_csv("positive_detections.csv", index=False)
log("Saved detection logs: detection_summary.csv, positive_detections.csv")
save_checkpoint(df, "detection_logging", 1, 1, time.time())


# ----------------------------
# CHI-SQUARE TEST
# ----------------------------
contingency = pd.crosstab(df["sentiment_category"], df["is_ai"])
log("\nContingency Table:")
log(str(contingency))

chi2 = p = dof = expected = None
if contingency.shape[1] == 2 and contingency.to_numpy().sum() > 0:
    chi2, p, dof, expected = chi2_contingency(contingency)
    log("\nChi-square Results:")
    log(f"Chi2: {chi2}")
    log(f"p-value: {p}")
    log(f"Degrees of Freedom: {dof}")
else:
    log("\nChi-square not valid (insufficient variation).")

contingency.to_csv("chi_square_contingency.csv")
if expected is not None:
    pd.DataFrame(expected, index=contingency.index, columns=contingency.columns).to_csv(
        "chi_square_expected.csv"
    )
pd.DataFrame(top_bigrams, columns=["bigram", "count"]).to_csv("top_bigrams.csv", index=False)
log("Saved analysis summary CSVs: chi_square_contingency.csv, chi_square_expected.csv (if available), top_bigrams.csv")
save_checkpoint(df, "chi_square_complete", 1, 1, time.time())


# ----------------------------
# PLOTS
# ----------------------------
plots_dir = Path("plots")
plots_dir.mkdir(parents=True, exist_ok=True)
important_plot_names = {
    "hybrid_sentiment_hist.png",
    "ai_probability_hist.png",
    "chi_square_contingency_heatmap.png",
    "hybrid_vs_ai_scatter.png",
    "top_bigrams_barh.png",
}
for old_plot in plots_dir.glob("*.png"):
    if old_plot.name not in important_plot_names:
        old_plot.unlink(missing_ok=True)

plt.figure(figsize=(8, 5))
df["hybrid_sentiment"].hist(bins=40)
plt.title("Hybrid Sentiment Distribution")
plt.xlabel("Hybrid Sentiment")
plt.ylabel("Count")
save_plot(plots_dir / "hybrid_sentiment_hist.png")

plt.figure(figsize=(8, 5))
df["ai_prob"].hist(bins=40)
plt.title("AI Probability Distribution")
plt.xlabel("AI Probability")
plt.ylabel("Count")
save_plot(plots_dir / "ai_probability_hist.png")

plt.figure(figsize=(7, 5))
heatmap_data = contingency.reindex(index=["Negative", "Neutral", "Positive"], fill_value=0)
plt.imshow(heatmap_data.values, cmap="Blues", aspect="auto")
plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
for r in range(heatmap_data.shape[0]):
    for c in range(heatmap_data.shape[1]):
        plt.text(c, r, int(heatmap_data.iloc[r, c]), ha="center", va="center", color="black")
plt.title("Contingency Table Heatmap (Sentiment vs AI Label)")
plt.xlabel("is_ai")
plt.ylabel("sentiment_category")
plt.colorbar()
save_plot(plots_dir / "chi_square_contingency_heatmap.png")

plt.figure(figsize=(8, 5))
plt.scatter(df["hybrid_sentiment"], df["ai_prob"], s=10, alpha=0.35)
xs = df["hybrid_sentiment"].astype(float).to_numpy()
ys = df["ai_prob"].astype(float).to_numpy()
mask = np.isfinite(xs) & np.isfinite(ys)
xs = xs[mask]
ys = ys[mask]
if len(xs) > 1 and np.unique(xs).size > 1:
    coeff = np.polyfit(xs, ys, 1)
    xvals = np.linspace(xs.min(), xs.max(), 200)
    yvals = coeff[0] * xvals + coeff[1]
    plt.plot(xvals, yvals, color="red", linewidth=1.5)
plt.title("Hybrid Sentiment vs AI Probability")
plt.xlabel("Hybrid Sentiment")
plt.ylabel("AI Probability")
save_plot(plots_dir / "hybrid_vs_ai_scatter.png")

if top_bigrams:
    labels = [f"{a} {b}" for (a, b), _ in top_bigrams]
    values = [count for _, count in top_bigrams]
    plt.figure(figsize=(10, 7))
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, values)
    plt.yticks(y_pos, labels)
    plt.gca().invert_yaxis()
    plt.title("Top 20 Bigrams")
    plt.xlabel("Count")
    save_plot(plots_dir / "top_bigrams_barh.png")
save_checkpoint(df, "plots_complete", 1, 1, time.time())

# ----------------------------
# SAVE RESULTS
# ----------------------------
df.to_csv("corpus_with_results.csv", index=False)
log("\nDone. Results saved to corpus_with_results.csv")
log(f"Plots saved to: {plots_dir.resolve()}")
