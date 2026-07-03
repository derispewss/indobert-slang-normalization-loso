# ============================================================
# config/settings.py
# Konfigurasi global untuk seluruh pipeline
# ============================================================

import os
from pathlib import Path

# ── Root project ─────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR   = ROOT_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR  = ROOT_DIR / "model"
LOG_DIR    = ROOT_DIR / "logs"

# Buat direktori jika belum ada
for d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── File paths ────────────────────────────────────────────────
TOKOPEDIA_SHOPS_FILE  = CONFIG_DIR / "tokopedia_shops.txt"
TOKOPEDIA_RAW_CSV     = RAW_DIR / "tokopedia_raw.csv"

# SHOPEE (Dataset Publik SmSA/IndoNLU)
SHOPEE_RAW_CSV        = RAW_DIR / "shopee_raw.csv"

MERGED_LABELED_CSV    = RAW_DIR / "merged_labeled.csv"
CLEAN_DATASET_CSV     = PROCESSED_DIR / "clean_dataset.csv"
TRAIN_CSV             = PROCESSED_DIR / "train.csv"
TEST_CSV              = PROCESSED_DIR / "test.csv"

# ── Model paths ───────────────────────────────────────────────
INDOBERT_MODEL_DIR    = MODEL_DIR / "indobert_single_platform"
SVM_MODEL_PATH        = MODEL_DIR / "baseline_single_platform" / "svm_model.pkl"
NB_MODEL_PATH         = MODEL_DIR / "baseline_single_platform" / "nb_model.pkl"
TFIDF_PATH            = MODEL_DIR / "baseline_single_platform" / "tfidf_vectorizer.pkl"
LSTM_MODEL_PATH       = MODEL_DIR / "baseline_single_platform" / "lstm_model.pt"

# ── Scraping: Tokopedia ───────────────────────────────────────
TOKOPEDIA_GQL_ENDPOINT = "https://gql.tokopedia.com/graphql/ReviewList"
TOKOPEDIA_HEADERS = {
    "accept": "*/*",
    "content-type": "application/json",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    ),
}
# Quota review per rating untuk balance dataset
TOKOPEDIA_RATING_QUOTA = {
    5: 1500,
    4: 750,
    3: 750,
    2: 750,
    1: 1250,
}
TOKOPEDIA_PAGE_SIZE       = 50    # review per request
TOKOPEDIA_CHECKPOINT_EVERY = 200  # simpan setiap N review
TOKOPEDIA_SLEEP_BETWEEN_PAGES = 0.5  # detik

# ── Label mapping ─────────────────────────────────────────────
LABEL2ID = {"Negatif": 0, "Netral": 1, "Positif": 2}
ID2LABEL = {0: "Negatif", 1: "Netral", 2: "Positif"}

def rating_to_label(rating: int) -> str:
    if rating in (1, 2):
        return "Negatif"
    if rating == 3:
        return "Netral"
    return "Positif"

# ── Preprocessing ─────────────────────────────────────────────
MIN_TEXT_LENGTH  = 5    # karakter minimum sebelum cleaning
MIN_TOKEN_COUNT  = 3    # token minimum setelah cleaning
# Kata negasi yang TIDAK boleh dihapus saat stopword removal
NEGATION_WORDS = {
    "tidak", "bukan", "belum", "jangan", "kurang",
    "tanpa", "tak", "tiada", "no", "anti",
}

# ── Training: IndoBERT ────────────────────────────────────────
INDOBERT_MODEL_NAME = "indobenchmark/indobert-base-p2"
INDOBERT_MAX_LENGTH = 128
INDOBERT_BATCH_SIZE = 8        # aman untuk RTX 3050 4GB
INDOBERT_LEARNING_RATE = 2e-5
INDOBERT_NUM_EPOCHS  = 5
INDOBERT_WARMUP_RATIO = 0.1
INDOBERT_WEIGHT_DECAY = 0.01
INDOBERT_DROPOUT     = 0.1
INDOBERT_EARLY_STOP_PATIENCE = 2
INDOBERT_GRADIENT_CLIP = 1.0
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ── Training: LSTM ────────────────────────────────────────────
LSTM_EMBEDDING_DIM = 256
LSTM_HIDDEN_SIZE   = 512
LSTM_DROPOUT       = 0.3
LSTM_LEARNING_RATE = 1e-3
LSTM_NUM_EPOCHS    = 10
LSTM_BATCH_SIZE    = 64
LSTM_EARLY_STOP_PATIENCE = 3

# ── TF-IDF (SVM & NaiveBayes) ────────────────────────────────
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE  = (1, 2)
CV_FOLDS           = 5

# ── FastAPI ───────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
