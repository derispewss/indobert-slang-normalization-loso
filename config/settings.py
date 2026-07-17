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
EXTERNAL_DIR  = DATA_DIR / "external"
MODEL_DIR  = ROOT_DIR / "models"     # <-- Telah diubah dari "model" ke "models"
LOG_DIR    = ROOT_DIR / "logs"
REPORTS_DIR = ROOT_DIR / "reports"

# Buat direktori jika belum ada
for d in [RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, MODEL_DIR, LOG_DIR, REPORTS_DIR]:
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

def rating_to_label(rating: int, text: str = "") -> str:
    """
    Mengubah rating bintang menjadi label sentimen (Negatif, Netral, Positif).
    Dilengkapi text-based fallback untuk mengatasi bias anomali pengguna e-commerce 
    (misal: memberikan bintang 3, 4, atau 5, tetapi teks ulasannya penuh kekecewaan).
    """
    text_lower = text.lower() if isinstance(text, str) else ""
    
    # Kumpulan kata kunci (keywords) yang secara inheren mengindikasikan kekecewaan berat
    # Kata-kata ini sangat jarang dipakai dalam konteks pujian atau netral.
    strong_negative_keywords = [
        "kecewa", "nyesel", "menyesal", "rugi", "rusak", "cacat", "hancur", "pecah", 
        "jelek", "buruk", "parah", "bohong", "penipu", "nipu", "palsu", "kw", 
        "lambat", "lama banget", "lelet", "gak sesuai", "tidak sesuai", "kurang",
        "mati", "bocor", "basi", "bau", "sobek", "patah", "penyok", "hilang",
        "jangan beli", "kapok", "sampah", "buang", "zonk", "males"
    ]
    
    # 1. Pastikan Bintang 1 dan 2 MUTLAK Negatif (apapun isi teksnya)
    if rating in (1, 2):
        return "Negatif"
        
    # 2. Text-Based Fallback untuk anomali Bintang 3, 4, 5
    # Jika teks mengandung salah satu kata negatif kuat, paksa jadi "Negatif"
    for keyword in strong_negative_keywords:
        if keyword in text_lower:
            return "Negatif"
            
    # 3. Jika aman dari kata negatif kuat, kembalikan ke logika rating bawaan
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

# Kata intensifier (penguat) yang TIDAK boleh dihapus karena membedakan sentimen
INTENSIFIER_WORDS = {
    "sangat", "sekali", "banget", "bgt", "amat", "super", 
    "terlalu", "paling", "makin", "makin"
}

# ── Training: IndoBERT ────────────────────────────────────────
INDOBERT_MODEL_NAME = "indobenchmark/indobert-base-p2"
INDOBERT_MAX_LENGTH = 128
INDOBERT_BATCH_SIZE = 8        # aman untuk RTX 3050 4GB
INDOBERT_LEARNING_RATE = 2e-5  # Digunakan sebagai fallback jika LLRD tidak jalan
INDOBERT_NUM_EPOCHS  = 8       # Dinaikkan dari 5 ke 8 agar data Shopee bisa konvergen
INDOBERT_WARMUP_RATIO = 0.1
INDOBERT_WEIGHT_DECAY = 0.01
INDOBERT_DROPOUT     = 0.2     # Dinaikkan dari 0.1 ke 0.2 untuk regularisasi ekstra (overfitting)
INDOBERT_EARLY_STOP_PATIENCE = 3 # Dinaikkan dari 2 ke 3 agar model bisa keluar local optima
INDOBERT_GRADIENT_CLIP = 1.0
RANDOM_STATE = 42
TEST_SIZE    = 0.15            # Split 70/15/15 (Train/Val/Test)
VAL_SIZE     = 0.15

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
