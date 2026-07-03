# PLANNING FINAL — Sentiment Analysis E-Commerce Indonesia
**Nama:** Deris Firmansyah | **NIM:** A11.2024.15624
**Mata Kuliah:** Pembelajaran Mesin — NLP

---

## Environment
| Komponen | Detail |
|---|---|
| OS | Linux |
| Python | 3.14.2 |
| GPU | NVIDIA RTX 3050 4GB (CUDA 13.2) |
| Chrome | 149.0.7827.196 |
| Package Manager | Miniconda |
| Conda Env Name | `nlp-sentiment` |

---

## Struktur Direktori
```
machine-learning-nlp/
├── planning/                        # dokumen perencanaan
│   └── PLANNING.md
├── config/
│   ├── settings.py                  # konstanta & konfigurasi global
│   ├── tokopedia_shops.txt          # 25+ URL toko tokopedia target
│   └── shopee_keywords.txt          # keyword pencarian shopee
├── data/
│   ├── raw/
│   │   ├── tokopedia_raw.csv        # hasil scraping tokopedia
│   │   ├── shopee_raw.csv           # hasil scraping shopee
│   │   └── merged_labeled.csv       # setelah merge + auto-label
│   └── processed/
│       ├── clean_dataset.csv        # setelah preprocessing
│       ├── train.csv
│       └── test.csv
├── src/
│   ├── scraping/
│   │   ├── tokopedia_scraper.py     # GraphQL API (requests only)
│   │   ├── shopee_scraper.py        # undetected-chromedriver
│   │   ├── scraper_utils.py         # helper: delay, checkpoint, log
│   │   └── run_scraping.py          # entry point
│   ├── preprocessing/
│   │   ├── merger.py                # merge + dedup + labeling
│   │   ├── cleaner.py               # text cleaning pipeline
│   │   ├── normalizer.py            # normalisasi slang per token
│   │   ├── slang_dict.py            # kamus slang 200+ entri
│   │   └── run_preprocessing.py    # entry point
│   ├── modeling/
│   │   ├── dataset.py               # PyTorch Dataset class
│   │   ├── train_indobert.py        # fine-tuning IndoBERT (GPU)
│   │   ├── train_baseline.py        # SVM, NaiveBayes, LSTM
│   │   ├── evaluate.py              # metrik + confusion matrix
│   │   └── inference.py             # load model & predict
│   └── utils/
│       ├── logger.py                # loguru setup
│       └── config.py                # hyperparameter dataclass
├── model/
│   ├── indobert/                    # saved IndoBERT weights
│   ├── svm_model.pkl
│   ├── nb_model.pkl
│   └── lstm_model.pt
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Experiment_Comparison.ipynb
├── app/
│   ├── main.py                      # FastAPI server
│   └── schemas.py                   # Pydantic models
├── streamlit_app.py
├── environment.yml                  # Miniconda environment
├── requirements.txt
└── Dockerfile
```

---

## STEP 1A — Scraping Tokopedia (GraphQL API)

**Metode:** HTTP POST ke `gql.tokopedia.com` (TANPA Selenium)
**Library:** `requests` saja
**Risiko bot detection:** RENDAH
**Target:** 5.000 review dari 25+ toko

### Flow
```
[1] Baca config/tokopedia_shops.txt (25+ URL toko)
[2] Ekstrak Shop ID dari HTML via regex bertingkat:
      Pola 1: r'ShopPageGetHeaderLayout\({\"shopID\":\"(\d+)\"}'
      Pola 2: r'"shopID":"(\d+)"'
[3] GraphQL POST per rating (5→4→3→2→1) untuk balance dataset:
      Endpoint: https://gql.tokopedia.com/graphql/ReviewList
      Variables: shopID, limit=50, page, filterBy="rating=N"
[4] Quota per rating (total 5.000):
      rating 5 → 1.500 | rating 4 → 750 | rating 3 → 750
      rating 2 → 750   | rating 1 → 1.250
[5] Filter: skip review kosong / < 5 karakter
[6] Checkpoint save setiap 200 review (anti data loss)
[7] Output: data/raw/tokopedia_raw.csv
```

### Daftar 25 Toko Target (multi-kategori)
```
Elektronik: samsung-id, xiaomi-official, realme-official-store
Fashion: hmid-official, uniqlo-indonesia, erigo-store, zara-official
Skincare: wardah-official, somethinc-id, skintific-id, emina-official
Makanan: indomie-official, khong-guan-official, ulker-official
Olahraga: specs-official, league-official, nike-indonesia
Rumah Tangga: philips-indonesia, oxone-official
Otomotif: federal-oil-official, castrol-indonesia
Buku: gramedia-official, periplus-official
Bayi: pigeon-indonesia, dr-browns-official
```

---

## STEP 1B — Scraping Shopee (undetected-chromedriver)

**Metode:** Browser automation dengan bypass fingerprint
**Library:** `undetected_chromedriver` + `selenium`
**Risiko bot detection:** RENDAH-SEDANG
**Target:** 5.000 review dari berbagai kategori

### Kenapa undetected-chromedriver?
Selenium biasa meninggalkan `navigator.webdriver=true` → Shopee langsung detect.
`undetected_chromedriver` mempatch CDP signatures, flags, dan fingerprint Chrome otomatis.

### Flow
```
[1] Setup Chrome dengan real user profile:
      --user-data-dir=/home/derispewsss/.config/google-chrome
      --profile-directory=Default
      --disable-blink-features=AutomationControlled
      headless=False (lebih aman dari bot detection)
[2] Load cookies dari sesi sebelumnya (cookies_shopee.pkl)
[3] CAPTCHA handler: pause + input() jika URL mengandung "login/verify"
[4] Baca config/shopee_keywords.txt
[5] Per keyword: scroll infinite → kumpulkan URL produk
[6] Per produk:
      → scroll ke section rating
      → klik filter per bintang (5→4→3→2→1)
      → ekstrak: review_text, rating, date, author
      → klik next page (fallback XPath berlapis)
      → random delay: 2-4 detik antar produk
[7] Rate limiting: max 2 produk/menit, istirahat 2 menit setiap 50 produk
[8] Checkpoint save setiap 100 review
[9] Save cookies sebelum quit
[10] Output: data/raw/shopee_raw.csv
```

---

## STEP 2 — Data Merging, Dedup & Labeling

**File:** `src/preprocessing/merger.py`

```
[1] Load tokopedia_raw.csv + shopee_raw.csv
[2] pd.concat() → harmonisasi kolom
[3] drop_duplicates(subset=['review_text_lower']) — exact match
[4] Auto label:
      rating 1-2 → "Negatif"
      rating 3   → "Netral"
      rating 4-5 → "Positif"
[5] Print distribusi kelas
[6] Output: data/raw/merged_labeled.csv
```

---

## STEP 3 — Preprocessing Pipeline

**Files:** `cleaner.py`, `normalizer.py`, `slang_dict.py`

```
[1] Filter awal: drop NaN, < 5 char, hanya angka/emoji
[2] Text Cleaning (urutan penting):
      a. lowercase
      b. hapus URL: re.sub(r'https?\S+', ' ', text)
      c. hapus @mention dan #hashtag
      d. normalisasi karakter berulang: "bagussss" → "baguss"
      e. normalisasi tanda baca berlebih: "!!!" → "!"
      f. hapus karakter non-latin
      g. normalisasi spasi
[3] Normalisasi Slang (200+ entri):
      tokenize → lookup kamus → re-join
[4] Stopword Removal (PySastrawi):
      PENTING: kata negasi di-whitelist (tidak, bukan, belum, dll)
[5] Post-filter: drop < 3 token setelah cleaning
[6] Simpan review_text_original + review_text_clean
[7] Output: data/processed/clean_dataset.csv
```

---

## STEP 4 — EDA (notebooks/01_EDA.ipynb)

```
[1] Distribusi kelas (pie + bar chart)
[2] Distribusi panjang teks per kelas (histogram + boxplot)
[3] Word Cloud per kelas (3 word cloud)
[4] Top 20 kata per kelas (bar chart horizontal)
[5] Analisis class imbalance → keputusan: class_weight
[6] Distribusi per platform (Tokopedia vs Shopee)
```

---

## STEP 5 — Data Split

```
Stratified 80/20 split (sklearn.model_selection)
X = review_text_clean
y = sentiment_label (encoded: Negatif=0, Netral=1, Positif=2)
random_state = 42
Output: data/processed/train.csv + test.csv
```

---

## STEP 6 — Training Model

### IndoBERT Fine-Tuning (GPU)
```
Model   : indobenchmark/indobert-base-p2
Device  : cuda (RTX 3050 4GB)
Batch   : 8 (bukan 16 karena VRAM 4GB)
LR      : 2e-5
Epochs  : 5 (dengan early stopping patience=2)
Max Len : 128 token
Loss    : CrossEntropyLoss + class_weight
Optim   : AdamW + linear warmup scheduler (10%)
Dropout : 0.1 pada classification head
Simpan  : model/indobert/
```

### Baseline Models
```
SVM        : LinearSVC + TF-IDF (unigram+bigram, max_features=50k)
             5-Fold Stratified CV
NaiveBayes : MultinomialNB (alpha=1.0) + TF-IDF
LSTM       : Embedding(256) → BiLSTM(512) → Linear(3)
             Adam lr=1e-3, 10 epoch, early stopping patience=3
```

---

## STEP 7 — Evaluasi & Perbandingan

```
Metrik per model:
  - Accuracy
  - Precision (macro)
  - Recall (macro)
  - F1-Score macro     ← METRIK UTAMA
  - F1-Score weighted
  - Confusion Matrix 3x3
  - Classification report per kelas
  - Learning curve IndoBERT (train loss vs val loss)
```

---

## STEP 8 — Deployment

```
FastAPI  : POST /predict, GET /health
           uvicorn app.main:app --host 0.0.0.0 --port 8000
Streamlit: Input teks → Label + confidence bar chart
           streamlit run streamlit_app.py
Docker   : python:3.11-slim, expose port 8000
```

---

## Urutan Eksekusi Final

```bash
# 0. Aktivasi environment
conda activate nlp-sentiment

# 1. Scraping Tokopedia (~1-2 jam)
python src/scraping/run_scraping.py --platform tokopedia

# 2. Scraping Shopee (~3-4 jam, perlu pantau)
python src/scraping/run_scraping.py --platform shopee

# 3. Merge + Label
python src/preprocessing/merger.py

# 4. Preprocessing
python src/preprocessing/run_preprocessing.py

# 5. EDA (Jupyter)
jupyter notebook notebooks/01_EDA.ipynb

# 6. Training baseline (~10-20 menit)
python src/modeling/train_baseline.py

# 7. Training IndoBERT GPU (~1-2 jam)
python src/modeling/train_indobert.py

# 8. Evaluasi
python src/modeling/evaluate.py

# 9. API
uvicorn app.main:app --reload

# 10. UI
streamlit run streamlit_app.py
```

---

## Estimasi Waktu Total
| Step | Waktu |
|---|---|
| Scraping Tokopedia | 1-2 jam |
| Scraping Shopee | 3-4 jam |
| Merge + Preprocessing | 15-30 menit |
| EDA | 30-60 menit |
| Training Baseline | 10-20 menit |
| Training IndoBERT (GPU) | 1-2 jam |
| Evaluasi | 15 menit |
| Deployment setup | 30 menit |
| **Total** | **~8-12 jam** |
