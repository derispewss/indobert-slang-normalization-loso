# Analisis Sentimen Ulasan E-Commerce Indonesia
### Pengaruh Normalisasi Kata Slang terhadap Generalisasi IndoBERT Lintas Platform (LOSO)

> **Deris Firmansyah** · A11.2024.15624 · Teknik Informatika · Universitas Dian Nuswantoro
> Mata Kuliah: Machine Learning · 2025/2026

---

## Daftar Isi

- [Tentang Penelitian](#tentang-penelitian)
- [Arsitektur Eksperimen](#arsitektur-eksperimen)
- [Hasil Evaluasi](#hasil-evaluasi)
- [Struktur Direktori](#struktur-direktori)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Instalasi](#instalasi)
- [Dataset](#dataset)
  - [Opsi A — Gunakan Dataset yang Sudah Disediakan (Direkomendasikan)](#opsi-a--gunakan-dataset-yang-sudah-disediakan-direkomendasikan)
  - [Opsi B — Scrape Sendiri dari Awal](#opsi-b--scrape-sendiri-dari-awal)
- [Download Model IndoBERT](#download-model-indobert)
- [Menjalankan Pipeline](#menjalankan-pipeline)
- [REST API](#rest-api)
- [Dashboard Streamlit](#dashboard-streamlit)
- [Docker](#docker)
- [Teknologi](#teknologi)
- [Keterbatasan](#keterbatasan)

---

## Tentang Penelitian

Penelitian ini mengkaji **pengaruh Normalisasi Kata Slang** terhadap kemampuan generalisasi model **IndoBERT** (`indobenchmark/indobert-base-p2`) dalam mengklasifikasikan sentimen ulasan e-commerce Indonesia lintas platform.

**Pertanyaan Penelitian:**
> *"Apakah normalisasi kata slang mampu meningkatkan kemampuan model IndoBERT yang dilatih di satu platform e-commerce untuk menggeneralisasi ke platform lain yang belum pernah dilihat sebelumnya?"*

**Pendekatan:**
- Skema evaluasi **Leave-One-Site-Out (LOSO)** — model yang dilatih di Tokopedia diuji ke Shopee, dan sebaliknya
- Perbandingan **Baseline** (tanpa normalisasi) vs **Proposed** (dengan normalisasi kamus slang)
- Klasifikasi **3 kelas**: Negatif · Netral · Positif
- Metrik utama: **F1-Macro** (lebih adil untuk kelas tidak seimbang)

---

## Arsitektur Eksperimen

```
DATA MENTAH
├── Tokopedia (16.441 ulasan — GraphQL API)
└── Shopee    (90.302 ulasan — 4 dataset Kaggle)
        │
        ▼
PREPROCESSING (2 jalur paralel)
├── Baseline  → cleaning biasa (tanpa normalisasi slang)
└── Proposed  → cleaning + kamus translasi slang Indonesia
        │
        ▼
4 MODEL INDOBERT DILATIH SECARA INDEPENDEN
├── indobert_tokopedia_baseline
├── indobert_tokopedia_proposed
├── indobert_shopee_baseline
└── indobert_shopee_proposed
        │
        ▼
EVALUASI LOSO (8 kombinasi pengujian)
├── In-Domain  : model diuji di platform yang sama (batas atas)
└── Cross-Platform : model diuji di platform asing (zero-shot)
```

---

## Hasil Evaluasi

### In-Domain (Kontrol)

| Model Dilatih Di | Diuji Di | Kondisi | F1-Macro | Accuracy |
|:---|:---|:---|:---:|:---:|
| Tokopedia | Tokopedia | Baseline | 0.6634 | 0.6995 |
| Tokopedia | Tokopedia | **Proposed** | **0.6696** | 0.6987 |
| Shopee | Shopee | Baseline | 0.6493 | 0.6782 |
| Shopee | Shopee | **Proposed** | **0.6552** | 0.6909 |

### Cross-Platform / LOSO (Zero-Shot)

| Model Dilatih Di | Diuji Di | Kondisi | F1-Macro | Accuracy | Degradasi |
|:---|:---|:---|:---:|:---:|:---:|
| Tokopedia | Shopee | Baseline | 0.5588 | 0.6141 | -10.46% |
| Tokopedia | Shopee | Proposed | 0.5506 | 0.6179 | -11.94% |
| Shopee | Tokopedia | Baseline | 0.5840 | 0.6553 | -6.53% |
| Shopee | Tokopedia | **Proposed** | **0.5931** | **0.6635** | -6.21% |

### Temuan Kunci

| # | Temuan | Dampak |
|:---:|:---|:---:|
| 1 | **Domain Shift terbukti nyata** — degradasi F1 hingga -10.46% saat lintas platform | Terkonfirmasi |
| 2 | **Normalisasi berhasil** pada skenario Shopee→Tokopedia (+0.0091 F1) | Positif |
| 3 | **Anomali** — normalisasi justru merusak skenario Tokopedia→Shopee (-0.0083 F1) karena context loss pada kata domain-specific Shopee (COD, ongkir, kurir) | Negatif |

---

## Struktur Direktori

```text
machine-learning-nlp/
│
├── app/
│   ├── main.py                      # FastAPI REST API
│   └── schemas.py                   # Pydantic request/response schema
│
├── config/
│   ├── settings.py                  # Konfigurasi global & hyperparameter
│   └── tokopedia_shops.txt          # Daftar URL toko Tokopedia untuk scraping
│
├── data/
│   ├── raw/                         # Dataset mentah (tidak disertakan di repo)
│   │   ├── tokopedia_raw.csv        # 16.441 ulasan Tokopedia
│   │   └── shopee_raw.csv           # 90.302 ulasan Shopee (gabungan 4 Kaggle)
│   └── processed/
│       └── experiment_splits/       # 8 file CSV hasil preprocessing (generated)
│
├── model/
│   ├── indobert_raw/                # Base pre-trained IndoBERT (tidak disertakan)
│   ├── indobert_tokopedia_baseline/ # Fine-tuned (generated setelah training)
│   ├── indobert_tokopedia_proposed/ # Fine-tuned (generated setelah training)
│   ├── indobert_shopee_baseline/    # Fine-tuned (generated setelah training)
│   └── indobert_shopee_proposed/    # Fine-tuned (generated setelah training)
│
├── notebooks/
│   └── 01_EDA.ipynb                 # EDA: OOV Rate & Jensen-Shannon Divergence
│
├── planning/
│   └── PLANNING.md                  # Blueprint eksperimen V3
│
├── src/
│   ├── modeling/
│   │   ├── dataset.py               # PyTorch Dataset wrapper
│   │   ├── evaluate.py              # LOSO evaluator
│   │   ├── inference.py             # Dynamic model loader
│   │   └── train_indobert.py        # Pipeline training 4 skenario
│   ├── preprocessing/
│   │   ├── cleaner.py               # Cleaning pipeline (dual-path)
│   │   ├── merger.py                # Merge, deduplikasi, downsampling
│   │   ├── normalizer.py            # Normalisasi slang & karakter
│   │   ├── run_preprocessing.py     # Eksekutor 4 skenario preprocessing
│   │   └── slang_dict.py            # Kamus translasi slang Indonesia
│   └── scraping/
│       ├── tokopedia_scraper.py     # GraphQL API scraper
│       ├── shopee_kaggle.py         # Kaggle dataset downloader
│       └── run_scraping.py          # Entry point scraping
│
├── logs/                            # Log runtime (tidak disertakan di repo)
├── streamlit_app.py                 # Dashboard demo interaktif
├── Dockerfile                       # Container deployment
├── environment.yml                  # Conda environment (GPU)
├── requirements.txt                 # Python dependencies
├── EVALUATION_REPORT.md             # Laporan hasil evaluasi lengkap
└── README.md
```

---

## Persyaratan Sistem

| Komponen | Minimum | Digunakan di Penelitian |
|:---|:---|:---|
| Python | 3.10+ | 3.11 |
| GPU VRAM | 4GB | NVIDIA RTX 3050 4GB |
| CUDA | 11.8+ | 12.x |
| RAM | 8GB | 16GB |
| Storage | 10GB | ~15GB (model + data) |
| OS | Linux / Windows / macOS | Ubuntu 22.04 |

> **Catatan:** Training tanpa GPU tetap bisa berjalan namun sangat lambat (~10x lebih lama). Inference dan evaluasi bisa dijalankan di CPU.

---

## Instalasi

### Langkah 1 — Clone Repository

```bash
git clone https://github.com/derisfirmansyah/machine-learning-nlp.git
cd machine-learning-nlp
```

### Langkah 2 — Buat Conda Environment

```bash
# Buat environment dari file spec (sudah termasuk PyTorch + CUDA)
conda env create -f environment.yml

# Aktifkan environment
conda activate nlp-sentiment
```

> Jika tidak menggunakan Conda, install manual via pip:
> ```bash
> pip install -r requirements.txt
> # Install PyTorch dengan CUDA secara terpisah:
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### Langkah 3 — Verifikasi GPU

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Tidak terdeteksi')"
```

Output yang diharapkan:
```
CUDA: True
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

---

## Dataset

Ada **dua opsi** untuk mendapatkan data — pilih salah satu sesuai kebutuhan.

---

### Opsi A — Gunakan Dataset yang Sudah Disediakan (Direkomendasikan)

Dataset gabungan Tokopedia + Shopee sudah tersedia di Kaggle dan siap pakai tanpa perlu scraping.

**Link Dataset:**
> [https://www.kaggle.com/datasets/derisfirmansyah/dataset-ulasan-product-platform-shopee-tokopedia](https://www.kaggle.com/datasets/derisfirmansyah/dataset-ulasan-product-platform-shopee-tokopedia)

**Informasi Dataset:**

| File | Platform | Jumlah Baris | Keterangan |
|:---|:---|:---:|:---|
| `tokopedia_raw.csv` | Tokopedia | 16.441 | Ulasan produk fisik (15 toko, 5 kategori) |
| `shopee_raw.csv` | Shopee | 90.302 | Gabungan 4 sumber Kaggle publik |
| `merged_labeled.csv` | Keduanya | 74.137 | Master dataset (post-merge + downsampling) |

**Cara Download:**

**Cara 1 — Kaggle CLI (paling cepat):**

```bash
# Install kaggle CLI jika belum ada
pip install kaggle

# Letakkan kaggle.json di ~/.kaggle/ (download dari https://kaggle.com/settings > API)
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d derisfirmansyah/dataset-ulasan-product-platform-shopee-tokopedia

# Ekstrak ke folder yang tepat
unzip dataset-ulasan-product-platform-shopee-tokopedia.zip -d data/raw/
```

**Cara 2 — Download Manual:**

1. Buka [link dataset Kaggle](https://www.kaggle.com/datasets/derisfirmansyah/dataset-ulasan-product-platform-shopee-tokopedia)
2. Klik tombol **Download** (pojok kanan atas)
3. Ekstrak file ZIP
4. Salin file ke dalam folder proyek:

```bash
# Struktur folder yang diharapkan setelah ekstrak:
data/
└── raw/
    ├── tokopedia_raw.csv
    ├── shopee_raw.csv
    └── merged/
        └── multi_platform/
            └── merged_labeled.csv
```

**Setelah dataset tersedia, langsung lompat ke [Menjalankan Pipeline](#menjalankan-pipeline).**

---

### Opsi B — Scrape Sendiri dari Awal

Pilih opsi ini jika ingin mengumpulkan data segar atau menambahkan toko baru.

#### B.1 — Scraping Tokopedia (GraphQL API)

**Konfigurasi Toko Target:**

Edit file `config/tokopedia_shops.txt` — tambahkan URL halaman review toko yang diinginkan, satu URL per baris:

```text
# Format: https://www.tokopedia.com/<nama-toko>/review

# Elektronik
https://www.tokopedia.com/logitech-g/review
https://www.tokopedia.com/anker-indonesia/review

# Fashion
https://www.tokopedia.com/aerostreet/review
https://www.tokopedia.com/3second/review

# Tambahkan toko baru di sini:
https://www.tokopedia.com/<nama-toko-anda>/review
```

> **Cara menemukan URL toko:** Buka halaman toko di Tokopedia → klik tab "Ulasan" → salin URL dari address bar browser.

**Jalankan Scraper:**

```bash
python src/scraping/run_scraping.py --mode scrape --platform tokopedia
```

Output: `data/raw/tokopedia_raw.csv`

Konfigurasi kuota per rating bisa diubah di `config/settings.py`:

```python
TOKOPEDIA_RATING_QUOTA = {
    5: 1500,   # Maksimum ulasan bintang 5 per toko
    4: 750,
    3: 750,
    2: 750,
    1: 1250,
}
```

#### B.2 — Download Dataset Shopee dari Kaggle

Dataset Shopee diambil dari 4 sumber Kaggle publik secara otomatis menggunakan `kagglehub`:

| Sumber | Dataset Kaggle | Jumlah |
|:---|:---|:---:|
| Ahmad Selo Abadi | `ahmadseloabadi/shoppe-app-reviews-from-google-play-store` | 85.500 |
| Taqiyya Ghazi | `taqiyyaghazi/indonesian-marketplace-product-reviews` | 831 |
| Md Himas Pamungkas | `mdhimaspamungkas/review-product-shopee` | 3.020 |
| Alvian Ardiansyah | `alvianardiansyah/dataset-ulasan-pengguna-shopee` | 1.840 |

**Siapkan Kaggle API Key:**

```bash
# Download kaggle.json dari: https://www.kaggle.com/settings > API > Create New Token
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Jalankan Downloader:**

```bash
python src/scraping/run_scraping.py --mode scrape --platform shopee
```

Output: `data/raw/shopee_raw.csv`

#### B.3 — Merge Kedua Dataset

```bash
python src/preprocessing/merger.py --mode multi
```

Output: `data/raw/merged/multi_platform/merged_labeled.csv` (~74.137 baris)

---

## Download Model IndoBERT

Model base **IndoBERT** (`indobenchmark/indobert-base-p2`) harus diunduh terlebih dahulu sebelum training. Model tidak disertakan di repository karena ukurannya ~475MB.

### Cara 1 — Download Otomatis via Script (Direkomendasikan)

```bash
python -c "
from transformers import AutoTokenizer, AutoModel
import os

save_dir = 'model/indobert_raw'
os.makedirs(save_dir, exist_ok=True)

print('Mengunduh tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
tokenizer.save_pretrained(save_dir)

print('Mengunduh model weights...')
model = AutoModel.from_pretrained('indobenchmark/indobert-base-p2')
model.save_pretrained(save_dir)

print(f'Selesai! Model tersimpan di: {save_dir}')
"
```

### Cara 2 — Download Manual via HuggingFace CLI

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Download model ke folder lokal
huggingface-cli download indobenchmark/indobert-base-p2 \
    --local-dir model/indobert_raw \
    --local-dir-use-symlinks False
```

### Cara 3 — Download Manual via Browser

1. Buka [https://huggingface.co/indobenchmark/indobert-base-p2](https://huggingface.co/indobenchmark/indobert-base-p2)
2. Klik tab **Files and versions**
3. Download file berikut satu per satu dan letakkan di `model/indobert_raw/`:
   - `config.json`
   - `model.safetensors` (atau `pytorch_model.bin`)
   - `tokenizer_config.json`
   - `vocab.txt`
   - `special_tokens_map.json`

**Verifikasi download berhasil:**

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('model/indobert_raw', local_files_only=True)
print('Tokenizer OK — vocab size:', tokenizer.vocab_size)
print('Model siap digunakan untuk fine-tuning.')
"
```

Output yang diharapkan:
```
Tokenizer OK — vocab size: 32000
Model siap digunakan untuk fine-tuning.
```

---

## Menjalankan Pipeline

Setelah dataset dan model base tersedia, jalankan pipeline berikut secara berurutan:

### Step 1 — Preprocessing (Hasilkan 8 File Eksperimen)

```bash
python src/preprocessing/run_preprocessing.py
```

Proses ini akan menghasilkan 8 file CSV di `data/processed/experiment_splits/`:

```
train_tokopedia_baseline.csv  |  test_tokopedia_baseline.csv
train_tokopedia_proposed.csv  |  test_tokopedia_proposed.csv
train_shopee_baseline.csv     |  test_shopee_baseline.csv
train_shopee_proposed.csv     |  test_shopee_proposed.csv
```

Estimasi waktu: ~5–10 menit (bergantung pada jumlah data)

### Step 2 — Training 4 Model IndoBERT

```bash
# Training semua skenario sekaligus (direkomendasikan)
python src/modeling/train_indobert.py --platform all --norm both
```

Opsi argumen tersedia:

```bash
# Training hanya satu platform
python src/modeling/train_indobert.py --platform tokopedia --norm both
python src/modeling/train_indobert.py --platform shopee --norm both

# Training hanya satu kondisi
python src/modeling/train_indobert.py --platform all --norm baseline
python src/modeling/train_indobert.py --platform all --norm proposed

# Training satu skenario spesifik
python src/modeling/train_indobert.py --platform tokopedia --norm baseline
```

Estimasi waktu training per skenario (RTX 3050 4GB):

| Skenario | Data Train | Waktu/Epoch | Total (4 Epoch) |
|:---|:---:|:---:|:---:|
| Tokopedia Baseline | 19.792 | ~10.7 menit | ~43 menit |
| Tokopedia Proposed | 19.778 | ~10.7 menit | ~43 menit |
| Shopee Baseline | 38.218 | ~20.8 menit | ~83 menit |
| Shopee Proposed | 38.216 | ~20.8 menit | ~83 menit |
| **Total** | | | **~4.3 jam** |

Model terbaik (berdasarkan Val F1-Macro) disimpan otomatis ke `model/indobert_{platform}_{kondisi}/`.

### Step 3 — Evaluasi LOSO

```bash
python src/modeling/evaluate.py
```

Output tabel evaluasi akan ditampilkan di terminal dan dicatat ke `logs/evaluate_cross_platform_<timestamp>.log`:

```
==========================================================================================
TABEL EVALUASI: IN-DOMAIN EVALUATION (Kontrol / Batas Atas)
==========================================================================================
Model (Dilatih di)        | Diuji di (Dataset Test)   | Kondisi Data | F1-Macro | Accuracy
------------------------------------------------------------------------------------------
Tokopedia                 | Tokopedia                 | Baseline     | 0.6634   | 0.6995
...

==========================================================================================
TABEL EVALUASI: CROSS-PLATFORM (LEAVE-ONE-SITE-OUT) EVALUATION
==========================================================================================
...
```

### Step 4 — Jalankan Dashboard Interaktif

```bash
streamlit run streamlit_app.py
```

Buka browser di `http://localhost:8501`

Dashboard memungkinkan demo langsung efek normalisasi slang — masukkan teks ulasan, pilih platform model, toggle Baseline/Proposed, dan lihat perbedaan prediksi secara real-time.

---

## REST API

### Menjalankan Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs otomatis tersedia di:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Endpoint

#### `POST /predict` — Prediksi Sentimen

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "barang rusak pas nyampe, penjual ga respon sama sekali",
    "target_platform": "tokopedia",
    "use_normalization": true
  }'
```

Response:

```json
{
  "label": "Negatif",
  "confidence": 0.9231,
  "probabilities": {
    "Negatif": 0.9231,
    "Netral": 0.0512,
    "Positif": 0.0257
  },
  "text_original": "barang rusak pas nyampe, penjual ga respon sama sekali",
  "text_cleaned": "barang rusak sampai penjual tidak respons sama sekali"
}
```

**Parameter Request:**

| Parameter | Tipe | Wajib | Keterangan |
|:---|:---|:---:|:---|
| `text` | `string` | Ya | Teks ulasan yang ingin diprediksi |
| `target_platform` | `string` | Ya | `"tokopedia"` atau `"shopee"` |
| `use_normalization` | `boolean` | Ya | `true` = Proposed, `false` = Baseline |

#### `GET /health` — Status Server

```bash
curl http://localhost:8000/health
```

Response:

```json
{ "status": "ok" }
```

---

## Docker

### Build Image

```bash
docker build -t nlp-sentiment .
```

### Jalankan Container

```bash
docker run -d \
  --name nlp-api \
  -p 8000:8000 \
  --gpus all \
  nlp-sentiment
```

> Hapus `--gpus all` jika tidak memiliki GPU.

### Cek Status

```bash
docker logs nlp-api
docker ps
```

---

## Teknologi

| Kategori | Library / Tool | Versi |
|:---|:---|:---|
| Deep Learning | PyTorch | ≥2.3.0 |
| NLP Framework | HuggingFace Transformers | ≥4.40.0 |
| Base Model | IndoBERT (`indobert-base-p2`) | — |
| NLP Indonesia | PySastrawi | ≥1.0.0 |
| Data Processing | Pandas, NumPy | ≥2.2.0, ≥1.26.4 |
| ML Utilities | scikit-learn | ≥1.4.0 |
| Kaggle Downloader | kagglehub | latest |
| Scraping | Requests (GraphQL) | ≥2.31.0 |
| REST API | FastAPI + Uvicorn | ≥0.110.0 |
| Dashboard | Streamlit | ≥1.31.0 |
| Logging | Loguru | ≥0.7.2 |
| Visualisasi | Matplotlib, Seaborn | ≥3.8.3, ≥0.13.2 |
| Environment | Conda + Python | 3.11 |
| Container | Docker | — |

---

## Keterbatasan

| # | Keterbatasan | Dampak |
|:---:|:---|:---:|
| 1 | **Label berbasis rating bintang** — bintang 3 langsung jadi "Netral" tanpa membaca teks | F1 stagnan di ~0.66 |
| 2 | **Kamus slang statis (1-to-1)** — tidak memahami konteks kalimat, menyebabkan context loss | Anomali cross-platform |
| 3 | **Topic mismatch** — sebagian data Shopee berisi ulasan aplikasi Play Store, bukan produk fisik | Domain shift -10% |
| 4 | **Hanya 2 platform** — skema LOSO terbatas pada 2 arah pengujian | Generalisasi terbatas |

---

## Lisensi

Proyek ini dibuat untuk keperluan tugas akademik **Mata Kuliah Machine Learning**, Teknik Informatika, Universitas Dian Nuswantoro.

- Data Tokopedia dikumpulkan melalui API publik untuk keperluan penelitian non-komersial.
- Dataset Shopee bersumber dari dataset publik Kaggle milik masing-masing kontributor.
- Model IndoBERT merupakan karya `indobenchmark` yang tersedia di HuggingFace.
