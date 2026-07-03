# PLANNING V3 — Pengaruh Normalisasi Slang & Evaluasi Cross-Platform
**Nama:** Deris Firmansyah | **NIM:** A11.2024.15624
**Mata Kuliah:** Machine Learning
**Fokus Penelitian:** Generalisasi IndoBERT, Cross-Platform Evaluation (LOSO), & Slang Normalization.

---

## 🏗️ 1. Perubahan Paradigma Arsitektur (V3)
Dokumen *plan* ini merepresentasikan perombakan total dari arsitektur V2. Berdasarkan analisis pada rumusan masalah terbaru, penelitian **tidak lagi membandingkan algoritma model** (SVM, NB, BiLSTM telah dihapus).

Fokus penelitian kini 100% beralih pada **Eksperimen Intervensi Linguistik** menggunakan satu arsitektur mutakhir: **IndoBERT**. Arsitektur diekspansi untuk mengukur *Domain Shift* (heterogenitas) dan mengevaluasi apakah perbaikan kata gaul (*Slang Normalization*) mampu menyelamatkan performa model ketika dipaksa mengklasifikasikan data dari e-commerce yang belum pernah ia pelajari (*Zero-Shot Cross-Domain*).

### Konfigurasi Lingkungan (Environment)
| Komponen | Spesifikasi |
|---|---|
| Bahasa | Python 3.11 |
| Framework | PyTorch (GPU CUDA 13.2), Transformers (HuggingFace) |
| Hardware | NVIDIA RTX 3050 4GB (Max VRAM Constraints Handled) |

---

## 📂 2. Struktur Direktori Baru
```text
machine-learning-nlp/
├── planning/
│   ├── PLANNING.md                  # Plan V3 Saat Ini (Eksperimen LOSO & Slang)
│   └── v1_single_platform/          # Backup eksperimen lama
├── data/
│   ├── raw/
│   │   ├── tokopedia_raw.csv        # Dataset Tokopedia
│   │   └── shopee_raw.csv           # Dataset Shopee (4 Sumber Kaggle)
│   └── processed/
│       └── experiment_splits/       # (BARU) 8 File kombinasi eksperimen (Train/Test)
├── src/
│   ├── scraping/
│   │   └── (Skrip pengumpul data via GraphQL & Kagglehub)
│   ├── preprocessing/
│   │   ├── cleaner.py               # (BARU) Mendukung 2 jalur: `use_normalization` (True/False)
│   │   ├── normalizer.py            # Mesin translasi slang & typo
│   │   └── run_preprocessing.py     # (BARU) Pemecah data menjadi 4 Kondisi Eksperimen
│   └── modeling/
│       ├── train_indobert.py        # (BARU) Mesin pelatih otomatis 4 Varian IndoBERT
│       ├── evaluate.py              # (BARU) Kalkulator Metrik Silang (LOSO Evaluator)
│       └── inference.py             # Router inferensi dinamis untuk Dashboard
├── notebooks/
│   └── 01_EDA.ipynb                 # (BARU) Ditambahkan Uji OOV & Jensen-Shannon Divergence
├── model/
│   ├── indobert_raw/                # Base model (indobenchmark/indobert-base-p2)
│   ├── indobert_tokopedia_baseline/ # (BARU) Bobot model Toped (Tanpa Normalisasi)
│   ├── indobert_tokopedia_proposed/ # (BARU) Bobot model Toped (Dengan Normalisasi)
│   ├── indobert_shopee_baseline/    # (BARU) Bobot model Shopee (Tanpa Normalisasi)
│   └── indobert_shopee_proposed/    # (BARU) Bobot model Shopee (Dengan Normalisasi)
└── streamlit_app.py                 # (BARU) Dasbor presentasi efek Normalisasi & Domain Shift
```

---

## ⚙️ 3. Skema Eksperimental Lintas Domain (Cross-Platform)

Seluruh logika *preprocessing* dan *training* telah dirombak untuk mendukung Matriks Pengujian Silang yang ketat. Data dipecah berdasarkan platform asal dan perlakuan linguistiknya:

### A. Kondisi Linguistik (Treatment)
1.  **Baseline (Tanpa Normalisasi):** Ulasan dibersihkan dari noise (URL, simbol), namun struktur slang/typo/singkatan dibiarkan utuh (misal: "brg jelek bgt").
2.  **Proposed (Dengan Normalisasi):** Ulasan disuntikkan modul *Slang Translation* sehingga kata non-baku dikonversi menjadi formal (misal: "barang jelek banget").

### B. Matriks Pelatihan (Training Generative)
Sistem (`train_indobert.py`) akan secara berurutan melatih dan mencetak 4 buah versi "Otak" IndoBERT:
*   `Train: Tokopedia` $\times$ `Kondisi: Baseline`
*   `Train: Tokopedia` $\times$ `Kondisi: Proposed`
*   `Train: Shopee`    $\times$ `Kondisi: Baseline`
*   `Train: Shopee`    $\times$ `Kondisi: Proposed`

### C. Metrik Pengujian (Evaluation Scheme)
Skrip `evaluate.py` akan menyilangkan model yang telah dilatih dengan data uji (*test set*). Terdapat dua blok evaluasi:
1.  **In-Domain (Kontrol):** Model diuji pada platform tempat ia dilatih (Misal: Model Tokopedia diuji dengan kalimat Tokopedia). Bertindak sebagai batas atas akurasi.
2.  **Leave-One-Site-Out / LOSO (Generalisasi):** Model diuji pada platform asing (Misal: Model Tokopedia diuji menggunakan bahasa dari Shopee). Metrik krusial adalah memantau seberapa parah F1-Macro jatuh (*Domain Shift Degradation*) dan apakah **Proposed (Normalisasi)** mampu meredam kejatuhan akurasi tersebut.

---

## 📊 4. Integrasi Uji Heterogenitas (Bukti Akademis)
Untuk menjustifikasi mengapa pengujian LOSO itu diperlukan, ditambahkan fungsi pembuktian statistik di dalam modul EDA (`01_EDA.ipynb`):
1.  **OOV (Out-Of-Vocabulary) Fragmentation Rate:** Menghitung seberapa banyak Tokenizer IndoBERT "pecah" akibat gagal mengenali kata gaul di ulasan Shopee dibandingkan Tokopedia.
2.  **Jensen-Shannon Divergence (JSD):** Mengukur jarak probabilitas distribusi kata. Skor JSD yang signifikan (mendekati 1) membuktikan secara konkret kepada penguji bahwa *Gaya Bahasa Tokopedia secara empiris BERBEDA dari Shopee*.

---

## 🚀 5. Urutan Eksekusi Sistem (Pipeline Command)

Proyek ini telah dikondisikan agar dapat dieksekusi secara otomatis dari hulu ke hilir. Jalankan urutan perintah berikut pada terminal:

```bash
# 0. Siapkan Lingkungan Kerja
conda activate nlp-sentiment

# 1. Pecah dan Bersihkan Data (Hasilkan 8 File Eksperimen)
python src/preprocessing/run_preprocessing.py

# 2. Latih Ke-4 Versi IndoBERT (Otomasi Berurutan, VRAM Flushed per epoch)
python src/modeling/train_indobert.py --platform all --norm both

# 3. Hitung Matriks Akurasi Silang & Kalkulasi Selisih Normalisasi (Delta F1)
python src/modeling/evaluate.py

# 4. Angkat Dasbor Visualisasi Interaktif
streamlit run streamlit_app.py
```

### Panduan Presentasi (*Dashboard Demo*)
Saat sidang, antarmuka `streamlit_app.py` telah didesain untuk mendemonstrasikan teori.
*   Pilih **Model Tokopedia**.
*   Masukkan teks: *"gila x ya ini tko nipu pelnggn. psn wrna mrh dtg htm."* (Teks bahasa Shopee yang berat).
*   Klik **Analisis (Baseline)** $\rightarrow$ Akurasi prediksi akan kacau (OOV memicu disorientasi model).
*   Ubah saklar ke **Proposed (Normalisasi)** $\rightarrow$ Dasbor akan menampilkan teks terjemahan *"gila kali ya ini toko menipu pelanggan..."* lalu masuk ke model. Akurasi akan kembali tajam, membuktikan hipotesis penelitian secara *real-time*!
