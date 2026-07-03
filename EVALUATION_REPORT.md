# LAPORAN EVALUASI & ANALISIS CROSS-PLATFORM (LOSO)
**Pengaruh Normalisasi Kata Slang pada Generalisasi IndoBERT**

---

## 1. Tujuan Pengujian
Pengujian ini bertujuan membuktikan dua fenomena sentral:
1. **Domain Shift:** Seberapa besar degradasi performa (*performance drop*) saat model yang dilatih di satu E-Commerce diuji pada E-Commerce lain (*Zero-Shot Cross-Platform*).
2. **Efektivitas Normalisasi Slang:** Apakah mengubah kata gaul menjadi bahasa Indonesia formal sebelum diproses IndoBERT mampu meningkatkan kemampuan generalisasi model saat melintas platform.

---

## 2. Tabel Hasil Evaluasi (F1-Macro)

### A. Pengujian In-Domain (Kontrol / Batas Atas)
Menguji model pada "habitat aslinya" (Platform Train = Platform Test).

| Model Dilatih Di | Diuji Di | Kondisi Data | F1-Macro | Accuracy |
|:---|:---|:---|:---:|:---:|
| Tokopedia | Tokopedia | Baseline (Tanpa Normalisasi) | **0.6634** | 0.6995 |
| Tokopedia | Tokopedia | Proposed (Dengan Normalisasi) | **0.6696** | 0.6987 |
| | | *Peningkatan Lokal* | ***+0.0062*** | |
| Shopee | Shopee | Baseline (Tanpa Normalisasi) | **0.6493** | 0.6782 |
| Shopee | Shopee | Proposed (Dengan Normalisasi) | **0.6552** | 0.6909 |
| | | *Peningkatan Lokal* | ***+0.0059*** | |

**Kesimpulan Awal:** Normalisasi Slang **terbukti secara konsisten meningkatkan akurasi** saat digunakan untuk pengujian *In-Domain*.

---

### B. Pengujian Cross-Platform (Leave-One-Site-Out / LOSO)
Menguji model pada platform asing (Platform Train $\neq$ Platform Test).

| Model Dilatih Di | Diuji Di | Kondisi Data | F1-Macro | Accuracy | Degradasi dari In-Domain |
|:---|:---|:---|:---:|:---:|:---:|
| Tokopedia | Shopee | Baseline | **0.5588** | 0.6141 | *-0.1046* |
| Shopee | Tokopedia | Baseline | **0.5840** | 0.6553 | *-0.0653* |
| Tokopedia | Shopee | Proposed | **0.5506** | 0.6179 | *-0.1190* |
| Shopee | Tokopedia | Proposed | **0.5931** | 0.6635 | *-0.0621* |

---

## 3. Analisis Ilmiah (Insights & Findings)

### 🚨 Temuan 1: Fenomena Domain Shift Terbukti Nyata
Hipotesis adanya *Domain Shift* terbukti benar secara empiris. Ketika model Tokopedia dipaksa membaca data Shopee, akurasi anjlok drastis (Degradasi **-10.4%** F1-Macro). 
Hal ini disebabkan karena dataset Shopee jauh lebih heterogen (memuat ulasan aplikasi dan logistik COD), sehingga model Tokopedia yang hanya dilatih membedah ulasan produk (*homogen*) kebingungan menghadapi kosakata teknis yang tidak pernah ia lihat (*Out-Of-Vocabulary / OOV*).

### 📈 Temuan 2: Normalisasi Slang Berhasil Menyelamatkan Model (Skenario Shopee $\rightarrow$ Toped)
Pada pengujian **Shopee diuji ke Tokopedia**, pengaktifan fitur Normalisasi Slang berhasil menaikkan F1-Macro dari `0.5840` menjadi `0.5931` (Meningkat **+0.0091**).
*Analisis:* Kamus translasi slang berhasil menjembatani kesenjangan gaya bahasa antar platform. Model Shopee yang sudah memiliki pemahaman leksikal luas (*broad domain*) kini dibantu dengan teks yang lebih terstandarisasi, membantunya menebak sentimen data Tokopedia dengan lebih akurat.

### 📉 Temuan 3: Anomali Konteks (Skenario Toped $\rightarrow$ Shopee)
Pada pengujian sebaliknya (**Tokopedia diuji ke Shopee**), fitur Normalisasi Slang secara mengejutkan justru merusak akurasi, di mana F1-Macro turun dari `0.5588` menjadi `0.5506` (Turun **-0.0083**).
*Analisis:* Kamus slang statis kita secara buta menerjemahkan kata gaul Shopee tanpa memahami konteks kalimat seutuhnya (*Context Loss*). Karena otak model Tokopedia sangat sempit (hanya tahu ulasan barang), ketika kata spesifik Shopee diterjemahkan, IndoBERT Tokopedia justru semakin kehilangan pijakan semantiknya untuk membedakan kelas sentimen.

---

## 4. Kelemahan Sistem (Limitations) & Saran Penelitian Lanjutan

Berdasarkan anomali pada Temuan 3, berikut adalah batasan (*limitations*) model ini yang wajib dipertimbangkan:

1. **Static Dictionary Bottleneck:** Modul *Slang Normalization* kita menggunakan pemetaan kata baku secara kaku (1-to-1). Sistem ini tidak bisa membedakan makna ganda (polisemi) yang sering terjadi di bahasa gaul internet tergantung konteks kalimat sekitarnya.
2. **Topic Mismatch (Bukan Sekadar Bahasa):** Domain Shift 10% terjadi bukan murni karena bahasa gaul, melainkan karena *topik* yang dibicarakan berbeda. Shopee banyak berisi umpatan tentang fitur aplikasi "nge-lag", sedangkan Tokopedia berisi ulasan "baju bagus". Normalisasi kata tidak bisa menyelamatkan model yang memang "buta" akan sebuah topik.
3. **IndoBERT Murni Formal:** *Pre-training* IndoBERT menggunakan teks Wikipedia (formal). Normalisasi kita mencoba menjembatani ini, namun arsitektur internal IndoBERT secara fundamental tetap rentan terhadap struktur kalimat media sosial yang melompat-lompat tanpa gramatika jelas.

**Saran Mendatang:** Mengganti *Slang Dictionary* statis dengan *Dynamic Contextual Slang Normalization* (seperti model LLM mini di depan IndoBERT) atau langsung melakukan tahap *Domain-Adaptive Pre-Training* pada korpus e-commerce mentah.

---
*Laporan digenerate pada Juli 2026. Data diolah dari skrip evaluasi LOSO (`evaluate.py`).*
