# ============================================================
# src/preprocessing/merger.py
# Step 2: Merge Data multi-platform berdasarkan Argumen Mode
# Auto-Labeling, Deduplikasi, Standardisasi Missing Values & Downsampling
# Usage: python src/preprocessing/merger.py --mode [single/multi]
# ============================================================

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import LOG_DIR, TOKOPEDIA_RAW_CSV, rating_to_label
from src.scraping.scraper_utils import setup_logger

REQUIRED_COLUMNS = [
    "review_text", "rating", "product_name",
    "review_date", "reviewer_name", "source_platform",
]

COLUMN_DEFAULTS = {
    "review_text":     "",
    "rating":          3,
    "product_name":    "unknown",
    "review_date":     "unknown",
    "reviewer_name":   "anonymous",
    "source_platform": "unknown",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "multi"], default="single",
                        help="single (1 platform) atau multi (Tokopedia + Shopee)")
    parser.add_argument("--platform", choices=["tokopedia", "shopee"], default="tokopedia",
                        help="Target platform (wajib jika --mode single)")
    return parser.parse_args()


def get_output_path(mode: str) -> Path:
    base = ROOT / "data" / "raw" / "merged"
    folder = "single_platform" if mode == "single" else "multi_platform"
    path = base / folder / "merged_labeled.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_csv(filepath: Path, platform: str) -> pd.DataFrame:
    if not filepath.exists():
        logger.warning(f"  File tidak ditemukan: {filepath} — dilewati")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(filepath, encoding="latin1")
    except Exception as e:
        logger.error(f"  Gagal membaca {filepath}: {e}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    logger.info(f"  {platform}: {len(df)} baris dari {filepath.name}")

    # Standardisasi kolom: drop extra columns, add missing with defaults
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = COLUMN_DEFAULTS[col]
            logger.info(f"    └─ Kolom '{col}' tidak ditemukan — diisi default: '{COLUMN_DEFAULTS[col]}'")

    # Hanya pertahankan kolom yang diperlukan
    df = df[REQUIRED_COLUMNS].copy()

    # Bersihkan missing values per kolom
    for col in REQUIRED_COLUMNS:
        missing = df[col].isna().sum()
        if missing > 0:
            df[col] = df[col].fillna(COLUMN_DEFAULTS[col])
            logger.info(f"    └─ {missing} nilai kosong di '{col}' — diisi default: '{COLUMN_DEFAULTS[col]}'")

    # rating: pastikan numerik, invalid → 3
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3).astype(int)
    df["rating"] = df["rating"].clip(1, 5)

    # source_platform: timpa sesuai argumen (konsisten)
    df["source_platform"] = platform

    # review_text: hapus baris yang benar-benar kosong
    before = len(df)
    df = df[df["review_text"].str.strip().ne("")]
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"    └─ {dropped} baris dengan review_text kosong — dihapus")

    return df


def filter_empty_and_noise(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    from config.settings import MIN_TEXT_LENGTH, MIN_TOKEN_COUNT

    before = len(df)
    # Hapus review yang terlalu pendek setelah strip
    df = df[df["review_text"].str.strip().str.len() >= MIN_TEXT_LENGTH]
    logger.info(f"  Filter <{MIN_TEXT_LENGTH} char: {before - len(df)} baris dihapus")

    if mode == "multi":
        # Denoising Netral: rating=3 dengan teks < 50 char dihapus (meaningless neutral)
        # Serta membuang ulasan netral yang isinya hanya kata-kata filler pendek
        meaningless_keywords = ["oke", "ok", "biasa", "lumayan", "bagus", "mantap", "sip", "standar", "sesuai"]
        
        # Kondisi 1: Rating 3 dan karakter sangat pendek (<50)
        cond_length = (df["rating"] == 3) & (df["review_text"].str.strip().str.len() < 50)
        
        # Kondisi 2: Rating 3 dan teksnya KESELURUHAN (setelah di-lower dan strip) hanya ada di daftar meaningless
        cond_meaningless = (df["rating"] == 3) & (df["review_text"].str.lower().str.strip().isin(meaningless_keywords))
        
        cond = cond_length | cond_meaningless
        n_neutral_noise = cond.sum()
        df = df[~cond]
        logger.info(f"  Denoising Netral (<50 char & meaningless): {n_neutral_noise} baris dihapus")

    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["review_text"], keep="first")
    logger.info(f"  Deduplikasi: {before - len(df)} baris duplikat dihapus")
    return df


def balance_classes_downsample(df: pd.DataFrame) -> pd.DataFrame:
    # Pastikan label sudah dibuat SEBELUM melakukan downsampling
    if "label" not in df.columns:
        df = auto_label(df)
        
    label_counts = df["label"].value_counts()
    target = int(label_counts.median())
    logger.info(f"  Downsampling: target {target} baris per kelas sentimen (Negatif/Netral/Positif)")

    chunks = []
    for label_val in sorted(df["label"].unique()):
        subset = df[df["label"] == label_val]
        if len(subset) > target:
            subset = subset.sample(n=target, random_state=42)
        chunks.append(subset)

    out = pd.concat(chunks, ignore_index=True)
    logger.info(f"  Hasil downsampling: {len(out)} baris (dari {len(df)})")
    return out


def auto_label(df: pd.DataFrame) -> pd.DataFrame:
    # Ubah apply untuk menerima dua kolom: rating dan review_text
    df["label"] = df.apply(lambda row: rating_to_label(row["rating"], row["review_text"]), axis=1)
    return df


def print_stats(df: pd.DataFrame) -> None:
    logger.info("─" * 40)
    logger.info("STATISTIK DATASET")
    logger.info(f"  Total baris:     {len(df)}")
    logger.info(f"  Distribusi rating:\n{df['rating'].value_counts().sort_index().to_string()}")
    logger.info(f"  Distribusi label:\n{df['label'].value_counts().to_string()}")
    logger.info(f"  Source platform:\n{df['source_platform'].value_counts().to_string()}")
    logger.info(f"  Sample review (3):")
    for _, r in df.sample(n=min(3, len(df))).iterrows():
        logger.info(f"    [{r['label']}] {r['review_text'][:80]}...")
    logger.info("─" * 40)


def main():
    args = parse_args()
    setup_logger(LOG_DIR, "merger")
    logger.info("=" * 60)
    logger.info(f"STEP 2: MERGING, DEDUPLICATION & LABELING (MODE: {args.mode.upper()})")

    if args.mode == "single":
        logger.info(f"TARGET PLATFORM: {args.platform.upper()}")
    logger.info("=" * 60)

    dfs = []

    if args.mode == "single":
        if args.platform == "tokopedia":
            dfs.append(load_csv(TOKOPEDIA_RAW_CSV, "tokopedia"))
        elif args.platform == "shopee":
            from config.settings import SHOPEE_RAW_CSV
            dfs.append(load_csv(SHOPEE_RAW_CSV, "shopee"))

    elif args.mode == "multi":
        dfs.append(load_csv(TOKOPEDIA_RAW_CSV, "tokopedia"))
        from config.settings import SHOPEE_RAW_CSV
        dfs.append(load_csv(SHOPEE_RAW_CSV, "shopee"))

    valid_dfs = [d for d in dfs if not d.empty]
    if not valid_dfs:
        logger.error("Tidak ada data yang bisa dimuat. Periksa argumen --platform Anda!")
        sys.exit(1)

    df = pd.concat(valid_dfs, ignore_index=True)

    df = filter_empty_and_noise(df, args.mode)
    df = deduplicate(df)
    
    # Label harus dibuat SEBELUM downsampling agar balancing berdasarkan label, bukan rating
    df = auto_label(df)

    if args.mode == "multi":
        df = balance_classes_downsample(df)

    print_stats(df)

    output_path = get_output_path(args.mode)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    logger.info(f"Tersimpan ke: {output_path}")
    logger.info(f"Selanjutnya: python src/preprocessing/run_preprocessing.py --mode {args.mode}")


if __name__ == "__main__":
    main()
