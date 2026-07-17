# ============================================================
# src/preprocessing/run_preprocessing.py
# Memisahkan dan membersihkan data menjadi 4 Kondisi Eksperimen:
# 1. Tokopedia Baseline (Non-Norm)
# 2. Tokopedia Proposed (Norm)
# 3. Shopee Baseline (Non-Norm)
# 4. Shopee Proposed (Norm)
# ============================================================

import sys
import argparse
from pathlib import Path
import pandas as pd
from loguru import logger
from tqdm import tqdm
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import RANDOM_STATE, TEST_SIZE, VAL_SIZE
from src.preprocessing.cleaner import clean_text_pipeline
from src.scraping.scraper_utils import setup_logger

def get_io_paths():
    input_csv = ROOT / "data" / "raw" / "merged" / "multi_platform" / "merged_labeled.csv"
    output_dir = ROOT / "data" / "processed" / "experiment_splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_csv, output_dir

def process_dataset(df: pd.DataFrame, platform_name: str, use_normalization: bool, output_dir: Path):
    condition = "proposed" if use_normalization else "baseline"
    logger.info(f"\n--- Memproses {platform_name.upper()} | Kondisi: {condition.upper()} ---")
    
    # Filter dataset berdasarkan platform
    df_platform = df[df["source_platform"] == platform_name].copy()
    logger.info(f"Jumlah awal: {len(df_platform)} baris")

    # Cleaning pipeline
    tqdm.pandas(desc=f"Cleaning {platform_name}-{condition}")
    df_platform["review_text_clean"] = df_platform["review_text"].astype(str).progress_apply(
        lambda x: clean_text_pipeline(x, use_normalization=use_normalization)
    )

    before = len(df_platform)
    df_platform = df_platform.dropna(subset=["review_text_clean"])
    df_platform = df_platform[df_platform["review_text_clean"].str.strip() != ""]
    df_platform = df_platform.reset_index(drop=True)
    after = len(df_platform)
    logger.info(f"Filter pasca cleaning: {before} → {after} baris (drop {before - after})")

    # Prepare final columns
    if "sentiment_label" not in df_platform.columns:
        if "label" in df_platform.columns:
            df_platform = df_platform.rename(columns={"label": "sentiment_label"})
    
    output_cols = ["review_text_clean", "review_text", "rating", "sentiment_label", "source_platform"]
    df_platform = df_platform[output_cols]

    # Split 70% Train, 15% Val, 15% Test Stratified
    # Pertama pisahkan Train (70%) dan Sisa (30%)
    train_df, temp_df = train_test_split(
        df_platform,
        test_size=(TEST_SIZE + VAL_SIZE),
        random_state=RANDOM_STATE,
        stratify=df_platform["sentiment_label"],
    )
    
    # Lalu pisahkan sisa (30%) menjadi Val (15%) dan Test (15%)
    val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_ratio,
        random_state=RANDOM_STATE,
        stratify=temp_df["sentiment_label"],
    )
    
    train_file = output_dir / f"train_{platform_name}_{condition}.csv"
    val_file   = output_dir / f"val_{platform_name}_{condition}.csv"
    test_file  = output_dir / f"test_{platform_name}_{condition}.csv"
    
    train_df.to_csv(train_file, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_file, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_file, index=False, encoding="utf-8-sig")
    
    logger.info(f"Train set: {len(train_df)} baris → {train_file.name}")
    logger.info(f"Val set  : {len(val_df)} baris → {val_file.name}")
    logger.info(f"Test set : {len(test_df)} baris → {test_file.name}")

def run():
    input_csv, output_dir = get_io_paths()
    setup_logger(ROOT / "logs", "preprocessing_experiments")
    logger.info("=" * 60)
    logger.info("STEP 3: TEXT PREPROCESSING (4 SKENARIO EKSPERIMEN LOSO)")
    logger.info("=" * 60)

    if not input_csv.exists():
        logger.error(f"File input tidak ditemukan: {input_csv}")
        sys.exit(1)

    df_master = pd.read_csv(input_csv, encoding="utf-8-sig")
    logger.info(f"Master Data dimuat: {len(df_master)} baris")

    # Jalankan 4 skenario
    platforms = ["tokopedia", "shopee"]
    conditions = [False, True] # False=Baseline, True=Proposed

    for plat in platforms:
        for norm in conditions:
            process_dataset(df_master, plat, norm, output_dir)
            
    logger.info("\nLangkah berikutnya: Eksekusi src/modeling/train_indobert.py")

if __name__ == "__main__":
    run()
