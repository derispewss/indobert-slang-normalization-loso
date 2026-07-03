# ============================================================
# src/scraping/shopee_kaggle.py
# Download & standardisasi 4 dataset Shopee dari Kaggle
#
# Sumber:
#   1. Ahmad Selo Abadi — Shopee Play Store ID (85.500)
#   2. Taqiyya Ghazi    — Marketplace reviews (831)
#   3. Md Himas Pamungkas — Review produk Shopee (3.020)
#   4. Alvian Ardiansyah — Ulasan COD Shopee (1.840)
# ============================================================

import sys
import pandas as pd
from pathlib import Path
from loguru import logger
import kagglehub

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import SHOPEE_RAW_CSV

REQUIRED = ["review_text", "rating", "product_name", "review_date", "reviewer_name", "source_platform"]


def download_ahmad() -> pd.DataFrame:
    path = kagglehub.dataset_download("ahmadseloabadi/shoppe-app-reviews-from-google-play-store")
    df = pd.read_csv(Path(path) / "scrapped_Shopee 12.12_ID_.csv")
    out = pd.DataFrame({
        "review_text":      df["content"],
        "rating":           df["score"],
        "product_name":     "shopee_app",
        "review_date":      pd.to_datetime(df["at"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "reviewer_name":    df["userName"],
        "source_platform":  "shopee",
    })
    logger.info(f"  Ahmad: {len(out)} baris (Play Store ID)")
    return out


def download_taqiyya() -> pd.DataFrame:
    path = kagglehub.dataset_download("taqiyyaghazi/indonesian-marketplace-product-reviews")
    df = pd.read_csv(Path(path) / "reviews.csv")
    rating_map = {0.0: 1, 1.0: 5}
    out = pd.DataFrame({
        "review_text":      df["reviews"],
        "rating":           df["label"].map(rating_map).fillna(3).astype(int),
        "product_name":     "marketplace_generic",
        "review_date":      "unknown",
        "reviewer_name":    "anonymous",
        "source_platform":  "shopee",
    })
    logger.info(f"  Taqiyya: {len(out)} baris (binary label -> rating 1/5)")
    return out


def download_himas() -> pd.DataFrame:
    path = kagglehub.dataset_download("mdhimaspamungkas/review-product-shopee")
    df = pd.read_csv(Path(path) / "data.csv")
    out = pd.DataFrame({
        "review_text":      df["comment"],
        "rating":           df["rating"],
        "product_name":     "shopee_product",
        "review_date":      "unknown",
        "reviewer_name":    df["username"].fillna("anonymous"),
        "source_platform":  "shopee",
    })
    logger.info(f"  Himas: {len(out)} baris (produk Shopee)")
    return out


def download_alvian() -> pd.DataFrame:
    path = kagglehub.dataset_download("alvianardiansyah/dataset-ulasan-pengguna-shopee")
    df = pd.read_csv(Path(path) / "Data ulasan Shopee tentang COD.csv", encoding="latin1")
    out = pd.DataFrame({
        "review_text":      df["content"],
        "rating":           df["score"],
        "product_name":     "shopee_cod",
        "review_date":      pd.to_datetime(df["at"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "reviewer_name":    df["userName"],
        "source_platform":  "shopee",
    })
    logger.info(f"  Alvian: {len(out)} baris (ulasan COD)")
    return out


def run() -> int:
    logger.info("Mengunduh & menstandardisasi 4 dataset Shopee dari Kaggle...")

    funcs = [
        ("Ahmad (Play Store ID)",    download_ahmad),
        ("Taqiyya (Marketplace)",    download_taqiyya),
        ("Himas (Review Produk)",    download_himas),
        ("Alvian (COD)",             download_alvian),
    ]

    all_dfs = []
    for name, fn in funcs:
        try:
            df = fn()
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"  Gagal memuat {name}: {e}")
            continue

    if not all_dfs:
        logger.error("Tidak ada dataset yang berhasil dimuat.")
        return 0

    df_concat = pd.concat(all_dfs, ignore_index=True)

    df_concat = df_concat.dropna(subset=["review_text"])
    df_concat = df_concat[df_concat["review_text"].str.strip().ne("")]
    df_concat = df_concat.drop_duplicates(subset=["review_text"])
    df_concat["rating"] = df_concat["rating"].astype(int)

    SHOPEE_RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_concat.to_csv(SHOPEE_RAW_CSV, index=False, encoding="utf-8-sig")

    total = len(df_concat)
    dist = df_concat["rating"].value_counts().sort_index().to_dict()
    logger.info(f"Berhasil menyimpan {total} ulasan ke {SHOPEE_RAW_CSV}")
    logger.info(f"Distribusi rating: {dist}")
    return total


if __name__ == "__main__":
    run()
