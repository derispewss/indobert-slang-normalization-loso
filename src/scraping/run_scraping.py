# ============================================================
# src/scraping/run_scraping.py
# Entry point untuk menjalankan scraper Tokopedia & Shopee (publik)
# Usage:
#   python src/scraping/run_scraping.py --mode single --platform tokopedia
#   python src/scraping/run_scraping.py --mode single --platform shopee
#   python src/scraping/run_scraping.py --mode multi
# ============================================================

import argparse
import sys
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runner scraping ulasan produk e-commerce Indonesia",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="Mode eksekusi: 'single' untuk 1 platform (butuh --platform), 'multi' otomatis kedua platform.",
    )
    parser.add_argument(
        "--platform",
        choices=["tokopedia", "shopee"],
        default="tokopedia",
        help="Target platform (Hanya berlaku di --mode single)",
    )
    return parser.parse_args()

def run_tokopedia() -> int:
    logger.info("━" * 60)
    logger.info("SCRAPING TOKOPEDIA (GraphQL API)")
    logger.info("━" * 60)
    from src.scraping.tokopedia_scraper import run
    return run()

def run_shopee() -> int:
    logger.info("━" * 60)
    logger.info("SHOPEE — 4 Dataset dari Kaggle")
    logger.info("━" * 60)
    try:
        from src.scraping.shopee_kaggle import run
        return run()
    except ImportError as e:
        logger.error(f"Modul shopee_kaggle tidak ditemukan: {e}")
        return 0

def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  SENTIMENT ANALYSIS — DATA COLLECTION PIPELINE")
    logger.info(f"  Mode: {args.mode.upper()}")
    if args.mode == "single":
        logger.info(f"  Platform: {args.platform.upper()}")
    logger.info("=" * 60)

    total = 0

    if args.mode == "single":
        if args.platform == "tokopedia":
            total = run_tokopedia()
        elif args.platform == "shopee":
            total = run_shopee()

    elif args.mode == "multi":
        logger.info("Menjalankan Pengumpulan Data Lintas Platform (Tokopedia -> Shopee)...")
        total += run_tokopedia()
        total += run_shopee()

    logger.info("=" * 60)
    logger.info(f"SELESAI — Total ulasan terkumpul sesi ini: {total}")
    logger.info(f"Langkah berikutnya: python src/preprocessing/merger.py --mode {args.mode}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
