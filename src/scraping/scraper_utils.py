# ============================================================
# src/scraping/scraper_utils.py
# Helper functions: logging, delay, checkpoint, user-agent
# ============================================================

import csv
import os
import random
import time
from datetime import datetime
from pathlib import Path
from curl_cffi import requests as cffi_requests

from loguru import logger

# ── User-Agent Pool ───────────────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
]

CSV_COLUMNS = [
    "review_text",
    "rating",
    "product_name",
    "review_date",
    "reviewer_name",
    "source_platform",
]


def setup_logger(log_dir: Path, name: str) -> None:
    """Setup loguru logger ke file dan stdout."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"{name}_{timestamp}.log"
    logger.add(str(log_file), rotation="50 MB", retention="7 days", level="INFO")
    logger.info(f"Logger initialized → {log_file}")


def random_delay(min_sec: float = 1.0, max_sec: float = 3.0) -> None:
    """Tidur random antara min_sec dan max_sec detik."""
    delay = random.uniform(min_sec, max_sec)
    time.sleep(delay)


def get_random_ua() -> str:
    """Ambil User-Agent acak dari pool."""
    return random.choice(USER_AGENTS)

def validate_url_cffi(url: str, headers: dict) -> bool:
    """Mengecek apakah URL bisa diakses (HTTP 200) menggunakan curl_cffi."""
    try:
        resp = cffi_requests.get(url, headers=headers, impersonate="chrome120", timeout=10)
        if resp.status_code == 200:
            return True
        else:
            logger.warning(f"URL Tidak Valid (Status {resp.status_code}) — Skip: {url}")
            return False
    except Exception as e:
        logger.warning(f"Gagal memvalidasi URL (Error DNS/Timeout) — Skip: {url}")
        return False

def ensure_csv(filepath: Path, columns: list[str]) -> None:
    """Buat file CSV dengan header jika belum ada."""
    if not filepath.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
        logger.info(f"CSV baru dibuat: {filepath}")


def checkpoint_save(filepath: Path, rows: list[dict], columns: list[str]) -> None:
    """Append rows ke CSV (checkpoint — anti data loss)."""
    if not rows:
        return
    with open(filepath, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerows(rows)
    logger.info(f"Checkpoint: {len(rows)} baris disimpan ke {filepath.name}")


def count_existing_rows(filepath: Path) -> int:
    """Hitung jumlah baris data yang sudah ada di CSV (tidak termasuk header)."""
    if not filepath.exists():
        return 0
    with open(filepath, "r", encoding="utf-8-sig") as f:
        return max(0, sum(1 for _ in f) - 1)


def load_shop_urls(filepath: Path) -> list[str]:
    """Baca daftar URL toko dari file teks (skip baris komentar dan kosong)."""
    urls = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    logger.info(f"Loaded {len(urls)} URL dari {filepath.name}")
    return urls


def load_keywords(filepath: Path) -> list[str]:
    """Baca daftar keyword dari file teks (skip komentar dan kosong)."""
    keywords = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                keywords.append(line)
    logger.info(f"Loaded {len(keywords)} keyword dari {filepath.name}")
    return keywords


def print_summary(platform: str, total: int, filepath: Path) -> None:
    """Print ringkasan hasil scraping."""
    logger.info("=" * 50)
    logger.info(f"SCRAPING SELESAI — {platform.upper()}")
    logger.info(f"Total review terkumpul : {total}")
    logger.info(f"Disimpan ke            : {filepath}")
    logger.info("=" * 50)
