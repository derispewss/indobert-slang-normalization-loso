# ============================================================
# src/scraping/tokopedia_scraper.py
# Scraping review Tokopedia via GraphQL API internal
# TANPA Selenium — hanya menggunakan requests
# Risiko bot detection: RENDAH
# ============================================================

import re
import sys
import time
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

# Tambahkan root ke sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import (
    CONFIG_DIR,
    LOG_DIR,
    TOKOPEDIA_CHECKPOINT_EVERY,
    TOKOPEDIA_GQL_ENDPOINT,
    TOKOPEDIA_HEADERS,
    TOKOPEDIA_PAGE_SIZE,
    TOKOPEDIA_RATING_QUOTA,
    TOKOPEDIA_RAW_CSV,
    TOKOPEDIA_SHOPS_FILE,
    TOKOPEDIA_SLEEP_BETWEEN_PAGES,
)
from src.scraping.scraper_utils import (
    CSV_COLUMNS,
    checkpoint_save,
    count_existing_rows,
    ensure_csv,
    load_shop_urls,
    print_summary,
    random_delay,
    setup_logger,
)

# ── GraphQL Query ─────────────────────────────────────────────
REVIEW_LIST_QUERY = """
query ReviewList($shopID:String!,$limit:Int!,$page:Int!,$filterBy:String,$sortBy:String){
  productrevGetShopReviewReadingList(
    shopID:$shopID,
    limit:$limit,
    page:$page,
    filterBy:$filterBy,
    sortBy:$sortBy
  ){
    list{
      id:reviewID
      product{
        productID
        productName
        productPageURL
      }
      rating
      reviewTime
      reviewText
      reviewerID
      reviewerName
    }
    hasNext
    shopName
    totalReviews
  }
}
""".strip()

# ── Regex untuk ekstrak Shop ID ───────────────────────────────
SHOP_ID_PATTERNS = [
    re.compile(r'ShopPageGetHeaderLayout\(\{\\\"shopID\\\":\\\"(\d+)\\\"'),
    re.compile(r'productrevGetShopReviewReadingList.*?shopID.*?(\d+)'),
    re.compile(r'"shopID":"(\d+)"'),
]


def get_shop_id(session: requests.Session, review_url: str) -> str | None:
    """
    Ekstrak Shop ID dari halaman HTML toko Tokopedia.
    Menggunakan 3 pola regex bertingkat sebagai fallback.
    """
    try:
        resp = session.get(review_url, headers=TOKOPEDIA_HEADERS, timeout=30)
        
        # Validasi status code
        if resp.status_code != 200:
            logger.warning(f"URL tidak valid atau mengembalikan Error {resp.status_code}: {review_url}")
            return None
            
        html = resp.text

        for pattern in SHOP_ID_PATTERNS:
            match = pattern.search(html)
            if match:
                shop_id = match.group(1)
                logger.info(f"Shop ID ditemukan: {shop_id} dari {review_url}")
                return shop_id

        logger.warning(f"Shop ID tidak ditemukan untuk: {review_url}")
        return None

    except requests.RequestException as e:
        logger.error(f"Gagal fetch halaman toko {review_url}: {e}")
        return None


def fetch_review_page(
    session: requests.Session,
    shop_id: str,
    page: int,
    rating: int,
) -> dict | None:
    """
    Fetch satu halaman review dari GraphQL endpoint Tokopedia.
    Return dict data atau None jika gagal.
    """
    payload = {
        "operationName": "ReviewList",
        "query": REVIEW_LIST_QUERY,
        "variables": {
            "shopID": shop_id,
            "limit": TOKOPEDIA_PAGE_SIZE,
            "page": page,
            "filterBy": f"rating={rating}",
            "sortBy": "create_time desc",
        },
    }

    try:
        resp = session.post(
            TOKOPEDIA_GQL_ENDPOINT,
            json=payload,
            headers=TOKOPEDIA_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if "data" not in data or "productrevGetShopReviewReadingList" not in data["data"]:
            logger.warning(f"Respons tidak valid untuk shop {shop_id} rating {rating} page {page}")
            return None

        return data["data"]["productrevGetShopReviewReadingList"]

    except requests.RequestException as e:
        logger.error(f"Request gagal shop {shop_id} rating {rating} page {page}: {e}")
        return None


def is_valid_review(review_text: str | None) -> bool:
    """Cek apakah review layak disimpan."""
    if not review_text:
        return False
    text = review_text.strip()
    if len(text) < 5:
        return False
    # Skip jika hanya berisi angka
    if text.isdigit():
        return False
    return True


def scrape_shop_by_rating(
    session: requests.Session,
    shop_id: str,
    shop_name: str,
    rating: int,
    quota: int,
) -> list[dict]:
    """
    Scrape review dari satu toko untuk satu rating tertentu.
    Berhenti saat quota terpenuhi atau tidak ada halaman berikutnya.
    """
    collected = []
    page = 1

    with tqdm(total=quota, desc=f"  [{shop_name}] ★{rating}", leave=False) as pbar:
        while len(collected) < quota:
            data = fetch_review_page(session, shop_id, page, rating)
            if data is None:
                break

            items = data.get("list") or []
            if not items:
                break

            for item in items:
                if len(collected) >= quota:
                    break

                review_text = (item.get("reviewText") or "").strip()
                if not is_valid_review(review_text):
                    continue

                product = item.get("product") or {}
                row = {
                    "review_text":    review_text,
                    "rating":         item.get("rating", rating),
                    "product_name":   (product.get("productName") or "").strip(),
                    "review_date":    (item.get("reviewTime") or "").strip(),
                    "reviewer_name":  (item.get("reviewerName") or "").strip(),
                    "source_platform": "tokopedia",
                }
                collected.append(row)
                pbar.update(1)

            # Cek pagination
            if not data.get("hasNext", False):
                break

            page += 1
            time.sleep(TOKOPEDIA_SLEEP_BETWEEN_PAGES)

    logger.info(f"  [{shop_name}] ★{rating}: {len(collected)}/{quota} review terkumpul")
    return collected


def run(shops_file: Path = TOKOPEDIA_SHOPS_FILE, output_file: Path = TOKOPEDIA_RAW_CSV) -> int:
    """
    Entry point scraping Tokopedia.
    Return total jumlah review yang berhasil dikumpulkan.
    """
    setup_logger(LOG_DIR, "tokopedia_scraper")
    logger.info("=" * 60)
    logger.info("MULAI SCRAPING TOKOPEDIA (GraphQL API)")
    logger.info("=" * 60)

    # Buat CSV dengan header jika belum ada
    ensure_csv(output_file, CSV_COLUMNS)
    already_collected = count_existing_rows(output_file)
    logger.info(f"Review yang sudah ada: {already_collected}")

    # Load daftar toko
    shop_urls = load_shop_urls(shops_file)
    if not shop_urls:
        logger.error("Tidak ada URL toko. Cek config/tokopedia_shops.txt")
        return 0

    session = requests.Session()
    total_collected = already_collected
    buffer: list[dict] = []

    for shop_url in tqdm(shop_urls, desc="Toko Tokopedia"):
        # Ekstrak Shop ID
        shop_id = get_shop_id(session, shop_url)
        if not shop_id:
            logger.warning(f"Skip toko: {shop_url}")
            continue

        shop_name = shop_url.rstrip("/").split("/")[-2]  # ambil nama dari URL
        logger.info(f"\nScraping toko: {shop_name} (ID: {shop_id})")

        # Scrape per rating untuk balance dataset
        for rating, quota in TOKOPEDIA_RATING_QUOTA.items():
            reviews = scrape_shop_by_rating(session, shop_id, shop_name, rating, quota)
            buffer.extend(reviews)
            total_collected += len(reviews)

            # Checkpoint save setiap N review
            if len(buffer) >= TOKOPEDIA_CHECKPOINT_EVERY:
                checkpoint_save(output_file, buffer, CSV_COLUMNS)
                buffer.clear()

            # Delay sopan antar rating
            random_delay(0.5, 1.5)

        logger.info(f"Total terkumpul sejauh ini: {total_collected}")

    # Simpan sisa buffer
    if buffer:
        checkpoint_save(output_file, buffer, CSV_COLUMNS)

    print_summary("tokopedia", total_collected, output_file)
    return total_collected


if __name__ == "__main__":
    run()
