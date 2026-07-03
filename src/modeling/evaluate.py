# ============================================================
# src/modeling/evaluate.py
# Skrip Evaluasi Cross-Platform (LOSO) dan In-Domain
# Mengukur performa generalisasi IndoBERT dengan/tanpa Normalisasi Slang
# ============================================================

import sys
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import ID2LABEL, INDOBERT_BATCH_SIZE, INDOBERT_MAX_LENGTH, LABEL2ID, LOG_DIR
from src.modeling.dataset import ReviewDataset
from src.scraping.scraper_utils import setup_logger

try:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

def get_test_paths(platform: str, condition: str) -> Path:
    return ROOT / "data" / "processed" / "experiment_splits" / f"test_{platform}_{condition}.csv"

def get_model_path(platform: str, condition: str) -> Path:
    return ROOT / "model" / f"indobert_{platform}_{condition}"

def load_test_data(csv_path: Path) -> tuple[list[str], list[int]]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    texts = df["review_text_clean"].astype(str).tolist()
    labels = df["sentiment_label"].map(LABEL2ID).tolist()
    return texts, labels

def evaluate_model_on_dataset(model_path: Path, test_texts: list[str], test_labels: list[int]) -> dict:
    if not model_path.exists():
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path / "tokenizer"))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    dataset = ReviewDataset(test_texts, test_labels, tokenizer, INDOBERT_MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=INDOBERT_BATCH_SIZE, shuffle=False)

    all_preds = []
    start = time.time()
    
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)

    elapsed_ms = (time.time() - start) / len(test_texts) * 1000
    
    f1m = f1_score(test_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(test_labels, all_preds)

    # Bebaskan memori GPU
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "accuracy": acc,
        "f1_macro": f1m,
        "inference_ms": elapsed_ms
    }

def print_result_table(results: list, title: str):
    logger.info(f"\n{'='*90}")
    logger.info(f"TABEL EVALUASI: {title}")
    logger.info(f"{'='*90}")
    header = f"{'Model (Dilatih di)':<25} | {'Diuji di (Dataset Test)':<25} | {'Kondisi Data':<12} | {'F1-Macro':<8} | {'Accuracy':<8}"
    logger.info(header)
    logger.info("-" * 90)
    
    for r in results:
        row = f"{r['train_src']:<25} | {r['test_src']:<25} | {r['condition']:<12} | {r['f1_macro']:.4f}   | {r['accuracy']:.4f}"
        logger.info(row)
    logger.info("=" * 90)


def evaluate():
    setup_logger(LOG_DIR, "evaluate_cross_platform")
    logger.info("=" * 60)
    logger.info("STEP 7: CROSS-PLATFORM & IN-DOMAIN EVALUATION (INDOBERT)")
    logger.info("=" * 60)

    platforms = ["tokopedia", "shopee"]
    conditions = ["baseline", "proposed"]
    
    in_domain_results = []
    cross_domain_results = []

    for condition in conditions:
        logger.info(f"\n{'#'*40} MENGUJI KONDISI: {condition.upper()} {'#'*40}")
        
        for train_plat in platforms:
            model_path = get_model_path(train_plat, condition)
            if not model_path.exists():
                logger.warning(f"Lewati: Model {train_plat}-{condition} belum ada.")
                continue

            for test_plat in platforms:
                test_path = get_test_paths(test_plat, condition)
                if not test_path.exists():
                    continue

                logger.info(f"Menguji Model [{train_plat.upper()}] pada data [{test_plat.upper()}] ...")
                texts, labels = load_test_data(test_path)
                metrics = evaluate_model_on_dataset(model_path, texts, labels)

                if metrics:
                    res_dict = {
                        "train_src": train_plat.title(),
                        "test_src": test_plat.title(),
                        "condition": condition.title(),
                        "f1_macro": metrics["f1_macro"],
                        "accuracy": metrics["accuracy"],
                    }
                    
                    if train_plat == test_plat:
                        in_domain_results.append(res_dict)
                    else:
                        cross_domain_results.append(res_dict)

    if in_domain_results:
        print_result_table(in_domain_results, "IN-DOMAIN EVALUATION (Kontrol / Batas Atas)")
    if cross_domain_results:
        print_result_table(cross_domain_results, "CROSS-PLATFORM (LEAVE-ONE-SITE-OUT) EVALUATION")
        
        # Analisis Dampak Normalisasi Slang pada Generalisasi
        if len(cross_domain_results) == 4: # Pastikan ke-4 skenario silang selesai
            logger.info("\n--- ANALISIS KONTRIBUSI NORMALISASI SLANG PADA KEMAMPUAN GENERALISASI ---")
            
            # Toped -> Shopee
            ts_base = next(r["f1_macro"] for r in cross_domain_results if r["train_src"]=="Tokopedia" and r["condition"]=="Baseline")
            ts_prop = next(r["f1_macro"] for r in cross_domain_results if r["train_src"]=="Tokopedia" and r["condition"]=="Proposed")
            delta_ts = ts_prop - ts_base
            
            # Shopee -> Toped
            st_base = next(r["f1_macro"] for r in cross_domain_results if r["train_src"]=="Shopee" and r["condition"]=="Baseline")
            st_prop = next(r["f1_macro"] for r in cross_domain_results if r["train_src"]=="Shopee" and r["condition"]=="Proposed")
            delta_st = st_prop - st_base
            
            logger.info(f"Skenario: Model Tokopedia diuji ke Shopee (Zero-Shot)")
            logger.info(f"  F1 Baseline : {ts_base:.4f}")
            logger.info(f"  F1 Proposed : {ts_prop:.4f}")
            logger.info(f"  Peningkatan : {delta_ts:+.4f} F1-Macro")
            
            logger.info(f"\nSkenario: Model Shopee diuji ke Tokopedia (Zero-Shot)")
            logger.info(f"  F1 Baseline : {st_base:.4f}")
            logger.info(f"  F1 Proposed : {st_prop:.4f}")
            logger.info(f"  Peningkatan : {delta_st:+.4f} F1-Macro")

if __name__ == "__main__":
    evaluate()
