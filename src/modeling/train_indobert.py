import os
os.environ["TORCH_LOAD_RESTRICTED"] = "0"
# ============================================================
# src/modeling/train_indobert.py
# Melatih 4 versi model IndoBERT secara mandiri atau otomatis:
# 1. Tokopedia - Baseline (Tanpa Normalisasi Slang)
# 2. Tokopedia - Proposed (Dengan Normalisasi Slang)
# 3. Shopee    - Baseline
# 4. Shopee    - Proposed
# ============================================================

import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import (
    ID2LABEL,
    INDOBERT_BATCH_SIZE,
    INDOBERT_DROPOUT,
    INDOBERT_EARLY_STOP_PATIENCE,
    INDOBERT_GRADIENT_CLIP,
    INDOBERT_LEARNING_RATE,
    INDOBERT_MAX_LENGTH,
    INDOBERT_NUM_EPOCHS,
    INDOBERT_WARMUP_RATIO,
    INDOBERT_WEIGHT_DECAY,
    LABEL2ID,
    LOG_DIR,
    RANDOM_STATE,
)
from src.modeling.dataset import ReviewDataset
from src.scraping.scraper_utils import setup_logger

try:
    from sklearn.metrics import f1_score
    from sklearn.utils.class_weight import compute_class_weight
    from torch.optim import AdamW 
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["tokopedia", "shopee", "all"], default="all",
                        help="Platform data latih yang akan digunakan. Pilih 'all' untuk melatih seluruh skenario.")
    parser.add_argument("--norm", choices=["baseline", "proposed", "both"], default="both",
                        help="Kondisi normalisasi data latih. 'baseline'=no slang, 'proposed'=slang norm, 'both'=keduanya.")
    return parser.parse_args()


def load_split(csv_path: Path) -> tuple[list[str], list[int]]:
    df     = pd.read_csv(csv_path, encoding="utf-8-sig")
    texts  = df["review_text_clean"].astype(str).tolist()
    labels = df["sentiment_label"].map(LABEL2ID).tolist()
    return texts, labels


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"Menggunakan GPU: {torch.cuda.get_device_name(0)}")
    return device


def compute_class_weights(labels: list[int]) -> torch.Tensor:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=np.array(labels),
    )
    return torch.tensor(weights, dtype=torch.float)


def evaluate_epoch(model, loader, criterion, device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, f1


def train_single_scenario(platform: str, condition: str, device: torch.device):
    logger.info("━" * 60)
    logger.info(f"🚀 MEMULAI TRAINING: {platform.upper()} | KONDISI: {condition.upper()}")
    
    base_dir = ROOT / "data" / "processed" / "experiment_splits"
    train_csv = base_dir / f"train_{platform}_{condition}.csv"
    test_csv  = base_dir / f"test_{platform}_{condition}.csv"
    final_model_dir = ROOT / "models" / f"indobert_{platform}_{condition}"

    if not train_csv.exists() or not test_csv.exists():
        logger.error(f"Dataset tidak ditemukan di: {train_csv.parent}")
        logger.error("Jalankan 'run_preprocessing.py' terlebih dahulu.")
        return

    train_texts, train_labels = load_split(train_csv)
    test_texts,  test_labels  = load_split(test_csv)
    logger.info(f"Dataset | Train: {len(train_texts)} sampel | Val: {len(test_texts)} sampel")

    # Load Tokeizer & Model
    LOCAL_MODEL_DIR = str(ROOT / "models" / "indobert_raw")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
    
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, INDOBERT_MAX_LENGTH)
    test_dataset  = ReviewDataset(test_texts,  test_labels,  tokenizer, INDOBERT_MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=INDOBERT_BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=INDOBERT_BATCH_SIZE, shuffle=False, num_workers=2)

    model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_MODEL_DIR,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        hidden_dropout_prob=INDOBERT_DROPOUT,
        attention_probs_dropout_prob=INDOBERT_DROPOUT,
        ignore_mismatched_sizes=True,
        local_files_only=True
    )
    model.to(device)

    class_weights = compute_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Pisahkan parameter untuk LLRD (Layer-wise Learning Rate Decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # 1. Classifier Head (LR paling besar)
        {'params': [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': INDOBERT_WEIGHT_DECAY, 'lr': 3e-5},
        {'params': [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': 3e-5},
        
        # 2. Transformer Layers (LR menengah)
        {'params': [p for n, p in model.bert.encoder.layer.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': INDOBERT_WEIGHT_DECAY, 'lr': 1e-5},
        {'params': [p for n, p in model.bert.encoder.layer.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': 1e-5},
         
        # 3. Embeddings (LR paling kecil)
        {'params': [p for n, p in model.bert.embeddings.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': INDOBERT_WEIGHT_DECAY, 'lr': 5e-6},
        {'params': [p for n, p in model.bert.embeddings.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': 5e-6}
    ]

    optimizer = AdamW(optimizer_grouped_parameters)

    total_steps   = len(train_loader) * INDOBERT_NUM_EPOCHS
    warmup_steps  = int(INDOBERT_WARMUP_RATIO * total_steps)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_f1   = 0.0
    patience_count = 0

    for epoch in range(1, INDOBERT_NUM_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        epoch_start = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{INDOBERT_NUM_EPOCHS} [{platform[:3]}-{condition[:4]}]"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), INDOBERT_GRADIENT_CLIP)

            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, test_loader, criterion, device)
        elapsed = time.time() - epoch_start
        
        logger.info(
            f"Epoch {epoch} | T_Loss: {avg_train_loss:.4f} | "
            f"V_Loss: {val_loss:.4f} | V_F1: {val_f1:.4f} | {elapsed:.0f}s"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_count = 0
            final_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(final_model_dir))
            tokenizer.save_pretrained(str(final_model_dir / "tokenizer"))
            logger.info(f"  ⭐ Model Disimpan (F1={best_val_f1:.4f})")
        else:
            patience_count += 1
            if patience_count >= INDOBERT_EARLY_STOP_PATIENCE:
                logger.info("  ⏹️ Early stopping triggered!")
                break

    # Bebaskan memori GPU untuk sesi training berikutnya
    del model
    del optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
        
    logger.info(f"Selesai: {platform}-{condition} | Best Val F1: {best_val_f1:.4f}")
    logger.info("━" * 60)


def main():
    args = parse_args()
    setup_logger(LOG_DIR, "train_indobert_loso")
    logger.info("=" * 60)
    logger.info("PIPELINE TRAINING INDOBERT (CROSS-PLATFORM EVALUATION)")
    logger.info("=" * 60)

    torch.manual_seed(RANDOM_STATE)
    device = get_device()

    platforms_to_run = ["tokopedia", "shopee"] if args.platform == "all" else [args.platform]
    conditions_to_run = ["baseline", "proposed"] if args.norm == "both" else [args.norm]

    for plat in platforms_to_run:
        for cond in conditions_to_run:
            train_single_scenario(plat, cond, device)
            
    logger.info("\nLangkah selanjutnya: Evaluasi model silang dengan 'python src/modeling/evaluate.py'")

if __name__ == "__main__":
    main()
