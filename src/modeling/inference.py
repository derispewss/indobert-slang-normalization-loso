# ============================================================
# src/modeling/inference.py
# Load model IndoBERT untuk 4 Skenario Evaluasi:
# - Tokopedia (Baseline vs Proposed)
# - Shopee (Baseline vs Proposed)
# ============================================================

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import ID2LABEL, INDOBERT_MAX_LENGTH, MODEL_DIR
from src.preprocessing.cleaner import clean_text_pipeline

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    logger.error("transformers tidak terinstall.")
    raise

# ── Global Cache Tracker ───────────────────────────
_loaded_platform = None  # "tokopedia" atau "shopee"
_loaded_norm     = None  # True (Proposed) atau False (Baseline)
_tokenizer       = None
_model           = None
_device          = None


def load_model(target_platform: str = "tokopedia", use_normalization: bool = True) -> None:
    """Load model IndoBERT dan tokenizer berdasarkan platform dan status normalisasi."""
    global _tokenizer, _model, _device, _loaded_platform, _loaded_norm

    # Jika model yang direquest sama dengan yang aktif di memory, skip re-load.
    if _model is not None and _loaded_platform == target_platform and _loaded_norm == use_normalization:
        logger.info(f"Model IndoBERT ({target_platform} | norm={use_normalization}) sudah aktif di memory.")
        return

    norm_str = "proposed" if use_normalization else "baseline"
    model_folder = f"indobert_{target_platform}_{norm_str}"
    
    logger.info(f"🔄 Switching Model ke: {model_folder.upper()} ...")

    model_path = MODEL_DIR / model_folder
    if not model_path.exists():
        msg = f"Model tidak ditemukan di {model_path}. Jalankan train_indobert.py untuk skenario ini."
        logger.error(msg)
        raise FileNotFoundError(msg)

    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(str(model_path / "tokenizer"))
    _model     = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    _model.to(_device)
    _model.eval()

    _loaded_platform = target_platform
    _loaded_norm = use_normalization
    logger.info(f"✅ Model {model_folder} berhasil dimuat ke {_device}")


def predict(raw_text: str, target_platform: str = "tokopedia", use_normalization: bool = True) -> dict:
    """
    Prediksi sentimen ulasan menggunakan IndoBERT.

    Return:
        {
            "sentiment":          str,
            "confidence":         float,
            "probabilities":      dict,
            "processing_time_ms": float,
            "cleaned_text":       str,
            "active_model":       str,
        }
    """
    load_model(target_platform, use_normalization)

    start = time.time()

    # Preprocessing teks sesuai setting (baseline/proposed)
    cleaned = clean_text_pipeline(raw_text, use_normalization=use_normalization)
    if not cleaned:
        cleaned = raw_text.lower().strip()  # fallback: minimal cleaning

    encoding = _tokenizer(
        cleaned,
        max_length=INDOBERT_MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids      = encoding["input_ids"].to(_device)
    attention_mask = encoding["attention_mask"].to(_device)

    with torch.no_grad():
        outputs = _model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = F.softmax(outputs.logits, dim=-1).squeeze()

    pred_id    = torch.argmax(probs).item()
    confidence = probs[pred_id].item()
    probabilities = {ID2LABEL[i]: round(probs[i].item(), 4) for i in range(len(ID2LABEL))}

    sentiment = ID2LABEL[pred_id]
    elapsed_ms = (time.time() - start) * 1000
    
    norm_str = "Proposed (Slang Norm)" if use_normalization else "Baseline (No Norm)"

    return {
        "sentiment":          sentiment,
        "confidence":         round(confidence, 4),
        "probabilities":      probabilities,
        "processing_time_ms": round(elapsed_ms, 2),
        "cleaned_text":       cleaned,
        "active_model":       f"IndoBERT Trained on {target_platform.title()} [{norm_str}]",
    }
