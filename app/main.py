# ============================================================
# app/main.py
# FastAPI server — Sentiment Analysis API
# Usage: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# ============================================================

import sys
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.schemas import HealthResponse, ReviewRequest, SentimentResponse
from src.modeling.inference import load_model, predict

# ── Inisialisasi FastAPI ──────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "API untuk analisis sentimen ulasan produk e-commerce Indonesia. "
        "Fokus pada model IndoBERT dengan opsi Normalisasi Slang dan Cross-Platform."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup: load model ke memory ─────────────────────────────
@app.on_event("startup")
async def startup_event() -> None:
    logger.info("API startup — loading IndoBERT model (Default: Tokopedia, Normalized)...")
    try:
        load_model(target_platform="tokopedia", use_normalization=True)
        logger.info("Model berhasil dimuat. API siap menerima request.")
    except FileNotFoundError as e:
        logger.warning(f"Model belum tersedia: {e}")
        logger.warning("Jalankan train_indobert.py sebelum menggunakan API.")


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root() -> dict:
    return {
        "message": "Sentiment Analysis API — E-Commerce Indonesia (IndoBERT Cross-Platform)",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health() -> HealthResponse:
    """Cek status API dan model yang sedang aktif di memory."""
    from src.modeling.inference import _model, _loaded_model_name
    device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        active_model_name=str(_loaded_model_name),
        device=device,
    )


@app.post("/predict", response_model=SentimentResponse, tags=["Prediction"])
async def predict_sentiment(request: ReviewRequest) -> SentimentResponse:
    """
    Prediksi sentimen dari teks ulasan.

    - **review_text**: Teks ulasan dalam bahasa Indonesia
    - **target_platform**: "tokopedia" atau "shopee"
    - **use_normalization**: boolean (True = normalisasi slang aktif)
    - **Returns**: Kelas sentimen, confidence score, processing time, dan hasil cleaning
    """
    try:
        result = predict(
            request.review_text, 
            target_platform=request.target_platform, 
            use_normalization=request.use_normalization
        )
        return SentimentResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model belum tersedia: {e}")
    except Exception as e:
        logger.error(f"Error saat prediksi: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
