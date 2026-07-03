# ============================================================
# app/schemas.py
# Pydantic request/response models untuk FastAPI
# ============================================================

from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    review_text: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Teks ulasan produk dalam bahasa Indonesia",
        examples=["Produk bagus banget, pengiriman cepat, packing aman!"],
    )
    # Target Platform (Domain)
    target_platform: str = Field(
        "tokopedia", 
        description="Situs asal model: 'tokopedia' atau 'shopee'."
    )
    # Mode Normalisasi
    use_normalization: bool = Field(
        True,
        description="Apakah model menggunakan data yang telah dinormalisasi slang-nya? (True=Proposed, False=Baseline)"
    )


class SentimentResponse(BaseModel):
    sentiment: str = Field(..., description="Kelas sentimen: Positif / Negatif / Netral")
    confidence: float = Field(..., description="Probabilitas kelas prediksi tertinggi (0-1)")
    probabilities: dict[str, float] = Field(..., description="Probabilitas ke-3 kelas sentimen")
    processing_time_ms: float = Field(..., description="Waktu inferensi dalam milidetik")
    cleaned_text: str = Field(..., description="Teks hasil preprocessing (cleaning & normalisasi)")
    active_model: str = Field(..., description="Kombinasi arsitektur dan mode platform yang digunakan")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    active_model_name: str
    device: str
