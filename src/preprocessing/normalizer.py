# ============================================================
# src/preprocessing/normalizer.py
# Normalisasi slang per token menggunakan SLANG_DICT
# ============================================================

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocessing.slang_dict import SLANG_DICT


def normalize_slang(text: str) -> str:
    """
    Normalisasi slang/singkatan menggunakan kamus SLANG_DICT.
    Proses per token (kata per kata).
    Token yang ada di kamus dengan nilai "" akan dihapus (noise).
    Mengabaikan kata-kata domain-specific e-commerce (whitelist).
    """
    if not text or not isinstance(text, str):
        return ""

    # Whitelist kata-kata teknis e-commerce yang TIDAK BOLEH dinormalisasi
    # untuk mencegah hilangnya konteks kalimat (Context Loss)
    DOMAIN_WHITELIST = {
        "cod", "kurir", "ongkir", "ongkos", "kirim", "aplikasi", "app", 
        "playstore", "update", "akun", "seller", "toko", "admin", "lag",
        "lemot", "error", "bug", "crash"
    }

    tokens = text.split()
    normalized = []

    for token in tokens:
        token_lower = token.lower()
        
        # Jika token ada di whitelist, biarkan apa adanya
        if token_lower in DOMAIN_WHITELIST:
            normalized.append(token)
            continue
            
        # Cek kamus (case-insensitive)
        if token_lower in SLANG_DICT:
            replacement = SLANG_DICT[token_lower]
            if replacement:  # hanya tambahkan jika bukan string kosong
                normalized.append(replacement)
        else:
            normalized.append(token)

    return " ".join(normalized)


def normalize_repeated_chars(text: str) -> str:
    """
    Normalisasi karakter berulang lebih dari 2x menjadi 1x.
    Contoh: "bagussss" → "bagus", "mantaaap" → "mantap"
    Ini untuk memastikan Tokenizer IndoBERT mengenali kata tersebut (tidak OOV).
    """
    return re.sub(r"(.)\1{2,}", r"\1", text)


def normalize_punctuation(text: str) -> str:
    """
    Normalisasi tanda baca berlebih.
    "!!!" → "!", "???" → "?", "..." → "."
    """
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"-{2,}", "-", text)
    return text
