# ============================================================
# src/preprocessing/cleaner.py
# Step 3: Full text cleaning pipeline (2 Jalur: Baseline vs Proposed)
# ============================================================

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import MIN_TEXT_LENGTH, MIN_TOKEN_COUNT, NEGATION_WORDS, INTENSIFIER_WORDS
from src.preprocessing.normalizer import (
    normalize_punctuation,
    normalize_repeated_chars,
    normalize_slang,
)

try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
        ArrayOfStopWordsRemover,
        StopWordRemoverFactory,
    )
    _SASTRAWI_AVAILABLE = True
except ImportError:
    _SASTRAWI_AVAILABLE = False

_stop_word_remover = None

def _get_stop_word_remover():
    global _stop_word_remover
    if _stop_word_remover is None:
        if not _SASTRAWI_AVAILABLE:
            return None
        factory  = StopWordRemoverFactory()
        stopwords = set(factory.get_stop_words())
        
        # Mengecualikan kata negasi dan kata penguat dari penghapusan Sastrawi
        stopwords -= NEGATION_WORDS
        stopwords -= INTENSIFIER_WORDS
        
        _stop_word_remover = ArrayOfStopWordsRemover(list(stopwords))
    return _stop_word_remover

_URL_PATTERN       = re.compile(r"https?://\S+|www\.\S+")
_MENTION_PATTERN   = re.compile(r"@\w+")
_HASHTAG_PATTERN   = re.compile(r"#\w+")
_NON_LATIN_PATTERN = re.compile(r"[^\w\s.,!?']")
_MULTI_SPACE       = re.compile(r"\s+")
_EMOJI_PATTERN     = re.compile(
    "[\U00010000-\U0010ffff"
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\u2600-\u26FF\u2700-\u27BF]+",
    flags=re.UNICODE,
)


def is_meaningful(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < MIN_TEXT_LENGTH:
        return False
    if text.isdigit():
        return False
    text_no_emoji = _EMOJI_PATTERN.sub("", text).strip()
    if len(text_no_emoji) < MIN_TEXT_LENGTH:
        return False
    return True


def clean_text(text: str, use_normalization: bool = True) -> str:
    """
    Pipeline pembersihan teks dengan parameter switch untuk normalisasi slang.
    use_normalization=False -> Baseline (Tanpa perbaikan typo/slang)
    use_normalization=True  -> Proposed (Dengan perbaikan slang via Sastrawi/Custom)
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _MENTION_PATTERN.sub(" ", text)
    text = _HASHTAG_PATTERN.sub(" ", text)
    text = _EMOJI_PATTERN.sub(" ", text)
    
    text = normalize_repeated_chars(text)
    text = normalize_punctuation(text)

    # ── JALUR EKSPERIMEN: NORMALISASI SLANG ──
    if use_normalization:
        text = normalize_slang(text)
        
    text = _NON_LATIN_PATTERN.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()

    return text


def remove_stopwords(text: str) -> str:
    remover = _get_stop_word_remover()
    if remover is None:
        return text
    return remover.remove(text)


def is_too_short_after_cleaning(text: str) -> bool:
    return len(text.split()) < MIN_TOKEN_COUNT


def clean_text_pipeline(raw_text: str, use_normalization: bool = True) -> str | None:
    """
    Jalankan full cleaning pipeline pada satu teks berdasarkan jalur normalisasi.
    """
    if not is_meaningful(raw_text):
        return None

    cleaned = clean_text(raw_text, use_normalization=use_normalization)
    cleaned = remove_stopwords(cleaned)

    if is_too_short_after_cleaning(cleaned):
        return None

    return cleaned
