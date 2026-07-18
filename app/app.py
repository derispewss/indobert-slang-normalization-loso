# ============================================================
# streamlit_app.py
# Enterprise Dashboard UI — Slang Normalization Impact
# Usage: streamlit run streamlit_app.py
# ============================================================

import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="IndoBERT Generalization Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Pengecekan Ketersediaan Model ────────────────────────────
from huggingface_hub import list_repo_files

def check_model_exists(platform: str, condition: str) -> bool:
    """
    Cek apakah model ada di folder lokal 'models/' 
    ATAU ada di repositori Hugging Face 'derispewsss/...'.
    """
    model_name = f"indobert_{platform}_{condition}"
    
    # 1. Cek lokal
    local_path = ROOT / "models" / model_name
    if local_path.exists():
        return True
        
    # 2. Cek Hugging Face (untuk Streamlit Cloud)
    hf_repo_id = f"derispewsss/{model_name}"
    try:
        # Jika berhasil menarik daftar file, berarti repo eksis
        files = list_repo_files(hf_repo_id)
        return len(files) > 0
    except:
        return False

# ── Helper Load Model (cached) ────────────────────────────────
@st.cache_resource
def load_model_cached(target_platform: str, use_normalization: bool) -> None:
    from src.modeling.inference import load_model
    load_model(target_platform, use_normalization)

# ── SIDEBAR: Konfigurasi Sistem ───────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🤖 IndoBERT Generalization</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Cross-Platform Evaluation Engine</p>", unsafe_allow_html=True)
    st.divider()

    st.subheader("⚙️ Konfigurasi Eksperimen")

    # 1. Pilih Platform Model Dilatih (Domain Asal)
    st.markdown("**1. Situs Asal Pelatihan Model (Source Domain)**")
    platform_options = {"Tokopedia": "tokopedia", "Shopee": "shopee"}
    selected_platform_label = st.radio(
        "Pilih korpus di mana model dilatih:",
        options=list(platform_options.keys()),
        index=0,
        label_visibility="collapsed"
    )
    active_platform = platform_options[selected_platform_label]

    st.divider()

    # 2. Toggle Normalisasi Slang (Baseline vs Proposed)
    st.markdown("**2. Kondisi Linguistik (Treatment)**")
    norm_options = {
        "❌ Baseline (Tanpa Normalisasi Slang)": False,
        "✅ Proposed (Dengan Normalisasi Slang)": True
    }
    selected_norm_label = st.radio(
        "Terapkan perbaikan kata gaul / typo?",
        options=list(norm_options.keys()),
        index=1,
    )
    use_norm = norm_options[selected_norm_label]
    condition_str = "proposed" if use_norm else "baseline"

    st.info("💡 **Tips Pengujian (LOSO):** Cobalah melatih model di **Tokopedia**, lalu ujikan kalimat slang dari **Shopee** untuk melihat dampak *Domain Shift*. Lalu, nyalakan *Proposed* untuk melihat bagaimana normalisasi menyelamatkan akurasi prediksi.")

    st.divider()

    # Cek ketersediaan file bobot
    st.markdown("**Status Ketersediaan File Bobot:**")
    model_exists = check_model_exists(active_platform, condition_str)

    if model_exists:
        st.success(f"✅ Bobot `indobert_{active_platform}_{condition_str}` siap digunakan.")
    else:
        st.error(f"❌ Bobot `indobert_{active_platform}_{condition_str}` belum di-training!")

    st.divider()
    st.caption("© 2026 | Pembelajaran Mesin — NLP")


# ── MAIN CONTENT AREA ─────────────────────────────────────────

st.markdown("<h1 style='color: #2c3e50;'>📊 Evaluasi Efek Normalisasi Slang terhadap Generalisasi Model</h1>", unsafe_allow_html=True)
st.markdown(
    "Dasbor ini mendemonstrasikan fenomena *Domain Shift* secara *real-time*. "
    "Ketikkan kalimat dengan bahasa gaul/typo yang berat, lalu amati perbedaan prediksi saat teks tersebut diproses mentah-mentah (*Baseline*) "
    "versus saat teks tersebut distandardisasi (*Proposed*) sebelum dimasukkan ke dalam **IndoBERT**."
)
st.divider()

if not model_exists:
    st.warning(
        f"⚠️ **Perhatian**: Model **{active_platform.title()}** ({condition_str.title()}) "
        f"tidak ditemukan di folder lokal `models/` dan gagal ditarik dari Hugging Face Hub.  \n\n"
        f"**Solusi Lokal**: Silakan jalankan skrip training:\n"
        f"`python src/modeling/train_indobert.py --platform {active_platform} --norm {condition_str}`"
    )
    st.stop()

# ── PRESETS & INPUT AREA ──────────────────────────────────────

PRESETS = {
    "-- Ketik ulasan e-commerce (dengan kata slang/typo) di bawah --": "",
    "Slang Campuran (Tantangan):": "Wah gila sihhh sebagus itu, brg dtng cpt bgt, se worth it itu bahannya. Kirain bakal nerawang ternyata engga sm skali. mantull dehh!!",
    "Typo Berat & Emosi (Tantangan):": "gila x ya ini tko nipu pelnggn. psn wrna mrh dtg htm. kcewa bgt sumpah, jgn bli dsni rugi bndr!!",
    "Netral / Pertanyaan (Ambigu):": "brg udh nympe, packing rapi, tp blm sy cb smoga aja awet krn hrganya lumyn pnjng umurnya.",
    "Formal (Kontrol - Harusnya sama-sama akurat):": "Barang sudah saya terima dalam kondisi yang sangat baik. Kualitas produk sesuai dengan deskripsi yang ada pada etalase. Terima kasih."
}

col_input1, col_input2 = st.columns([2, 1])

with col_input1:
    preset_choice = st.selectbox("💡 **Pilih sampel ulasan eksperimen:**", list(PRESETS.keys()))

preset_text = PRESETS[preset_choice]

review_input = st.text_area(
    "📝 **Teks Ulasan Mentah (Raw Input):**",
    value=preset_text,
    height=140,
    placeholder="Ketik ulasan dengan bahasa gaul (cth: 'mantull', 'bgt', 'cepet') untuk menguji efektivitas sistem...",
    max_chars=2000,
)

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    analyze_btn = st.button("🚀 Prediksi Sentimen IndoBERT", type="primary", use_container_width=True)
with col_btn2:
    mode_display = f"Trained on {active_platform.title()} | Norm: {'ON' if use_norm else 'OFF'}"
    st.caption(f"Jumlah Karakter: **{len(review_input)}** / 2000 | Konfigurasi Model: **[{mode_display}]**")


# ── EXECUTION & RESULT DISPLAY ────────────────────────────────

if analyze_btn:
    if not review_input.strip():
        st.warning("⚠️ Masukkan teks ulasan terlebih dahulu.")
    elif len(review_input.strip()) < 3:
        st.warning("⚠️ Teks terlalu pendek. Silakan masukkan minimal 3 karakter.")
    else:
        with st.spinner(f"⚡ Memproses melalui IndoBERT ({active_platform.title()} - {condition_str})..."):
            try:
                load_model_cached(active_platform, use_norm)
                from src.modeling.inference import predict
                
                # Simulasi sedikit delay agar efek spinner terlihat
                time.sleep(0.3)
                
                result = predict(review_input, target_platform=active_platform, use_normalization=use_norm)

                sentiment  = result["sentiment"]
                confidence = result["confidence"]
                probs      = result["probabilities"]
                ms         = result["processing_time_ms"]
                cleaned    = result["cleaned_text"]
                act_model  = result["active_model"]

                st.divider()

                # TATA LETAK HASIL (2 Kolom)
                res_col1, res_col2 = st.columns([1, 2])

                # Kolom 1: Box Sentimen
                with res_col1:
                    st.subheader("🎯 Keputusan IndoBERT")
                    color_map = {"Positif": "#2ecc71", "Negatif": "#e74c3c", "Netral": "#f39c12"}
                    border_map = {"Positif": "#27ae60", "Negatif": "#c0392b", "Netral": "#d35400"}
                    icon_map = {"Positif": "🌟", "Negatif": "⚠️", "Netral": "⚖️"}

                    bg_color = color_map.get(sentiment, "#3498db")
                    border_color = border_map.get(sentiment, "#2980b9")
                    icon = icon_map.get(sentiment, "📌")

                    # HTML Card
                    card_html = f"""
                    <div style="background-color: {bg_color}; border: 3px solid {border_color}; 
                                padding: 25px; border-radius: 12px; text-align: center; color: white; 
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <span style="font-size: 3rem;">{icon}</span>
                        <h1 style="margin: 10px 0; font-size: 2.5rem; color: white;">{sentiment.upper()}</h1>
                        <p style="font-size: 1.2rem; margin: 0;">Skor Keyakinan: <b>{confidence*100:.1f}%</b></p>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

                # Kolom 2: Grafik & Analisis Teks
                with res_col2:
                    st.subheader("📊 Analisis Distribusi Otak Transformer")

                    # Bar chart Probabilitas
                    prob_df = pd.DataFrame({
                        "Kelas Sentimen": list(probs.keys()),
                        "Probabilitas (%)": [v * 100 for v in probs.values()],
                    }).set_index("Kelas Sentimen")
                    st.bar_chart(prob_df, height=150)
                    
                    st.caption(f"⏱️ Waktu Inferensi: **{ms:.2f} ms**")

                st.divider()

                # KOMPARASI TEKS (SANGAT PENTING UNTUK DEMO PDF)
                st.subheader("🔬 Hasil Translasi & Transformasi Teks (Preprocessing Insight)")
                if use_norm:
                    st.success("✅ **Normalisasi Slang: AKTIF**. Kata gaul dan typo diterjemahkan ke bahasa Indonesia baku, membantu IndoBERT memahami makna sebenarnya.")
                else:
                    st.error("❌ **Normalisasi Slang: NON-AKTIF**. Model dipaksa mencerna singkatan dan kata slang secara mentah, berpotensi menurunkan keyakinan prediksi (OOV).")
                
                compare_col1, compare_col2 = st.columns(2)
                with compare_col1:
                    st.markdown("**1. Teks Mentah (Raw User Input):**")
                    st.info(review_input)
                with compare_col2:
                    st.markdown("**2. Teks Yang Disuapkan ke Model (Cleaned Input):**")
                    if use_norm and cleaned.strip() != review_input.strip().lower():
                        st.success(cleaned)
                    else:
                        st.info(cleaned)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses inferensi: {e}")

# ── FOOTER ────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:gray; font-size:13px;'>"
    "<b>Dasbor Evaluasi Pengaruh Normalisasi Kata Slang pada IndoBERT</b> | Dibangun menggunakan FastAPI & Streamlit  <br>"
    "Deris Firmansyah | NIM: A11.2024.15624 | Pembelajaran Mesin — Natural Language Processing"
    "</p>",
    unsafe_allow_html=True,
)
