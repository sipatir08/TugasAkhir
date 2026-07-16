import streamlit as st

st.set_page_config(
    page_title="HR Performance Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4FC3F7;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #B0B8C1;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="main-title">👥 HR Performance Prediction System</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-title">Sistem Prediksi Kinerja Karyawan Berbasis Machine Learning & Explainable AI</p>',
    unsafe_allow_html=True
)

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📌 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dirancang untuk membantu **tim HRD** dalam menilai dan memprediksi
    kinerja karyawan secara lebih **objektif, transparan, dan berbasis data**.

    Selama ini penilaian kinerja karyawan sering dilakukan secara manual dan subjektif,
    yang berpotensi menimbulkan ketidakadilan. Aplikasi ini hadir sebagai solusi dengan
    memanfaatkan teknologi **Machine Learning** dan **Explainable AI (SHAP)** untuk:

    - ✅ Memprediksi performance rating karyawan secara akurat
    - ✅ Menjelaskan **alasan di balik setiap prediksi** (bukan black box!)
    - ✅ Memberikan rekomendasi tindakan HR yang tepat
    - ✅ Mendukung keputusan promosi, pelatihan, dan kenaikan gaji
    - ✅ Fleksibel dilatih ulang menggunakan dataset organisasi masing-masing
    """)

with col2:
    st.markdown("### 🏷️ Kategori Kinerja")
    st.markdown("""
    | Kategori | Keterangan |
    |----------|-----------|
    | ⭐ Outstanding | Jauh melampaui ekspektasi |
    | 🟢 Good | Sesuai ekspektasi |
    | 🔴 Low | Perlu perhatian |
    """)

st.divider()

st.markdown("### 🚀 Fitur Utama")

METRIC_CARD_STYLE = (
    "background-color:#16213e; border-radius:10px; padding:1rem; "
    "text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.4); color:#E0E0E0;"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div style="{METRIC_CARD_STYLE}">
        <h2 style="color:#E0E0E0; margin:0.3rem 0;">📊</h2>
        <h4 style="color:#E0E0E0; margin:0.3rem 0;">EDA</h4>
        <p style="color:#E0E0E0; margin:0.3rem 0;">Analisis dan visualisasi data karyawan</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="{METRIC_CARD_STYLE}">
        <h2 style="color:#E0E0E0; margin:0.3rem 0;">🤖</h2>
        <h4 style="color:#E0E0E0; margin:0.3rem 0;">Modeling</h4>
        <p style="color:#E0E0E0; margin:0.3rem 0;">Training & evaluasi model ML</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="{METRIC_CARD_STYLE}">
        <h2 style="color:#E0E0E0; margin:0.3rem 0;">🔮</h2>
        <h4 style="color:#E0E0E0; margin:0.3rem 0;">Prediksi</h4>
        <p style="color:#E0E0E0; margin:0.3rem 0;">Prediksi + penjelasan SHAP</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div style="{METRIC_CARD_STYLE}">
        <h2 style="color:#E0E0E0; margin:0.3rem 0;">📈</h2>
        <h4 style="color:#E0E0E0; margin:0.3rem 0;">Dashboard</h4>
        <p style="color:#E0E0E0; margin:0.3rem 0;">Overview kinerja karyawan</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.markdown("### 📖 Cara Penggunaan")

CARD_STYLE = (
    "background-color:#1a1a2e; border-radius:12px; padding:1.5rem; "
    "border-left:4px solid #4FC3F7; margin-bottom:1rem; color:#E0E0E0;"
)

st.markdown(f"""
<div style="{CARD_STYLE}">
    <b style="color:#4FC3F7;">Langkah 1 — EDA (Exploratory Data Analysis)</b><br>
    Lihat gambaran umum data karyawan, distribusi kategori kinerja,
    dan korelasi antar variabel sebelum melakukan prediksi.
</div>
<div style="{CARD_STYLE}">
    <b style="color:#4FC3F7;">Langkah 2 — Modeling</b><br>
    Lakukan training model Machine Learning (Random Forest, XGBoost, LightGBM)
    dan lihat perbandingan performanya. Model terbaik dipilih otomatis.
</div>
<div style="{CARD_STYLE}">
    <b style="color:#4FC3F7;">Langkah 3 — Prediksi Karyawan</b><br>
    Input data karyawan yang ingin dinilai, lalu dapatkan prediksi kategori kinerja
    beserta penjelasan faktor-faktor yang mempengaruhinya (SHAP).
</div>
<div style="{CARD_STYLE}">
    <b style="color:#4FC3F7;">Langkah 4 — Dashboard</b><br>
    Lihat overview kinerja seluruh karyawan, distribusi kategori,
    dan karyawan dengan performa terbaik/terburuk.
</div>
""", unsafe_allow_html=True)

st.divider()

st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem; padding: 1rem;">
    Muhammad Fathir | 714220021 | D4 Teknik Informatika | ULBI | 2026<br>
    <i>Tugas Akhir: Implementasi Explainable AI (SHAP) untuk Prediksi Kinerja Karyawan</i>
</div>
""", unsafe_allow_html=True)