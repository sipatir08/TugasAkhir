import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import load_data, guess_target_column, guess_exclude_columns

st.set_page_config(page_title="EDA - HR Performance", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #4FC3F7; margin-bottom: 0.3rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
st.markdown("Analisis dan visualisasi data karyawan sebelum dilakukan pemodelan.")
st.divider()

# ============================================
# PILIH SUMBER DATA (tampilan sama seperti awal, key ditambahin biar gak reset)
# ============================================
st.markdown("### 📂 Sumber Data")

data_option_list = ["Gunakan dataset default (INX Future Inc)", "Upload dataset untuk dieksplorasi"]

# Ambil posisi terakhir yang dipilih user, default ke posisi 0 kalau belum pernah pilih
if 'eda_data_option_saved' not in st.session_state:
    st.session_state['eda_data_option_saved'] = data_option_list[0]

data_option = st.radio(
    "Pilih sumber dataset:",
    data_option_list,
    index=data_option_list.index(st.session_state['eda_data_option_saved']),
    horizontal=True
)

# Simpan pilihan terbaru, biar keinget pas pindah halaman
st.session_state['eda_data_option_saved'] = data_option

df = None

if data_option == "Gunakan dataset default (INX Future Inc)":
    DATA_PATH = "data/INX_Future_Inc_Employee_Performance.csv"
    if os.path.exists(DATA_PATH):
        df = load_data(DATA_PATH)
        st.session_state['eda_df'] = df
        st.session_state['eda_source'] = 'default'
        st.success(f"✅ Dataset default dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    else:
        st.error(f"⚠️ File tidak ditemukan di `{DATA_PATH}`")
else:
    uploaded_file = st.file_uploader("Upload file CSV apapun untuk dieksplorasi", type="csv", key="eda_uploader")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['eda_df'] = df
        st.session_state['eda_source'] = 'upload'
        st.session_state['eda_filename'] = uploaded_file.name
        st.success(f"✅ Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    elif st.session_state.get('eda_source') == 'upload' and 'eda_df' in st.session_state:
        # Belum upload file baru di sesi render ini, tapi ada data upload sebelumnya tersimpan
        df = st.session_state['eda_df']
        st.info(f"📌 Menggunakan dataset yang sebelumnya diupload: "
                f"**{st.session_state.get('eda_filename', 'dataset')}** "
                f"({df.shape[0]} baris, {df.shape[1]} kolom). "
                f"Upload file baru di atas untuk menggantinya.")

if df is None:
    st.info("👆 Pilih atau upload dataset terlebih dahulu untuk melanjutkan.")
    st.stop()

st.divider()

# ============================================
# TENTUKAN KOLOM TARGET
# ============================================
st.markdown("### 🎯 Kolom Target")

all_columns = list(df.columns)
guessed_target = guess_target_column(df)

target_col = st.selectbox(
    "Pilih kolom yang merepresentasikan kinerja/performa (opsional, untuk analisis distribusi kategori):",
    ["-- Tidak ada --"] + all_columns,
    index=(all_columns.index(guessed_target) + 1) if guessed_target in all_columns else 0,
    key="eda_target_col"
)

st.divider()

# ============================================
# STATISTIK DESKRIPTIF
# ============================================
st.markdown("### 📋 Statistik Deskriptif")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Baris", df.shape[0])
col2.metric("Jumlah Kolom", df.shape[1])
col3.metric("Missing Values", int(df.isnull().sum().sum()))
col4.metric("Data Duplikat", int(df.duplicated().sum()))

with st.expander("Lihat 10 Baris Data Pertama"):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("Lihat Statistik Deskriptif Lengkap (Kolom Numerik)"):
    st.dataframe(df.describe(), use_container_width=True)

st.divider()

numeric_cols = df.select_dtypes(include='number').columns.tolist()
categorical_cols = [c for c in df.columns if c not in numeric_cols]

# ============================================
# DISTRIBUSI KOLOM TARGET
# ============================================
if target_col != "-- Tidak ada --":
    st.markdown(f"### 🏷️ Distribusi Kolom Target: `{target_col}`")

    col1, col2 = st.columns([1, 2])

    with col1:
        dist = df[target_col].value_counts().reset_index()
        dist.columns = ['Nilai', 'Jumlah']
        st.dataframe(
            dist.style.bar(subset=['Jumlah'], color='#4FC3F7'),
            hide_index=True, use_container_width=True
        )

    with col2:
        dist['Nilai'] = dist['Nilai'].astype(str)
        fig = px.bar(
            dist, x='Nilai', y='Jumlah',
            title=f"Distribusi Nilai pada Kolom {target_col}"
        )
        st.plotly_chart(fig, use_container_width=True)

    dist_pct = df[target_col].value_counts(normalize=True) * 100
    if (dist_pct.max() - dist_pct.min()) > 10:
        st.info("📌 Data ini bersifat **imbalanced**. Teknik SMOTE dapat diterapkan pada tahap pemodelan "
                "untuk menyeimbangkan distribusi kelas.")

    st.divider()

    if target_col in numeric_cols and len(numeric_cols) > 1:
        st.markdown(f"### 🔗 Korelasi Fitur Numerik terhadap `{target_col}`")

        corr = df[numeric_cols].corr()[target_col].drop(target_col).sort_values(ascending=False)
        corr_df = corr.reset_index()
        corr_df.columns = ['Fitur', 'Korelasi']

        fig = px.bar(
            corr_df, x='Korelasi', y='Fitur', orientation='h',
            title=f"Korelasi Fitur Numerik ke {target_col}",
            color='Korelasi', color_continuous_scale='Blues'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.caption("⚠️ Korelasi ini hanya mencakup fitur numerik. Analisis SHAP pada halaman Prediksi "
                    "memberikan gambaran feature importance yang lebih lengkap, termasuk fitur kategorikal.")

        st.divider()

    other_cat_cols = [c for c in categorical_cols if c != target_col]
    if other_cat_cols:
        st.markdown(f"### 🏢 Distribusi `{target_col}` berdasarkan Kolom Lain")

        breakdown_col = st.selectbox("Pilih kolom untuk rincian:", other_cat_cols, key="eda_breakdown_col")

        breakdown_dist = df.groupby([breakdown_col, target_col]).size().reset_index(name='Jumlah')
        breakdown_dist[target_col] = breakdown_dist[target_col].astype(str)

        fig = px.bar(
            breakdown_dist, x=breakdown_col, y='Jumlah', color=target_col,
            title=f"Distribusi {target_col} berdasarkan {breakdown_col}",
            barmode='stack'
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("ℹ️ Pilih kolom target di atas untuk melihat analisis distribusi kategori, korelasi, dan breakdown.")

st.divider()

# ============================================
# DISTRIBUSI SEMUA FITUR NUMERIK
# ============================================
st.markdown("### 📈 Distribusi Fitur Numerik")

if numeric_cols:
    selected_numeric = st.selectbox("Pilih fitur numerik untuk dilihat distribusinya:", numeric_cols, key="eda_numeric_col")
    fig = px.histogram(df, x=selected_numeric, nbins=30, title=f"Distribusi {selected_numeric}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Tidak ada kolom numerik pada dataset ini.")