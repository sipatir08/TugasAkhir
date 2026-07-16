import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.modeling import load_model

st.set_page_config(page_title="Dashboard - HR Performance", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #4FC3F7; margin-bottom: 0.3rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📈 Dashboard Kinerja Karyawan</p>', unsafe_allow_html=True)
st.markdown("Overview kinerja seluruh karyawan berdasarkan dataset yang terakhir digunakan pada halaman Modeling.")
st.divider()

# ============================================
# LOAD DATA REFERENSI (mengikuti dataset terakhir di-training)
# ============================================
reference_df = load_model("reference_data.pkl")

if reference_df is None or len(reference_df) == 0:
    st.error("⚠️ Belum ada data yang tersedia. Silakan lakukan training terlebih dahulu di halaman "
             "**Modeling**, lalu klik tombol **Simpan Model Terbaik**.")
    st.stop()

df = reference_df.copy()

TARGET_COL = 'Kategori_Kinerja'
if TARGET_COL not in df.columns:
    st.error("⚠️ Data referensi tidak memiliki kolom kategori kinerja. Silakan training ulang di halaman Modeling.")
    st.stop()

categories = sorted(df[TARGET_COL].dropna().unique().tolist())
palette = px.colors.qualitative.Set2
color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

categorical_cols = [c for c in df.columns if c != TARGET_COL and not pd.api.types.is_numeric_dtype(df[c])]
numeric_cols = [c for c in df.columns if c != TARGET_COL and pd.api.types.is_numeric_dtype(df[c])]

st.success(f"✅ Menampilkan data dengan **{len(df)} baris** dan **{len(df.columns)-1} fitur**.")

# ============================================
# FILTER
# ============================================
st.markdown("### 🔍 Filter Data")

filtered_df = df.copy()

filter_candidates = categorical_cols[:3]  # maksimal 3 kolom kategorikal jadi filter, biar gak kepanjangan
n_filter_cols = len(filter_candidates) + 1
filter_cols = st.columns(n_filter_cols)

for i, col in enumerate(filter_candidates):
    with filter_cols[i]:
        options = ["Semua"] + sorted(df[col].dropna().unique().tolist())
        selected = st.selectbox(col, options, key=f"filter_{col}")
        if selected != "Semua":
            filtered_df = filtered_df[filtered_df[col] == selected]

with filter_cols[-1]:
    cat_options = ["Semua"] + categories
    selected_cat = st.selectbox("Kategori Kinerja", cat_options)
    if selected_cat != "Semua":
        filtered_df = filtered_df[filtered_df[TARGET_COL] == selected_cat]

st.caption(f"Menampilkan **{len(filtered_df)}** dari **{len(df)}** total data (sesuai filter di atas)")

st.divider()

if len(filtered_df) == 0:
    st.warning("Tidak ada data yang sesuai dengan filter yang dipilih.")
    st.stop()

# ============================================
# KEY METRICS
# ============================================
st.markdown("### 📊 Ringkasan")

col1, col2, col3 = st.columns(3)
col1.metric("Total Data", len(filtered_df))

top_category = filtered_df[TARGET_COL].value_counts().idxmax()
top_category_pct = (filtered_df[TARGET_COL] == top_category).mean() * 100
col2.metric(f"Kategori Terbanyak", f"{top_category} ({top_category_pct:.1f}%)")

if numeric_cols:
    ref_numeric_col = numeric_cols[0]
    col3.metric(f"Rata-rata {ref_numeric_col}", f"{filtered_df[ref_numeric_col].mean():.2f}")

st.divider()

# ============================================
# DISTRIBUSI & VISUALISASI
# ============================================
st.markdown("### 🏷️ Distribusi Kategori Kinerja")

col1, col2 = st.columns(2)

with col1:
    dist = filtered_df[TARGET_COL].value_counts().reset_index()
    dist.columns = ['Kategori', 'Jumlah']
    fig = px.pie(
        dist, names='Kategori', values='Jumlah',
        color='Kategori', color_discrete_map=color_map,
        title="Proporsi Kategori Kinerja", hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    if categorical_cols:
        breakdown_col = st.selectbox("Rincian berdasarkan:", categorical_cols)
        breakdown_dist = filtered_df.groupby([breakdown_col, TARGET_COL]).size().reset_index(name='Jumlah')
        fig = px.bar(
            breakdown_dist, x=breakdown_col, y='Jumlah', color=TARGET_COL,
            color_discrete_map=color_map, barmode='stack',
            title=f"Kategori Kinerja berdasarkan {breakdown_col}"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Tidak ada kolom kategorikal lain untuk rincian tambahan.")

st.divider()

# ============================================
# TOP & BOTTOM PERFORMERS
# ============================================
st.markdown("### 🏅 Top & Bottom Data")

if numeric_cols:
    sort_col = st.selectbox("Urutkan berdasarkan:", numeric_cols)

    display_cols = categorical_cols[:3] + [TARGET_COL, sort_col]
    display_cols = list(dict.fromkeys(display_cols))  # hapus duplikat, jaga urutan

    tab1, tab2 = st.tabs(["🔝 Top 10", "🔻 Bottom 10"])

    with tab1:
        top10 = filtered_df.sort_values(sort_col, ascending=False).head(10)[display_cols]
        st.dataframe(top10, use_container_width=True, hide_index=True)

    with tab2:
        bottom10 = filtered_df.sort_values(sort_col, ascending=True).head(10)[display_cols]
        st.dataframe(bottom10, use_container_width=True, hide_index=True)
else:
    st.info("Tidak ada kolom numerik untuk pengurutan.")

st.divider()

# ============================================
# PENCARIAN DATA
# ============================================
st.markdown("### 🔎 Cari Data")

if categorical_cols:
    search_col = st.selectbox("Cari berdasarkan kolom:", categorical_cols, key="search_col")
    search_term = st.text_input(f"Kata kunci pencarian pada kolom '{search_col}'")

    if search_term:
        search_result = filtered_df[
            filtered_df[search_col].astype(str).str.contains(search_term, case=False, na=False)
        ]
        if len(search_result) > 0:
            st.dataframe(search_result, use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ditemukan data yang sesuai.")
else:
    st.info("Tidak ada kolom kategorikal untuk pencarian.")

st.divider()

# ============================================
# TABEL DATA LENGKAP & DOWNLOAD
# ============================================
st.markdown("### 📋 Seluruh Data Terfilter")
st.dataframe(filtered_df, use_container_width=True, hide_index=True)

csv_data = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Data Terfilter (CSV)",
    data=csv_data,
    file_name="hasil_filter_kinerja_karyawan.csv",
    mime="text/csv",
    use_container_width=True
)