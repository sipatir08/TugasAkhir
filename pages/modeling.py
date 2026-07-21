import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import sys
import os
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import (
    load_data, guess_target_column, guess_exclude_columns, adaptive_preprocess, split_data
)
from utils.modeling import (
    train_random_forest, train_xgboost, train_lightgbm,
    evaluate_model, save_model, compare_models, get_best_model,
    calculate_roc_auc, get_roc_curve_data
)

st.set_page_config(page_title="Modeling - HR Performance", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4FC3F7;
        margin-bottom: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🤖 Modeling</p>', unsafe_allow_html=True)
st.markdown("Training dan evaluasi model Machine Learning untuk prediksi kinerja karyawan. "
            "Mendukung dataset apapun — cukup tentukan kolom target dan kolom yang dikecualikan.")
st.divider()

# ============================================
# PILIH SUMBER DATA
# ============================================
st.markdown("### 📂 Sumber Data")

data_option_list = ["Gunakan dataset default (INX Future Inc)", "Upload dataset baru"]

if 'modeling_data_option_saved' not in st.session_state:
    st.session_state['modeling_data_option_saved'] = data_option_list[0]

data_option = st.radio(
    "Pilih sumber dataset:",
    data_option_list,
    index=data_option_list.index(st.session_state['modeling_data_option_saved']),
    horizontal=True
)
st.session_state['modeling_data_option_saved'] = data_option

df = None

if data_option == "Gunakan dataset default (INX Future Inc)":
    DATA_PATH = "data/INX_Future_Inc_Employee_Performance.csv"
    if os.path.exists(DATA_PATH):
        df = load_data(DATA_PATH)
        st.session_state['modeling_uploaded_df'] = None  # reset cache upload kalau balik ke default
        st.success(f"✅ Dataset default dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    else:
        st.error(f"⚠️ File tidak ditemukan di `{DATA_PATH}`")
else:
    uploaded_file = st.file_uploader("Upload file CSV apapun (struktur bebas)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['modeling_uploaded_df'] = df
        st.session_state['modeling_uploaded_filename'] = uploaded_file.name
        st.success(f"✅ Dataset baru dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    elif st.session_state.get('modeling_uploaded_df') is not None:
        df = st.session_state['modeling_uploaded_df']
        st.info(f"📌 Menggunakan dataset yang sebelumnya diupload: "
                f"**{st.session_state.get('modeling_uploaded_filename', 'dataset')}** "
                f"({df.shape[0]} baris, {df.shape[1]} kolom). "
                f"Upload file baru di atas untuk menggantinya.")

if df is None:
    st.info("👆 Pilih atau upload dataset terlebih dahulu untuk melanjutkan.")
    st.stop()

with st.expander("👁️ Lihat Preview Data"):
    st.dataframe(df.head(10), use_container_width=True)

st.divider()

# ============================================
# PEMETAAN KOLOM (COLUMN MAPPING)
# ============================================
st.markdown("### 🗂️ Konfigurasi Dataset")
st.markdown("Tentukan kolom mana yang menjadi **target (kinerja)** dan kolom mana yang **dikecualikan** dari proses training.")

all_columns = list(df.columns)
guessed_target = guess_target_column(df)
guessed_exclude = guess_exclude_columns(df)

col1, col2 = st.columns(2)

with col1:
    target_col = st.selectbox(
        "Kolom Target (Kinerja)",
        options=all_columns,
        index=all_columns.index(guessed_target) if guessed_target in all_columns else 0,
        help="Kolom yang berisi nilai/rating kinerja karyawan yang ingin diprediksi."
    )

with col2:
    exclude_cols = st.multiselect(
        "Kolom yang Dikecualikan (ID, tanggal, dll)",
        options=[c for c in all_columns if c != target_col],
        default=[c for c in guessed_exclude if c != target_col],
        help="Kolom yang tidak relevan sebagai fitur prediksi, misalnya ID karyawan atau tanggal masuk kerja."
    )

custom_labels = None
target_is_numeric = pd.api.types.is_numeric_dtype(df[target_col])

if target_is_numeric:
    st.markdown("#### 🏷️ Beri Label Kategori untuk Nilai Target")

    unique_values = sorted(df[target_col].dropna().unique().tolist())
    n_unique = len(unique_values)

    if n_unique > 10:
        st.warning(f"⚠️ Target memiliki {n_unique} nilai unik. "
                    "Pertimbangkan memilih kolom target lain, atau lanjutkan jika ini memang sesuai kebutuhan.")

    # Template label umum, otomatis disarankan sesuai jumlah kategori unik
    TEMPLATE_PRESETS = {
        2: ["Low", "High"],
        3: ["Low", "Good", "Outstanding"],
        4: ["Poor", "Average", "Good", "Excellent"],
        5: ["Poor", "Below Average", "Average", "Good", "Excellent"],
    }

    suggested_labels = TEMPLATE_PRESETS.get(n_unique)

    if suggested_labels:
        st.success(f"✅ Terdeteksi {n_unique} kategori — label otomatis disarankan di bawah ini. "
                    "Silakan edit jika ingin menggunakan istilah lain.")
    else:
        st.info(f"ℹ️ Terdeteksi {n_unique} kategori (tidak ada template otomatis untuk jumlah ini). "
                "Silakan isi label secara manual.")

    custom_labels = {}
    label_cols = st.columns(min(n_unique, 5))
    for i, val in enumerate(unique_values):
        with label_cols[i % len(label_cols)]:
            default_val = suggested_labels[i] if suggested_labels else ""
            label = st.text_input(
                f"Label untuk nilai {val}",
                value=default_val,
                placeholder="contoh: Low/Good/dst",
                key=f"label_{val}"
            )
            custom_labels[val] = label.strip() if label.strip() else str(val)

    labels_still_numeric = [
        val for val, label in custom_labels.items()
        if label == str(val)
    ]
    if labels_still_numeric:
        st.warning(f"⚠️ Label untuk nilai {labels_still_numeric} belum diisi — akan tetap ditampilkan sebagai angka "
                    "jika Anda lanjut training.")
else:
    st.info(f"✅ Kolom target berupa teks/kategori, akan digunakan apa adanya: "
            f"{sorted(df[target_col].dropna().unique().tolist())}")

st.divider()

# ============================================
# PREPROCESSING & CEK DISTRIBUSI
# ============================================
try:
    X, y, label_encoders = adaptive_preprocess(df, target_col, exclude_cols, custom_labels)
except Exception as e:
    st.error(f"⚠️ Terjadi kesalahan saat memproses data: {e}")
    st.stop()

st.markdown("### Distribusi Kelas Target")

dist = y.value_counts()
dist_pct = y.value_counts(normalize=True) * 100

col1, col2 = st.columns([1, 2])
with col1:
    dist_df = pd.DataFrame({'Kategori': dist.index, 'Jumlah': dist.values, 'Persentase': dist_pct.values})
    st.dataframe(
        dist_df.style.bar(subset=['Jumlah'], color='#4FC3F7').format({'Persentase': '{:.1f}%'}),
        hide_index=True, use_container_width=True
    )

is_imbalanced = (dist_pct.max() - dist_pct.min()) > 10

with col2:
    if is_imbalanced:
        st.warning(f"⚠️ **Data terdeteksi imbalanced** (selisih proporsi kelas: {dist_pct.max()-dist_pct.min():.1f}%). "
                    "SMOTE akan diterapkan secara otomatis pada tahap training.")
    else:
        st.info(f"✅ **Data terdeteksi cukup seimbang** (selisih proporsi kelas: {dist_pct.max()-dist_pct.min():.1f}%). "
                "SMOTE tidak diperlukan.")

use_smote = is_imbalanced

st.markdown(f"**Jumlah fitur yang akan digunakan:** {X.shape[1]}")
with st.expander("👁️ Lihat Daftar Fitur"):
    st.write(list(X.columns))

st.divider()

# ============================================
# TOMBOL MULAI TRAINING
# ============================================
st.markdown("### 🚀 Training Model")
st.markdown("Klik tombol di bawah untuk melatih 3 model sekaligus (Random Forest, XGBoost, LightGBM) dengan hyperparameter tuning otomatis.")

if st.button("▶️ Mulai Training", type="primary", use_container_width=True):

    with st.spinner("Membagi data training dan testing..."):
        X_train, X_test, y_train, y_test = split_data(X, y)

        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['X_full'] = X
        st.session_state['y_full'] = y

    results_list = []
    models_dict = {}

    progress_bar = st.progress(0, text="Training Random Forest...")
    model_rf, params_rf, score_rf = train_random_forest(X_train, y_train, use_smote=use_smote)
    result_rf = evaluate_model(model_rf, X_test, y_test, model_name="Random Forest")
    results_list.append(result_rf)
    models_dict["Random Forest"] = model_rf

    progress_bar.progress(33, text="Training XGBoost...")
    model_xgb, le_target, params_xgb, score_xgb = train_xgboost(X_train, y_train, use_smote=use_smote)
    result_xgb = evaluate_model(model_xgb, X_test, y_test, model_name="XGBoost", label_encoder=le_target)
    results_list.append(result_xgb)
    models_dict["XGBoost"] = model_xgb
    st.session_state['xgb_label_encoder'] = le_target

    progress_bar.progress(66, text="Training LightGBM...")
    model_lgb, params_lgb, score_lgb = train_lightgbm(X_train, y_train, use_smote=use_smote)
    result_lgb = evaluate_model(model_lgb, X_test, y_test, model_name="LightGBM")
    results_list.append(result_lgb)
    models_dict["LightGBM"] = model_lgb

    progress_bar.progress(100, text="Training selesai!")

    st.session_state['results_list'] = results_list
    st.session_state['models_dict'] = models_dict
    st.session_state['label_encoders'] = label_encoders
    st.session_state['feature_names'] = list(X.columns)
    st.session_state['class_names'] = sorted(y.unique().tolist())
    reference_df = df[list(X.columns)].reset_index(drop=True)
    reference_df['Kategori_Kinerja'] = y.reset_index(drop=True)
    st.session_state['reference_df'] = reference_df
    st.session_state['training_done'] = True

    st.success("✅ Training selesai untuk ketiga model!")

# ============================================
# TAMPILKAN HASIL
# ============================================
if st.session_state.get('training_done', False):
    st.divider()
    st.markdown("### 📊 Hasil Evaluasi")

    results_list = st.session_state['results_list']
    models_dict = st.session_state['models_dict']
    class_names = st.session_state['class_names']

    comparison = compare_models(results_list)
    st.dataframe(
        comparison.style.bar(subset=['Accuracy', 'F1-Weighted', 'F1-Macro'], color='#4FC3F7').format({
            'Accuracy': '{:.4f}', 'F1-Weighted': '{:.4f}', 'F1-Macro': '{:.4f}'
        }),
        hide_index=True, use_container_width=True
    )

    best_name, best_model = get_best_model(results_list, models_dict)
    best_result = next(r for r in results_list if r['model_name'] == best_name)

    st.markdown(f"### 🏆 Model Terbaik: **{best_name}**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{best_result['accuracy']:.2%}")
    col2.metric("F1-Weighted", f"{best_result['f1_weighted']:.4f}")
    col3.metric("F1-Macro", f"{best_result['f1_macro']:.4f}")

    with st.expander("📄 Lihat Classification Report Lengkap"):
        for r in results_list:
            st.markdown(f"**{r['model_name']}**")
            st.text(r['report'])

    st.markdown("### 🔢 Confusion Matrix")
    tabs = st.tabs([r['model_name'] for r in results_list])
    for tab, r in zip(tabs, results_list):
        with tab:
            cm = r['confusion_matrix']
            fig = ff.create_annotated_heatmap(
                z=cm, x=class_names, y=class_names,
                colorscale='Blues', showscale=True
            )
            fig.update_layout(
                title=f"Confusion Matrix — {r['model_name']}",
                xaxis_title="Predicted", yaxis_title="Actual"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # ============================================
    # ROC-AUC (ONE-VS-REST)
    # ============================================
    st.markdown("### 📈 ROC-AUC (One-vs-Rest)")
    st.caption("Mengukur seberapa baik model membedakan tiap kelas dari kelas lainnya (skor 0,5 = acak, 1,0 = sempurna).")

    roc_tabs = st.tabs([r['model_name'] for r in results_list])
    for tab, r in zip(roc_tabs, results_list):
        with tab:
            model_for_roc = models_dict[r['model_name']]
            label_enc = st.session_state.get('xgb_label_encoder') if r['model_name'] == 'XGBoost' else None

            try:
                roc_results = calculate_roc_auc(
                    model_for_roc, st.session_state['X_test'], st.session_state['y_test'],
                    class_names, label_encoder=label_enc
                )

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Macro AUC", f"{roc_results['macro_auc']:.4f}")
                cols = [col2, col3, col4]
                for i, (cls, auc) in enumerate(roc_results['per_class'].items()):
                    cols[i % 3].metric(f"AUC ({cls})", f"{auc:.4f}")

                # ============================================
                # KURVA ROC
                # ============================================
                curve_data = get_roc_curve_data(
                    model_for_roc, st.session_state['X_test'], st.session_state['y_test'],
                    class_names, label_encoder=label_enc
                )

                fig_roc = go.Figure()
                for cls, (fpr, tpr, auc_score) in curve_data.items():
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode='lines',
                        name=f'{cls} (AUC={auc_score:.3f})'
                    ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode='lines',
                    line=dict(dash='dash', color='gray'),
                    name='Random (AUC=0.5)'
                ))
                fig_roc.update_layout(
                    title=f'Kurva ROC — {r["model_name"]}',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    width=700, height=500
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            except Exception as e:
                st.error(f"⚠️ Gagal menghitung ROC-AUC: {e}")

    st.divider()

    # ============================================
    # SHAP FEATURE IMPORTANCE (GLOBAL) - SEMUA MODEL
    # ============================================
    st.markdown("### 🔍 Analisis SHAP — Feature Importance")
    st.caption("Menampilkan kontribusi setiap fitur terhadap prediksi, dihitung untuk ketiga model.")

    if st.button("🔍 Hitung SHAP Feature Importance (Semua Model)", use_container_width=True):
        with st.spinner("Menghitung SHAP values untuk semua model, mohon tunggu..."):
            from utils.shap_utils import create_explainer, get_global_importance, get_per_class_importance

            X_test_saved = st.session_state['X_test']
            shap_global_all = {}
            shap_per_class_all = {}
            errors = {}

            for model_name, model in models_dict.items():
                try:
                    explainer = create_explainer(model)
                    final_model_for_shap = model.named_steps['clf'] if hasattr(model, 'named_steps') else model
                    shap_values = explainer.shap_values(X_test_saved)

                    if model_name == "XGBoost" and 'xgb_label_encoder' in st.session_state:
                        class_labels = st.session_state['xgb_label_encoder'].inverse_transform(final_model_for_shap.classes_)
                    else:
                        class_labels = final_model_for_shap.classes_

                    global_importance = get_global_importance(shap_values, list(X_test_saved.columns))
                    per_class_importance = get_per_class_importance(
                        shap_values, list(X_test_saved.columns), class_labels
                    )

                    shap_global_all[model_name] = global_importance
                    shap_per_class_all[model_name] = per_class_importance
                except Exception as e:
                    errors[model_name] = str(e)

            st.session_state['shap_global_all_models'] = shap_global_all
            st.session_state['shap_per_class_all_models'] = shap_per_class_all

            if errors:
                for model_name, err in errors.items():
                    st.error(f"⚠️ Gagal menghitung SHAP untuk {model_name}: {err}")
            if shap_global_all:
                st.success(f"✅ Perhitungan SHAP selesai untuk {len(shap_global_all)} model!")

    if 'shap_global_all_models' in st.session_state and st.session_state['shap_global_all_models']:
        st.markdown("#### 🏆 Ranking Feature Importance (Global) — Perbandingan 3 Model")

        shap_tabs = st.tabs(list(st.session_state['shap_global_all_models'].keys()))
        for tab, model_name in zip(shap_tabs, st.session_state['shap_global_all_models'].keys()):
            with tab:
                df_global = st.session_state['shap_global_all_models'][model_name]
                st.dataframe(
                    df_global.style.bar(
                        subset=['Rata-rata |SHAP value|'], color='#4FC3F7'
                    ).format({'Rata-rata |SHAP value|': '{:.4f}'}),
                    hide_index=True, use_container_width=True
                )

                with st.expander(f"📊 Lihat Feature Importance per Kategori — {model_name}"):
                    df_per_class = st.session_state['shap_per_class_all_models'][model_name]
                    numeric_subset = [c for c in df_per_class.columns if c != 'Fitur']
                    st.dataframe(
                        df_per_class.style.bar(
                            subset=numeric_subset, color='#FF8A65'
                        ).format({c: '{:.4f}' for c in numeric_subset}),
                        hide_index=True, use_container_width=True
                    )

    st.divider()

    # ============================================
    # UJI STABILITAS MODEL (REPEATED K-FOLD)
    # ============================================
    st.markdown("### 🔄 Uji Stabilitas Model")
    st.caption("Menguji apakah performa model terbaik konsisten jika data dibagi ulang berkali-kali "
               "dengan cara berbeda-beda (Repeated Stratified K-Fold: 5-fold, 10 pengulangan, total 50 kali pengujian).")

    if st.button("🔄 Jalankan Uji Stabilitas", use_container_width=True):
        with st.spinner("Menjalankan 50 kali pengujian, mohon tunggu..."):
            from sklearn.model_selection import RepeatedStratifiedKFold
            from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
            from sklearn.base import clone

            X_full = st.session_state['X_full']
            y_full = st.session_state['y_full']

            fresh_pipeline = clone(best_model)

            rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

            accs, precs, recs, f1s = [], [], [], []
            progress = st.progress(0, text="Memulai pengujian...")

            for i, (train_idx, test_idx) in enumerate(rskf.split(X_full, y_full), start=1):
                X_tr, X_te = X_full.iloc[train_idx], X_full.iloc[test_idx]
                y_tr, y_te = y_full.iloc[train_idx], y_full.iloc[test_idx]

                fresh_pipeline.fit(X_tr, y_tr)
                y_pred = fresh_pipeline.predict(X_te)

                accs.append(accuracy_score(y_te, y_pred))
                precs.append(precision_score(y_te, y_pred, average='macro', zero_division=0))
                recs.append(recall_score(y_te, y_pred, average='macro', zero_division=0))
                f1s.append(f1_score(y_te, y_pred, average='macro', zero_division=0))

                progress.progress(i / 50, text=f"Pengujian ke-{i}/50")

            st.session_state['stability_results'] = {
                'accuracy': accs, 'precision': precs, 'recall': recs, 'f1': f1s
            }
            st.success("✅ Uji stabilitas selesai!")

    if 'stability_results' in st.session_state:
        results = st.session_state['stability_results']

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{np.mean(results['accuracy']):.4f}", f"± {np.std(results['accuracy']):.4f}")
        col2.metric("Precision", f"{np.mean(results['precision']):.4f}", f"± {np.std(results['precision']):.4f}")
        col3.metric("Recall", f"{np.mean(results['recall']):.4f}", f"± {np.std(results['recall']):.4f}")
        col4.metric("F1-Score", f"{np.mean(results['f1']):.4f}", f"± {np.std(results['f1']):.4f}")

        stability_df = pd.DataFrame({
            'Percobaan': list(range(1, 51)) * 4,
            'Skor': results['accuracy'] + results['precision'] + results['recall'] + results['f1'],
            'Metrik': ['Accuracy'] * 50 + ['Precision'] * 50 + ['Recall'] * 50 + ['F1-Score'] * 50
        })

        fig = px.box(stability_df, x='Metrik', y='Skor', color='Metrik',
                     title='Distribusi Skor dari 50 Kali Pengujian (Repeated Stratified K-Fold)',
                     points='all')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ============================================
    # SIMPAN MODEL
    # ============================================
    st.markdown("### 💾 Simpan Model")
    st.markdown(f"Model terbaik (**{best_name}**) akan disimpan dan digunakan pada halaman Prediksi.")
    st.warning("⚠️ Menyimpan model baru akan **menggantikan** model sebelumnya di folder `model/`.")

    if st.button("💾 Simpan Model Terbaik", use_container_width=True):
        save_model(best_model, "best_model.pkl")
        save_model(st.session_state['label_encoders'], "label_encoders.pkl")
        save_model(st.session_state['feature_names'], "feature_names.pkl")
        save_model(class_names, "class_names.pkl")
        save_model(st.session_state['reference_df'], "reference_data.pkl")
        st.success(f"✅ Model **{best_name}** berhasil disimpan ke folder `model/`. Silakan lanjut ke halaman Prediksi.")