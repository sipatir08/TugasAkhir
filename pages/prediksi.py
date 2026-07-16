import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.modeling import load_model
from utils.preprocessing import get_performance_info
from utils.shap_utils import create_explainer, explain_single_prediction, get_top_features, generate_narrative_explanation, generate_personalized_recommendations


st.set_page_config(page_title="Prediksi Kinerja Karyawan", page_icon="🔮", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #4FC3F7; margin-bottom: 0.3rem; }
    .result-card { border-radius: 12px; padding: 1.5rem; margin: 1rem 0; text-align: center; }
    .result-card h1 { font-size: 3rem; margin: 0; }
    .result-card h2 { margin: 0.3rem 0; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🔮 Prediksi Kinerja Karyawan</p>', unsafe_allow_html=True)
st.markdown("Masukkan data karyawan untuk memprediksi kategori kinerja beserta penjelasannya.")
st.divider()

# ============================================
# LOAD MODEL & KOMPONEN
# ============================================
model = load_model("best_model.pkl")
label_encoders = load_model("label_encoders.pkl")
feature_names = load_model("feature_names.pkl")
class_names = load_model("class_names.pkl")
reference_df = load_model("reference_data.pkl")

if model is None or feature_names is None:
    st.error("⚠️ Model belum tersedia. Silakan lakukan training terlebih dahulu di halaman **Modeling**, "
             "lalu klik tombol **Simpan Model Terbaik**.")
    st.stop()

label_encoders = label_encoders or {}
class_names = class_names or []

st.success(f"✅ Model siap digunakan. Fitur: {len(feature_names)} | Kategori: {', '.join(map(str, class_names))}")

# ============================================
# ISI OTOMATIS DARI DATA YANG SUDAH ADA (OPSIONAL)
# ============================================
selected_key = "manual"

if reference_df is not None and len(reference_df) > 0:
    st.markdown("### ⚡ Isi Otomatis dari Data yang Sudah Ada (Opsional)")
    st.caption("Pilih salah satu karyawan untuk mengisi form otomatis, lalu edit field yang ingin diubah.")

    options = ["-- Isi manual dari nol --"] + [f"Baris {i}" for i in range(len(reference_df))]
    selected_option = st.selectbox("Pilih karyawan:", options)

    if selected_option != "-- Isi manual dari nol --":
        selected_idx = int(selected_option.replace("Baris ", ""))
        prefill = reference_df.iloc[selected_idx].to_dict()
        selected_key = str(selected_idx)
        with st.expander("👁️ Lihat data lengkap karyawan ini"):
            st.dataframe(reference_df.iloc[[selected_idx]], use_container_width=True)
    else:
        prefill = {}

    st.divider()
else:
    prefill = {}
    st.info("ℹ️ Belum ada data referensi tersimpan. Silakan isi form secara manual, "
            "atau lakukan training ulang di halaman Modeling untuk mengaktifkan fitur isi otomatis.")

# ============================================
# FORM INPUT DATA KARYAWAN (DINAMIS, sesuai fitur model yang tersimpan)
# ============================================
st.markdown("### 📝 Input Data Karyawan")
st.caption("Form berikut menyesuaikan otomatis dengan fitur yang digunakan model saat ini.")

with st.form("prediction_form"):
    input_dict = {}
    cols = st.columns(3)

    for i, feat in enumerate(feature_names):
        widget_key = f"input_{feat}_{selected_key}"
        with cols[i % 3]:
            if feat in label_encoders:
                options = list(label_encoders[feat].classes_)
                default_val = prefill.get(feat, options[0])
                default_index = options.index(default_val) if default_val in options else 0
                input_dict[feat] = st.selectbox(feat, options, index=default_index, key=widget_key)
            else:
                default_val = prefill.get(feat, 0)
                try:
                    default_val = float(default_val)
                except (TypeError, ValueError):
                    default_val = 0.0
                input_dict[feat] = st.number_input(feat, value=default_val, key=widget_key)

    submitted = st.form_submit_button("🔮 Prediksi Kinerja", use_container_width=True)


def decode_prediction(pred, class_names):
    """
    Beberapa model (XGBoost) mengembalikan hasil prediksi berupa angka
    (indeks kelas), bukan label teks. Fungsi ini menerjemahkan kembali
    ke label asli jika diperlukan.
    """
    if isinstance(pred, (int, np.integer)) and class_names:
        sorted_classes = sorted(class_names)
        if 0 <= pred < len(sorted_classes):
            return sorted_classes[pred]
    return pred


# ============================================
# PROSES PREDIKSI
# ============================================
if submitted:
    encoded_input = {}
    for feat, val in input_dict.items():
        if feat in label_encoders:
            encoded_input[feat] = label_encoders[feat].transform([val])[0]
        else:
            encoded_input[feat] = val

    input_df = pd.DataFrame([encoded_input])[feature_names]

    raw_prediction = model.predict(input_df)[0]
    prediction = decode_prediction(raw_prediction, class_names)

    try:
        probabilities = model.predict_proba(input_df)[0]
        proba_available = True
    except Exception:
        proba_available = False

    info = get_performance_info(str(prediction))

    st.divider()
    st.markdown("### 📊 Hasil Prediksi")

    st.markdown(f"""
    <div class="result-card" style="background-color:{info['color']}22; border: 2px solid {info['color']};">
        <h1>{info['emoji']}</h1>
        <h2 style="color:{info['color']};">{info['label']}</h2>
        <p style="color:#E0E0E0;">{info['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    if proba_available:
        st.markdown("#### Probabilitas Tiap Kategori")
        model_classes = getattr(model, 'classes_', sorted(class_names) if class_names else [])
        prob_labels = [decode_prediction(c, class_names) if isinstance(c, (int, np.integer)) else c for c in model_classes]
        prob_df = pd.DataFrame({'Kategori': prob_labels, 'Probabilitas': probabilities}).sort_values('Probabilitas', ascending=False)
        st.dataframe(
            prob_df.style.bar(subset=['Probabilitas'], color='#4FC3F7').format({'Probabilitas': '{:.2%}'}),
            use_container_width=True, hide_index=True
        )

    st.markdown("#### 💡 Rekomendasi Tindakan HR")
    st.caption("Rekomendasi akan diperbarui otomatis berdasarkan analisis SHAP di bawah setelah tersedia.")
    st.session_state['_pending_recommendations'] = info['recommendations']
    recommendation_placeholder = st.empty()
    with recommendation_placeholder.container():
        for rec in info['recommendations']:
            st.markdown(f"- {rec}")

    # ============================================
    # PENJELASAN SHAP
    # ============================================
    st.divider()
    st.markdown("### 🔍 Penjelasan Prediksi (SHAP)")

    try:
        explainer = create_explainer(model)
        final_model = model.named_steps['clf'] if hasattr(model, 'named_steps') else model
        model_classes_for_shap = final_model.classes_

        explain_df = explain_single_prediction(
            model, explainer, input_df,
            predicted_class=raw_prediction,
            class_names=model_classes_for_shap
        )

        display_df = explain_df.copy()
        for feat, le in label_encoders.items():
            mask = display_df['Fitur'] == feat
            if mask.any():
                val = display_df.loc[mask, 'Nilai Fitur'].values[0]
                try:
                    display_df.loc[mask, 'Nilai Fitur'] = le.inverse_transform([int(val)])[0]
                except Exception:
                    pass

        positive_sentences, negative_sentences = generate_narrative_explanation(
            display_df, prediction_label=str(prediction), top_n=5
        )

        combined_recs, personalized_recs = generate_personalized_recommendations(
            display_df, info['recommendations'], top_n=3
        )
        with recommendation_placeholder.container():
            for rec in combined_recs:
                if rec in personalized_recs:
                    st.markdown(f"- 🎯 **{rec}** *(berdasarkan analisis data karyawan ini)*")
                else:
                    st.markdown(f"- {rec}")

        col_pos, col_neg = st.columns(2)
        positive, negative = get_top_features(display_df, top_n=5)

        with col_pos:
            st.markdown("**✅ Faktor Pendukung**")
            st.dataframe(positive[['Fitur', 'Nilai Fitur', 'Kontribusi SHAP']], hide_index=True, use_container_width=True)
            with st.expander("📝 Lihat penjelasan"):
                for sentence in positive_sentences:
                    st.markdown(f"- {sentence}")

        with col_neg:
            st.markdown("**⚠️ Faktor Penghambat**")
            st.dataframe(negative[['Fitur', 'Nilai Fitur', 'Kontribusi SHAP']], hide_index=True, use_container_width=True)
            with st.expander("📝 Lihat penjelasan"):
                for sentence in negative_sentences:
                    st.markdown(f"- {sentence}")

        st.markdown("**Rincian Lengkap Kontribusi Semua Fitur**")
        st.dataframe(
            display_df.style.bar(subset=['Kontribusi SHAP'], color='#FF8A65', align='mid').format({'Kontribusi SHAP': '{:.4f}'}),
            use_container_width=True, hide_index=True
        )
    except Exception as e:
        st.warning(f"⚠️ Penjelasan SHAP tidak dapat ditampilkan untuk model ini: {e}")