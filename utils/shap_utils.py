import shap
import numpy as np
import pandas as pd
import re


def get_final_estimator(model):
    """
    Ambil model classifier murni dari dalam Pipeline (kalau ada).
    SHAP butuh model classifier langsung, bukan Pipeline.
    """
    if hasattr(model, 'named_steps'):
        return model.named_steps['clf']
    return model


def create_explainer(model):
    """
    Bikin SHAP TreeExplainer dari model.
    Otomatis ambil classifier murni kalau model masih berupa Pipeline.
    """
    final_model = get_final_estimator(model)
    explainer = shap.TreeExplainer(final_model)
    return explainer


def calculate_shap_values(explainer, X):
    """
    Hitung SHAP values untuk data X.
    """
    shap_values = explainer.shap_values(X)
    return shap_values


def _to_3d_array(shap_values):
    """
    Helper internal: normalisasi berbagai kemungkinan format shap_values
    (list per kelas, atau array 3D) jadi 1 format array 3D standar:
    (n_samples, n_features, n_classes)
    """
    if isinstance(shap_values, list):
        return np.stack(shap_values, axis=-1)
    elif len(np.array(shap_values).shape) == 3:
        return np.array(shap_values)
    else:
        return np.array(shap_values)[..., np.newaxis]


def get_global_importance(shap_values, feature_names):
    """
    Hitung ranking feature importance global (rata-rata seluruh kelas & sampel).
    Return: DataFrame diurutkan dari yang paling berpengaruh.
    """
    shap_array = _to_3d_array(shap_values)
    mean_abs_shap = np.abs(shap_array).mean(axis=(0, 2))

    df = pd.DataFrame({
        'Fitur': feature_names,
        'Rata-rata |SHAP value|': mean_abs_shap
    }).sort_values('Rata-rata |SHAP value|', ascending=False).reset_index(drop=True)

    return df


def get_per_class_importance(shap_values, feature_names, class_names):
    """
    Hitung feature importance terpisah untuk tiap kelas.
    Return: DataFrame dengan 1 kolom skor per kelas.
    """
    shap_array = _to_3d_array(shap_values)

    df = pd.DataFrame({'Fitur': feature_names})
    for i, cls in enumerate(class_names):
        df[f'SHAP_{cls}'] = np.abs(shap_array[:, :, i]).mean(axis=0)

    sort_cols = [c for c in df.columns if c != 'Fitur']
    df = df.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    return df


def explain_single_prediction(model, explainer, X_single, predicted_class, class_names):
    """
    Jelaskan 1 prediksi individu: kontribusi tiap fitur ke kelas yang diprediksi.

    Return: DataFrame kontribusi tiap fitur, diurutkan dari yang paling besar pengaruhnya
    """
    shap_values_single = explainer.shap_values(X_single)
    pred_class_idx = list(class_names).index(predicted_class)

    shap_array = _to_3d_array(shap_values_single)
    sv = shap_array[0, :, pred_class_idx]

    df = pd.DataFrame({
        'Fitur': X_single.columns,
        'Nilai Fitur': X_single.iloc[0].values,
        'Kontribusi SHAP': sv
    })
    df['Arah'] = df['Kontribusi SHAP'].apply(
        lambda x: '⬆️ Menaikkan' if x > 0 else '⬇️ Menurunkan'
    )
    df = df.reindex(
        df['Kontribusi SHAP'].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    return df


def get_top_features(explain_df, top_n=5):
    """
    Pisahkan top-N faktor pendukung (positif) dan penghambat (negatif).
    Return: (positive_features_df, negative_features_df)
    """
    positive = explain_df[explain_df['Kontribusi SHAP'] > 0].head(top_n)
    negative = explain_df[explain_df['Kontribusi SHAP'] < 0].head(top_n)
    return positive, negative


FEATURE_LABELS = {
    'Age': 'Usia',
    'Gender': 'Jenis Kelamin',
    'EducationBackground': 'Latar Belakang Pendidikan',
    'MaritalStatus': 'Status Pernikahan',
    'EmpDepartment': 'Departemen',
    'EmpJobRole': 'Posisi/Jabatan',
    'BusinessTravelFrequency': 'Frekuensi Perjalanan Dinas',
    'DistanceFromHome': 'Jarak Rumah ke Kantor',
    'EmpEducationLevel': 'Jenjang Pendidikan',
    'EmpEnvironmentSatisfaction': 'Kepuasan Lingkungan Kerja',
    'EmpHourlyRate': 'Tarif per Jam',
    'EmpJobInvolvement': 'Keterlibatan Kerja',
    'EmpJobLevel': 'Level Jabatan',
    'EmpJobSatisfaction': 'Kepuasan Kerja',
    'NumCompaniesWorked': 'Jumlah Perusahaan Sebelumnya',
    'OverTime': 'Status Lembur',
    'EmpLastSalaryHikePercent': 'Persentase Kenaikan Gaji Terakhir',
    'EmpRelationshipSatisfaction': 'Kepuasan Hubungan Kerja',
    'TotalWorkExperienceInYears': 'Total Pengalaman Kerja',
    'TrainingTimesLastYear': 'Frekuensi Pelatihan Tahun Lalu',
    'EmpWorkLifeBalance': 'Work-Life Balance',
    'ExperienceYearsAtThisCompany': 'Lama Kerja di Perusahaan Ini',
    'ExperienceYearsInCurrentRole': 'Lama di Posisi Saat Ini',
    'YearsSinceLastPromotion': 'Tahun Sejak Promosi Terakhir',
    'YearsWithCurrManager': 'Lama Bersama Atasan Saat Ini',
}


def humanize_feature_name(feature):
    """
    Terjemahkan nama kolom teknis jadi label yang mudah dibaca manusia.
    """
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]

    s = re.sub(r'(?<!^)(?=[A-Z])', ' ', feature)
    s = s.replace('_', ' ')
    return s.strip().title()


def generate_narrative_explanation(explain_df, prediction_label, top_n=5):
    """
    Ubah tabel kontribusi SHAP jadi kalimat naratif bahasa Indonesia,
    terpisah untuk faktor pendukung dan penghambat.

    Return: (list_kalimat_positif, list_kalimat_negatif)
    """
    positive_rows = explain_df[explain_df['Kontribusi SHAP'] > 0].head(top_n)
    negative_rows = explain_df[explain_df['Kontribusi SHAP'] < 0].head(top_n)

    positive_sentences = []
    for _, row in positive_rows.iterrows():
        label = humanize_feature_name(row['Fitur'])
        sentence = (
            f"**{label}** (nilai: {row['Nilai Fitur']}) — faktor ini **mendukung** "
            f"prediksi menuju kategori **{prediction_label}** "
            f"(kontribusi SHAP: {row['Kontribusi SHAP']:.4f})."
        )
        positive_sentences.append(sentence)

    negative_sentences = []
    for _, row in negative_rows.iterrows():
        label = humanize_feature_name(row['Fitur'])
        sentence = (
            f"**{label}** (nilai: {row['Nilai Fitur']}) — faktor ini **menghambat/menarik menjauh** "
            f"dari kategori **{prediction_label}**, meski kalah kuat dibanding faktor pendukung lainnya "
            f"(kontribusi SHAP: {row['Kontribusi SHAP']:.4f})."
        )
        negative_sentences.append(sentence)

    return positive_sentences, negative_sentences

FEATURE_RECOMMENDATIONS = {
    'EmpEnvironmentSatisfaction': "Tingkatkan kepuasan lingkungan kerja melalui perbaikan suasana kerja, fasilitas, atau budaya tim.",
    'EmpJobSatisfaction': "Lakukan sesi 1-on-1 untuk memahami sumber ketidakpuasan kerja dan cari solusi yang sesuai.",
    'EmpWorkLifeBalance': "Evaluasi beban kerja dan pertimbangkan fleksibilitas jam kerja untuk memperbaiki work-life balance.",
    'EmpRelationshipSatisfaction': "Fasilitasi team building atau mediasi untuk memperbaiki hubungan kerja dengan rekan/atasan.",
    'OverTime': "Tinjau kembali beban kerja yang menyebabkan lembur berlebihan, karena dapat memicu kelelahan.",
    'EmpLastSalaryHikePercent': "Pertimbangkan peninjauan kompensasi, terutama jika sudah lama tidak ada penyesuaian gaji.",
    'YearsSinceLastPromotion': "Evaluasi jenjang karir karyawan ini, karena sudah cukup lama sejak promosi terakhir.",
    'TrainingTimesLastYear': "Tingkatkan frekuensi pelatihan untuk mendukung pengembangan kompetensi.",
    'EmpJobInvolvement': "Libatkan karyawan lebih aktif dalam proyek atau pengambilan keputusan tim.",
    'DistanceFromHome': "Pertimbangkan opsi kerja remote/hybrid jika jarak tempat tinggal menjadi kendala.",
    'BusinessTravelFrequency': "Evaluasi frekuensi perjalanan dinas yang mungkin memengaruhi keseimbangan kerja karyawan.",
    'NumCompaniesWorked': "Perhatikan potensi ketidakstabilan karir; pastikan program onboarding dan engagement berjalan baik.",
}


def generate_personalized_recommendations(explain_df, base_recommendations, top_n=3):
    """
    Gabungkan rekomendasi umum (berdasarkan kategori) dengan rekomendasi
    spesifik berdasarkan faktor SHAP yang paling menghambat untuk karyawan ini.

    explain_df: hasil dari explain_single_prediction()
    base_recommendations: list rekomendasi generik dari get_performance_info()
    top_n: jumlah faktor penghambat teratas yang dipertimbangkan

    Return: (combined_recommendations, personalized_only)
    """
    negative_rows = explain_df[explain_df['Kontribusi SHAP'] < 0].head(top_n)

    personalized = []
    for _, row in negative_rows.iterrows():
        feat = row['Fitur']
        if feat in FEATURE_RECOMMENDATIONS:
            rec_text = FEATURE_RECOMMENDATIONS[feat]
            if rec_text not in personalized:
                personalized.append(rec_text)

    combined = personalized + [r for r in base_recommendations if r not in personalized]
    return combined, personalized