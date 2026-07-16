import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(filepath="data/INX_Future_Inc_Employee_Performance.csv"):
    """
    Load dataset mentah dari file CSV.
    """
    df = pd.read_csv(filepath)
    return df


def categorize(rating):
    """
    Konversi PerformanceRating (angka 2/3/4) jadi kategori teks.
    Threshold ini FIX, jangan diubah tanpa diskusi.
    """
    mapping = {2: "Low", 3: "Good", 4: "Outstanding"}
    return mapping.get(rating, "Good")


def convert_performance_to_category(df):
    """
    Tambahkan kolom 'performance_category' berdasarkan PerformanceRating.
    """
    df = df.copy()
    df['performance_category'] = df['PerformanceRating'].apply(categorize)
    return df


def preprocess_data(df):
    """
    Drop kolom yang tidak dipakai, encode kolom kategorikal,
    lalu pisahkan jadi X (fitur) dan y (target).
    Khusus untuk struktur dataset INX.

    Return: X, y, label_encoders (dict berisi encoder tiap kolom kategorikal)
    """
    df_model = df.drop(columns=['EmpNumber', 'PerformanceRating', 'Attrition'])

    df_model['OverTime'] = df_model['OverTime'].map({'Yes': 1, 'No': 0})

    label_encoders = {}
    cat_cols = ['Gender', 'EducationBackground', 'MaritalStatus',
                'EmpDepartment', 'EmpJobRole', 'BusinessTravelFrequency']

    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

    X = df_model.drop(columns=['performance_category'])
    y = df_model['performance_category']

    return X, y, label_encoders


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data jadi train dan test, dengan stratifikasi
    supaya proporsi tiap kelas tetap seimbang.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state=42):
    """
    Terapkan SMOTE ke data training untuk menyeimbangkan kelas minoritas.
    """
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote


def get_feature_names(X):
    """
    Return daftar nama kolom fitur, sesuai urutan yang dipakai saat training.
    """
    return list(X.columns)


def encode_user_input(input_dict, label_encoders):
    """
    Encode input dari form HRD (dictionary) menggunakan encoder
    yang SAMA seperti saat training, supaya konsisten.
    """
    encoded = input_dict.copy()
    for col, encoder in label_encoders.items():
        if col in encoded:
            encoded[col] = encoder.transform([encoded[col]])[0]
    return encoded


def get_performance_info(category):
    """
    Return info lengkap (emoji, warna, deskripsi, rekomendasi)
    untuk kategori kinerja. Mendukung 3 kategori standar (Low/Good/Outstanding),
    dengan fallback netral untuk kategori lain di luar itu.
    """
    info = {
        "Low": {
            "emoji": "🔴", "color": "#FF4B4B", "label": "Low",
            "description": "Kinerja di bawah ekspektasi, perlu perhatian dan pendampingan.",
            "recommendations": [
                "Berikan pelatihan/coaching intensif sesuai gap kompetensi",
                "Lakukan evaluasi rutin (mingguan/bulanan) dengan atasan langsung",
                "Identifikasi hambatan personal/lingkungan kerja yang mempengaruhi kinerja",
                "Pertimbangkan mentoring dari rekan kerja berkinerja baik"
            ]
        },
        "Good": {
            "emoji": "🟢", "color": "#00CC44", "label": "Good",
            "description": "Kinerja sesuai ekspektasi, konsisten dan dapat diandalkan.",
            "recommendations": [
                "Pertahankan konsistensi kinerja melalui apresiasi berkala",
                "Berikan kesempatan pengembangan skill lanjutan",
                "Libatkan dalam proyek yang lebih menantang",
                "Evaluasi kesiapan untuk promosi jangka menengah"
            ]
        },
        "Outstanding": {
            "emoji": "⭐", "color": "#1E90FF", "label": "Outstanding",
            "description": "Kinerja jauh melampaui ekspektasi, berpotensi menjadi talent kunci.",
            "recommendations": [
                "Pertimbangkan promosi jabatan atau kenaikan gaji",
                "Libatkan sebagai mentor bagi karyawan lain",
                "Berikan proyek strategis atau kepemimpinan tim",
                "Susun rencana pengembangan karir jangka panjang (talent retention)"
            ]
        }
    }

    if category in info:
        return info[category]

    # Fallback netral untuk kategori di luar 3 standar (misal dataset lain diupload)
    return {
        "emoji": "📌",
        "color": "#4FC3F7",
        "label": str(category),
        "description": f"Karyawan diprediksi masuk kategori '{category}'.",
        "recommendations": [
            "Tinjau lebih lanjut faktor-faktor yang mempengaruhi kategori ini melalui analisis SHAP di bawah.",
        ]
    }


# ============================================
# FUNGSI TAMBAHAN — PREPROCESSING ADAPTIF
# (untuk mendukung dataset apapun yang diupload user)
# ============================================

def get_column_types(df, exclude_cols=None):
    """
    Pisahkan kolom jadi numerik dan kategorikal.
    """
    exclude_cols = exclude_cols or []
    cols = [c for c in df.columns if c not in exclude_cols]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols, categorical_cols


def guess_target_column(df):
    """
    Coba tebak otomatis kolom mana yang kemungkinan target kinerja,
    berdasarkan nama kolom yang mengandung kata kunci umum.
    """
    keywords = ['performance', 'rating', 'kinerja', 'score']
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            return col
    return df.columns[0]


def guess_exclude_columns(df):
    """
    Coba tebak otomatis kolom yang sebaiknya dikecualikan
    (ID unik, atau kolom tanggal).
    """
    guesses = []
    for col in df.columns:
        if 'id' in col.lower() or 'number' in col.lower():
            guesses.append(col)
        elif df[col].nunique() == len(df):
            guesses.append(col)
        elif 'date' in col.lower() or 'hire' in col.lower():
            guesses.append(col)
    return guesses


def build_target_category(df, target_col, custom_labels=None):
    """
    Bikin kolom 'target_category' dari kolom target asli.
    """
    df = df.copy()
    if custom_labels:
        df['target_category'] = df[target_col].map(custom_labels)
    else:
        df['target_category'] = df[target_col].astype(str)
    return df


def adaptive_preprocess(df, target_col, exclude_cols, custom_labels=None):
    """
    Preprocessing fleksibel: bekerja untuk struktur dataset apapun.

    Return: X, y, label_encoders
    """
    df = build_target_category(df, target_col, custom_labels)

    drop_cols = list(set(exclude_cols + [target_col]))
    drop_cols = [c for c in drop_cols if c in df.columns]
    df_model = df.drop(columns=drop_cols)

    y = df_model['target_category']
    X = df_model.drop(columns=['target_category'])

    label_encoders = {}
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    return X, y, label_encoders