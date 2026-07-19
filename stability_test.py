import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from utils.preprocessing import load_data, convert_performance_to_category, preprocess_data

# ============================================
# 1. LOAD & PREPROCESS DATA (sama kayak biasa)
# ============================================
df = load_data("data/INX_Future_Inc_Employee_Performance.csv")
df = convert_performance_to_category(df)
X, y, label_encoders = preprocess_data(df)

print(f"Total data: {X.shape[0]} baris, {X.shape[1]} fitur")
print(f"Distribusi kelas: {y.value_counts().to_dict()}")
print()

# ============================================
# 2. SETUP PIPELINE DENGAN HYPERPARAMETER TERBAIK
# (hasil GridSearchCV sebelumnya: n_estimators=200, max_depth=10, min_samples_split=2)
# ============================================
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        random_state=42
    ))
])

# ============================================
# 3. REPEATED STRATIFIED K-FOLD (5-fold, 10 repeat = 50 kali pengujian)
# ============================================
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

accuracies, precisions, recalls, f1s = [], [], [], []

print("Menjalankan 50 kali pengujian (5-fold x 10 repeat)...")
for fold_num, (train_idx, test_idx) in enumerate(rskf.split(X, y), start=1):
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

    pipeline.fit(X_train_fold, y_train_fold)
    y_pred_fold = pipeline.predict(X_test_fold)

    accuracies.append(accuracy_score(y_test_fold, y_pred_fold))
    precisions.append(precision_score(y_test_fold, y_pred_fold, average='macro', zero_division=0))
    recalls.append(recall_score(y_test_fold, y_pred_fold, average='macro', zero_division=0))
    f1s.append(f1_score(y_test_fold, y_pred_fold, average='macro', zero_division=0))

    if fold_num % 10 == 0:
        print(f"  ...selesai {fold_num}/50")

# ============================================
# 4. RINGKASAN HASIL (mean ± std)
# ============================================
print()
print("=" * 50)
print("HASIL UJI STABILITAS (50 kali pengujian)")
print("=" * 50)
print(f"Accuracy   : {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Precision  : {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall     : {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"F1-Score   : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print()
print(f"Rentang Accuracy : {np.min(accuracies):.4f} - {np.max(accuracies):.4f}")
print(f"Rentang F1-Score : {np.min(f1s):.4f} - {np.max(f1s):.4f}")