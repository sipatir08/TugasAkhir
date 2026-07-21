import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

import xgboost as xgb
import lightgbm as lgb


def train_random_forest(X_train, y_train, use_smote=True):
    """
    Training Random Forest dengan GridSearchCV.
    use_smote=True -> pakai Pipeline(SMOTE + RF), buat data imbalanced.
    use_smote=False -> RF langsung tanpa SMOTE, buat data yang sudah balance.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }

    if use_smote:
        estimator = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf', RandomForestClassifier(random_state=42))
        ])
        param_grid = {f'clf__{k}': v for k, v in param_grid.items()}
    else:
        estimator = RandomForestClassifier(random_state=42)

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=10,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def train_xgboost(X_train, y_train, use_smote=True):
    """
    Training XGBoost dengan GridSearchCV.
    XGBoost butuh target numerik, jadi y_train di-encode dulu.
    Return: model, label_encoder_target, best_params, best_score
    """
    le_target = LabelEncoder()
    y_train_enc = le_target.fit_transform(y_train)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.05, 0.1]
    }

    if use_smote:
        estimator = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf', xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'))
        ])
        param_grid = {f'clf__{k}': v for k, v in param_grid.items()}
    else:
        estimator = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=10,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train_enc)

    return grid.best_estimator_, le_target, grid.best_params_, grid.best_score_


def train_lightgbm(X_train, y_train, use_smote=True):
    """
    Training LightGBM dengan GridSearchCV.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 50]
    }

    if use_smote:
        estimator = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf', lgb.LGBMClassifier(random_state=42, verbose=-1))
        ])
        param_grid = {f'clf__{k}': v for k, v in param_grid.items()}
    else:
        estimator = lgb.LGBMClassifier(random_state=42, verbose=-1)

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=10,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def evaluate_model(model, X_test, y_test, model_name="Model", label_encoder=None):
    """
    Evaluasi model di data test.
    Kalau label_encoder gak None (kasus XGBoost), hasil prediksi
    di-decode balik ke label asli sebelum dibandingkan.
    """
    y_pred = model.predict(X_test)

    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'report': report,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }
    return results


def save_model(model, filename, folder="model"):
    """
    Simpan model (atau objek lain seperti encoder) ke file .pkl
    """
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    joblib.dump(model, filepath)
    return filepath


def load_model(filename, folder="model"):
    """
    Load model dari file .pkl. Return None kalau file tidak ditemukan.
    """
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        return None
    return joblib.load(filepath)


def compare_models(results_list):
    """
    Bikin tabel perbandingan dari list hasil evaluate_model().
    """
    import pandas as pd
    comparison = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Accuracy': r['accuracy'],
            'F1-Weighted': r['f1_weighted'],
            'F1-Macro': r['f1_macro']
        }
        for r in results_list
    ]).sort_values('F1-Macro', ascending=False).reset_index(drop=True)
    return comparison


def get_best_model(results_list, models_dict):
    """
    Pilih model terbaik berdasarkan F1-Macro tertinggi.
    results_list: list dari dict hasil evaluate_model()
    models_dict: dict {model_name: model_object}

    Return: (best_model_name, best_model_object)
    """
    best_result = max(results_list, key=lambda r: r['f1_macro'])
    best_name = best_result['model_name']
    return best_name, models_dict[best_name]
    
def calculate_roc_auc(model, X_test, y_test, class_names, label_encoder=None):
    """
    Hitung ROC-AUC untuk klasifikasi multi-kelas pakai pendekatan one-vs-rest.
    Return: dict berisi AUC per kelas + macro-average AUC
    """
    y_proba = model.predict_proba(X_test)

    if label_encoder is not None:
        y_test_encoded = label_encoder.transform(y_test)
    else:
        y_test_encoded = np.array([list(class_names).index(v) for v in y_test])

    y_test_bin = label_binarize(y_test_encoded, classes=range(len(class_names)))

    auc_per_class = {}
    for i, cls in enumerate(class_names):
        auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        auc_per_class[cls] = auc

    macro_auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')

    return {
        'per_class': auc_per_class,
        'macro_auc': macro_auc
    }

def get_roc_curve_data(model, X_test, y_test, class_names, label_encoder=None):
    """
    Hitung data kurva ROC (FPR, TPR) untuk tiap kelas, one-vs-rest.
    Return: dict {nama_kelas: (fpr, tpr, auc)}
    """
    y_proba = model.predict_proba(X_test)

    if label_encoder is not None:
        y_test_encoded = label_encoder.transform(y_test)
    else:
        y_test_encoded = np.array([list(class_names).index(v) for v in y_test])

    y_test_bin = label_binarize(y_test_encoded, classes=range(len(class_names)))

    curve_data = {}
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        auc_score = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        curve_data[cls] = (fpr, tpr, auc_score)

    return curve_data