# ================================================================
# File: trustscore_unsw_cicids_eval.py
# Author: [Eunsu Jeong]
# Description:
#   Evaluate Trust Score model performance on two benchmark datasets:
#     - UNSW-NB15
#     - CICIDS2017
#   Outputs common classification metrics:
#     Accuracy, Precision, Recall, F1-Score, ROC-AUC
# ================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def load_and_preprocess(csv_path, label_col='label', normal_label_values=None):
    """
    Load dataset and preprocess features/labels (Binary Classification: Normal=0, Attack=1)
    """
    print(f"\n[INFO] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.dropna()

    # 🔹 라벨 값 확인 (디버깅용)
    print("[DEBUG] Unique values in label column:", df[label_col].unique()[:20])

    # 🔹 Binary label 변환 (문자열일 경우만 변환)
    if normal_label_values is not None:
        if df[label_col].dtype == object:  # 문자열 라벨이면 변환
            df[label_col] = df[label_col].apply(lambda x: 0 if x in normal_label_values else 1)
    else:
        if df[label_col].dtype == object:  # 숫자형 라벨은 그대로 둠
            df[label_col] = df[label_col].apply(lambda x: 0 if str(x).lower() in ['normal', 'benign'] else 1)

    # 🔹 라벨 분포 출력
    print("[INFO] Label distribution after binarization:")
    print(df[label_col].value_counts())

    # 🔹 불필요한 컬럼 제거
    drop_cols = [label_col, 'attack_cat', 'id']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[label_col]

    # 🔹 범주형 컬럼 인코딩
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le_col = LabelEncoder()
        X[col] = le_col.fit_transform(X[col].astype(str))

    # 🔹 스케일링
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def evaluate_model(X, y):
    """
    Train and evaluate RandomForest model
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # AUC 계산 (binary일 때만)
    if model.n_classes_ == 2:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = np.nan

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": auc
    }
    return results


def pretty_print_results(dataset_name, results):
    """
    Print evaluation results in a formatted table
    """
    print(f"\n=== {dataset_name} Results ===")
    print(f"{'Metric':<10} | {'Value':>8}")
    print("-" * 22)
    for k, v in results.items():
        print(f"{k:<10} | {v:8.4f}")


if __name__ == "__main__":
    # 🔹 UNSW-NB15
    unsw_csv = "UNSW_NB15_training-set.csv"   # 실제 경로로 수정
    X_unsw, y_unsw = load_and_preprocess(
        unsw_csv, label_col='label', normal_label_values=['Normal']
    )
    unsw_results = evaluate_model(X_unsw, y_unsw)
    pretty_print_results("UNSW-NB15", unsw_results)

    # 🔹 CICIDS2017
    cicids_csv = "CICIDS2017_wednesday.csv"   # 실제 경로로 수정
    X_cic, y_cic = load_and_preprocess(
        cicids_csv, label_col='Label', normal_label_values=['BENIGN']
    )
    cic_results = evaluate_model(X_cic, y_cic)
    pretty_print_results("CICIDS2017", cic_results)
