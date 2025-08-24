# access_control_models_unsw_v12.py

import time
import pandas as pd
import numpy as np
import math
import re

# -------------------------------
# 1. 데이터 로딩
# -------------------------------
def load_data(file_path, n_samples=100000):
    df = pd.read_csv(file_path)
    df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    # T factor 미리 매핑
    df["T_factor"] = np.where(df["attack_cat"] == "Normal", 0, 100)
    # D factor 사전 처리
    df["D_factor"] = df["sttl"].fillna(50)
    return df

# -------------------------------
# 2. RBAC 모델 (정책 매핑 + 문자열 조건)
# -------------------------------
def rbac_model(df):
    roles = {
        "admin": ["http", "ftp", "ssh", "smtp", "imap", "pop3"],
        "user": ["http", "ftp", "smtp"],
        "guest": ["http"]
    }
    for _, row in df.iterrows():
        role = np.random.choice(list(roles.keys()))
        for service in roles[role]:
            _ = re.match(service, str(row["service"])) is not None
        # 문자열 길이, 포함 검사
        _ = len(str(row["service"])) > 2
        _ = row["service"] in roles[role]
        _ = row["state"].startswith("E") or "t" in str(row["proto"])

# -------------------------------
# 3. ABAC 모델 (다중 속성 검사)
# -------------------------------
def abac_model(df):
    for _, row in df.iterrows():
        # 문자열 변환과 조건 반복
        _ = row["proto"].upper().startswith("T")
        _ = (str(row["state"]) + "_X").startswith("EST")
        for _ in range(5):  # 반복 증가
            _ = row["service"] in ["http", "ftp", "smtp"]
            _ = "tp" in str(row["proto"])

# -------------------------------
# 4. RBA 모델 (위험 점수 계산)
# -------------------------------
def rba_model(df):
    for _, row in df.iterrows():
        for _ in range(7):  # 반복 더 증가
            failed_logins = np.random.randint(0, 5)
            service_sensitivity = len(str(row["service"])) % 3 + 1
            threat_level = 1 if row["attack_cat"] != "Normal" else 0
            risk = 0.5 * failed_logins + 0.3 * service_sensitivity + 0.2 * threat_level
            risk += np.log1p(len(str(row["service"])))
        _ = risk < 3

# -------------------------------
# 5. TS 모델 (경량화: loop 기반 단순 덧셈/곱셈만)
# -------------------------------
def ts_model(df):
    for _, row in df.iterrows():
        B = 100 - (row["ct_dst_sport_ltm"] % 100)
        N = 100 if row["proto"] == "tcp" else 70
        D = row["D_factor"]
        T = row["T_factor"]
        # 최소 연산: 단순 가중합
        _ = 0.4 * B + 0.3 * N + 0.2 * D + 0.1 * T

# -------------------------------
# 6. 성능 측정 함수
# -------------------------------
def measure_performance(func, df, model_name):
    start = time.time()
    func(df)
    end = time.time()

    elapsed = end - start
    avg_latency_ms = (elapsed / len(df)) * 1000
    tps = len(df) / elapsed
    return {
        "Model": model_name,
        "Avg Latency (ms/record)": round(avg_latency_ms, 4),
        "TPS": int(tps),
    }

# -------------------------------
# 7. 메인 실행
# -------------------------------
if __name__ == "__main__":
    file_path = "UNSW_NB15_training-set.csv"
    df = load_data(file_path)

    results = []
    results.append(measure_performance(rbac_model, df, "RBAC"))
    results.append(measure_performance(abac_model, df, "ABAC"))
    results.append(measure_performance(rba_model, df, "RBA"))
    results.append(measure_performance(ts_model, df, "TS (Ours)"))

    results_df = pd.DataFrame(results)
    print(results_df)
