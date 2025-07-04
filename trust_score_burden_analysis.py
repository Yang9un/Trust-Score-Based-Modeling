
import pandas as pd
import numpy as np
import time
import psutil

# Load dataset
df = pd.read_csv("UNSW_NB15_training-set.csv")

# Extend dataset to 1 million rows
df_extended = pd.concat([df] * (1000000 // len(df) + 1), ignore_index=True).iloc[:1000000]

# Trust Score component mapping
def map_features_to_ts_components(df_subset):
    df_subset = df_subset.copy()
    df_subset['B'] = (df_subset['ct_state_ttl'].fillna(0) + df_subset['ct_dst_sport_ltm'].fillna(0)) / 30
    df_subset['N'] = df_subset['proto'].apply(lambda x: 1 if x in ['tcp', 'udp'] else 0.5)
    df_subset['N'] *= df_subset['service'].apply(lambda x: 1 if x in ['http', 'dns'] else 0.5)
    df_subset['D'] = (
        df_subset['dpkts'].fillna(0) / 1000 +
        df_subset['dbytes'].fillna(0) / 10000 +
        df_subset['sttl'].fillna(0) / 255
    ) / 3
    threat_score_map = {
        'Normal': 1.0, 'Reconnaissance': 0.6, 'Fuzzers': 0.3, 'DoS': 0.1,
        'Exploits': 0.2, 'Backdoor': 0.2, 'Generic': 0.4,
        'Shellcode': 0.1, 'Worms': 0.1, np.nan: 0.5
    }
    df_subset['T'] = df_subset['attack_cat'].map(threat_score_map).fillna(0.5)
    return df_subset[['B', 'N', 'D', 'T']]

# Trust Score calculation
def calculate_trust_score(row, weights):
    return weights['B'] * row['B'] + weights['N'] * row['N'] + weights['D'] * row['D'] + weights['T'] * row['T']

# Evaluation function
def evaluate_computational_burden_extended(df_extended, data_size, weights):
    df_sample = df_extended.sample(n=data_size, random_state=42)
    df_prepared = map_features_to_ts_components(df_sample)

    start_time = time.time()
    ts_scores = df_prepared.apply(lambda row: calculate_trust_score(row, weights), axis=1)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_row = total_time / data_size * 1000
    memory_used = psutil.Process().memory_info().rss / (1024 * 1024)
    tps = int(data_size / total_time)

    return {
        'Data Size': data_size,
        'Total Time (sec)': round(total_time, 2),
        'Avg Time (ms/row)': round(avg_time_per_row, 4),
        'Memory Usage (MB)': round(memory_used, 2),
        'TPS': tps
    }

# Run experiments
weights = {'B': 0.4, 'N': 0.3, 'D': 0.2, 'T': 0.1}
results = []
for size in [10000, 50000, 100000, 1000000]:
    result = evaluate_computational_burden_extended(df_extended, size, weights)
    results.append(result)
    print(result)
