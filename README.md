# Trust Score-Based Modeling – Experimental Code Repository

This repository provides the experimental code used in the paper:
"A Trust Score-Based Access Control Model for Zero Trust Architecture: Design, Sensitivity Analysis, and Real-World Performance Evaluation"

---

## Experiments
The repository contains three main types of experiments:

1. Cross-Dataset Performance Evaluation (UNSW-NB15 vs. CICIDS2017)  
   - Evaluate the robustness of the TS model across different datasets.  
   - Datasets: UNSW-NB15 (general-purpose attacks) and CICIDS2017 Wednesday subset (benign + DoS/DDoS).  
   - Metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.  

2. Computational Burden and Scalability of the TS Model  
   - Measure latency, throughput, and memory usage.  
   - Tested across dataset sizes up to 1,000,000 records to demonstrate scalability.
   
3. Comparative Benchmarking of Legacy Access Control Models vs. the TS Model
   - Models included:  
      - RBAC (Role-Based Access Control)  
      - ABAC (Attribute-Based Access Control)  
      - RBA (Risk-Based Access Control)  
    - Compared with the proposed Trust Score (TS) model.  
   - Metrics: average latency (ms/record) and throughput (TPS). 


---

## Repository Structure
- evaluate_datasets_unsw_cicids.py  
  Code for cross-dataset performance evaluation using UNSW-NB15 and CICIDS2017.  
  - Includes preprocessing, train/test split, and classification performance analysis.
   
- ts_computational_burden.py  
  Code for measuring computational burden and scalability of the TS model.  

- access_control_models_unsw_v12.py  
  Code for comparative benchmarking of RBAC, ABAC, RBA, and the TS model.  

---

## Dataset
- UNSW-NB15 dataset** (already included in this repository for reproducibility)  
  Original reference: [UNSW-NB15 – UNSW Canberra](https://research.unsw.edu.au/projects/unsw-nb15-dataset)  

- CICIDS2017 dataset** (Wednesday subset used for evaluation)  
  Original reference: [CICIDS2017 – Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)

※ The original datasets (UNSW-NB15 and CICIDS2017) are not included in this repository due to GitHub upload size limitations, but they can be obtained directly from the authors upon request.

---

## Requirements
- Python 3.10+  
- pandas  
- numpy  
- scikit-learn  

Install required packages:  
```bash
pip install pandas numpy scikit-learn
