# src/drift_analysis.py

def compare_performance(clean_metrics, drift_metrics):

    print("\n--- PERFORMANCE DROP ---")

    metric_names = ["Accuracy", "F1", "ROC-AUC"]

    for name, clean, drift in zip(metric_names, clean_metrics, drift_metrics):
        drop = clean - drift
        print(f"{name} Drop: {drop:.4f}")