# src/evaluate.py

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate(model, X, y, label="DATA"):

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc = roc_auc_score(y, y_prob)

    print(f"\n--- {label} PERFORMANCE ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")

    return accuracy, f1, roc