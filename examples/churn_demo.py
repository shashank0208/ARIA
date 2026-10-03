"""
churn_demo.py — End-to-end ARIA demo using a synthetic customer churn dataset.

Shows:
  - 3-line integration
  - Drift detection with LLM root cause diagnosis
  - Works with llm=None fallback (no Ollama required)

Run:
    cd aria-monitor
    pip install -e ".[xgboost,shap]"
    python examples/churn_demo.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ── 1. Create synthetic churn dataset ─────────────────────────────────────────
np.random.seed(42)
n = 2000

X = pd.DataFrame({
    "tenure":          np.random.exponential(24, n),
    "monthly_charges": np.random.normal(65, 20, n),
    "num_products":    np.random.randint(1, 5, n),
    "support_calls":   np.random.poisson(2, n),
    "age":             np.random.normal(42, 12, n),
    "contract_length": np.random.choice([1, 12, 24], n),
})
y = (
    (X["tenure"] < 12) &
    (X["monthly_charges"] > 70) &
    (X["support_calls"] > 2)
).astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ── 2. Train model ────────────────────────────────────────────────────────────
clf = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric="logloss")
clf.fit(X_train, y_train)

# ── 3. Simulate production drift — new customers have higher charges ──────────
X_production = X_test.copy()
X_production["monthly_charges"] += np.random.normal(25, 5, len(X_test))   # shifted up
X_production["tenure"]          *= 0.6                                      # shorter tenures

print("=" * 60)
print("  ARIA — Autonomous ML Monitoring Demo")
print("  Use case: Customer Churn Prediction")
print("=" * 60)
print(f"  Training data:    {len(X_train)} samples")
print(f"  Production data:  {len(X_production)} samples")
print(f"  Drift simulated:  monthly_charges +25, tenure -40%")
print()

# ── THE 3-LINE INTEGRATION ────────────────────────────────────────────────────
from aria import ARIA

aria = ARIA(model=clf, feature_names=X_train.columns.tolist(), config="aria.yaml")
result = aria.monitor(X_train, X_production, y_true=y_test)

# ── Output ────────────────────────────────────────────────────────────────────
print(result.summary)
print()

# Detailed agent results
for agent_name, agent_result in result.agent_results.items():
    print(f"\n{'─'*50}")
    print(f"  {agent_name} [{agent_result.status.upper()}]")
    print(f"{'─'*50}")

    if agent_name == "drift" and agent_result.metrics.get("psi_scores"):
        psi = agent_result.metrics["psi_scores"]
        print("  PSI Scores (top 5):")
        for feat, score in sorted(psi.items(), key=lambda x: x[1], reverse=True)[:5]:
            bar = "█" * min(int(score * 50), 30)
            status = agent_result.metrics["feature_statuses"].get(feat, "ok")
            print(f"    {feat:<20} {score:.4f}  {bar} [{status}]")

    if agent_result.explanation:
        print(f"\n  Diagnosis:\n  {agent_result.explanation}")

    if agent_result.recommendations:
        print(f"\n  Recommendations:")
        for rec in agent_result.recommendations:
            print(f"    • {rec}")

print(f"\n{'='*60}")
print(f"  Overall Status: {result.status.upper()}")
print(f"{'='*60}\n")
