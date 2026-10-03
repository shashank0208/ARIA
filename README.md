p# aria-monitor

**Autonomous ML model monitoring with agentic root cause diagnosis.**

ARIA detects when your production ML model starts degrading — and tells you *why* in plain English, using an injected LLM agent that reasons over raw drift signals before any action is taken.

## Install

```bash
pip install aria-monitor
```

## Quickstart — 3 lines

```python
from aria import ARIA

aria = ARIA(model=clf, feature_names=X_train.columns.tolist(), config="aria.yaml")
result = aria.monitor(X_train, X_production)
print(result.summary)
```

## What it does

- **Drift Detection** — PSI + KS tests per feature, computed from scratch (no Evidently dependency)
- **Performance Monitoring** — tracks accuracy, F1, ROC-AUC degradation over time
- **Explainability** — SHAP-based feature importance shifts between training and production
- **Alerting** — console, JSON, or Slack alerts
- **LLM Diagnosis** — an LLM agent receives raw signals and reasons about root cause autonomously

## Works without an LLM

```python
# No Ollama, no API key needed — drift scores still returned
aria = ARIA(model=clf, feature_names=features)
result = aria.monitor(X_train, X_new)
```

## Swap LLM backends

```yaml
# aria.yaml
llm:
  backend: "ollama"    # or "openai" or "claude" or "none"
  model: "qwen2.5:3b"
```

One line at the entry point. All agents upgrade instantly.

## Supports any sklearn-compatible model

```python
from aria import ARIA
from aria.adapters import XGBoostAdapter, SklearnAdapter

# Auto-detected for XGBoost and sklearn
aria = ARIA(model=your_model, feature_names=features)

# Or pass adapter explicitly for custom models
aria = ARIA(model=your_model, feature_names=features, adapter=YourCustomAdapter(your_model, features))
```

## Optional dependencies

```bash
pip install aria-monitor[xgboost]    # XGBoost support
pip install aria-monitor[shap]       # SHAP explainability
pip install aria-monitor[openai]     # OpenAI LLM backend
pip install aria-monitor[anthropic]  # Claude LLM backend
pip install aria-monitor[all]        # Everything
```

## License

MIT
