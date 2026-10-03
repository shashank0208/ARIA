import numpy as np
import pandas as pd
from aria.agents.base import BaseAgent, AgentConfig, AgentResult


class ExplainabilityAgent(BaseAgent):
    """
    Explains model behaviour using SHAP values and feature importance shifts.

    Deterministic layer: computes SHAP values and importance rankings — numbers only.
    LLM layer: receives SHAP deltas + drift context and reasons about which features
               are driving prediction changes and whether the model is still behaving
               as expected. This is the core of the research contribution.

    Requires shap to be installed. Degrades gracefully if not available.
    """

    def run(self, adapter, context: dict) -> AgentResult:
        X_train: pd.DataFrame = context["X_train"]
        X_new:   pd.DataFrame = context["X_new"]
        drift_result  = context.get("drift_result")
        perf_result   = context.get("performance_result")

        top_n = self.config.extra.get("top_n_features", 10)
        sample_size = min(200, len(X_new))   # SHAP is expensive — cap sample

        # ── Step 1: Feature importance from adapter (always available) ────────
        importance = adapter.get_feature_importance()
        top_features_by_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # ── Step 2: SHAP values (optional, degrades gracefully) ───────────────
        shap_available = False
        shap_train_mean: dict[str, float] = {}
        shap_new_mean:   dict[str, float] = {}
        shap_deltas:     dict[str, float] = {}

        try:
            import shap

            X_sample_train = X_train.sample(min(100, len(X_train)), random_state=42)
            X_sample_new   = X_new.sample(sample_size, random_state=42)

            # TreeExplainer for tree-based models (XGBoost, RF, etc.)
            try:
                explainer = shap.TreeExplainer(adapter.model)
            except Exception:
                # Fallback to KernelExplainer for any model type
                explainer = shap.KernelExplainer(
                    adapter.predict_proba,
                    shap.sample(X_sample_train, 50),
                )

            shap_vals_train = explainer.shap_values(X_sample_train)
            shap_vals_new   = explainer.shap_values(X_sample_new)

            # For binary classifiers shap returns list[array] — take class 1
            if isinstance(shap_vals_train, list):
                shap_vals_train = shap_vals_train[1]
                shap_vals_new   = shap_vals_new[1]

            feat_names = adapter.get_feature_names()
            shap_train_mean = {f: float(np.mean(np.abs(shap_vals_train[:, i])))
                               for i, f in enumerate(feat_names)}
            shap_new_mean   = {f: float(np.mean(np.abs(shap_vals_new[:, i])))
                               for i, f in enumerate(feat_names)}
            shap_deltas     = {f: round(shap_new_mean[f] - shap_train_mean[f], 5)
                               for f in feat_names}
            shap_available = True

        except ImportError:
            pass  # SHAP not installed — fall through gracefully
        except Exception:
            pass  # SHAP computation failed — fall through gracefully

        # ── Step 3: Detect importance shifts ─────────────────────────────────
        shift_threshold = self.config.thresholds.get("importance_shift", 0.01)
        shifted_features = (
            {f: d for f, d in shap_deltas.items() if abs(d) >= shift_threshold}
            if shap_available else {}
        )

        overall_status = (
            "warning" if shifted_features else "ok"
        )

        metrics = {
            "feature_importance": {k: round(v, 5) for k, v in importance.items()},
            "top_features": [f for f, _ in top_features_by_importance],
            "shap_available": shap_available,
            "shap_train_mean": {k: round(v, 5) for k, v in shap_train_mean.items()},
            "shap_new_mean":   {k: round(v, 5) for k, v in shap_new_mean.items()},
            "shap_deltas":     {k: round(v, 5) for k, v in shap_deltas.items()},
            "shifted_features": shifted_features,
        }

        # ── Step 4: LLM reasons about what the importance shifts MEAN ─────────
        explanation = self._diagnose_with_llm(
            top_features=top_features_by_importance,
            shap_deltas=shap_deltas,
            shifted_features=shifted_features,
            shap_available=shap_available,
            drift_result=drift_result,
            perf_result=perf_result,
        )

        return AgentResult(
            agent_name="ExplainabilityAgent",
            status=overall_status,
            metrics=metrics,
            explanation=explanation,
            recommendations=self._build_recommendations(shifted_features, drift_result),
        )

    def _diagnose_with_llm(
        self,
        top_features: list,
        shap_deltas: dict,
        shifted_features: dict,
        shap_available: bool,
        drift_result,
        perf_result,
    ) -> str | None:

        if not shap_available:
            top_block = "\n".join(f"  {i+1}. {f} (importance={v:.4f})"
                                  for i, (f, v) in enumerate(top_features[:10]))
            prompt = f"""You are an MLOps expert interpreting model behaviour.

SHAP is not installed. Feature importances from model internals:
{top_block}

Based on these top features, explain what the model is likely relying on most heavily
and what to watch for if these features experience distribution shift.
Maximum 100 words."""
        else:
            # Sort by magnitude of shift
            top_shifts = sorted(shifted_features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            shift_block = "\n".join(
                f"  - {f}: SHAP delta={d:+.5f} ({'increased' if d > 0 else 'decreased'} importance)"
                for f, d in top_shifts
            ) if top_shifts else "  No significant SHAP shifts detected."

            drift_context = ""
            if drift_result and drift_result.metrics.get("drifted_features"):
                drift_context = f"\nFeatures with data drift: {', '.join(drift_result.metrics['drifted_features'][:5])}"

            perf_context = ""
            if perf_result and perf_result.metrics.get("degraded_metrics"):
                perf_context = f"\nDegraded performance metrics: {', '.join(perf_result.metrics['degraded_metrics'])}"

            prompt = f"""You are an MLOps expert diagnosing model behaviour change via SHAP analysis.

SHAP IMPORTANCE SHIFTS (production vs training):
{shift_block}{drift_context}{perf_context}

Reason about:
1. Which shifted features are most concerning and why?
2. Do the SHAP shifts align with the data drift signals? What does that alignment (or misalignment) tell you?
3. Is the model compensating for drift in one feature by over-relying on another?
4. What does this mean for prediction reliability?

Be specific. Maximum 150 words."""

        system = "You are an MLOps expert. Interpret SHAP values and feature importance shifts to diagnose model behaviour changes."
        return self._ask_llm(prompt, system=system)

    def _build_recommendations(self, shifted_features: dict, drift_result) -> list[str]:
        recs = []
        if shifted_features:
            top = sorted(shifted_features.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            recs.append(f"Model reliance has shifted most on: {', '.join(f for f, _ in top)}")
            recs.append("Validate that these features are still reliable in production.")
        if drift_result and drift_result.status == "critical":
            recs.append("Critical drift + SHAP shifts together strongly suggest model needs retraining.")
        return recs
