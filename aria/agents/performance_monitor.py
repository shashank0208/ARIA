import numpy as np
import pandas as pd
from aria.agents.base import BaseAgent, AgentConfig, AgentResult


class PerformanceMonitorAgent(BaseAgent):
    """
    Monitors model performance metrics over time.

    Deterministic layer: computes metrics via adapter.get_metrics() — numbers only.
    LLM layer: receives metric deltas + drift context and reasons about whether
               performance degradation is caused by drift, label shift, or model staleness.

    Requires y_true in context. If absent, skips metric computation gracefully.
    Metrics to track come from config — never hardcoded here.
    """

    def run(self, adapter, context: dict) -> AgentResult:
        X_new  = context.get("X_new")
        y_true = context.get("y_true")
        drift_result = context.get("drift_result")   # from DriftDetectionAgent if ran first

        # Metrics to track come from config, not hardcoded
        metrics_to_track = self.config.metrics or ["accuracy", "f1", "roc_auc"]

        # ── Graceful degradation if no labels available ──────────────────────
        if y_true is None:
            return AgentResult(
                agent_name="PerformanceMonitorAgent",
                status="ok",
                metrics={"note": "y_true not provided — performance metrics skipped"},
                explanation="No ground truth labels available. Provide y_true to aria.monitor() to enable performance tracking.",
            )

        warn_thresh = self._threshold("degradation_warning",  0.05)
        crit_thresh = self._threshold("degradation_critical", 0.10)

        # ── Step 1: Compute current metrics (deterministic) ──────────────────
        try:
            current_metrics = adapter.get_metrics(X_new, y_true, metrics_to_track)
        except Exception as e:
            return AgentResult(
                agent_name="PerformanceMonitorAgent",
                status="error",
                metrics={},
                explanation=f"Metric computation failed: {e}",
            )

        # ── Step 2: Compute baseline from X_train if available ───────────────
        baseline_metrics: dict[str, float] = context.get("baseline_metrics", {})
        if not baseline_metrics:
            # Estimate baseline from training data distribution via adapter
            # In production, users pass baseline_metrics= via context or config
            baseline_metrics = {m: context.get(f"baseline_{m}", float("nan")) for m in metrics_to_track}

        # ── Step 3: Compute deltas ────────────────────────────────────────────
        deltas: dict[str, float] = {}
        degraded: list[str] = []
        critical: list[str] = []

        for metric, current_val in current_metrics.items():
            baseline_val = baseline_metrics.get(metric, float("nan"))
            if np.isnan(baseline_val) or np.isnan(current_val):
                deltas[metric] = float("nan")
                continue

            delta = current_val - baseline_val
            deltas[metric] = round(delta, 4)

            # Negative delta = degradation for accuracy/f1/auc
            # Positive delta = degradation for mse/loss type metrics
            is_loss_metric = metric in ("mse", "mae", "loss")
            degraded_val = delta if is_loss_metric else -delta

            if degraded_val >= crit_thresh:
                critical.append(metric)
                degraded.append(metric)
            elif degraded_val >= warn_thresh:
                degraded.append(metric)

        overall_status = (
            "critical" if critical else
            "warning"  if degraded  else
            "ok"
        )

        metrics_out = {
            "current": {k: round(v, 4) for k, v in current_metrics.items()},
            "baseline": {k: round(v, 4) if not np.isnan(v) else None for k, v in baseline_metrics.items()},
            "deltas": deltas,
            "degraded_metrics": degraded,
            "critical_metrics": critical,
        }

        # ── Step 4: LLM reasoning about WHY performance degraded ─────────────
        explanation = self._diagnose_with_llm(
            current_metrics=current_metrics,
            deltas=deltas,
            degraded=degraded,
            critical=critical,
            drift_result=drift_result,
        )

        return AgentResult(
            agent_name="PerformanceMonitorAgent",
            status=overall_status,
            metrics=metrics_out,
            explanation=explanation,
            recommendations=self._build_recommendations(overall_status, critical, degraded, drift_result),
        )

    def _diagnose_with_llm(
        self,
        current_metrics: dict,
        deltas: dict,
        degraded: list,
        critical: list,
        drift_result,
    ) -> str | None:
        if not degraded:
            perf_line = ", ".join(f"{k}={v:.4f}" for k, v in current_metrics.items() if not np.isnan(v))
            return f"Performance is stable. Current metrics: {perf_line}"

        metric_block = "\n".join(
            f"  - {m}: current={current_metrics.get(m, 'N/A'):.4f}, "
            f"delta={deltas.get(m, 'N/A')}"
            for m in degraded
        )

        drift_context = ""
        if drift_result and drift_result.metrics.get("drifted_features"):
            top_drifted = drift_result.metrics["drifted_features"][:5]
            drift_context = f"\nActive data drift detected in: {', '.join(top_drifted)}"

        prompt = f"""You are an MLOps expert diagnosing production model performance degradation.

PERFORMANCE DEGRADATION DETECTED:
{metric_block}{drift_context}

Reason about:
1. Is this degradation caused by data drift, concept drift (label distribution change), or model staleness?
2. How severe is this — is the model still usable or unreliable?
3. What is the most likely root cause given the pattern of metric changes?
4. What action should be taken first?

Be direct and specific. Maximum 120 words."""

        system = "You are an MLOps expert. Diagnose model performance issues from raw metric signals. Be actionable."

        return self._ask_llm(prompt, system=system)

    def _build_recommendations(self, status: str, critical: list, degraded: list, drift_result) -> list[str]:
        recs = []
        if status == "critical":
            recs.append(f"CRITICAL: {critical} metrics have degraded beyond threshold — model reliability is at risk.")
        if degraded:
            recs.append("Evaluate whether model needs retraining on recent production data.")
        if drift_result and drift_result.status in ("warning", "critical"):
            recs.append("Drift was detected — performance degradation is likely distribution-related, not concept drift.")
        return recs
