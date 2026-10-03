import numpy as np
import pandas as pd
from scipy import stats
from aria.agents.base import BaseAgent, AgentConfig, AgentResult

# ── Deterministic math lives here, not in the LLM ──────────────────────────

def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index. PSI < 0.1: stable. 0.1–0.2: warning. >0.2: critical."""
    expected = np.array(expected, dtype=float)
    actual   = np.array(actual,   dtype=float)

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    exp_counts = np.histogram(expected, bins=breakpoints)[0].astype(float)
    act_counts = np.histogram(actual,   bins=breakpoints)[0].astype(float)

    exp_pct = np.where(exp_counts == 0, 1e-4, exp_counts / len(expected))
    act_pct = np.where(act_counts == 0, 1e-4, act_counts / len(actual))

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _ks(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """KS test. Returns (statistic, p_value)."""
    stat, pval = stats.ks_2samp(expected, actual)
    return float(stat), float(pval)


def _classify_psi(psi: float, warn: float, crit: float) -> str:
    if psi >= crit:   return "critical"
    if psi >= warn:   return "warning"
    return "ok"


# ── Agent ───────────────────────────────────────────────────────────────────

class DriftDetectionAgent(BaseAgent):
    """
    Detects data drift between reference (X_train) and production (X_new) distributions.

    Deterministic layer: computes PSI + KS per feature — raw numbers only.
    LLM layer: receives those raw numbers and reasons about WHICH features drifted,
               WHY it likely happened, and what the business impact might be.
               The LLM decides — it is not narrating decisions already made by code.
    """

    def run(self, adapter, context: dict) -> AgentResult:
        X_train: pd.DataFrame = context["X_train"]
        X_new:   pd.DataFrame = context["X_new"]
        features: list[str]   = adapter.get_feature_names()

        warn_psi = self._threshold("psi_warning",  0.1)
        crit_psi = self._threshold("psi_critical", 0.2)
        warn_ks  = self._threshold("ks_warning",   0.05)
        crit_ks  = self._threshold("ks_critical",  0.1)

        # ── Step 1: Raw computation (deterministic, no LLM) ─────────────────
        psi_scores: dict[str, float] = {}
        ks_scores:  dict[str, dict]  = {}
        feature_statuses: dict[str, str] = {}

        for feat in features:
            if feat not in X_train.columns or feat not in X_new.columns:
                continue

            ref = X_train[feat].dropna().values
            cur = X_new[feat].dropna().values

            psi_val = _psi(ref, cur)
            ks_stat, ks_pval = _ks(ref, cur)

            psi_scores[feat] = round(psi_val, 4)
            ks_scores[feat]  = {"statistic": round(ks_stat, 4), "p_value": round(ks_pval, 4)}

            psi_status = _classify_psi(psi_val, warn_psi, crit_psi)
            ks_status  = "critical" if ks_stat >= crit_ks else ("warning" if ks_stat >= warn_ks else "ok")
            feature_statuses[feat] = max(psi_status, ks_status,
                                         key=lambda s: {"ok": 0, "warning": 1, "critical": 2}[s])

        drifted_features = [f for f, s in feature_statuses.items() if s != "ok"]
        critical_features = [f for f, s in feature_statuses.items() if s == "critical"]
        max_psi = max(psi_scores.values(), default=0.0)
        overall_status = (
            "critical" if critical_features else
            "warning"  if drifted_features  else
            "ok"
        )

        metrics = {
            "psi_scores": psi_scores,
            "ks_scores": ks_scores,
            "feature_statuses": feature_statuses,
            "drifted_features": drifted_features,
            "critical_features": critical_features,
            "max_psi": round(max_psi, 4),
            "n_drifted": len(drifted_features),
            "n_critical": len(critical_features),
        }

        # ── Step 2: LLM as upstream reasoner ────────────────────────────────
        # We hand the LLM raw signal and let it reason about root cause.
        # The code has NOT made any diagnosis — only classified thresholds.
        explanation = self._diagnose_with_llm(
            psi_scores=psi_scores,
            ks_scores=ks_scores,
            feature_statuses=feature_statuses,
            drifted_features=drifted_features,
            critical_features=critical_features,
            warn_psi=warn_psi,
            crit_psi=crit_psi,
        )

        return AgentResult(
            agent_name="DriftDetectionAgent",
            status=overall_status,
            metrics=metrics,
            explanation=explanation,
            recommendations=self._build_recommendations(overall_status, critical_features, drifted_features),
        )

    def _diagnose_with_llm(
        self,
        psi_scores: dict,
        ks_scores: dict,
        feature_statuses: dict,
        drifted_features: list,
        critical_features: list,
        warn_psi: float,
        crit_psi: float,
    ) -> str | None:
        if not drifted_features:
            return "No significant drift detected. All features are stable."

        # Sort by PSI descending so LLM sees most drifted features first
        sorted_drifted = sorted(drifted_features, key=lambda f: psi_scores.get(f, 0), reverse=True)

        feature_block = "\n".join(
            f"  - {f}: PSI={psi_scores.get(f, 'N/A'):.4f}, "
            f"KS={ks_scores.get(f, {}).get('statistic', 'N/A'):.4f}, "
            f"status={feature_statuses.get(f, 'unknown')}"
            for f in sorted_drifted[:15]   # cap to avoid token explosion
        )

        prompt = f"""You are an expert MLOps engineer diagnosing data drift in a production ML system.

RAW DRIFT SIGNALS (computed by deterministic algorithms — NOT pre-analysed):
Threshold reference: PSI warning={warn_psi}, critical={crit_psi}

Drifted features ({len(drifted_features)} total, {len(critical_features)} critical):
{feature_block}

Your job: reason about these raw signals and produce a diagnosis. Do NOT just restate the numbers.
Answer these questions in your response:
1. Which features are the primary concern and why (considering both PSI magnitude and KS statistic)?
2. What real-world phenomena could explain this pattern of drift (seasonal changes, data pipeline issues, population shift, etc.)?
3. What is the likely business impact if this drift is not addressed?
4. What is the most urgent action required?

Be specific. Be concise. Maximum 150 words."""

        system = (
            "You are an MLOps expert. Analyse drift signals and provide root cause reasoning. "
            "Never repeat numbers verbatim — interpret them. Always suggest a concrete next step."
        )

        return self._ask_llm(prompt, system=system)

    def _build_recommendations(self, status: str, critical: list, drifted: list) -> list[str]:
        recs = []
        if status == "critical":
            recs.append(f"URGENT: {len(critical)} features have critical drift — investigate data pipeline immediately.")
            recs.append("Consider pausing model predictions until root cause is identified.")
        elif status == "warning":
            recs.append(f"{len(drifted)} features show distributional shift — monitor closely.")
            recs.append("Schedule model revalidation on recent production data.")
        if drifted:
            recs.append("Run ExplainabilityAgent to correlate drift with prediction changes.")
        return recs
