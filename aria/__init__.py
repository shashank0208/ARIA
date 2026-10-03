"""
aria-monitor — Autonomous ML model monitoring with agentic root cause diagnosis.

Usage:
    from aria import ARIA
    aria = ARIA(model=clf, feature_names=X_train.columns.tolist(), config="aria.yaml")
    result = aria.monitor(X_train, X_new)
    print(result.summary)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Any
from dataclasses import dataclass, field

from aria.config.loader import ARIAConfig
from aria.adapters.base import ModelAdapter
from aria.llm.base import LLMEngine


@dataclass
class MonitorResult:
    summary: str
    status: str
    agent_results: dict = field(default_factory=dict)
    report_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "status": self.status,
            "agents": {k: v.to_dict() for k, v in self.agent_results.items()},
            "reports": self.report_paths,
        }


class ARIA:
    """
    Single entry point for autonomous ML model monitoring.

    from aria import ARIA
    aria = ARIA(model=clf, feature_names=X_train.columns.tolist(), config="aria.yaml")
    result = aria.monitor(X_train, X_new)
    print(result.summary)
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        config: str | dict | None = None,
        adapter: Optional[ModelAdapter] = None,
        llm_engine: Optional[LLMEngine] = None,
    ):
        if isinstance(config, str):
            self._cfg = ARIAConfig(config_path=config)
        elif isinstance(config, dict):
            self._cfg = ARIAConfig()
            self._cfg._cfg = self._cfg._deep_merge(self._cfg._cfg, config)
        else:
            self._cfg = ARIAConfig()

        self._adapter = adapter if adapter is not None else self._auto_adapter(model, feature_names)
        self._llm = llm_engine if llm_engine is not None else self._build_llm()
        self._agents: dict = {}
        self._build_agents()

    def monitor(
        self,
        X_train: pd.DataFrame,
        X_new: pd.DataFrame,
        y_true: Optional[np.ndarray] = None,
    ) -> MonitorResult:
        """
        Run all enabled agents against reference vs production data.

        Args:
            X_train: Reference distribution the model was trained on
            X_new:   Production data the model is seeing now
            y_true:  Ground truth labels for X_new (optional, enables performance metrics)
        """
        context = {
            "X_train": X_train,
            "X_new": X_new,
            "y_true": y_true,
            "feature_names": self._adapter.get_feature_names(),
        }

        agent_results = {}
        statuses = []

        for name, agent in self._agents.items():
            try:
                result = agent.run(self._adapter, context)
                agent_results[name] = result
                context[f"{name}_result"] = result
                statuses.append(result.status)
            except Exception as e:
                from aria.agents.base import AgentResult
                agent_results[name] = AgentResult(
                    agent_name=name, status="error", metrics={},
                    explanation=f"Agent failed: {e}",
                )
                statuses.append("error")

        overall = self._aggregate_status(statuses)
        summary = self._build_summary(overall, agent_results)
        return MonitorResult(summary=summary, status=overall, agent_results=agent_results)

    def set_llm(self, llm_engine: LLMEngine) -> None:
        """Hot-swap LLM engine without rebuilding agents. One line to switch backends."""
        self._llm = llm_engine
        for agent in self._agents.values():
            agent.llm_engine = llm_engine

    def _auto_adapter(self, model: Any, feature_names: list[str]) -> ModelAdapter:
        model_type = type(model).__name__
        if "XGB" in model_type:
            from aria.adapters.xgboost_adapter import XGBoostAdapter
            return XGBoostAdapter(model, feature_names)
        if hasattr(model, "predict"):
            from aria.adapters.xgboost_adapter import SklearnAdapter
            return SklearnAdapter(model, feature_names)
        raise ValueError(f"Cannot auto-detect adapter for {model_type}. Pass adapter= explicitly.")

    def _build_llm(self) -> Optional[LLMEngine]:
        llm_cfg = self._cfg.llm_config()
        backend = llm_cfg.get("backend", "none").lower()
        if backend == "none":
            return None
        if backend == "ollama":
            from aria.llm.ollama_engine import OllamaEngine
            return OllamaEngine(
                model=llm_cfg.get("model", "qwen2.5:3b"),
                base_url=llm_cfg.get("base_url", "http://localhost:11434"),
                timeout=llm_cfg.get("timeout", 60),
            )
        if backend == "openai":
            from aria.llm.openai_engine import OpenAIEngine
            return OpenAIEngine(api_key=llm_cfg["api_key"], model=llm_cfg.get("model", "gpt-4o-mini"))
        if backend == "claude":
            from aria.llm.claude_engine import ClaudeEngine
            return ClaudeEngine(api_key=llm_cfg["api_key"], model=llm_cfg.get("model", "claude-haiku-4-5-20251001"))
        raise ValueError(f"Unknown LLM backend: '{backend}'")

    def _build_agents(self) -> None:
        from aria.agents.drift_detection import DriftDetectionAgent
        from aria.agents.performance_monitor import PerformanceMonitorAgent
        from aria.agents.explainability import ExplainabilityAgent
        from aria.agents.alerting import AlertingAgent

        for key, (AgentClass, config_key) in {
            "drift":          (DriftDetectionAgent,     "drift"),
            "performance":    (PerformanceMonitorAgent,  "performance"),
            "explainability": (ExplainabilityAgent,      "explainability"),
            "alerting":       (AlertingAgent,            "alerting"),
        }.items():
            cfg = self._cfg.agent_config(config_key)
            if cfg.enabled:
                self._agents[key] = AgentClass(config=cfg, llm_engine=self._llm)

    def _aggregate_status(self, statuses: list[str]) -> str:
        if "critical" in statuses: return "critical"
        if "error"    in statuses: return "error"
        if "warning"  in statuses: return "warning"
        return "ok"

    def _build_summary(self, overall: str, results: dict) -> str:
        lines = [f"ARIA Monitor — Status: {overall.upper()}"]
        for name, r in results.items():
            lines.append(f"  [{r.status.upper()}] {name}: {r.explanation or 'No explanation available'}")
        return "\n".join(lines)
