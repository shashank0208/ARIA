import yaml
from pathlib import Path
from aria.agents.base import AgentConfig


DEFAULT_CONFIG = {
    "llm": {
        "backend": "none",           # "ollama" | "openai" | "claude" | "none"
        "model": "qwen2.5:3b",
        "base_url": "http://localhost:11434",
        "api_key": None,
        "timeout": 60,
    },
    "drift": {
        "enabled": True,
        "method": "psi",             # "psi" | "ks" | "both"
        "thresholds": {
            "psi_warning": 0.1,
            "psi_critical": 0.2,
            "ks_warning": 0.05,
            "ks_critical": 0.1,
        },
    },
    "performance": {
        "enabled": True,
        "metrics": ["accuracy", "f1", "roc_auc"],
        "thresholds": {
            "degradation_warning": 0.05,
            "degradation_critical": 0.1,
        },
    },
    "explainability": {
        "enabled": True,
        "top_n_features": 10,
    },
    "retraining": {
        "enabled": False,            # Off by default — explicit opt-in
        "buffer_size": 1000,
        "trigger_threshold": "critical",
    },
    "alerting": {
        "enabled": True,
        "channels": ["console"],     # "console" | "email" | "slack" | "pagerduty"
    },
    "output": {
        "format": ["console"],       # "console" | "json" | "html"
        "report_path": "./aria_report",
    },
}


class ARIAConfig:
    def __init__(self, config_path: str | None = None):
        self._cfg = dict(DEFAULT_CONFIG)

        if config_path:
            path = Path(config_path)
            if path.exists():
                with open(path) as f:
                    user_cfg = yaml.safe_load(f) or {}
                self._cfg = self._deep_merge(self._cfg, user_cfg)

    def _deep_merge(self, base: dict, override: dict) -> dict:
        result = dict(base)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    def get(self, *keys, default=None):
        node = self._cfg
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def llm_config(self) -> dict:
        return self.get("llm") or {}

    def agent_config(self, agent_name: str) -> AgentConfig:
        section = self.get(agent_name) or {}
        return AgentConfig(
            enabled=section.get("enabled", True),
            thresholds=section.get("thresholds", {}),
            metrics=section.get("metrics", []),
            extra={k: v for k, v in section.items() if k not in ("enabled", "thresholds", "metrics")},
        )
