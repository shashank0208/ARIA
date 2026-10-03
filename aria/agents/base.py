from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime


@dataclass
class AgentResult:
    """
    Standardized output from every agent.
    Agents never return raw dicts — always this structure.
    The LLM explanation is optional: if no LLM is configured, explanation is None.
    """
    agent_name: str
    status: str                          # "ok" | "warning" | "critical" | "error"
    metrics: dict[str, Any]              # Raw numeric outputs (PSI scores, accuracies, etc.)
    explanation: Optional[str] = None    # Plain-English LLM diagnosis. None if LLM disabled.
    recommendations: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent_name,
            "status": self.status,
            "metrics": self.metrics,
            "explanation": self.explanation,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class AgentConfig:
    """
    Per-agent configuration slice. Populated from aria.yaml.
    Agents never read config files directly — they receive this object.
    """
    enabled: bool = True
    thresholds: dict[str, float] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Contract all agents must satisfy.

    Key rules enforced here:
    - Agent receives llm_engine as a parameter, never creates one
    - Agent receives adapter, never touches model directly
    - If llm_engine is None, explanation step is skipped — agent still works
    """

    def __init__(self, config: AgentConfig, llm_engine=None):
        self.config = config
        self.llm_engine = llm_engine   # Can be None — agents must handle this gracefully

    @abstractmethod
    def run(self, adapter, context: dict) -> AgentResult:
        """
        Execute the agent.

        Args:
            adapter: ModelAdapter instance — only way to interact with any model
            context: Shared data from previous agents (e.g. drift scores for explainability agent)

        Returns:
            AgentResult with status, metrics, and optional LLM explanation
        """
        ...

    def _ask_llm(self, prompt: str, system: str = "") -> Optional[str]:
        """
        Safe LLM call. Returns None if LLM is disabled or unavailable.
        Agents call this — never call llm_engine.complete() directly.
        """
        if self.llm_engine is None:
            return None
        try:
            if not self.llm_engine.is_available():
                return None
            return self.llm_engine.complete(prompt, system=system)
        except Exception as e:
            return f"[LLM unavailable: {e}]"

    def _threshold(self, key: str, default: float = 0.2) -> float:
        """Fetch a threshold from config. Falls back to default if not set."""
        return self.config.thresholds.get(key, default)
