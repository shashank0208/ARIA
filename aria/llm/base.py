from abc import ABC, abstractmethod


class LLMEngine(ABC):
    """
    Contract every LLM backend must satisfy.
    Agents never import any LLM library directly — they call this interface only.
    Swapping backends (Ollama → Claude → OpenAI) is one line at the entry point.
    """

    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> str:
        """
        Send a prompt, get a string response back.
        All formatting, retries, timeouts live inside the concrete engine — never in agents.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Health check. Returns False if the backend is unreachable.
        Used by ARIA to decide whether to skip LLM steps gracefully.
        """
        ...
