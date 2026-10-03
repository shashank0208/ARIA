from aria.llm.base import LLMEngine


class ClaudeEngine(LLMEngine):
    """Anthropic Claude backend. Requires: pip install anthropic"""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001", timeout: int = 30):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Run: pip install anthropic")

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.timeout = timeout

    def complete(self, prompt: str, system: str = "") -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text.strip()

    def is_available(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
