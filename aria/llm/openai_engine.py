from aria.llm.base import LLMEngine


class OpenAIEngine(LLMEngine):
    """OpenAI API backend. Requires: pip install openai"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", timeout: int = 30):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        self.client = OpenAI(api_key=api_key)  # type: ignore
        self.model = model
        self.timeout = timeout

    def complete(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            timeout=self.timeout,
        )
        return response.choices[0].message.content.strip()

    def is_available(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
