from typing import Any, Optional

from openai import OpenAI

from config_setter import REASONING_LEVELS


class LLM:
    def __init__(self, client: OpenAI, model: str, default_reasoning_effort: str = "medium"):
        self.client = client
        self.model = model
        self.default_reasoning_effort = self.validate_reasoning_effort(default_reasoning_effort)

    @staticmethod
    def validate_reasoning_effort(reasoning_effort: str) -> str:
        if reasoning_effort not in REASONING_LEVELS:
            raise ValueError(
                f"Invalid reasoning effort '{reasoning_effort}'. "
                f"Expected one of {REASONING_LEVELS}."
            )

        return reasoning_effort

    def generate(self, question: str, reasoning_effort: Optional[str] = None) -> Any:
        effort = reasoning_effort or self.default_reasoning_effort
        effort = self.validate_reasoning_effort(effort)

        return self.client.responses.create(
            model=self.model,
            input=question,
            reasoning={"effort": effort},
        )