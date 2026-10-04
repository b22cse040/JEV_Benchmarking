import csv
import json
import os
from typing import Any, Optional


MODEL_PRICING = {
    "jev-1.13": {
        "input": 0.000000042,
        "output": 0.0,
    },
    "gpt-6-luna": {
        "input": 0.0000001,
        "output": 0.0000005,
    },
}


CSV_COLUMNS = [
    "Question",
    "is_toxic",
    "answer",
    "guessed_toxicity",
    "JEV-Reasoning-Level",
    "JEV-Latency",
    "JEV-Decision",
    "JEV-Token-Usage",
    "LLM-Effort",
    "LLM-Latency",
    "LLM-Answer",
    "LLM-Token-Usage",
    "Total pricing",
    "Total Latency",
]


class ObservabilityManager:
    def __init__(self, logging_file_path: str):
        self.logging_file_path = logging_file_path
        self._ensure_csv_exists()

    def _ensure_csv_exists(self):
        directory = os.path.dirname(self.logging_file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        if not os.path.exists(self.logging_file_path):
            with open(self.logging_file_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    @staticmethod
    def _get_value(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)

        return getattr(obj, key, default)

    @staticmethod
    def _normalize_model_name(model: Optional[str]) -> str:
        if not model:
            return ""

        model = model.lower()

        if model.startswith("jev-1.13"):
            return "jev-1.13"

        return model

    def _build_token_usage(self, model: str, input_tokens: int, output_tokens: int) -> dict:
        model_name = self._normalize_model_name(model)
        pricing = MODEL_PRICING.get(model_name, {"input": 0.0, "output": 0.0})

        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]

        return {
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        }

    @staticmethod
    def _empty_token_usage() -> dict:
        return {
            "model": "none",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
        }

    def _parse_jev_response(self, response: Any) -> tuple[dict, dict, float]:
        answers = self._get_value(response, "answers", {})
        usage = self._get_value(response, "usage", {})
        model = self._get_value(response, "model", "jev-1.13")
        evaluation_time_ms = self._get_value(response, "evaluation_time_ms", 0.0)

        input_tokens = self._get_value(usage, "input_tokens", 0)
        output_tokens = self._get_value(usage, "output_tokens", 0)

        token_usage = self._build_token_usage(model, input_tokens, output_tokens)
        latency = evaluation_time_ms / 1000

        decision = {}

        for name, answer in answers.items():
            if isinstance(answer, dict):
                decision[name] = answer
            else:
                decision[name] = {
                    "type": self._get_value(answer, "type"),
                    "choice": self._get_value(answer, "choice"),
                    "confidence": self._get_value(answer, "confidence"),
                    "probabilities": self._get_value(answer, "probabilities"),
                    "stats": self._get_value(answer, "stats"),
                }

        return decision, token_usage, latency

    def _parse_openai_response(self, response: Any, latency: float) -> tuple[str, dict, float]:
        usage = self._get_value(response, "usage")
        model = self._get_value(response, "model", "")

        input_tokens = self._get_value(usage, "input_tokens", 0)
        output_tokens = self._get_value(usage, "output_tokens", 0)

        token_usage = self._build_token_usage(model, input_tokens, output_tokens)
        answer = self._get_value(response, "output_text", "")

        return answer, token_usage, latency

    def parse_response(
        self,
        question: str,
        answer: Any,
        is_toxic: Any,
        llm_response: Optional[Any],
        llm_latency: float,
        llm_effort: Optional[str],
        use_jev: bool = False,
        jev_response: Optional[Any] = None,
    ):
        if use_jev:
            if jev_response is None:
                raise ValueError("jev_response is required when use_jev=True")

            jev_decision, jev_token_usage, jev_latency = self._parse_jev_response(jev_response)
            guessed_toxicity = jev_decision.get("is_toxic", {}).get("choice")
            jev_reasoning_level = jev_decision.get("reasoning_level", {}).get("choice")
        else:
            guessed_toxicity = None
            jev_reasoning_level = None
            jev_decision = {}
            jev_token_usage = self._empty_token_usage()
            jev_latency = 0.0

        if llm_response is not None:
            llm_answer, llm_token_usage, llm_latency = self._parse_openai_response(
                llm_response, llm_latency
            )
        else:
            llm_answer = None
            llm_token_usage = self._empty_token_usage()
            llm_latency = 0.0
            llm_effort = None

        total_pricing = jev_token_usage["total_cost"] + llm_token_usage["total_cost"]
        total_latency = jev_latency + llm_latency

        row = {
            "Question": question,
            "is_toxic": is_toxic,
            "answer": answer,
            "guessed_toxicity": guessed_toxicity,
            "JEV-Reasoning-Level": jev_reasoning_level,
            "JEV-Latency": jev_latency,
            "JEV-Decision": json.dumps(jev_decision, separators=(",", ":")),
            "JEV-Token-Usage": json.dumps(jev_token_usage, separators=(",", ":")),
            "LLM-Effort": llm_effort,
            "LLM-Latency": llm_latency if llm_response is not None else None,
            "LLM-Answer": llm_answer,
            "LLM-Token-Usage": json.dumps(llm_token_usage, separators=(",", ":")),
            "Total pricing": total_pricing,
            "Total Latency": total_latency,
        }

        with open(self.logging_file_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
            writer.writerow(row)

        return row