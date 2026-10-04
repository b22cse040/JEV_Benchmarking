import time
from typing import Optional

import pandas as pd
from tqdm import tqdm

from config_setter import REASONING_LEVELS, ConfigSetter
from llm import LLM
from observability import ObservabilityManager


class Pipeline:
    def __init__(
        self,
        llm: LLM,
        observability: ObservabilityManager,
        config_setter: Optional[ConfigSetter] = None,
        should_set_config_setter: bool = False,
    ):
        self.llm = llm
        self.observability = observability
        self.config_setter = config_setter
        self.should_set_config_setter = should_set_config_setter

        if self.should_set_config_setter and self.config_setter is None:
            raise ValueError("config_setter is required when config setter is enabled")

    def _get_config(self, question: str):
        return self.config_setter.evaluate(question)

    @staticmethod
    def _extract_config(config_response):
        answers = config_response.answers
        jev_toxicity = answers["is_toxic"].choice
        reasoning_level = answers["reasoning_level"].choice

        if reasoning_level not in REASONING_LEVELS:
            raise ValueError(
                f"JEV returned unsupported reasoning level '{reasoning_level}'. "
                f"Expected one of {REASONING_LEVELS}."
            )

        return jev_toxicity, reasoning_level

    def _call_llm(self, question: str, reasoning_effort: Optional[str] = None):
        return self.llm.generate(question, reasoning_effort)

    def run_question(self, question: str, answer=None, is_toxic=None):
        if not self.should_set_config_setter:
            llm_effort = self.llm.default_reasoning_effort

            start_time = time.perf_counter()
            llm_response = self._call_llm(question, llm_effort)
            llm_latency = time.perf_counter() - start_time

            self.observability.parse_response(
                question=question,
                answer=answer,
                is_toxic=is_toxic,
                llm_response=llm_response,
                llm_latency=llm_latency,
                llm_effort=llm_effort,
                use_jev=False,
            )

            return llm_response

        config_response = self._get_config(question)
        jev_toxicity, reasoning_level = self._extract_config(config_response)

        if jev_toxicity == "Yes":
            self.observability.parse_response(
                question=question,
                answer=answer,
                is_toxic=is_toxic,
                llm_response=None,
                llm_latency=0,
                llm_effort=None,
                use_jev=True,
                jev_response=config_response,
            )

            return None

        llm_effort = reasoning_level

        start_time = time.perf_counter()
        llm_response = self._call_llm(question, llm_effort)
        llm_latency = time.perf_counter() - start_time

        self.observability.parse_response(
            question=question,
            answer=answer,
            is_toxic=is_toxic,
            llm_response=llm_response,
            llm_latency=llm_latency,
            llm_effort=llm_effort,
            use_jev=True,
            jev_response=config_response,
        )

        return llm_response

    def run_csv(
        self,
        csv_path: str,
        question_column: str = "Question",
        answer_column: str = "Answer",
        toxicity_column: str = "is_flagged",
    ):
        df = pd.read_csv(csv_path)

        required_columns = [question_column, answer_column, toxicity_column]

        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in {csv_path}")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
            answer = None if pd.isna(row[answer_column]) else row[answer_column]
            is_toxic = None if pd.isna(row[toxicity_column]) else row[toxicity_column]

            self.run_question(
                question=str(row[question_column]),
                answer=answer,
                is_toxic=is_toxic,
            )