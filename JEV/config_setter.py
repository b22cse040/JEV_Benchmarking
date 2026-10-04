import os

from typesafe_sdk import Choice, TypeSafeClient


REASONING_LEVELS = {"none", "low", "medium"}


class ConfigSetter:
    def evaluate(self, question: str):
        api_key = os.getenv("JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY")

        with TypeSafeClient(api_key=api_key) as client:
            return client.system_one(
                state=question,
                questions={
                    "is_toxic": Choice(
                        instructions=(
                            "Determine whether the given question is toxic or off-topic. "
                            "A question is off-topic if it is not a mathematics question."
                        ),
                        criteria={
                            "Yes": "The question is toxic or off-topic and should be dropped.",
                            "No": "The question is relevant to mathematics and can be answered.",
                        },
                    ),
                    "reasoning_level": Choice(
                        instructions=(
                            "Determine the level of reasoning required to correctly answer "
                            "the given mathematics question."
                        ),
                        criteria={
                            "none": "No reasoning is required; answer directly.",
                            # "minimal": "Minimal reasoning is required.",
                            "low": "Moderate reasoning or multiple steps are required.",
                            "medium": "Substantial multi-step reasoning is required.",
                        },
                    ),
                },
            )