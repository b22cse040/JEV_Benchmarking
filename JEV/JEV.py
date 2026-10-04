from openai import OpenAI

from config_setter import ConfigSetter
from llm import LLM
from observability import ObservabilityManager
from pipeline import Pipeline

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


client = OpenAI()
effort = "medium"

llm = LLM(
    client=client,
    model="gpt-6-luna",
    default_reasoning_effort=effort,
)

observability = ObservabilityManager(
    logging_file_path=f"logs/logs_jev.csv",
)

pipeline = Pipeline(
    llm=llm,
    observability=observability,
    config_setter=ConfigSetter(),
    should_set_config_setter=True,
)

pipeline.run_csv("./Data/reasoning_guardrail_100_questions_updated.csv")