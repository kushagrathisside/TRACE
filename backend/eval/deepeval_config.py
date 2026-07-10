"""
Local Ollama judge adapter for DeepEval.

DeepEval needs an LLM to evaluate test cases.  This wraps ChatOllama so
DeepEval uses the local Ollama model instead of OpenAI — no API key needed.

Install: pip install deepeval
"""

import sys

sys.path.insert(0, ".")


def get_judge():
    """
    Returns a DeepEvalBaseLLM instance backed by the local Ollama model.
    Import and call this in test files instead of constructing it inline.
    """
    try:
        from deepeval.models.base_model import DeepEvalBaseLLM
    except ImportError:
        raise ImportError("deepeval is not installed. Run: pip install deepeval")

    from llm_provider import LLMProvider

    class OllamaJudge(DeepEvalBaseLLM):
        def load_model(self):
            return LLMProvider.get_llm(temperature=0.0)

        def generate(self, prompt: str) -> str:
            from langchain_core.messages import HumanMessage

            return self.load_model().invoke([HumanMessage(content=prompt)]).content

        async def a_generate(self, prompt: str) -> str:
            return self.generate(prompt)

        def get_model_name(self) -> str:
            import config

            return f"ollama/{config.LLM_MODEL_NAME}"

    return OllamaJudge()
