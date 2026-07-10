import os

# The system socks proxy causes httpx (used by langchain_ollama) to crash at
# import time.  Clear any proxy env vars before the import so the Ollama client
# can initialise cleanly.  This is a one-time side-effect at module load.
for _k in [k for k in os.environ if "proxy" in k.lower()]:
    del os.environ[_k]

import config  # noqa: E402
from langchain_ollama import ChatOllama, OllamaEmbeddings  # noqa: E402

# Module-level singletons — expensive to build, safe to reuse across requests.
_embeddings: OllamaEmbeddings | None = None
_llm: ChatOllama | None = None
_json_llm: ChatOllama | None = None


class LLMProvider:
    @staticmethod
    def get_llm(
        model_name: str = config.LLM_MODEL_NAME, temperature: float = 0.1
    ) -> ChatOllama:
        """
        Chat model for generation.  keep_alive=-1 keeps the model resident so
        its KV-cache (especially the static system-prompt prefix) is reused
        across requests — avoids re-encoding ~800 system-prompt tokens every call.
        """
        global _llm
        if _llm is None:
            _llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                keep_alive=config.OLLAMA_KEEP_ALIVE,
                num_ctx=config.OLLAMA_NUM_CTX,
                base_url=config.OLLAMA_BASE_URL,
            )
        return _llm

    @staticmethod
    def get_json_llm(model_name: str = config.LLM_MODEL_NAME) -> ChatOllama:
        """
        Same model with format='json' for structured output.
        temperature=0 maximises determinism for schema-constrained generation.
        Separate singleton because format is baked into the client object.
        """
        global _json_llm
        if _json_llm is None:
            _json_llm = ChatOllama(
                model=model_name,
                temperature=0.0,
                format="json",
                keep_alive=config.OLLAMA_KEEP_ALIVE,
                num_ctx=config.OLLAMA_NUM_CTX,
                base_url=config.OLLAMA_BASE_URL,
            )
        return _json_llm

    @staticmethod
    def get_embeddings() -> OllamaEmbeddings:
        """
        Singleton embedding model. Now runs locally via Ollama to bypass
        HuggingFace network timeouts.
        """
        global _embeddings
        if _embeddings is None:
            _embeddings = OllamaEmbeddings(
                model=config.EMBEDDING_MODEL_NAME,
                base_url=config.OLLAMA_BASE_URL,
            )
        return _embeddings
