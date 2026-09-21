import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

load_dotenv(Path(__file__).resolve().parent / ".env")

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.6-luna")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

if not API_KEY:
    raise ValueError(
        "API_KEY is not set. Copy .env.example to .env and fill in your credentials."
    )

_client_kwargs: dict[str, str] = {"api_key": API_KEY}
if BASE_URL:
    _client_kwargs["base_url"] = BASE_URL


def _with_openai_prefix(model_name: str) -> str:
    return model_name if ":" in model_name else f"openai:{model_name}"


llm = init_chat_model(_with_openai_prefix(LLM_MODEL), **_client_kwargs)
embedder = init_embeddings(_with_openai_prefix(EMBEDDING_MODEL), **_client_kwargs)
