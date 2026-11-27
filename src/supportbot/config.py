"""Cliente único: mismo código contra local y proveedores (API compatible OpenAI)."""
from dataclasses import dataclass
from os import getenv

from openai import AsyncOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    ollama_base_url: str = getenv("OLLAMA_BASE_URL", "http://llamacpp:8080/v1")


@dataclass(frozen=True, slots=True)
class Provider:
    base_url: str | None
    api_key: str
    model: str


def providers(s: Settings) -> dict[str, Provider]:
    # Verificá los nombres de modelo vigentes en la doc de cada proveedor.
    return {
        "local": Provider(s.ollama_base_url, "ollama", "qwen3:8b"),
        "vllm": Provider("http://localhost:8000/v1", "EMPTY", "Qwen/Qwen3-8B"),
        "openai": Provider(None, s.openai_api_key, "TODO-modelo-openai"),
        "gemini": Provider(
            "https://generativelanguage.googleapis.com/v1beta/openai/", s.gemini_api_key,
            "TODO-modelo-gemini",
        ),
        "anthropic": Provider("https://api.anthropic.com/v1/", s.anthropic_api_key, "claude-haiku-5-5"),
    }


def make_client(name: str = "local") -> tuple[AsyncOpenAI, str]:
    p = providers(Settings())[name]
    return AsyncOpenAI(base_url=p.base_url, api_key=p.api_key), p.model
