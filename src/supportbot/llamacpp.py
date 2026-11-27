"""Cliente mínimo para el llama-server (llama.cpp) del compose, vía API compatible OpenAI.

Lo usan los tests "live" (tests/test_live_llamacpp.py) para correr las katas contra el modelo
local en vez de fakes. Mismo código sirve para cualquier proveedor compatible OpenAI.

Variables de entorno:
- LLAMACPP_BASE_URL (o OLLAMA_BASE_URL, que es la que ya usa el compose): chat.
  Si no hay ninguna, prueba http://llamacpp:8080/v1 (dentro del compose) y
  http://localhost:8081/v1 (desde tu máquina, puerto publicado).
- LLAMACPP_MODEL: alias del modelo (por defecto "qwen3:8b", el --alias del compose).
- EMBED_BASE_URL / EMBED_MODEL: servidor de embeddings (servicio llamacpp-embed).
"""
import os
from typing import Any

import httpx
import numpy as np
from openai import AsyncOpenAI

CHAT_CANDIDATES = ["http://llamacpp:8080/v1", "http://localhost:8081/v1"]
EMBED_CANDIDATES = ["http://llamacpp-embed:8080/v1", "http://localhost:8082/v1"]

# Qwen3 "piensa" (<think>...</think>) por defecto: más lento y ensucia el JSON.
# llama-server con --jinja acepta chat_template_kwargs para apagarlo.
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def _candidates(env_names: list[str], defaults: list[str]) -> list[str]:
    explicit = [os.environ[n] for n in env_names if os.environ.get(n)]
    return explicit or defaults


def find_server(env_names: list[str], defaults: list[str], timeout: float = 2.0) -> str | None:
    """Devuelve la primera base_url que responde GET /models, o None."""
    for url in _candidates(env_names, defaults):
        try:
            if httpx.get(f"{url.rstrip('/')}/models", timeout=timeout).status_code == 200:
                return url
        except httpx.HTTPError:
            continue
    return None


def find_chat_server() -> str | None:
    return find_server(["LLAMACPP_BASE_URL", "OLLAMA_BASE_URL"], CHAT_CANDIDATES)


def find_embed_server() -> str | None:
    return find_server(["EMBED_BASE_URL"], EMBED_CANDIDATES)


class LlamaCpp:
    """think=False (default): rápido y determinístico, ideal para JSON y clasificación.
    think=True: Qwen3 razona antes de responder (llama-server lo separa en reasoning_content,
    así que `content` queda limpio). Para agentes con varios pasos hace falta: sin pensar,
    qwen3:8b tiende a ANUNCIAR la acción ("voy a abrir un ticket") en vez de llamar a la tool.
    Qwen recomienda temperature ~0.6 en modo thinking (con greedy puede entrar en loops).
    """

    def __init__(
        self, base_url: str, model: str | None = None, temperature: float | None = None,
        think: bool = False,
    ):
        # timeout generoso: la primera llamada puede tardar mientras el modelo "calienta".
        self.client = AsyncOpenAI(base_url=base_url, api_key="sk-no-key", timeout=300)
        self.model = model or os.environ.get("LLAMACPP_MODEL", "qwen3:8b")
        self.think = think
        self.temperature = temperature if temperature is not None else (0.6 if think else 0.0)
        self._extra = {"chat_template_kwargs": {"enable_thinking": think}}

    async def text(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Firma compatible con `LLMCall` de las katas: mensajes -> texto."""
        resp = await self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature,
            extra_body=self._extra, **kwargs,
        )
        return resp.choices[0].message.content or ""

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict:
        """Firma compatible con `ChatFn` del día 3: devuelve choices[0].message como dict."""
        resp = await self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools or None,
            temperature=self.temperature, extra_body=self._extra,
        )
        msg = resp.choices[0].message
        tool_calls = [tc.model_dump() for tc in msg.tool_calls] if msg.tool_calls else None
        return {"role": "assistant", "content": msg.content, "tool_calls": tool_calls}


class Embedder:
    def __init__(self, base_url: str, model: str | None = None):
        self.client = AsyncOpenAI(base_url=base_url, api_key="sk-no-key", timeout=120)
        self.model = model or os.environ.get("EMBED_MODEL", "nomic-embed-text")

    async def embed(self, texts: list[str]) -> np.ndarray:
        resp = await self.client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)
