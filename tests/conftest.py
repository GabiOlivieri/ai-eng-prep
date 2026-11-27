"""Fixtures para los tests "live" contra el llama-server del compose.

Si el servidor no responde, los tests live se SALTEAN (skip) en vez de fallar, así
`pytest` sin modelo sigue en verde con los tests de fakes. Para correr solo los live:
    pytest -m evals -v
y para excluirlos:
    pytest -m "not evals"
"""
import pytest

from supportbot.llamacpp import Embedder, LlamaCpp, find_chat_server, find_embed_server


@pytest.fixture(scope="session")
def chat_url() -> str:
    url = find_chat_server()
    if url is None:
        pytest.skip("llama-server no disponible (levantalo: docker compose --profile gpu up -d llamacpp)")
    return url


@pytest.fixture(scope="session")
def embed_url() -> str:
    url = find_embed_server()
    if url is None:
        pytest.skip("servidor de embeddings no disponible (docker compose --profile gpu up -d llamacpp-embed)")
    return url


@pytest.fixture
def llm(chat_url: str) -> LlamaCpp:
    # Un cliente por test: el cliente async queda atado al event loop de cada test.
    return LlamaCpp(chat_url)


@pytest.fixture
def embedder(embed_url: str) -> Embedder:
    return Embedder(embed_url)
