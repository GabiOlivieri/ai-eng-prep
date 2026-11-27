FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app
WORKDIR /app
# Solo dependencias: el código se monta como volumen para editarlo desde tu editor.
COPY pyproject.toml ./
RUN uv sync --group dev --no-install-project
CMD ["sleep", "infinity"]
