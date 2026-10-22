# Inference image: CPU-only torch, the gated model baked in, non-root.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.12-slim
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY artifacts/ensemble.pt2 artifacts/metrics.json artifacts/model_card.md /app/artifacts/
ENV PATH=/app/.venv/bin:$PATH GROUNDEFFECT_ARTIFACTS=/app/artifacts PYTHONUNBUFFERED=1
RUN useradd --uid 10001 --no-create-home svc
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "groundeffect.serve:app", "--host", "0.0.0.0", "--port", "8000"]
