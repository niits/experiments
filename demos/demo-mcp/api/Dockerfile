FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1

COPY ./pyproject.toml ./uv.lock /app/

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

FROM python:3.12-slim AS runtime

COPY --from=builder /bin/uv /bin/uv
COPY --from=builder /app/.venv /app/.venv

COPY . /app/

WORKDIR /app

RUN ls -la /app

# Run the application.
CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
