FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app

# dependencies first (cached until pyproject.toml / uv.lock change)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./
COPY data ./data
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
# create tables, load the sample books, start the API
CMD ["sh", "-c", "alembic upgrade head && python -m bookverse.migrate_json_to_db && uvicorn bookverse.main:app --host 0.0.0.0 --port 8000"]
