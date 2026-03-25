# check=skip=CopyIgnoredFile
FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq -y && \
    apt-get install --no-install-recommends curl make tree build-essential -qq -y && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/rag-youtube-transcripts/.venv \
    PATH="/rag-youtube-transcripts/.venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /rag-youtube-transcripts

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project --no-cache --no-dev

COPY . .

RUN uv sync --frozen --no-cache --no-dev

CMD ["uvicorn", "src.rag_youtube_transcripts.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
