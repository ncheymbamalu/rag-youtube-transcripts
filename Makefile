.PHONY: install clean check fix nltk etl update_artifacts api

.DEFAULT_GOAL:=etl

install: pyproject.toml
	uv sync

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .ruff_cache .pytest_cache

check: .venv
	uv run ruff check src

fix: .venv
	uv run ruff check --fix src

nltk: .venv
	uv add nltk && \
	mkdir -p .venv/nltk_data && \
	cd .venv/nltk_data && \
	uv run python -m nltk.downloader stopwords

etl:
	dvc pull
	uv run python -Wignore src/rag_youtube_transcripts/pipelines/etl.py

update_artifacts:
	dvc add ./artifacts && \
	dvc push && \
	git add artifacts.dvc && \
	git commit -m "Executing the ETL pipeline and updating ./artifacts.dvc" && \
	git push

api:
	uvicorn src.rag_youtube_transcripts.app.main:app --reload
