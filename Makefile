.PHONY: install clean check fix nltk etl update_artifacts api start_container stop_container

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
	git add artifacts.dvc && \
	git commit -m "Executing the ETL pipeline and updating ./artifacts.dvc" && \
	dvc push && \
	git push

api:
	uvicorn src.rag_youtube_transcripts.app.main:app --reload

start_container:
	docker desktop start && \
	docker compose up -d && \
	echo "RAG backend will be running on http://localhost:8080/docs shortly ..."

stop_container:
	docker stop temp && \
	docker rm temp && \
	docker desktop stop
