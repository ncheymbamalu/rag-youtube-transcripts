.PHONY: venv install check clean etl update_artifacts api

.DEFAULT_GOAL:=etl

venv: pyproject.toml
	uv venv && . .venv/bin/activate

install: pyproject.toml
	uv sync

nltk: .venv
	uv add nltk \
	&& mkdir -p .venv/nltk_data \
	&& cd .venv/nltk_data \
	&& uv run python -m nltk.downloader stopwords 

check: install
	uv run isort src ; uv run ruff check src

clean:
	rm -rf `find . -type d -name __pycache__` ; rm -rf .ruff_cache

etl:
	dvc pull && uv run src/pipelines/etl.py

update_artifacts:
	dvc add ./artifacts && \
	git add artifacts.dvc && \
	git commit -m "Executing the ETL pipeline" && \
	dvc push && \
	git push

api:
	uvicorn src.app.main:app --reload
