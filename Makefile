.PHONY: install nltk check clean etl update_artifacts api

.DEFAULT_GOAL:=etl

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
	git pull && dvc pull && uv run python -Wignore src/pipelines/etl.py

update_artifacts:
	dvc add ./artifacts && \
	git add artifacts.dvc && \
	git commit -m "Executing the ETL pipeline and updating ./artifacts.dvc" && \
	dvc push && \
	git push

api:
	uvicorn src.app.main:app --reload
