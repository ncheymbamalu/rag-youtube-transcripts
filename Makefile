.PHONY: venv install check clean etl api

.DEFAULT_GOAL:=etl

venv: pyproject.toml
	uv venv && . .venv/bin/activate

install: pyproject.toml
	uv sync

check: install
	uv run isort src
	uv run ruff check src

clean:
	rm -rf `find . -type d -name __pycache__`
	rm -rf .ruff_cache

etl:
	dvc pull
	uv run src/pipelines/etl.py

api:
	uvicorn src.app.main:app --reload
