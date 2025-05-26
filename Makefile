.PHONY: venv install check clean etl

.DEFAULT_GOAL:=etl

venv: pyproject.toml
	uv venv && . .venv/bin/activate

install: uv.lock
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
