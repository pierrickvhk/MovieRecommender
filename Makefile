.PHONY: install format lint type test check

PY ?= python3

install:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev,local]"


format:
	$(PY) -m ruff format .

lint:
	$(PY) -m ruff check .

type:
	$(PY) -m mypy .

test:
	$(PY) -m pytest -q

check: format lint type test
