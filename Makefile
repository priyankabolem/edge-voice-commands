PY := .venv/bin/python

.PHONY: stage0 env smoke

stage0: env smoke

env:
	$(PY) scripts/record_environment.py

smoke:
	$(PY) scripts/smoke_test.py

.PHONY: stage1 data features test lint

stage1: data features test

data:
	$(PY) scripts/download_data.py

features:
	$(PY) scripts/precompute_features.py

test:
	$(PY) -m pytest -q

lint:
	uvx ruff check . && uvx ruff format --check .

.PHONY: stage2 train evaluate

stage2: train evaluate

train:
	$(PY) -m kws.train

evaluate:
	$(PY) -m kws.evaluate
