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
	find cpp -path cpp/build -prune -o \( -name '*.cc' -o -name '*.h' \) -print | xargs .venv/bin/clang-format --dry-run --Werror

.PHONY: stage2 train evaluate

stage2: train evaluate

train:
	$(PY) -m kws.train

evaluate:
	$(PY) -m kws.evaluate

.PHONY: stage3 quantize quant-experiments

stage3: quantize quant-experiments

quantize:
	$(PY) -m kws.quantize

quant-experiments:
	$(PY) -m kws.quantize --experiments

.PHONY: stage4 export

stage4: export

export:
	$(PY) -m kws.export

.PHONY: stage5 assets cpp parity-cpp format

stage5: assets cpp parity-cpp

assets:
	$(PY) -m kws.export_assets

cpp:
	PATH="$(CURDIR)/.venv/bin:$$PATH" cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
		-DPython3_EXECUTABLE=$(CURDIR)/.venv/bin/python
	PATH="$(CURDIR)/.venv/bin:$$PATH" cmake --build cpp/build

parity-cpp:
	$(PY) -m kws.parity_cpp

format:
	uvx ruff format .
	find cpp -path cpp/build -prune -o \( -name '*.cc' -o -name '*.h' \) -print | xargs .venv/bin/clang-format -i

.PHONY: stage6 benchmark profile-ops

stage6: benchmark profile-ops

benchmark:
	$(PY) scripts/record_environment.py > /dev/null
	$(PY) -m kws.benchmark

profile-ops:
	$(PY) -m kws.profile_ops

.PHONY: stage7 streams evaluate-stream

stage7: streams evaluate-stream

streams:
	$(PY) scripts/make_stream.py --split validation
	$(PY) scripts/make_stream.py --split testing

evaluate-stream:
	$(PY) -m kws.evaluate_stream
