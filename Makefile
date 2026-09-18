# quasar
#
# Targets that write result records run inside the Docker image. Targets that only read or
# lint run on the host.

IMAGE   ?= quasar:v1
UID:= $(shell id -u 2>/dev/null || echo 1000)
GID:= $(shell id -g 2>/dev/null || echo 1000)

# Set to 1 to skip checks already run at the current commit.
QUASAR_RESUME ?=

DOCKER:= docker run --rm -v "$(CURDIR)":/work -w /work -u $(UID):$(GID) \
             -e QUASAR_IMAGE=$(IMAGE) -e PYTHONPATH=/work \
             -e QUASAR_RESUME=$(QUASAR_RESUME) $(IMAGE)

.PHONY: help setup test test-all gates provenance lint format docker shell lock sweep

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup:  ## local development environment
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	.venv/bin/pre-commit install

docker:  ## build the execution image
	docker build -t $(IMAGE) --build-arg UID=$(UID) --build-arg GID=$(GID) .

lock:  ## write the resolved environment to environment.lock.txt
	docker run --rm $(IMAGE) cat /opt/quasar/environment.lock.txt > environment.lock.txt
	@echo "wrote environment.lock.txt"

shell:  ## interactive shell inside the image
	docker run --rm -it -v "$(CURDIR)":/work -w /work -u $(UID):$(GID) \
	  -e QUASAR_IMAGE=$(IMAGE) -e PYTHONPATH=/work $(IMAGE) bash

test:  ## fast tests
	pytest -m fast

test-all:  ## all tests, inside the image
	$(DOCKER) pytest -m "fast or slow or gate"

gates:  ## run every check and write its record, inside the image
	$(DOCKER) python scripts/run_all_gates.py

provenance:  ## check that every committed record came from the image
	python scripts/check_results_provenance.py

lint:
	ruff check quasarstack experiments scripts tests
	black --check quasarstack experiments scripts tests
	mypy quasarstack

format:
	black quasarstack experiments scripts tests
	ruff check --fix quasarstack experiments scripts tests

sweep:  ## grid sweep, WP=<n>
	$(DOCKER) python scripts/sweep_runner.py --wp $(WP)
