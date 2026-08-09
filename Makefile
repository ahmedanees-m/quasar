# QUASAR - one-word entry points.
#
# Targets that produce result records run inside the pinned Docker image. Targets that only
# read or lint run wherever you are. See DECISIONS.md ADR-0006.

IMAGE   ?= quasar:v1
UID     := $(shell id -u 2>/dev/null || echo 1000)
GID     := $(shell id -g 2>/dev/null || echo 1000)
DOCKER  := docker run --rm -v "$(CURDIR)":/work -w /work -u $(UID):$(GID) $(IMAGE)

.PHONY: help setup test test-all gates figures claims lint format docker shell lock sweep disk sync-up sync-down

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup:  ## local dev env for authoring only, never for producing results
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	.venv/bin/pre-commit install

docker:  ## build the execution image on this machine
	docker build -t $(IMAGE) --build-arg UID=$(UID) --build-arg GID=$(GID) .

lock:  ## copy the resolved environment out of the image and commit it
	docker run --rm $(IMAGE) cat /opt/quasar/environment.lock.txt > environment.lock.txt
	@echo "wrote environment.lock.txt"

shell:  ## interactive shell inside the image
	docker run --rm -it -v "$(CURDIR)":/work -w /work -u $(UID):$(GID) $(IMAGE) bash

test:  ## fast tests, runs anywhere
	pytest -m fast

test-all:  ## everything including slow tests and gates, inside the image
	$(DOCKER) pytest -m "fast or slow or gate"

gates:  ## full reproduction: every pre-registered gate, inside the image
	$(DOCKER) python scripts/run_all_gates.py

figures:  ## regenerate every figure from committed results
	$(DOCKER) python scripts/make_figures.py

claims:  ## verify every CLAIMS.md entry resolves to an artefact
	python scripts/check_claims.py

provenance:  ## verify every committed result record came from the pinned image
	python scripts/check_results_provenance.py

lint:
	ruff check quasarstack experiments scripts tests
	black --check quasarstack experiments scripts tests
	mypy quasarstack

format:
	black quasarstack experiments scripts tests
	ruff check --fix quasarstack experiments scripts tests

sweep:  ## resumable grid sweep, WP=<n>
	$(DOCKER) python scripts/sweep_runner.py --wp $(WP)

disk:  ## check the VM stays inside its declared storage ceiling
	@python infra/disk_guard.py

sync-up:  ## push results from the VM to the Drive archive over SFTP
	python infra/sync.py up results

sync-down:  ## pull archived results from the Drive to the VM over SFTP
	python infra/sync.py down results
