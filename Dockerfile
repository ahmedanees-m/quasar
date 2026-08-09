# QUASAR execution environment.
#
# Everything that produces a result record runs inside this image. Nothing is installed
# onto the VM host. See DECISIONS.md ADR-0006.
#
# The image is CPU-first: this workload is sparse linear algebra and RAM bound. GPU
# acceleration is only relevant to the Wright-Fisher baseline and is added as a separate
# optional layer rather than bloating the base image.
#
# Build (on the VM):
#   docker build -t quasar:v1 --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
#
# Run:
#   docker run --rm -v "$PWD":/work -w /work quasar:v1 python scripts/run_all_gates.py

FROM python:3.12-slim-bookworm

ARG UID=1000
ARG GID=1000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# OMP/BLAS thread counts are pinned to 1 on purpose. The grid sweep parallelises across
# cells with an explicit worker count that is part of the declared compute budget; letting
# BLAS spawn its own threads underneath would make the budget protocol meaningless. A run
# that genuinely wants threaded BLAS overrides these at container start and records it.

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/quasar

COPY requirements.in ./requirements.in
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.in

# Record exactly what the resolver chose. This file is copied out of the image and
# committed as environment.lock.txt, so the image and the repository agree.
RUN pip freeze > /opt/quasar/environment.lock.txt

RUN groupadd -g ${GID} quasar 2>/dev/null || true \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash quasar 2>/dev/null || true

USER ${UID}:${GID}
WORKDIR /work

CMD ["python", "-c", "import qiskit, numpy, scipy; print('quasar image ready:', qiskit.__version__, numpy.__version__, scipy.__version__)"]
