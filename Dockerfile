# quasar execution image.
#
# Build:
#   docker build -t quasar:v1 --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
#
# Run:
#   docker run --rm -v "$PWD":/work -w /work quasar:v1 python scripts/run_all_gates.py

FROM python:3.12-slim-bookworm

ARG UID=1000
ARG GID=1000

# BLAS and OpenMP are single-threaded so that per-instance wall-clock time reflects one core.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/quasar

COPY requirements.in ./requirements.in
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.in

RUN pip freeze > /opt/quasar/environment.lock.txt

RUN groupadd -g ${GID} quasar 2>/dev/null || true \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash quasar 2>/dev/null || true

USER ${UID}:${GID}
WORKDIR /work

CMD ["python", "-c", "import qiskit, numpy, scipy; print('quasar image ready:', qiskit.__version__, numpy.__version__, scipy.__version__)"]
