# Reproducibility image for the revised evaluation (revision Phase 9).
# Built from the committed files only:   make docker-build   (git archive HEAD | docker build -)
#   docker run --rm debias-revision make test       # test suite
#   docker run --rm debias-revision make verify     # rebuild every table/figure, compare checksums
#   docker run --rm debias-revision make check      # rerun one fold per experiment (~1 h)
FROM python:3.12.3-slim-bookworm

# procps: pgrep, which joblib uses to stop the child processes of workers that the memory guard stops
RUN apt-get update && apt-get install -y --no-install-recommends make git procps \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements-lock.txt .
RUN python -m venv venv \
    && venv/bin/pip install --no-cache-dir --upgrade pip \
    && venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
         --extra-index-url https://pypi.org/simple -r requirements-lock.txt

# The detector's base model, at the snapshot used for the reported results, so runs work offline.
RUN venv/bin/python -c "from transformers import AutoTokenizer as T, AutoModelForSequenceClassification as M; \
r='86b5e0934494bd15c9632b12f734a8a67f723594'; T.from_pretrained('google-bert/bert-base-uncased', revision=r); \
M.from_pretrained('google-bert/bert-base-uncased', revision=r)"

COPY . .
ARG DEBIAS_GIT_COMMIT=unknown
ENV DEBIAS_GIT_COMMIT=${DEBIAS_GIT_COMMIT} HF_HUB_OFFLINE=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
CMD ["make", "test"]
