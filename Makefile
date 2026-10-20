# Reproduce every table and figure of the revised evaluation (revision Phase 9).
#
#   make env           create venv/ from the pinned lock file (Python 3.12, CPU-only torch)
#   make test          run the test suite (~1 min)
#   make summaries     rebuild results/summary, results/figures and the LaTeX tables
#                      from the saved raw outputs (~2 min)
#   make verify        make summaries, then list any output that differs from the committed version
#   make experiments   rerun every experiment from scratch (~13 h on a 14-core i5-14500 with 15 GB RAM)
#   make all           experiments + summaries
#   make check         rerun one fold of every experiment and compare it with the saved outputs (~1 h)
#   make docker-build / docker-test / docker-check     the same inside a container (see Dockerfile)
#
# Each experiment target can also be run on its own (make main, make timeseries, ...); later
# steps read the earlier ones' raw outputs (see the dependency lines).

PY      ?= venv/bin/python
PYTHON  ?= python3.12
WORK    ?= /tmp/debias-repro-check
IMAGE   ?= debias-revision
TORCH_INDEX = --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# Fixed metadata in PDF figures, so rebuilt figures are byte-identical.
export SOURCE_DATE_EPOCH = 0
# Matplotlib without a display.
export MPLBACKEND = Agg

.PHONY: env test summaries verify experiments all check data main timeseries baselines ablation \
        synthetic detector cost docker-build docker-test docker-check

env:
	$(PYTHON) -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install $(TORCH_INDEX) -r requirements-lock.txt

test:
	$(PY) -m pytest -q

# ---- experiments (raw outputs in results/raw/) ----------------------------------------------------------
data:
	$(PY) scripts/make_data_summary.py

main:
	$(PY) run_experiments.py --config configs/main.yaml

timeseries:
	$(PY) run_experiments.py --config configs/timeseries.yaml

baselines:
	$(PY) run_experiments.py --config configs/baselines.yaml

ablation:
	$(PY) run_experiments.py --config configs/ablation.yaml

synthetic: main
	$(PY) run_experiments.py --config configs/synthetic.yaml

detector:
	$(PY) scripts/detector_validation.py --config configs/detector.yaml

cost: main baselines
	$(PY) scripts/cost_run.py --config configs/cost.yaml

experiments: data main timeseries baselines ablation synthetic detector cost

# ---- summaries, tables, figures (read only results/raw/) -----------------------------------------------
summaries:
	$(PY) scripts/make_data_summary.py
	$(PY) scripts/compare_baselines.py
	$(PY) scripts/fairness_utility.py
	$(PY) scripts/synthetic_summary.py
	$(PY) scripts/uncertainty_summary.py
	$(PY) scripts/cost_summary.py
	$(PY) scripts/make_latex_tables.py

all: experiments summaries

# Checksums of every rebuilt output before and after `make summaries` (run records, which hold
# timestamps, are left out). Works without git, e.g. in the container.
OUTPUTS = results/summary results/figures results/raw/comparison results/raw/fairness_utility
verify:
	@find $(OUTPUTS) -type f ! -name '*_run.json' | sort | xargs sha256sum > /tmp/debias-outputs-before.sha256
	$(MAKE) summaries
	@find $(OUTPUTS) -type f ! -name '*_run.json' | sort | xargs sha256sum > /tmp/debias-outputs-after.sha256
	@diff /tmp/debias-outputs-before.sha256 /tmp/debias-outputs-after.sha256 \
	    && echo "verify: all $$(wc -l < /tmp/debias-outputs-after.sha256) rebuilt outputs are identical" \
	    || (echo "verify: the files above differ"; exit 1)

check:
	$(PY) scripts/check_reproduction.py --work $(WORK)

# ---- container ---------------------------------------------------------------------------------------------
# Built from the committed files only (like a clean clone); commit first.
docker-build:
	git archive --format=tar HEAD | docker build --build-arg DEBIAS_GIT_COMMIT=$$(git rev-parse HEAD) -t $(IMAGE) -

docker-test: docker-build
	docker run --rm $(IMAGE) make test

docker-check: docker-build
	docker run --rm $(IMAGE) sh -c 'make verify && make check'
