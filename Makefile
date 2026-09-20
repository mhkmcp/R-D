.PHONY: test lint m0 m0-quick m0-report

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

# SPEC §12 M-0 gate: full measurement (≥1000 timed runs per cell), then the exit record.
m0:
	uv run python -m fidnn bench m0

# Smoke run: 100 timed runs per cell, report under artifacts/ (not the exit record).
m0-quick:
	uv run python -m fidnn bench m0 --quick --out artifacts/m0_quick --report artifacts/m0_quick/M0_throughput.md

# Re-derive the verdict from existing measurements (e.g. after changing an Assumption).
m0-report:
	uv run python -m fidnn bench m0 --report-only
