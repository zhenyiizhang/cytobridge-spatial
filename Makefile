.PHONY: test docs wheel-smoke

test:
	PYTHONDONTWRITEBYTECODE=1 pytest -q

docs:
	NUMBA_DISABLE_JIT=1 sphinx-build -W --keep-going -E -b html docs docs/_build/html

wheel-smoke:
	python scripts/smoke_installed_wheel.py
