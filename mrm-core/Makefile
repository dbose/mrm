.PHONY: help install dev test format lint clean example docs

help:
	@echo "MRM Core - Development Commands"
	@echo ""
	@echo "make install    - Install package"
	@echo "make dev        - Install in development mode"
	@echo "make test       - Run tests"
	@echo "make format     - Format code with black and isort"
	@echo "make lint       - Run linters"
	@echo "make clean      - Clean build artifacts"
	@echo "make example    - Run example project"
	@echo "make docs       - Generate documentation"

install:
	pip install .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

format:
	black mrm/
	isort mrm/

lint:
	black --check mrm/
	isort --check mrm/
	mypy mrm/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf credit_risk_example/

example:
	python examples/example_usage.py

docs:
	@echo "Documentation generation not yet implemented"

build:
	python setup.py sdist bdist_wheel

publish-test:
	twine upload --repository testpypi dist/*

publish:
	twine upload dist/*
