# DASMatrix Justfile

# List available recipes
default:
    @just --list

# Run unit tests
test:
    uv run pytest tests

# Run linter (ruff)
lint:
    uv run ruff check .

# Format code (ruff)
format:
    uv run ruff format .

# Check types (mypy)
typecheck:
    uvx ty check

# Clean build artifacts
clean:
    rm -rf dist build .pytest_cache .ruff_cache site
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name "*.egg-info" -exec rm -rf {} +

# Build documentation
docs:
    uv run mkdocs build

# Serve documentation locally with hot-reload
docs-serve:
    uv run mkdocs serve

# Deploy docs to GitHub Pages
docs-deploy:
    uv run mkdocs gh-deploy

# Run all checks (lint, format check, typecheck, test)
check-all:
    just lint
    uv run ruff format --check .
    just typecheck
    just test
