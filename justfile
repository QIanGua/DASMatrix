# DASMatrix Justfile

# List available recipes
default:
    @just --list

# === Development Tasks ===

# Run unit tests
test:
    uv run pytest tests

# Run quick checks (for pre-push hooks - no fixes)
quick-check:
    @echo "🔍 Running quick validation checks..."
    uv run ruff check . --diff
    uvx ty check

# Run linter check only (no fixes)
lint:
    uv run ruff check .

# Run linter with fixes
lint-fix:
    uv run ruff check --fix .

# Format code
format:
    uv run ruff format .

# Check types (mypy)
typecheck:
    uvx ty check

# === Comprehensive Tasks ===

# Run all checks with fixes (for CI/manual validation)
check-all:
    @echo "🔧 Running comprehensive checks with fixes..."
    just lint-fix
    just format
    just typecheck
    just test

# Quick format and lint (for development)
fix-all:
    @echo "✨ Auto-fixing code issues..."
    just lint-fix
    just format

# === Build & Deploy ===

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
