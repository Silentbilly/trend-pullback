# ============================================================
# Trend Pullback Pro — Docker image
# Multi-stage build: slim production image (~200 MB)
# ============================================================

# ---- Build stage ----
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps
RUN pip install --upgrade pip hatchling

# Copy only package definition first (layer cache)
COPY pyproject.toml .
COPY src/ src/

# Install the package and all runtime deps into a prefix
RUN pip install --prefix=/install --no-cache-dir ".[dev]"

# ---- Runtime stage ----
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY src/       src/
COPY scripts/   scripts/
COPY configs/   configs/

# Create directories that the bot writes to at runtime
RUN mkdir -p state output logs data/raw data/processed

# Non-root user for security
RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

# Default command — overridden per-service in docker-compose.yml
CMD ["python", "scripts/run_live.py", "--config", "configs/live_testnet.yaml"]
