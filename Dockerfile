# =============================================================================
# Dockerfile — Churn Intelligence MCP Server
# =============================================================================
# Build:  docker build -t churn-mcp .
# Run:    docker run -p 8000:8000 churn-mcp
#
# Environment variables:
#   MCP_PORT             (default: 8000)
#   MCP_HOST             (default: 0.0.0.0)
#   MLFLOW_TRACKING_URI  (default: sqlite:///tracking/mlflow.db)
#   LOG_LEVEL            (default: INFO)
# =============================================================================

FROM python:3.12-slim

# Install uv — fast Python package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first (Docker layer caching)
# This means deps are re-installed ONLY when pyproject.toml changes,
# not on every code change.
COPY pyproject.toml uv.lock ./

# Install dependencies into the system Python (no venv needed in container)
RUN uv sync --frozen --no-dev

# Copy application code and config
COPY params.yaml main.py ./
COPY src/ ./src/

# Copy model artifacts and data needed at runtime
COPY models/ ./models/
COPY data/raw/ ./data/raw/

# Create remaining directories for generated artifacts
RUN mkdir -p tracking data/processed

# Expose default port (Heroku overrides via PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "from src.interfaces.mcp.server import mcp; print('ok')" || exit 1

# Start MCP server — reads PORT env var (set by Heroku) or falls back to 8000
CMD ["uv", "run", "python", "main.py", "serve"]
