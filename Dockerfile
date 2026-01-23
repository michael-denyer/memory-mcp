FROM python:3.11-slim

WORKDIR /app

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install the package
RUN uv sync --frozen --no-dev

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash memory && \
    mkdir -p /home/memory/.memory-mcp && \
    chown -R memory:memory /home/memory/.memory-mcp

USER memory

# Default database location (can be overridden with volume mount)
ENV MEMORY_MCP_DB_PATH=/home/memory/.memory-mcp/memory.db

# Run the MCP server
ENTRYPOINT ["uv", "run", "memory-mcp"]
