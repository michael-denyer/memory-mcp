FROM python:3.11-slim@sha256:61e01f72680f86a5279ad510818ab7b2ee9f049822589d4e34e33e044360356c

WORKDIR /app

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:0.13.9@sha256:d3b601f0c7b118bc1753febf7c41cc80981df678445e0254284eecf84db79a7d /uv /usr/local/bin/uv

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
