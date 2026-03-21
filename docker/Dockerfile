FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install langchain-adk with all extras
RUN uv pip install --system --no-cache \
    langchain-adk[composer,openai,anthropic,google,a2a,mcp] \
    uvicorn

# Generic entrypoint that loads compose.yaml from /app
COPY entrypoint.py /app/entrypoint.py

EXPOSE 8080

# Users mount their compose.yaml (and optional tools.py) to /app
# docker run -v ./compose.yaml:/app/compose.yaml -e OPENAI_API_KEY=$OPENAI_API_KEY nicolaimtlassen/langchain-adk
CMD ["python", "entrypoint.py"]
