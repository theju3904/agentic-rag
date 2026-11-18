# Pulled via the mirror.gcr.io pull-through cache; see docker-compose.yml for why.
FROM mirror.gcr.io/library/python:3.12-slim

WORKDIR /app

# build-essential is required to compile a couple of transitive C-extension deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# torch is installed from the official CPU-only wheel index *first*, deliberately,
# before the generic `pip install .`: this project never uses a GPU (see README's
# "What's simplified" section), but the default PyPI torch wheel on Linux pulls in
# several GB of transitive `nvidia-*` CUDA runtime packages it would never use.
# Satisfying `torch` from the CPU index up front means the later `pip install .`
# sees the requirement already met and skips the CUDA build entirely.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .

# The port comes from API_CONTAINER_PORT in .env; docker-compose.yml passes it as a build arg.
ARG API_CONTAINER_PORT
ENV API_CONTAINER_PORT=${API_CONTAINER_PORT}

EXPOSE ${API_CONTAINER_PORT}

CMD ["sh", "-c", "exec uvicorn agentic_rag.api.main:create_app --factory --host 0.0.0.0 --port ${API_CONTAINER_PORT}"]
