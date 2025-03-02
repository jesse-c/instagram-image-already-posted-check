FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

WORKDIR /app

ADD pyproject.toml uv.lock /app

RUN uv sync --frozen --no-install-project --no-dev

ADD . /app

RUN uv sync --frozen --no-dev

# Then, use a final image without uv
FROM python:3.12-slim-bookworm

# Copy the application from the builder
COPY --from=builder --chown=app:app /app /app

WORKDIR /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Download the model
RUN python -c "import torch; import torchvision.models as models; model = models.resnet50(pretrained=True); torch.save(model.state_dict(), 'resnet50_model.pth')"

# Run app.py when the container launches
CMD ["python", "src/main.py"]
