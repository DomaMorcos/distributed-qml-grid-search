# ---- Stage 1: Build dependencies in a venv ----
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: Lean runtime image ----
FROM python:3.11-slim

LABEL maintainer="Doma" \
      description="Distributed QML Grid Search – VQC Training" \
      org.opencontainers.image.source="https://github.com/doma/distributed-qml-grid-search"

# Copy only the installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

COPY train_vqc.py params.csv ./
COPY results/ ./results/

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

ENTRYPOINT ["python", "train_vqc.py"]
CMD ["--help"]
