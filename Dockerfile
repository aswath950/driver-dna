# ── Stage 1: builder — install all Python dependencies ──────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build tools needed for C-extension packages (numpy, shap, xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime — lean production image ────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Runtime system libraries for numpy / xgboost / shap
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ src/
COPY streamlit_app.py .
COPY .streamlit/config.toml .streamlit/config.toml

# Copy pre-generated model artefacts and data
# (These are committed on the deploy branch — see README for details)
COPY models/ models/
COPY data/dataset.parquet data/dataset.parquet
COPY data/dataset_meta.json data/dataset_meta.json
COPY data/circuits.json data/circuits.json

# FastF1 writes its HTTP cache here; mount a volume in production
ENV FASTF1_CACHE_DIR=/tmp/fastf1_cache

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
