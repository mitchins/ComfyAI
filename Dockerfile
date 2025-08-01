# ComfyAI Unified Server Docker Image
# Provides vision models, face comparison, and model management in one container

FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies needed for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt requirements-*.txt ./
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# Copy source code and install the package
COPY . .
RUN pip install --prefix=/install --no-cache-dir .

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create cache directory for models
RUN mkdir -p /app/.cache/huggingface

# Expose the default port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the unified server
CMD ["uvicorn", "apps.main:app", "--host", "0.0.0.0", "--port", "8000"]
