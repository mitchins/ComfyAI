# ComfyAI Deployment Guide

This guide covers deploying the unified ComfyAI server that provides vision models, face comparison, and model management in a single container.

## Quick Start

### Using the Deployment Script (Recommended)

```bash
# Build and run the server
./deploy.sh build
./deploy.sh run

# Access the services
# 🏠 Homepage: http://localhost:8000/
# 🌐 Management UI: http://localhost:8000/ui
# 🎯 Vision Test: http://localhost:8000/test
# 🔗 API Docs: http://localhost:8000/docs

# View logs
./deploy.sh logs

# Stop the server
./deploy.sh stop
```

### Using Docker Compose Directly

```bash
# Build and start
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Using Docker Directly

```bash
# Build
docker build -t comfyai:latest .

# Run
docker run -p 8000:8000 -v comfyai_models:/app/.cache/huggingface comfyai:latest
```

## Available Services

Once deployed, the following services are available:

### Web Interfaces
- **Homepage**: http://localhost:8000/ *(Easy navigation to all UIs)*
- **Model Management UI**: http://localhost:8000/ui *(or /manage/ui/)*
- **Vision Test Interface**: http://localhost:8000/test *(or /manage/ui/vision-test.html)*
- **API Documentation**: http://localhost:8000/docs

### API Endpoints
- **Face Comparison**: `POST /v1/image/compare_faces` *(Key differentiator - no standard solution exists)*
- **Vision/Chat API**: `POST /v1/chat/completions` *(OpenAI-compatible with image support)*
- **Model Management**: `GET /v1/models`, `POST /v1/models/download`
- **Health Check**: `GET /health`

**Note**: Face comparison is the primary reason to deploy this server, as no standard solution exists. ONNX vision models provide additional value-add inference capabilities.

## Supported Models

The server supports these vision-language models:

### Ultra-Lightweight (CPU-friendly)
- **SmolVLM-256M-Instruct**: Only 256M parameters, excellent for CPU inference
- **Smallest quantization**: ~200MB download

### High-Quality Vision Models
- **Qwen2-VL-2B-Instruct**: High-quality vision understanding
- **Gemma-3n-E2B-it-ONNX**: Multimodal with audio support
- **Phi-3.5-vision-instruct**: Microsoft's vision model

All models support multiple quantization levels (UINT8, INT8, Q4, FP16, FP32).

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HOME` | `/app/.cache/huggingface` | Model cache directory |
| `USE_MOCK_ONNX` | `false` | Use mock models for testing |
| `PYTHONUNBUFFERED` | `1` | Ensure logs appear in real-time |

### Model Cache

Models are automatically downloaded and cached on first use. The cache is persisted using Docker volumes:

```bash
# View cache usage
docker exec -it comfyai_comfyai_1 du -sh /app/.cache/huggingface

# Backup cache
docker cp comfyai_comfyai_1:/app/.cache/huggingface ./model_backup
```

## Development

### Local Development

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run with auto-reload
uvicorn apps.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ integration_tests/
```

### Development Mode with Docker

```bash
# Run with live code reloading
./deploy.sh dev
```

This mounts your local code directory and enables auto-reload for development.

## Production Deployment

### Resource Requirements

**Minimum (CPU-only with SmolVLM)**:
- CPU: 2 cores
- RAM: 4GB
- Disk: 10GB (for model cache)

**Recommended (GPU with larger models)**:
- CPU: 4+ cores  
- RAM: 8GB+
- GPU: 4GB+ VRAM
- Disk: 50GB+ (for multiple model quantizations)

### Performance Tuning

#### CPU Optimization
```bash
# Use CPU-optimized ONNX runtime
docker run -e OMP_NUM_THREADS=4 -p 8000:8000 comfyai:latest
```

#### GPU Support
```bash
# Build with GPU support (requires nvidia-docker)
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

### Load Balancing

For high-traffic deployments, run multiple instances:

```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  comfyai:
    scale: 3
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

## Monitoring

### Health Checks

The container includes built-in health checks:

```bash
# Check service health
curl http://localhost:8000/health

# Docker health status
docker ps --format "table {{.Names}}\t{{.Status}}"
```

### Logs

```bash
# Real-time logs
./deploy.sh logs

# Search logs
docker-compose logs | grep ERROR

# Export logs
docker-compose logs > comfyai.log
```

### Metrics

Monitor key metrics:

```bash
# Memory usage
docker stats comfyai_comfyai_1

# Model cache size
docker exec comfyai_comfyai_1 du -sh /app/.cache/huggingface

# API response times
curl -w "%{time_total}" http://localhost:8000/health
```

## Troubleshooting

### Common Issues

**1. Out of Memory**
```bash
# Use smaller models or reduce batch size
# Switch to SmolVLM-256M-Instruct for CPU deployment
```

**2. Model Download Failures**
```bash
# Check internet connection and HuggingFace access
docker exec -it comfyai_comfyai_1 ping huggingface.co

# Clear model cache
docker-compose down -v
```

**3. Slow Inference**
```bash
# Use smaller quantizations (UINT8, INT8)
# Enable GPU support
# Increase container memory limits
```

### Debug Mode

```bash
# Run with debug logging
docker run -e LOG_LEVEL=DEBUG -p 8000:8000 comfyai:latest

# Interactive shell
docker exec -it comfyai_comfyai_1 /bin/bash
```

## Security

### Network Security
- Use reverse proxy (nginx) for production
- Enable HTTPS/TLS termination
- Restrict access to management endpoints

### Model Security
- Models are downloaded from trusted sources (HuggingFace)
- Checksums are verified automatically
- Use read-only model cache volumes in production

## Backup and Recovery

### Backup Model Cache
```bash
# Create backup
docker run --rm -v comfyai_model_cache:/data -v $(pwd):/backup alpine tar czf /backup/models-backup.tar.gz -C /data .

# Restore backup
docker run --rm -v comfyai_model_cache:/data -v $(pwd):/backup alpine tar xzf /backup/models-backup.tar.gz -C /data
```

### Configuration Backup
```bash
# Backup entire deployment
tar czf comfyai-deployment-backup.tar.gz docker-compose.yml .env deploy.sh
```

## Support

For issues and questions:
- Check the logs: `./deploy.sh logs`
- Review API docs: http://localhost:8000/docs
- Test with the UI: http://localhost:8000/manage/ui/vision-test.html