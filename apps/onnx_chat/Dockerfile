FROM python:3.10-slim AS builder
WORKDIR /build
COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . /app

ARG DETECTOR_MODEL
ARG EMBEDDER_MODEL
ARG THRESHOLD=0.5
ARG ONNX_MODEL_PATH
ARG COMFYAI_ONNX_PORT=8000
ARG LOG_LEVEL=info

ENV DETECTOR_MODEL=${DETECTOR_MODEL}
ENV EMBEDDER_MODEL=${EMBEDDER_MODEL}
ENV THRESHOLD=${THRESHOLD}
ENV ONNX_MODEL_PATH=${ONNX_MODEL_PATH}
ENV COMFYAI_ONNX_PORT=${COMFYAI_ONNX_PORT}
ENV LOG_LEVEL=${LOG_LEVEL}

CMD ["python", "-m", "apps.onnx_chat.main"]
