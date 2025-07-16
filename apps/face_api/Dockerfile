FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --prefix=/install --no-cache-dir .

FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

ARG DETECTOR_MODEL=deepghs/real_face_detection
ARG DETECTOR_FILE=face_detect_v1.4_s/model.onnx
ARG EMBEDDER_MODEL_PATH=Xenova/clip-vit-base-patch32
ARG EMBEDDER_FILE=model.onnx
ENV DETECTOR_MODEL=$DETECTOR_MODEL \
    DETECTOR_FILE=$DETECTOR_FILE \
    EMBEDDER_MODEL_PATH=$EMBEDDER_MODEL_PATH \
    EMBEDDER_FILE=$EMBEDDER_FILE
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
