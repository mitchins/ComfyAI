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
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
