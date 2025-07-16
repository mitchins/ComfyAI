FROM python:3.10-slim
WORKDIR /app
COPY . /app
# Install FastAPI, InsightFace, python-multipart, etc.
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
