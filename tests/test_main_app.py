from fastapi import FastAPI
from starlette.testclient import TestClient
from apps.main import app

def test_main_app_root():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "ComfyAI API"}
