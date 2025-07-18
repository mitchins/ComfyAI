from fastapi import FastAPI
from starlette.testclient import TestClient
from apps.chat_server.router import router as chat_router
from apps.face_server.router import router as face_router

def test_chat_server_router():
    app = FastAPI()
    app.include_router(chat_router, prefix="/chat")
    client = TestClient(app)
    response = client.get("/chat")
    assert response.status_code == 200
    assert response.json() == {"message": "Chat server is running"}

def test_face_server_router():
    app = FastAPI()
    app.include_router(face_router, prefix="/face")
    client = TestClient(app)
    response = client.get("/face")
    assert response.status_code == 200
    assert response.json() == {"message": "Face server is running"}
