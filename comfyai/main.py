from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from apps.chat_server.router import router as chat_router
from apps.face_server.router import router as face_router
from apps.manage_api.router import router as manage_router
from common.logging import setup_logging

setup_logging()
app = FastAPI(title="ComfyAI Master API")

app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(face_router, prefix="/face", tags=["face"])
app.include_router(manage_router, prefix="/manage", tags=["manage"])
app.mount("/manage/ui", StaticFiles(directory="static/manage"), name="manage_ui")
