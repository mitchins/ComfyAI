from fastapi import APIRouter
from apps.onnx_chat.main import app as _app

router = APIRouter()

@router.get("/")
async def read_chat_root():
    return {"message": "Chat server is running"}

router.include_router(_app.router)