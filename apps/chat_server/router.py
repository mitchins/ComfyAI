from fastapi import APIRouter
from apps.onnx_chat.main import app as chat_app

router = APIRouter()
router.include_router(chat_app.router)
