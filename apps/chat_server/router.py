from fastapi import APIRouter, Request
from apps.onnx_chat import main as chat_main

router = APIRouter()

router.add_api_route("/health", chat_main.health_check, methods=["GET"])
router.add_api_route("/v1/chat/completions", chat_main.chat, methods=["POST"])
