from fastapi import APIRouter
from apps.face_api import main as face_main

router = APIRouter()

router.add_api_route(
    "/v1/image/compare_faces",
    face_main.compare_faces,
    methods=["POST"],
)
router.add_api_route("/health", face_main.health_check, methods=["GET"])
router.add_api_route("/models/info", face_main.models_info, methods=["GET"])
