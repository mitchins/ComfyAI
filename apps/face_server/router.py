from fastapi import APIRouter
from apps.face_api.main import app as face_app

router = APIRouter()
router.include_router(face_app.router)
