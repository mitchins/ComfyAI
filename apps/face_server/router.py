from fastapi import APIRouter
from apps.face_api.main import app as _app

router = APIRouter()

@router.get("/")
async def read_face_root():
    return {"message": "Face server is running"}

router.include_router(_app.router)