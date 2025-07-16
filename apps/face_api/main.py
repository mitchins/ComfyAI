from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from .face_model import get_embedding
from .utils import cosine_similarity

app = FastAPI()


@app.post("/v1/image/compare_faces")
async def compare_faces(image_a: UploadFile = File(...), image_b: UploadFile = File(...)):
    data_a = await image_a.read()
    data_b = await image_b.read()

    emb_a = get_embedding(data_a)
    emb_b = get_embedding(data_b)

    if emb_a is None or emb_b is None:
        return JSONResponse(status_code=422, content={"error": "face_not_detected"})

    similarity = cosine_similarity(emb_a, emb_b)
    return {"similarity": round(float(similarity), 4)}


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))


if __name__ == "__main__":
    main()
