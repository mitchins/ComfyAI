import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from apps.face_api.main import (
    ModelLoader,
    preprocess_for_detection,
    preprocess_for_embedding,
    extract_face_region,
    cosine_similarity,
    app,
    lifespan,
    get_embedding
)
from apps.face_api.config import Config as FaceApiConfig
from PIL import Image
import io

# Mock config for tests
@pytest.fixture(autouse=True)
def mock_config():
    with patch('apps.face_api.main.config') as mock_cfg:
        mock_cfg.detector_model = "test_detector_model"
        mock_cfg.detector_file = "test_detector_file"
        mock_cfg.embedder_model_path = "test_embedder_model_path"
        mock_cfg.embedder_file = "test_embedder_file"
        mock_cfg.threshold = 0.5
        mock_cfg.preload_models = False
        yield

# Test ModelLoader
def test_model_loader_get_providers(monkeypatch):
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    monkeypatch.setattr("apps.face_api.main.ort", mock_ort)
    loader = ModelLoader()
    providers = loader._get_providers()
    assert "CUDAExecutionProvider" in providers
    assert "CPUExecutionProvider" in providers

def test_model_loader_get_providers_no_cuda(monkeypatch):
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
    monkeypatch.setattr("apps.face_api.main.ort", mock_ort)
    loader = ModelLoader()
    providers = loader._get_providers()
    assert "CUDAExecutionProvider" not in providers
    assert "CPUExecutionProvider" in providers

def test_model_loader_load_detector_success(monkeypatch):
    mock_ort = MagicMock()
    mock_ort.InferenceSession.return_value = MagicMock()
    monkeypatch.setattr("apps.face_api.main.ort", mock_ort)
    monkeypatch.setattr("apps.face_api.main.download_model", MagicMock(return_value="/mock/path/detector.onnx"))
    loader = ModelLoader()
    detector = loader.load_detector()
    assert detector is not None
    mock_ort.InferenceSession.assert_called_once_with("/mock/path/detector.onnx", providers=loader._get_providers())

def test_model_loader_load_detector_no_ort(monkeypatch):
    monkeypatch.setattr("apps.face_api.main.ort", None)
    monkeypatch.setattr("apps.face_api.main.download_model", MagicMock())
    loader = ModelLoader()
    with pytest.raises(RuntimeError, match="huggingface_hub not available"):
        loader.load_detector()

def test_model_loader_load_detector_failure(monkeypatch):
    monkeypatch.setattr("apps.face_api.main.download_model", MagicMock(side_effect=Exception("Download failed")))
    loader = ModelLoader()
    with pytest.raises(Exception, match="Download failed"):
        loader.load_detector()

def test_model_loader_load_embedder_success(monkeypatch):
    mock_ort = MagicMock()
    mock_ort.InferenceSession.return_value = MagicMock()
    monkeypatch.setattr("apps.face_api.main.ort", mock_ort)
    monkeypatch.setattr("apps.face_api.main.download_model", MagicMock(return_value="/mock/path/embedder.onnx"))
    loader = ModelLoader()
    embedder = loader.load_embedder()
    assert embedder is not None
    mock_ort.InferenceSession.assert_called_once_with("/mock/path/embedder.onnx", providers=loader._get_providers())

def test_model_loader_load_embedder_no_ort(monkeypatch):
    monkeypatch.setattr("apps.face_api.main.ort", None)
    monkeypatch.setattr("apps.face_api.main.download_model", MagicMock())
    loader = ModelLoader()
    with pytest.raises(RuntimeError, match="huggingface_hub not available"):
        loader.load_embedder()

def test_model_loader_load_embedder_failure(monkeypatch):
    monkeypatch.setattr("apps.face_api.main.download_model", MagicMock(side_effect=Exception("Download failed")))
    loader = ModelLoader()
    with pytest.raises(Exception, match="Download failed"):
        loader.load_embedder()

# Test Preprocessing Functions
def test_preprocess_for_detection():
    image = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    processed_image = preprocess_for_detection(image)
    assert processed_image.shape == (1, 3, 640, 640)
    assert processed_image.dtype == np.float32

def test_preprocess_for_embedding():
    image = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    processed_image = preprocess_for_embedding(image)
    assert processed_image.shape == (1, 3, 224, 224)
    assert processed_image.dtype == np.float32

def test_extract_face_region():
    image = np.random.randint(0, 255, size=(200, 200, 3), dtype=np.uint8)
    bbox = (50, 50, 150, 150)
    face_region = extract_face_region(image, bbox)
    assert face_region.shape == (100, 100, 3)

def test_extract_face_region_out_of_bounds():
    image = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    bbox = (-10, -10, 120, 120)
    face_region = extract_face_region(image, bbox)
    assert face_region.shape == (100, 100, 3)

def test_extract_face_region_empty():
    image = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    bbox = (10, 10, 5, 5) # Invalid bbox
    face_region = extract_face_region(image, bbox)
    assert face_region.size == 0

# Test Cosine Similarity
def test_cosine_similarity():
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    vec3 = np.array([1, 0, 0])
    assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)
    assert cosine_similarity(vec1, vec3) == pytest.approx(1.0)

# Test API Endpoints
def test_health_check(client_with_mock_config, mock_config_fixture):
    response = client_with_mock_config.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": f"{mock_config_fixture.detector_model}/{mock_config_fixture.detector_file}"}

def test_models_info(client_with_mock_config, mock_config_fixture):
    response = client_with_mock_config.get("/models/info")
    assert response.status_code == 200
    assert response.json() == {
        "detector_model": f"{mock_config_fixture.detector_model}/{mock_config_fixture.detector_file}",
        "embedder_model": f"{mock_config_fixture.embedder_model_path}/{mock_config_fixture.embedder_file}",
        "detector_loaded": False,
        "embedder_loaded": False,
    }

# Test get_embedding
def test_get_embedding_no_faces(monkeypatch):
    monkeypatch.setattr("apps.face_api.main.detect_faces", MagicMock(return_value=[]))
    image_bytes = Image.new('RGB', (100, 100)).tobytes()
    embedding = get_embedding(image_bytes)
    assert embedding is None

def test_get_embedding_empty_face_region(monkeypatch):
    monkeypatch.setattr("apps.face_api.main.detect_faces", MagicMock(return_value=[((10, 10, 5, 5), 0.9, 'face')]))
    monkeypatch.setattr("apps.face_api.main.extract_face_region", MagicMock(return_value=np.array([])))
    image_bytes = Image.new('RGB', (100, 100)).tobytes()
    embedding = get_embedding(image_bytes)
    assert embedding is None

def test_get_embedding_embedder_none(monkeypatch):
    monkeypatch.setattr("apps.face_api.main.detect_faces", MagicMock(return_value=[((10, 10, 20, 20), 0.9, 'face')]))
    monkeypatch.setattr("apps.face_api.main.extract_face_region", MagicMock(return_value=np.zeros((10,10,3))))
    mock_model_loader = MagicMock()
    mock_model_loader.load_embedder.return_value = None
    monkeypatch.setattr("apps.face_api.main.model_loader", mock_model_loader)
    image_bytes = Image.new('RGB', (100, 100)).tobytes()
    embedding = get_embedding(image_bytes)
    assert embedding is None

def test_get_embedding_success(monkeypatch):
    mock_face_bbox = (10, 10, 20, 20)
    mock_face_region = np.zeros((10, 10, 3))
    mock_embedding_output = np.array([0.1, 0.2, 0.3])

    # Create a dummy image for testing
    dummy_image = Image.new('RGB', (100, 100), color = 'red')
    dummy_image_bytes = io.BytesIO()
    dummy_image.save(dummy_image_bytes, format='PNG')
    dummy_image_bytes.seek(0)
    monkeypatch.setattr("PIL.Image.open", MagicMock(return_value=dummy_image))

    monkeypatch.setattr("apps.face_api.main.detect_faces", MagicMock(return_value=[(mock_face_bbox, 0.9, 'face')]))
    monkeypatch.setattr("apps.face_api.main.extract_face_region", MagicMock(return_value=mock_face_region))
    monkeypatch.setattr("apps.face_api.main.preprocess_for_embedding", MagicMock(return_value=np.zeros((1,3,224,224))))

    mock_embedder_session = MagicMock()
    mock_embedder_session.get_inputs.return_value = [MagicMock(name="input")]
    mock_embedder_session.run.return_value = [mock_embedding_output]

    mock_model_loader = MagicMock()
    mock_model_loader.load_embedder.return_value = mock_embedder_session
    monkeypatch.setattr("apps.face_api.main.model_loader", mock_model_loader)

    image_bytes = Image.new('RGB', (100, 100)).tobytes()
    embedding = get_embedding(image_bytes)
    assert embedding is not None
    assert np.isclose(np.linalg.norm(embedding), 1.0)

def test_get_embedding_exception(monkeypatch):
    monkeypatch.setattr("apps.face_api.main.Image.open", MagicMock(side_effect=Exception("Image error")))
    image_bytes = b"invalid_image_data"
    embedding = get_embedding(image_bytes)
    assert embedding is None

# Test compare_faces endpoint
@pytest.mark.asyncio
async def test_compare_faces_success(monkeypatch):
    mock_embedding_a = np.array([1.0, 0.0, 0.0])
    mock_embedding_b = np.array([1.0, 0.0, 0.0])

    monkeypatch.setattr("apps.face_api.main.get_embedding", MagicMock(side_effect=[mock_embedding_a, mock_embedding_b]))
    monkeypatch.setattr("apps.face_api.main.cosine_similarity", MagicMock(return_value=0.99))

    client = TestClient(app)
    image_a_bytes = Image.new('RGB', (100, 100)).tobytes()
    image_b_bytes = Image.new('RGB', (100, 100)).tobytes()

    response = client.post(
        "/v1/image/compare_faces",
        files={
            "image_a": ("a.jpg", image_a_bytes, "image/jpeg"),
            "image_b": ("b.jpg", image_b_bytes, "image/jpeg"),
        },
    )
    assert response.status_code == 200
    assert response.json() == {"similarity": 0.99}

@pytest.mark.asyncio
async def test_compare_faces_no_face_detected(monkeypatch):
    monkeypatch.setattr("apps.face_api.main.get_embedding", MagicMock(side_effect=[None, np.array([1.0, 0.0, 0.0])]))

    client = TestClient(app)
    image_a_bytes = Image.new('RGB', (100, 100)).tobytes()
    image_b_bytes = Image.new('RGB', (100, 100)).tobytes()

    response = client.post(
        "/v1/image/compare_faces",
        files={
            "image_a": ("a.jpg", image_a_bytes, "image/jpeg"),
            "image_b": ("b.jpg", image_b_bytes, "image/jpeg"),
        },
    )
    assert response.status_code == 422
    assert response.json() == {"error": "face_not_detected", "message": "Could not detect face in one or both images"}

@pytest.mark.asyncio
async def test_compare_faces_internal_error(monkeypatch):
    monkeypatch.setattr("apps.face_api.main.get_embedding", MagicMock(side_effect=Exception("Test error")))

    client = TestClient(app)
    image_a_bytes = Image.new('RGB', (100, 100)).tobytes()
    image_b_bytes = Image.new('RGB', (100, 100)).tobytes()

    response = client.post(
        "/v1/image/compare_faces",
        files={
            "image_a": ("a.jpg", image_a_bytes, "image/jpeg"),
            "image_b": ("b.jpg", image_b_bytes, "image/jpeg"),
        },
    )
    assert response.status_code == 500
    assert response.json() == {"error": "internal_error", "message": "Test error"}

# Test lifespan context manager
@pytest.mark.asyncio
async def test_lifespan_preload_models_success(monkeypatch):
    mock_model_loader = MagicMock()
    monkeypatch.setattr("apps.face_api.main.model_loader", mock_model_loader)
    monkeypatch.setattr("apps.face_api.main.PRELOAD_MODELS", True)

    async with lifespan(app):
        mock_model_loader.load_detector.assert_called_once()
        mock_model_loader.load_embedder.assert_called_once()

@pytest.mark.asyncio
async def test_lifespan_preload_models_failure(monkeypatch):
    mock_model_loader = MagicMock()
    mock_model_loader.load_detector.side_effect = Exception("Preload failed")
    monkeypatch.setattr("apps.face_api.main.model_loader", mock_model_loader)
    monkeypatch.setattr("apps.face_api.main.PRELOAD_MODELS", True)

    with pytest.raises(Exception, match="Preload failed"):
        async with lifespan(app):
            pass

@pytest.mark.asyncio
async def test_lifespan_preload_models_disabled(monkeypatch):
    mock_model_loader = MagicMock()
    monkeypatch.setattr("apps.face_api.main.model_loader", mock_model_loader)
    monkeypatch.setattr("apps.face_api.main.PRELOAD_MODELS", False)

    async with lifespan(app):
        mock_model_loader.load_detector.assert_not_called()
        mock_model_loader.load_embedder.assert_not_called()
