PRESETS = {
    "photo": {
        "detector_repo": "deepghs/real_face_detection",
        "detector_file": "face_detect_v1.4_s/model.onnx",
        "embedder_repo": "openailab/onnx-arcface-resnet100-ms1m",
        "embedder_file": "model.onnx",
        "threshold": 0.446,
    },
    "anime": {
        "detector_repo": "deepghs/anime_face_detection",
        "detector_file": "face_detect_v1.4_s/model.onnx",
        "embedder_repo": "Xenova/clip-vit-base-patch32",
        "embedder_file": "onnx/vision_model.onnx",
        "threshold": 0.307,
    },
    "cg": {
        "detector_repo": "deepghs/real_face_detection",
        "detector_file": "face_detect_v1.4_n/model.onnx",
        "embedder_repo": "Xenova/clip-vit-base-patch32",
        "embedder_file": "onnx/vision_model.onnx",
        "threshold": 0.278,
    },
}
