import os

if os.getenv("UNIT_TEST_MODE") != "1":
    from .compare_faces import CompareFacesNode

    NODE_CLASS_MAPPINGS = {
        "CompareFacesNode": CompareFacesNode,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "CompareFacesNode": "Compare Faces",
    }
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

