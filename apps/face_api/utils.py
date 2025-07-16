import numpy as np


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two numpy vectors."""
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
