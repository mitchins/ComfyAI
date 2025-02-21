from collections import namedtuple

from dataclasses import dataclass

@dataclass
class TaskData:
    image_bytes: bytes
    reference_bytes: bytes
    text_query: str
    is_retried: bool = False  # ✅ Default retry flag