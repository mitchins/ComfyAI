from dataclasses import dataclass
from typing import Optional

@dataclass
class TaskData:
    image_bytes: Optional[bytes]
    reference_bytes: Optional[bytes]
    text_query: str
    is_retried: bool = False  # Default retry flag
