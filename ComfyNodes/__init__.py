from pathlib import Path
import sys

# Allow importing from the repository root
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from vllm_query import PersistentInferenceWorker  # noqa: F401

