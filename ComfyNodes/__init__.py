from importlib import import_module
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root.parent))
package_name = root.name
module = import_module(package_name)
globals().update(module.__dict__)

