"""Make `backend` importable when scripts are run as `python scripts/<name>.py` from the project root."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
