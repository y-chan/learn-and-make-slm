import sys
from pathlib import Path


# Add project root to import path so `models` can be imported from tests.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
