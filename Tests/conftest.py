"""
Shared pytest configuration for the test suite.

Ensures the core modules on the path are importable and pins the feature-output
location. Run the whole suite from this directory with:  pytest -v
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# core modules live beside the tests (copied in) or one level up
for p in [HERE, HERE.parent]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# default feature-output location if the caller did not set one
os.environ.setdefault("FEATURES_DIR", str(HERE.parent.parent / "feat"))
