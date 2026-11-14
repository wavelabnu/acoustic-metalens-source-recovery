"""Make `import recovery` work when running pytest from this folder without
an editable install (mirrors focusing/conftest.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
