"""Put the research harness on the import path.

The autoresearch scripts are written as flat scripts (``from model import
...``) and run as ``python3 research/autoresearch/<script>.py``.  The tests
that cover them need the same directory on the path.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESEARCH = os.path.join(ROOT, "research", "autoresearch")
for path in (ROOT, RESEARCH):
    if path not in sys.path:
        sys.path.insert(0, path)
