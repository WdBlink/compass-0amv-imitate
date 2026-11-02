"""Make the repository root importable, so the tests run without an install.

``pytest`` alone would only put ``tests/`` on the import path; the package
``compass_0amv`` lives in the repository root.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
