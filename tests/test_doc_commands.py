"""Every command quoted in the Markdown docs must point at something real.

``python -m X`` must name a module that resolves from the repository root,
and ``python path.py`` must name a file that exists (relative to the root or
to the Markdown file's own directory).  This catches docs left behind when
modules move, e.g. ``validation.`` -> ``research.``.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_CMD = re.compile(r"python3?\s+-m\s+([A-Za-z_][\w.]*)")
SCRIPT_CMD = re.compile(r"python3?\s+((?:[\w.-]+/)*[\w.-]+\.py)\b")
SKIP_DIRS = {".git", ".venv", "venv", "build", "dist", "node_modules"}


def _doc_commands():
    for md in sorted(ROOT.rglob("*.md")):
        rel = md.relative_to(ROOT)
        if SKIP_DIRS.intersection(rel.parts):
            continue
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for m in MODULE_CMD.finditer(line):
                yield pytest.param("module", m.group(1), md.parent, id=f"{rel.as_posix()}:{lineno}:{m.group(1)}")
            for m in SCRIPT_CMD.finditer(line):
                yield pytest.param("script", m.group(1), md.parent, id=f"{rel.as_posix()}:{lineno}:{m.group(1)}")


def _find_spec(target):
    sys.path.insert(0, str(ROOT))
    try:
        return importlib.util.find_spec(target)
    except ModuleNotFoundError:
        return None
    finally:
        sys.path.remove(str(ROOT))


@pytest.mark.parametrize("kind,target,doc_dir", list(_doc_commands()))
def test_documented_command_target_exists(kind, target, doc_dir):
    if kind == "script":
        assert (ROOT / target).is_file() or (doc_dir / target).is_file(), (
            f"documented script {target} does not exist"
        )
        return
    # Tools installed into the environment (pytest, pip, venv) only need
    # their top-level package; everything else must resolve from the repo.
    if target in {"pytest", "pip", "venv"} or target.split(".")[0] in sys.stdlib_module_names:
        spec = importlib.util.find_spec(target.split(".")[0])
    else:
        spec = _find_spec(target)
    assert spec is not None, f"documented module {target} is not importable from the repo root"
