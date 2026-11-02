"""Every command quoted in the Markdown docs must point at something real.

``python -m X`` must name a module that runs from the repository root:
it is resolved in a fresh interpreter (``sys.executable``, cwd = repo root,
``PYTHONPATH`` removed), so test-only ``sys.path`` changes in conftest do
not count, and a package must also have ``X.__main__``.  ``python path.py``
must name a file that exists (relative to the root or to the Markdown
file's own directory).  ``python`` may be written ``python3`` or
``python3.N``.  This catches docs left behind when modules move, e.g.
``validation.`` -> ``research.``.

Not checked: arguments and flags, whether the module is the right one,
``python -m \\`` continued on the next line, the ``py`` launcher, and
absolute script paths.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = r"python(?:3(?:\.\d+)?)?"
MODULE_CMD = re.compile(PYTHON + r"\s+-m\s+([A-Za-z_][\w.]*)")
SCRIPT_CMD = re.compile(PYTHON + r"\s+((?:[\w.-]+/)*[\w.-]+\.py)\b")
SKIP_DIRS = {".git", ".venv", "venv", "build", "dist", "node_modules"}

# Run in a child interpreter; exit 0 only if `python -m <argv[1]>` could start.
_PROBE = """\
import importlib.util, sys
name = sys.argv[1]
try:
    spec = importlib.util.find_spec(name)
    if spec is not None and spec.submodule_search_locations is not None:
        spec = importlib.util.find_spec(name + ".__main__")
except (ImportError, ValueError):
    spec = None
sys.exit(0 if spec is not None else 1)
"""


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


def _module_runs_from_root(target):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, target],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.returncode == 0


@pytest.mark.parametrize("kind,target,doc_dir", list(_doc_commands()))
def test_documented_command_target_exists(kind, target, doc_dir):
    if kind == "script":
        assert (ROOT / target).is_file() or (doc_dir / target).is_file(), (
            f"documented script {target} does not exist"
        )
        return
    assert _module_runs_from_root(target), (
        f"documented module {target} cannot be run with `python -m` from the repo root"
    )


def test_minor_version_interpreter_is_scanned():
    for cmd in ("python3.11 -m pkg.mod", "/opt/homebrew/bin/python3.11 -m pkg.mod", "python3.13 -m pkg.mod"):
        assert [m.group(1) for m in MODULE_CMD.finditer(cmd)] == ["pkg.mod"], cmd
    # observation_protocol.md:167 writes `/opt/homebrew/bin/python3.11 -m ...`;
    # it must produce a case like the plain `python -m` lines do.
    ids = [p.id for p in _doc_commands()]
    assert (
        "research/mechanism_research_20261030/observation_protocol.md:167:"
        "research.mechanism_research_20261030.observation_intake"
    ) in ids
