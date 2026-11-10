"""Shared helpers for the tests of the Python versions of the exercises.

The exercise folders are not Python packages (their names contain spaces
and Greek letters), so the scripts are loaded by file path with
``load_exercise``.
"""

import importlib.util
import os
import sys
from pathlib import Path

# Figures must never open a window during the tests
os.environ.setdefault("MPLBACKEND", "Agg")

REPO = Path(__file__).resolve().parent.parent

SET1 = REPO / "1ο Project"
SET2 = REPO / "2o Project"
SET3 = REPO / "3ο Project"
SET4 = REPO / "4o Project"

_loaded = {}


def load_exercise(path):
    """Import the Python script at ``path`` (absolute, or relative to the
    repository root) and return it as a module.

    The script's own folder is put on ``sys.path`` while it is imported, so
    that it can import helper modules that live next to it. Every script is
    imported once per test session.
    """
    path = Path(path)
    if not path.is_absolute():
        path = REPO / path
    path = path.resolve()
    if path in _loaded:
        return _loaded[path]

    # unique module name, e.g. "exercise_2o_Project_exercise2_7_svm_exercise1"
    rel = path.relative_to(REPO).with_suffix("")
    name = "exercise_" + "_".join(
        "".join(c if c.isascii() and c.isalnum() else "_" for c in part)
        for part in rel.parts)

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    _loaded[path] = module
    return module
