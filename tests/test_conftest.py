"""Sanity check of the loader used by all the other test files."""

from conftest import SET3, load_exercise


def test_load_exercise_imports_script_by_path():
    module = load_exercise(SET3 / "exercise3_1" / "ex3_1.py")
    assert hasattr(module, "knn_classify")
    # loading twice returns the same module object
    assert load_exercise(SET3 / "exercise3_1" / "ex3_1.py") is module
