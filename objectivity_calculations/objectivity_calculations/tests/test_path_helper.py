from pathlib import Path

from objectivity_calculations.misc.path_helper import get_project_root


def test_path_helper()->None:
    path = get_project_root()
    assert path.is_dir()
