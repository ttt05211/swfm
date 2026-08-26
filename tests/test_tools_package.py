from pathlib import Path
import tools
import tools.real_motion


def test_swfm_tools_package_resolves_to_repo():
    repo=Path(__file__).resolve().parents[1]
    assert Path(tools.__file__).resolve()==repo/'tools'/'__init__.py'
    assert Path(tools.real_motion.__file__).resolve()==repo/'tools'/'real_motion'/'__init__.py'
