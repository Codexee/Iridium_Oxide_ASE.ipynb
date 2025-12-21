import shutil

def test_xtb_available():
    assert shutil.which("xtb") is not None
