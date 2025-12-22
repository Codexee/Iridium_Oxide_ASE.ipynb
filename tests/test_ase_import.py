def test_ase_import_and_build():
    from ase import Atoms
    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    assert len(a) == 2
