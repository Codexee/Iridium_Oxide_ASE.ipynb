import pytest

@pytest.mark.slow
def test_xtb_energy_h2():
  from ase import Atoms
  from xtb.ase.calculator import XTB

  atoms = Atoms("H2", positions=[[0,0,0],[0,0,0]])
  atoms.calc = XTB(method="GFN2-xTB")
  e=atoms.get_potential_energy()
  assert e == e
