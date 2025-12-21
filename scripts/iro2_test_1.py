# pip install ase
# conda config --add channels conda-forge
# conda install xtb-python
# conda search xtb-python --channel conda-forge
# conda install qcelemental ase
from ase.io import read
import numpy as np
from ase.constraints import FixAtoms
from ase import Atom
from ase.visualize import view
from ase.io import write
from ase.visualize import view
from ase.data import covalent_radii
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list
from xtb.ase.calculator import XTB
from ase.optimize import BFGS

covalent_radii[1] = 0.6

slab = read("slab_clean_2x2.in", format="espresso-in")
print(slab)
print("Atoms:", len(slab))
print("Cell (Å):\n", slab.cell)
print("PBC:", slab.pbc)

# Get atomic symbols and positions
symbols = slab.get_chemical_symbols()
positions = slab.get_positions()

# Find top of slab
z_positions = positions[:, 2]
z_top = z_positions.max()

print(f"Top of slab z = {z_top:.3f} Å")

# Find surface O atoms within 1.5 Å of top
surface_O_indices = [
    i for i, s in enumerate(symbols)
    if s == "O" and (z_top - positions[i, 2]) < 1.5
]

print("Candidate surface O atoms:")
for i in surface_O_indices:
    x, y, z = positions[i]
    print(f"Index {i}: position = ({x:.3f}, {y:.3f}, {z:.3f})")

#Adding H* on top of Oxygen 20

o_idx = 20
O_pos = slab.positions[o_idx]

# Typical O–H bond length
OH_distance = 1.0  # Å

# Place H above O along +z
H_pos = O_pos + np.array([0.0, 0.0, OH_distance])

# Add H atom
slab.append(Atom("H", position=H_pos))

#for atom in slab:
    #if atom.symbol == "H":
        #atom.tag = 1
        
#set_highlight_atoms([i for i,a in enumerate(slab) if a.symbol=="H"])

print("Total atoms:", len(slab))
print("Last atom symbol:", slab[-1].symbol)
print("Last atom position:", slab[-1].position)

print(f"Added H at position {H_pos}")
print("Total atoms now:", len(slab))

write("slab_with_H.xyz", slab)

view(slab)

z = slab.positions[:, 2]
z_freeze = 20.0   # Å (tune if needed)

freeze_mask = z < z_freeze
slab.set_constraint(FixAtoms(mask=freeze_mask))

print("Frozen atoms:", int(freeze_mask.sum()), "/", len(slab))

iH = len(slab) - 1  # last atom is H
cut = 1.3           # Å check radius around H

i, j, d = neighbor_list("ijd", slab, cutoff=[cut]*len(slab))
close = [(jj, dd) for ii, jj, dd in zip(i, j, d) if ii == iH]



print("Neighbors within 1.3 Å of H:", close)

# Optimize structure with xTB
slab.calc = XTB(method="GFN2-xTB")
opt = BFGS(slab, )
